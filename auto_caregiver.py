#!/usr/bin/env python3
"""
auto_caregiver.py
=================
Automated "Caregiver" curriculum that bootstraps the gradient-free
Active Inference chatbot's grammar through 1 000 cycles of supervised
conversational practice.

Design rationale
────────────────
Human infants acquire language through a *caregiver feedback loop*:
the adult talks, the infant babbles, and the adult's contingent
reactions (smiles, corrections, repetitions) sculpt the babbling
into structured speech.  This script automates that loop.

Curriculum
──────────
A small dataset of Subject-Verb-Object (SVO) conversational pairs:

    Prompt:             "how are you"
    Expected concepts:  {"i", "am", "good"}

The Caregiver feeds each prompt to the agent, then evaluates the
agent's response against two heuristic criteria:

  1. **Diversity gate:**  The response must contain ≥ 2 unique
     non-trivial words.  Babbled repetition ("good good good")
     fails this gate.

  2. **Concept overlap:**  At least one word from the expected
     concept set must appear in the response.  This is a soft
     match — we don't require an exact sentence.

If both gates pass → **Type-I reward** (massive Dirichlet count
  reinforcement for the bigram transitions in the response).

If either gate fails → **Type-II penalty** (weaken those bigram
  transitions, mirroring an MDL-style pruning signal).

After 1 000 cycles, the script logs:
  • VFE decay curve  (expect VFE to drop from ~4.5 to < 2.0)
  • Unique-word ratio per response
  • Energy / affiliation traces

All learning is count-based.  No neural networks, no gradients.

Usage
─────
    cd AI_no_gradient
    python3 auto_caregiver.py               # default 1000 cycles
    python3 auto_caregiver.py --cycles 500  # override
    python3 auto_caregiver.py --plot         # save VFE plot to results/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ── Fix import path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import Config
from environment.chatbot_env import ChatbotEnv, tokenise, detokenise
from agent.core_agent import CognitiveAgent


# ═══════════════════════════════════════════════════════════════════════
# 1.  SVO CURRICULUM DATASET
# ═══════════════════════════════════════════════════════════════════════
# Each entry is (prompt_string, frozenset_of_expected_concept_words).
# The expected set is intentionally small — we want the agent to learn
# the *transitions*, not memorise exact strings.

CURRICULUM: list[tuple[str, frozenset[str]]] = [
    # ── Greetings ─────────────────────────────────────────────────────
    ("hello",                    frozenset({"hello", "hi", "hey"})),
    ("hi there",                 frozenset({"hello", "hi", "hey"})),
    ("hey",                      frozenset({"hello", "hi", "hey"})),

    # ── State queries ─────────────────────────────────────────────────
    ("how are you",              frozenset({"i", "am", "good", "fine", "well"})),
    ("how do you feel",          frozenset({"i", "am", "feel", "good", "fine"})),
    ("are you okay",             frozenset({"i", "am", "yes", "okay", "fine"})),

    # ── Identity ──────────────────────────────────────────────────────
    ("what is your name",        frozenset({"i", "am", "my", "name", "is"})),
    ("who are you",              frozenset({"i", "am"})),

    # ── Desires / preferences ─────────────────────────────────────────
    ("what do you like",         frozenset({"i", "like", "enjoy", "love"})),
    ("what do you want",         frozenset({"i", "want", "need", "like"})),

    # ── Affirmation / negation ────────────────────────────────────────
    ("do you understand",        frozenset({"yes", "i", "understand", "no"})),
    ("is that right",            frozenset({"yes", "right", "no", "correct"})),
    ("can you help me",          frozenset({"yes", "i", "can", "help"})),

    # ── Actions ───────────────────────────────────────────────────────
    ("tell me something",        frozenset({"i", "know", "think", "something"})),
    ("say something nice",       frozenset({"you", "are", "nice", "good"})),

    # ── Farewells ─────────────────────────────────────────────────────
    ("goodbye",                  frozenset({"goodbye", "bye", "see"})),
    ("see you later",            frozenset({"goodbye", "bye", "see", "later"})),
]


# ═══════════════════════════════════════════════════════════════════════
# 2.  RESPONSE EVALUATION HEURISTICS
# ═══════════════════════════════════════════════════════════════════════

# Tokens that don't count toward "unique meaningful words".
_STOP_TOKENS: frozenset[str] = frozenset({
    ".", ",", "!", "?", ":", ";", "-", "'", '"',
    "a", "an", "the",
    "<BOS>", "<EOS>", "<UNK>",
})

def _meaningful_unique(tokens: list[str]) -> set[str]:
    """Return the set of non-stop unique tokens in *tokens*."""
    return {t for t in tokens if t.lower() not in _STOP_TOKENS}


def evaluate_response(
    response_tokens: list[str],
    expected_concepts: frozenset[str],
    min_unique: int = 2,
) -> tuple[bool, bool, set[str]]:
    """
    Assess the agent's response against the curriculum entry.

    Returns
    -------
    diversity_ok : bool
        True if the response contains ≥ *min_unique* unique meaningful
        words.  Fails on pure babbling ("good good good").
    concept_ok : bool
        True if at least one expected concept word appears in the
        response.  A soft semantic match.
    unique_words : set[str]
        The meaningful unique words extracted (for logging).
    """
    unique = _meaningful_unique(response_tokens)
    diversity_ok = len(unique) >= min_unique
    concept_ok = bool(unique & expected_concepts)
    return diversity_ok, concept_ok, unique


# ═══════════════════════════════════════════════════════════════════════
# 3.  CAREGIVER FEEDBACK INJECTION
# ═══════════════════════════════════════════════════════════════════════

# Strength multipliers (much larger than normal conversation feedback
# to bootstrap the flat Dirichlet matrix quickly).
TYPE_I_STRENGTH: float  = 5.0    # Joy / Affiliation reward
TYPE_II_STRENGTH: float = 3.0    # Sadness / pruning penalty


def inject_reward(agent: CognitiveAgent, strength: float) -> None:
    """
    Massive Type-I reward: reinforce the Dirichlet transition counts
    for every bigram in the agent's last generated response.

    Also boost the Active Inference A-matrix at the agent's current
    MAP state to reinforce the state→obs mapping.

    No gradients — just additive integer/float count increments.
    """
    # Reinforce policy-layer bigrams
    for src, dst in agent.policy._last_bigrams:
        old = agent.policy._get_count(src, dst)
        agent.policy._set_count(src, dst, old + strength)

    # Reinforce AI engine A-matrix at current state
    state = agent.inference.most_likely_state()
    obs = agent._last_obs_idx % agent.inference.num_obs
    agent.inference.apply_social_feedback(
        sentiment=+1,
        current_state=state,
        obs_idx=obs,
        strength=strength,
    )


def inject_penalty(agent: CognitiveAgent, strength: float) -> None:
    """
    Type-II penalty: weaken the Dirichlet transition counts for every
    bigram in the agent's last generated response.  This teaches the
    agent that repeating / babbling is undesirable.

    Counts are floored at prior * 0.1 to avoid degenerate zeroes.
    """
    prior_floor = agent.policy.dirichlet_prior * 0.1
    for src, dst in agent.policy._last_bigrams:
        old = agent.policy._get_count(src, dst)
        agent.policy._set_count(src, dst, max(old - strength, prior_floor))

    # Weaken AI engine A-matrix at current state
    state = agent.inference.most_likely_state()
    obs = agent._last_obs_idx % agent.inference.num_obs
    agent.inference.apply_social_feedback(
        sentiment=-1,
        current_state=state,
        obs_idx=obs,
        strength=strength * 0.5,
    )


# ═══════════════════════════════════════════════════════════════════════
# 4.  MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_caregiver(
    total_cycles: int = 1000,
    seed: int = 42,
    save_plot: bool = False,
    plot_dir: str = "results",
) -> dict:
    """
    Run the Automated Caregiver curriculum.

    Parameters
    ----------
    total_cycles : int
        Number of prompt-response-feedback iterations.
    seed : int
        Reproducibility seed.
    save_plot : bool
        If True, save the VFE decay plot to *plot_dir*.
    plot_dir : str
        Directory for plots (created if absent).

    Returns
    -------
    metrics : dict
        "vfe"           : List[float]  per-cycle VFE
        "unique_ratio"  : List[float]  unique-words / total-tokens
        "energy"        : List[int]
        "affiliation"   : List[float]
        "reward_count"  : int
        "penalty_count" : int
    """
    cfg = Config()
    rng = np.random.default_rng(seed)

    env = ChatbotEnv(cfg.chat, rng)
    agent = CognitiveAgent(cfg, rng)

    # ── Metric accumulators ───────────────────────────────────────────
    vfe_log:     list[float] = []
    unique_log:  list[float] = []
    energy_log:  list[int]   = []
    affil_log:   list[float] = []
    reward_count  = 0
    penalty_count = 0

    num_prompts = len(CURRICULUM)
    t0 = time.perf_counter()

    print()
    print("  Automated Caregiver — Gradient-Free Language Bootstrap")
    print(f"  Cycles: {total_cycles}  |  Curriculum size: {num_prompts}")
    print("  " + "=" * 56)
    print()

    for cycle in range(1, total_cycles + 1):
        # ── Pick a curriculum entry (round-robin with jitter) ─────────
        idx = (cycle - 1) % num_prompts
        prompt, expected_concepts = CURRICULUM[idx]

        # ── Feed prompt to the agent via environment ──────────────────
        obs = env.observe(prompt)
        response_tokens, vfe = agent.chat_observe(obs)
        info = env.act(response_tokens)

        # ── Evaluate response ─────────────────────────────────────────
        diversity_ok, concept_ok, unique_words = evaluate_response(
            response_tokens, expected_concepts
        )

        # ── Inject Caregiver feedback ─────────────────────────────────
        if diversity_ok and concept_ok:
            # Response is structurally acceptable → reward!
            inject_reward(agent, TYPE_I_STRENGTH)
            reward_count += 1
            feedback_tag = "✓ REWARD"
        else:
            # Babbling or off-topic → penalise
            inject_penalty(agent, TYPE_II_STRENGTH)
            penalty_count += 1
            feedback_tag = "✗ PENALTY"

        # ── Record metrics ────────────────────────────────────────────
        vfe_log.append(vfe)
        n_tok = max(len(response_tokens), 1)
        unique_log.append(len(unique_words) / n_tok)
        energy_log.append(info["energy"])
        affil_log.append(info["affiliation"])

        # ── Progress logging (every 100 cycles) ──────────────────────
        if cycle % 100 == 0 or cycle == 1:
            w = min(50, cycle)
            avg_vfe = np.mean(vfe_log[-w:])
            avg_uniq = np.mean(unique_log[-w:])
            resp_str = detokenise(response_tokens)[:60]
            elapsed = time.perf_counter() - t0
            print(
                f"  [{cycle:>5d}/{total_cycles}]  "
                f"VFE={avg_vfe:6.3f}  "
                f"uniq={avg_uniq:.2f}  "
                f"E={info['energy']:>3d}  "
                f"A={info['affiliation']:5.1f}  "
                f"{feedback_tag:>10s}  "
                f"| {resp_str}"
            )

        # ── Respawn energy if depleted (keep the loop going) ─────────
        # In the real chatbot the session would end, but for training
        # we reset homeostatic drives to keep the curriculum running.
        if not env.is_alive():
            env.energy = cfg.chat.initial_energy
            env.affiliation = cfg.chat.initial_affiliation

    elapsed = time.perf_counter() - t0
    print()
    print("  " + "=" * 56)
    print(f"  Done.  {total_cycles} cycles in {elapsed:.1f}s")
    print(f"  Rewards: {reward_count}  |  Penalties: {penalty_count}")
    final_vfe = np.mean(vfe_log[-50:]) if len(vfe_log) >= 50 else np.mean(vfe_log)
    print(f"  Final avg VFE (last 50): {final_vfe:.3f}")
    diag = agent.get_diagnostics()
    print(f"  Graph nodes: {diag['num_memory_nodes']}  |  "
          f"Bigram entries: {diag['bigram_entries']}")
    print()

    metrics = {
        "vfe": vfe_log,
        "unique_ratio": unique_log,
        "energy": energy_log,
        "affiliation": affil_log,
        "reward_count": reward_count,
        "penalty_count": penalty_count,
    }

    # ── Optional plot ─────────────────────────────────────────────────
    if save_plot:
        _plot_caregiver_results(metrics, total_cycles, plot_dir)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 5.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def _plot_caregiver_results(
    metrics: dict,
    total_cycles: int,
    plot_dir: str,
) -> None:
    """Save a 3-panel summary plot of the caregiver training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Smooth helper
    def _smooth(arr, w=50):
        a = np.asarray(arr, dtype=np.float64)
        if len(a) < w:
            return a
        return np.convolve(a, np.ones(w) / w, mode="valid")

    xs_vfe = np.arange(len(_smooth(metrics["vfe"])))
    xs_uniq = np.arange(len(_smooth(metrics["unique_ratio"])))
    xs_e = np.arange(len(_smooth(metrics["energy"])))

    # Panel 1: VFE decay
    axes[0].plot(xs_vfe, _smooth(metrics["vfe"]), color="crimson", lw=1.2)
    axes[0].set_ylabel("VFE (smoothed)")
    axes[0].set_title("Automated Caregiver — VFE Decay Over Training")
    axes[0].axhline(3.0, color="grey", ls="--", lw=0.8, label="arousal threshold")
    axes[0].legend(fontsize=8)

    # Panel 2: Unique-word ratio
    axes[1].plot(xs_uniq, _smooth(metrics["unique_ratio"]),
                 color="steelblue", lw=1.2)
    axes[1].set_ylabel("Unique-word ratio")
    axes[1].axhline(0.5, color="grey", ls="--", lw=0.8, label="diversity target")
    axes[1].legend(fontsize=8)

    # Panel 3: Homeostatic drives
    axes[2].plot(xs_e, _smooth(metrics["energy"]),
                 color="green", lw=1.0, label="Energy")
    axes[2].plot(
        np.arange(len(_smooth(metrics["affiliation"]))),
        _smooth(metrics["affiliation"]),
        color="orange", lw=1.0, label="Affiliation",
    )
    axes[2].set_ylabel("Drive level")
    axes[2].set_xlabel("Training cycle")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(plot_dir, "caregiver_training.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
# 6.  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Automated Caregiver — bootstraps chatbot grammar"
    )
    p.add_argument("--cycles", type=int, default=1000,
                   help="Number of training cycles (default: 1000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--plot", action="store_true",
                   help="Save VFE decay plot to results/")
    p.add_argument("--output", type=str, default="results",
                   help="Plot output directory (default: results)")
    args = p.parse_args()

    run_caregiver(
        total_cycles=args.cycles,
        seed=args.seed,
        save_plot=args.plot,
        plot_dir=args.output,
    )


if __name__ == "__main__":
    main()
