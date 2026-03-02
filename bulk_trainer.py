#!/usr/bin/env python3
"""
bulk_trainer.py
===============
High-throughput corpus training loop for the gradient-free Active
Inference chatbot.  Feeds raw English text through the agent at
maximum speed, extracting semantic triplets, learning co-occurrence
statistics, and providing automated Type-I / Type-II feedback.

Design rationale
────────────────
The ``auto_caregiver.py`` script uses a small hand-crafted SVO
curriculum (≈30 prompt/response pairs).  The **bulk trainer** scales
this to *arbitrary text corpora* — books, Wikipedia dumps, dialogue
datasets — enabling the agent to acquire broad vocabulary, grammar
patterns, and semantic triplet knowledge.

Training algorithm
──────────────────
For each line in the corpus file:

  1. Tokenise the line.
  2. Feed it through ``ChatbotEnv.observe()`` → ``agent.chat_observe()``.
  3. The agent's internal pipeline automatically:
       • Grounds tokens → activates lexical nodes → learns co-occurrence
       • Extracts SVO triplets → creates predicate edges
       • Runs spreading activation → generates response
       • TM gets grammar-aware update with syntactic features
  4. Evaluate response quality:
       a. **Triplet reconstruction:**  If the agent's response contains
          at least one component of a triplet extracted from the input,
          apply Type-I reward (strength = 5.0).
       b. **Diversity gate:**  If the response has ≥ 2 unique words
          and overlaps ≥ 1 input word → Type-I.
       c. Otherwise → Type-II penalty (strength = 3.0).
  5. Periodically checkpoint the agent state.

Complexity
──────────
  Per-sentence:  O(K·T) where K = TopK, T = token length.
  Total:         O(N·K·T) for N sentences.
  Memory:        Grows sub-linearly via MDL pruning.

Usage
─────
    cd AI_no_gradient

    # Train on a text file
    python3 bulk_trainer.py --file corpus.txt --cycles 5000

    # Resume from checkpoint
    python3 bulk_trainer.py --file corpus.txt --load agent.pkl --save agent.pkl

    # Quick self-test with built-in mini corpus
    python3 bulk_trainer.py --selftest

All learning is count-based.  No neural networks, no gradients.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

# ── Fix import path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import Config
from environment.chatbot_env import ChatbotEnv, tokenise, detokenise
from agent.core_agent import CognitiveAgent
from agent.memory_graph import extract_triplets
from persistence import (
    save_agent_state,
    load_agent_state,
    restore_agent_state,
    checkpoint_exists,
    DEFAULT_SAVE_PATH,
)


# ═══════════════════════════════════════════════════════════════════════
# 1.  BUILT-IN MINI CORPUS (for --selftest)
# ═══════════════════════════════════════════════════════════════════════

MINI_CORPUS: List[str] = [
    "The cat eats fish every morning.",
    "Dogs love playing in the park.",
    "She reads books about history.",
    "He runs fast in the race.",
    "Birds fly over the tall trees.",
    "The teacher explains the lesson clearly.",
    "Children play games after school.",
    "The sun rises in the east.",
    "Rain falls from the dark clouds.",
    "The farmer grows vegetables and fruits.",
    "I like apples and oranges.",
    "You are a good friend.",
    "They went to the market yesterday.",
    "We eat dinner at seven.",
    "The dog chases the cat around.",
    "She sings beautiful songs every night.",
    "He writes letters to his family.",
    "The river flows through the valley.",
    "Stars shine brightly at night.",
    "The baby sleeps in the crib.",
    "My mother cooks delicious food.",
    "The students study hard for exams.",
    "Fish swim in the clear water.",
    "The wind blows from the north.",
    "Flowers bloom in the spring garden.",
    "The boy kicks the ball hard.",
    "She drives her car to work.",
    "He paints pictures of mountains.",
    "The clock ticks on the wall.",
    "Bees make honey from flower nectar.",
    "The president gives a long speech.",
    "Wolves hunt in the deep forest.",
    "The mechanic fixes broken engines.",
    "Scientists discover new species every year.",
    "The pianist plays classical music beautifully.",
    "Pilots fly planes across the ocean.",
    "The chef prepares gourmet meals daily.",
    "Engineers design bridges and buildings.",
    "Doctors treat patients at the hospital.",
    "The artist creates stunning sculptures.",
]


# ═══════════════════════════════════════════════════════════════════════
# 2.  EVALUATION HEURISTICS
# ═══════════════════════════════════════════════════════════════════════

def evaluate_response(
    response_tokens: List[str],
    input_tokens: List[str],
    input_triplets: List[Tuple[str, str, str]],
) -> int:
    """
    Evaluate the quality of the agent's response.

    Returns
    -------
    sentiment : int
        +1  if the response demonstrates grammatical competence.
        -1  if the response is babbling or irrelevant.
         0  if neutral (empty response).
    """
    if not response_tokens:
        return 0

    unique_words = set(w.lower() for w in response_tokens)
    input_words = set(w.lower() for w in input_tokens)

    # Gate 1: Diversity — must have ≥ 2 unique non-trivial words
    trivial = {"the", "a", "an", "is", "am", "are", "i", "to", "of"}
    non_trivial = unique_words - trivial
    if len(non_trivial) < 2:
        return -1

    # Gate 2: Triplet reconstruction — does the response contain
    # at least one component from any input triplet?
    triplet_match = False
    for subj, pred, obj in input_triplets:
        components = {subj.lower(), obj.lower()}
        # Also check verb stems (rough: first 4 chars)
        pred_words = pred.lower().split()
        for pw in pred_words:
            components.add(pw)
        if unique_words & components:
            triplet_match = True
            break

    # Gate 3: Input overlap — at least 1 meaningful input word in response
    meaningful_input = input_words - trivial
    overlap = unique_words & meaningful_input

    if triplet_match:
        return +1
    elif len(overlap) >= 1:
        return +1
    else:
        return -1


# ═══════════════════════════════════════════════════════════════════════
# 3.  BULK TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train_on_corpus(
    agent: CognitiveAgent,
    env: ChatbotEnv,
    sentences: List[str],
    max_cycles: int = 5000,
    checkpoint_interval: int = 500,
    save_path: str = DEFAULT_SAVE_PATH,
    verbose: bool = True,
    feedback_strength_positive: float = 5.0,
    feedback_strength_negative: float = 3.0,
) -> dict:
    """
    High-throughput corpus training loop.

    Parameters
    ----------
    agent : CognitiveAgent
    env : ChatbotEnv
    sentences : List[str]
        Corpus sentences (one per element).
    max_cycles : int
        Total training cycles (wraps around the corpus).
    checkpoint_interval : int
        Save agent state every N cycles.
    save_path : str
        File path for checkpoint saves.
    verbose : bool
        Print progress every 100 cycles.
    feedback_strength_positive : float
        Dirichlet count boost on Type-I feedback.
    feedback_strength_negative : float
        Dirichlet count decrement on Type-II feedback.

    Returns
    -------
    stats : dict
        Training statistics: vfe_history, reward_history, triplet_count.
    """
    if not sentences:
        raise ValueError("Corpus is empty")

    vfe_history: List[float] = []
    reward_history: List[int] = []
    triplet_total: int = 0
    type1_count: int = 0
    type2_count: int = 0

    t_start = time.time()
    n_sentences = len(sentences)

    # Save original feedback strength, override for bulk training
    orig_fb_strength = agent.policy.feedback_strength

    for cycle in range(max_cycles):
        # Cycle through corpus
        sentence = sentences[cycle % n_sentences]
        tokens_raw = tokenise(sentence)

        if len(tokens_raw) < 2:
            continue

        # Extract triplets from input for evaluation
        input_triplets = extract_triplets(sentence)
        triplet_total += len(input_triplets)

        # Feed through the environment → agent pipeline
        obs = env.observe(sentence)
        response_tokens, vfe = agent.chat_observe(obs)
        vfe_history.append(vfe)

        # Evaluate response quality
        sentiment = evaluate_response(
            response_tokens, tokens_raw, input_triplets,
        )
        reward_history.append(sentiment)

        # Apply calibrated feedback
        if sentiment > 0:
            agent.policy.feedback_strength = feedback_strength_positive
            agent.policy.apply_feedback(+1)
            type1_count += 1
        elif sentiment < 0:
            agent.policy.feedback_strength = feedback_strength_negative
            agent.policy.apply_feedback(-1)
            type2_count += 1

        # Restore original feedback strength
        agent.policy.feedback_strength = orig_fb_strength

        # Periodic reporting
        if verbose and (cycle + 1) % 100 == 0:
            recent_vfe = np.mean(vfe_history[-100:])
            recent_reward = np.mean(reward_history[-100:])
            elapsed = time.time() - t_start
            speed = (cycle + 1) / elapsed
            diag = agent.get_diagnostics()
            print(
                f"[Bulk {cycle+1:>6d}/{max_cycles}] "
                f"VFE={recent_vfe:6.3f}  "
                f"reward={recent_reward:+.2f}  "
                f"nodes={diag['num_memory_nodes']:>4d}  "
                f"triplets={diag['triplet_count']:>4d}  "
                f"bigrams={diag['bigram_entries']:>5d}  "
                f"bits={diag['model_bits']:8.1f}  "
                f"{speed:.1f} cyc/s"
            )

        # Periodic checkpoint
        if checkpoint_interval > 0 and (cycle + 1) % checkpoint_interval == 0:
            if save_path:
                save_agent_state(agent, env, filepath=save_path)
                if verbose:
                    print(f"  [checkpoint saved → {save_path}]")

    elapsed_total = time.time() - t_start

    stats = {
        "total_cycles": max_cycles,
        "elapsed_seconds": elapsed_total,
        "cycles_per_second": max_cycles / max(elapsed_total, 0.001),
        "final_avg_vfe": float(np.mean(vfe_history[-100:])) if vfe_history else 0.0,
        "type1_reward_count": type1_count,
        "type2_penalty_count": type2_count,
        "total_triplets_seen": triplet_total,
        "final_triplet_count": agent.memory.triplet_count,
        "final_node_count": agent.memory.num_nodes,
        "final_bigram_entries": agent.policy.total_bigram_entries(),
    }

    return stats


# ═══════════════════════════════════════════════════════════════════════
# 4.  CORPUS FILE LOADER
# ═══════════════════════════════════════════════════════════════════════

def load_corpus(filepath: str) -> List[str]:
    """
    Load a text file into a list of non-empty stripped lines.

    Parameters
    ----------
    filepath : str
        Path to a plain-text file (one sentence per line, or
        paragraph-style — each line is treated as one training sample).

    Returns
    -------
    List[str]
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    sentences: List[str] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 5:  # skip very short lines
                sentences.append(line)

    if not sentences:
        raise ValueError(f"Corpus file is empty or has no usable lines: {filepath}")

    return sentences


# ═══════════════════════════════════════════════════════════════════════
# 5.  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk corpus trainer for gradient-free Active Inference chatbot.",
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to training corpus (one sentence per line).",
    )
    parser.add_argument(
        "--cycles", "-c", type=int, default=5000,
        help="Total training cycles (default: 5000).",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=500,
        help="Checkpoint interval (default: every 500 cycles).",
    )
    parser.add_argument(
        "--save", "-s", type=str, default=DEFAULT_SAVE_PATH,
        help=f"Path to save agent state (default: {DEFAULT_SAVE_PATH}).",
    )
    parser.add_argument(
        "--load", "-l", type=str, default=None,
        help="Path to load pre-trained agent state.",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run a quick self-test with the built-in mini corpus.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────
    cfg = Config()
    rng = np.random.default_rng(args.seed)
    env = ChatbotEnv(cfg.chat, rng)
    agent = CognitiveAgent(cfg, rng)

    # Load checkpoint if requested
    if args.load and checkpoint_exists(args.load):
        state = load_agent_state(args.load)
        restore_agent_state(agent, env, checkpoint=state)
        if not args.quiet:
            print(f"Loaded agent state from {args.load}")

    # ── Determine corpus ──────────────────────────────────────────────
    if args.selftest:
        sentences = MINI_CORPUS
        cycles = min(args.cycles, 500)
        if not args.quiet:
            print(f"Self-test mode: {len(sentences)} sentences, {cycles} cycles")
    elif args.file:
        sentences = load_corpus(args.file)
        cycles = args.cycles
        if not args.quiet:
            print(f"Loaded {len(sentences)} sentences from {args.file}")
    else:
        parser.error("Specify --file <corpus.txt> or --selftest")
        return

    # ── Train ─────────────────────────────────────────────────────────
    if not args.quiet:
        print(f"Training for {cycles} cycles...")
        print("=" * 72)

    stats = train_on_corpus(
        agent=agent,
        env=env,
        sentences=sentences,
        max_cycles=cycles,
        checkpoint_interval=args.checkpoint,
        save_path=args.save,
        verbose=not args.quiet,
    )

    # ── Final report ──────────────────────────────────────────────────
    if not args.quiet:
        print("=" * 72)
        print("TRAINING COMPLETE")
        print(f"  Total cycles:        {stats['total_cycles']}")
        print(f"  Elapsed:             {stats['elapsed_seconds']:.1f}s")
        print(f"  Speed:               {stats['cycles_per_second']:.1f} cycles/s")
        print(f"  Final avg VFE:       {stats['final_avg_vfe']:.4f}")
        print(f"  Type-I rewards:      {stats['type1_reward_count']}")
        print(f"  Type-II penalties:   {stats['type2_penalty_count']}")
        print(f"  Triplets seen:       {stats['total_triplets_seen']}")
        print(f"  Triplets stored:     {stats['final_triplet_count']}")
        print(f"  Memory nodes:        {stats['final_node_count']}")
        print(f"  Bigram entries:      {stats['final_bigram_entries']}")

    # Final save
    if args.save:
        save_agent_state(agent, env, filepath=args.save)
        if not args.quiet:
            print(f"\nAgent state saved → {args.save}")


if __name__ == "__main__":
    main()
