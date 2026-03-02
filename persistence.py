"""
persistence.py
==============
Graph-and-state serialisation for the gradient-free Active Inference
chatbot.  Enables the agent to **sleep** (save all acquired vocabulary,
Dirichlet transition counts, memory graph structure, and homeostatic
drive levels) and later **wake up** without amnesia.

Serialisation strategy
──────────────────────
We use Python's built-in ``pickle`` protocol (protocol 5, the fastest
binary protocol available in Python ≥ 3.8).  The on-disk payload is a
single dict:

    {
        "version"       : int,
        "memory_graph"  : { node labels, activation, raw counts, W, … },
        "inference"     : { A, B, qs, C, VFE, num_states, num_obs },
        "policy"        : { bigram table, bos/eos IDs },
        "tsetlin"       : { automaton array, clause usage, dimensions },
        "structure"     : { rolling VFE, step counter, cooldown },
        "agent_state"   : { prev_state, prev_action, step, last_obs_idx },
        "homeostasis"   : { energy, affiliation, turn, recent_responses },
    }

Everything in this dict is either a plain Python scalar/dict/list or a
NumPy ndarray — all of which pickle handles natively.

No neural-network weights, no gradient buffers — just integer count
tables and graph adjacencies.  A 100k-node vocabulary typically
serialises to < 5 MB.

Security note
─────────────
``pickle.load`` is inherently unsafe on untrusted data.  The
``load_agent_state`` function validates the payload version and key
presence before returning, but callers must only load files they
themselves created.

Usage
─────
    from persistence import save_agent_state, load_agent_state

    # Save
    save_agent_state(agent, env, "checkpoints/agent_v1.pkl")

    # Load
    payload = load_agent_state("checkpoints/agent_v1.pkl")
    restore_agent_state(agent, env, payload)
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

PERSISTENCE_VERSION: int = 1          # Bump on any schema change
DEFAULT_SAVE_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "agent_state.pkl",
)


# ═══════════════════════════════════════════════════════════════════════
# 1.  SAVE  —  serialise the full cognitive state
# ═══════════════════════════════════════════════════════════════════════

def save_agent_state(
    agent,
    env=None,
    filepath: str = DEFAULT_SAVE_PATH,
) -> str:
    """
    Serialise the agent's entire cognitive state to a binary file.

    Parameters
    ----------
    agent : CognitiveAgent
        The live agent whose state we want to persist.
    env : ChatbotEnv, optional
        If provided, homeostatic drive levels (Energy, Affiliation)
        and the conversation turn counter are saved so the agent
        wakes up in the exact physiological state it went to sleep in.
    filepath : str
        Destination path.  Parent directories are created automatically.

    Returns
    -------
    filepath : str
        The absolute path where the checkpoint was written.
    """
    # ── 1a.  Memory Graph ─────────────────────────────────────────────
    mg = agent.memory
    n = mg.num_nodes
    memory_payload: Dict[str, Any] = {
        "num_nodes":    n,
        "labels":       dict(mg._labels),                       # {int: str}
        "activation":   mg._activation[:n].copy(),              # ndarray
        "visit_counts": mg._visit_counts[:n].copy(),            # ndarray
        "raw_counts":   mg._raw_counts[:n, :n].copy(),          # ndarray
        "W":            mg._W[:n, :n].copy(),                   # ndarray
    }

    # ── 1b.  Active Inference Engine (Dirichlet count tables) ─────────
    ie = agent.inference
    inference_payload: Dict[str, Any] = {
        "A":          ie._A.copy(),           # (S, O)  Dirichlet likelihood
        "B":          ie._B.copy(),           # (S, A, S') Dirichlet transition
        "qs":         ie._qs.copy(),          # (S,)  posterior belief
        "C":          ie._C.copy(),           # (O,)  preference prior
        "last_vfe":   ie.last_vfe,
        "num_states": ie.num_states,
        "num_obs":    ie.num_obs,
    }

    # ── 1c.  SBCP Policy Layer (bigram Dirichlet table) ───────────────
    pol = agent.policy
    policy_payload: Dict[str, Any] = {
        "bigram":       {src: dict(dsts) for src, dsts in pol._bigram.items()},
        "bos_id":       pol.bos_id,
        "eos_id":       pol.eos_id,
        "last_bigrams": list(pol._last_bigrams),
    }

    # ── 1d.  Tsetlin Machine ─────────────────────────────────────────
    tm = agent.tsetlin
    tsetlin_payload: Dict[str, Any] = {
        "ta":                  tm._ta.copy(),
        "clause_usage":        tm.clause_usage.copy(),
        "current_num_clauses": tm.current_num_clauses,
        "current_num_literals": tm.current_num_literals,
        "int_ops":             tm.int_ops,
    }

    # ── 1e.  Structure Learner ────────────────────────────────────────
    sl = agent.structure
    structure_payload: Dict[str, Any] = {
        "vfe_window":            list(sl._vfe_window),
        "step":                  sl._step,
        "steps_since_expansion": sl._steps_since_expansion,
    }

    # ── 1f.  Agent-level bookkeeping ──────────────────────────────────
    agent_payload: Dict[str, Any] = {
        "prev_state":   agent._prev_state,
        "prev_action":  agent._prev_action,
        "step":         agent._step,
        "last_obs_idx": agent._last_obs_idx,
        "bos_id":       agent._bos_id,
        "eos_id":       agent._eos_id,
    }

    # ── 1g.  Homeostatic drives (environment) ─────────────────────────
    homeostasis_payload: Optional[Dict[str, Any]] = None
    if env is not None:
        homeostasis_payload = {
            "energy":           env.energy,
            "affiliation":      env.affiliation,
            "turn":             env.turn,
            "recent_responses": list(env._recent_responses),
            "vocab":            dict(env.vocab),
        }

    # ── Assemble top-level checkpoint ─────────────────────────────────
    checkpoint: Dict[str, Any] = {
        "version":      PERSISTENCE_VERSION,
        "memory_graph": memory_payload,
        "inference":    inference_payload,
        "policy":       policy_payload,
        "tsetlin":      tsetlin_payload,
        "structure":    structure_payload,
        "agent_state":  agent_payload,
        "homeostasis":  homeostasis_payload,
    }

    # ── Write to disk ─────────────────────────────────────────────────
    dirpath = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dirpath, exist_ok=True)

    # Atomic write: dump to a temp file, then rename (prevents
    # half-written files on power failure / Ctrl-C during I/O).
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, filepath)

    return os.path.abspath(filepath)


# ═══════════════════════════════════════════════════════════════════════
# 2.  LOAD  —  deserialise and validate
# ═══════════════════════════════════════════════════════════════════════

def load_agent_state(filepath: str = DEFAULT_SAVE_PATH) -> Dict[str, Any]:
    """
    Load a previously saved checkpoint from disk.

    Parameters
    ----------
    filepath : str
        Path to the ``.pkl`` file produced by ``save_agent_state``.

    Returns
    -------
    checkpoint : dict
        The full payload.  Pass this to ``restore_agent_state()``
        to apply it to a live agent+env pair.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file is corrupt or has an incompatible version.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint at: {filepath}")

    with open(filepath, "rb") as f:
        checkpoint = pickle.load(f)  # noqa: S301  — trusted local file

    # ── Validate schema ───────────────────────────────────────────────
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dict — file may be corrupt.")

    version = checkpoint.get("version")
    if version is None or version > PERSISTENCE_VERSION:
        raise ValueError(
            f"Checkpoint version {version} is not supported "
            f"(this code supports up to v{PERSISTENCE_VERSION})."
        )

    required_keys = {
        "memory_graph", "inference", "policy",
        "tsetlin", "structure", "agent_state",
    }
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {missing}")

    return checkpoint


# ═══════════════════════════════════════════════════════════════════════
# 3.  RESTORE  —  apply a checkpoint to a live agent and environment
# ═══════════════════════════════════════════════════════════════════════

def restore_agent_state(
    agent,
    env=None,
    checkpoint: Optional[Dict[str, Any]] = None,
    filepath: str = DEFAULT_SAVE_PATH,
) -> None:
    """
    Overwrite the agent's (and optionally the environment's) internal
    state with a previously saved checkpoint.

    You may pass either a pre-loaded *checkpoint* dict or a *filepath*;
    if both are given, *checkpoint* takes precedence.

    Parameters
    ----------
    agent : CognitiveAgent
        Target agent.  Its sub-modules will be mutated in-place.
    env : ChatbotEnv, optional
        If provided **and** the checkpoint contains homeostasis data,
        the environment's energy / affiliation / turn counter are
        restored.
    checkpoint : dict, optional
        A payload previously returned by ``load_agent_state()``.
    filepath : str
        Fallback path if *checkpoint* is None.
    """
    if checkpoint is None:
        checkpoint = load_agent_state(filepath)

    # ── 3a.  Memory Graph ─────────────────────────────────────────────
    mp = checkpoint["memory_graph"]
    mg = agent.memory
    n = mp["num_nodes"]

    # Ensure backing arrays are large enough
    mg._ensure_capacity(n)

    mg._num_nodes = n
    mg._labels = {int(k): v for k, v in mp["labels"].items()}
    mg._activation[:] = 0.0
    mg._activation[:n] = mp["activation"]
    mg._visit_counts[:] = 0
    mg._visit_counts[:n] = mp["visit_counts"]
    mg._raw_counts[:, :] = 0
    mg._raw_counts[:n, :n] = mp["raw_counts"]
    mg._W[:, :] = 0.0
    mg._W[:n, :n] = mp["W"]

    # ── 3b.  Active Inference Engine ──────────────────────────────────
    ip = checkpoint["inference"]
    ie = agent.inference

    ie._A = ip["A"].copy()
    ie._B = ip["B"].copy()
    ie._qs = ip["qs"].copy()
    ie._C = ip["C"].copy()
    ie.last_vfe = ip["last_vfe"]
    ie.num_states = ip["num_states"]
    ie.num_obs = ip["num_obs"]

    # ── 3c.  SBCP Policy Layer ────────────────────────────────────────
    pp = checkpoint["policy"]
    pol = agent.policy

    pol._bigram = {
        int(src): {int(dst): float(cnt) for dst, cnt in dsts.items()}
        for src, dsts in pp["bigram"].items()
    }
    pol.bos_id = pp["bos_id"]
    pol.eos_id = pp["eos_id"]
    pol._last_bigrams = [tuple(pair) for pair in pp.get("last_bigrams", [])]
    pol._refractory.clear()

    # ── 3d.  Tsetlin Machine ─────────────────────────────────────────
    tp = checkpoint["tsetlin"]
    tm = agent.tsetlin

    tm._ta = tp["ta"].copy()
    tm.clause_usage = tp["clause_usage"].copy()
    tm.current_num_clauses = tp["current_num_clauses"]
    tm.current_num_literals = tp["current_num_literals"]
    tm.int_ops = tp["int_ops"]

    # ── 3e.  Structure Learner ────────────────────────────────────────
    sp = checkpoint["structure"]
    sl = agent.structure

    sl._vfe_window = deque(sp["vfe_window"], maxlen=sl.cfg.rolling_window_T)
    sl._step = sp["step"]
    sl._steps_since_expansion = sp["steps_since_expansion"]

    # ── 3f.  Agent-level bookkeeping ──────────────────────────────────
    ap = checkpoint["agent_state"]
    agent._prev_state = ap["prev_state"]
    agent._prev_action = ap["prev_action"]
    agent._step = ap["step"]
    agent._last_obs_idx = ap["last_obs_idx"]
    agent._bos_id = ap["bos_id"]
    agent._eos_id = ap["eos_id"]

    # ── 3g.  Homeostatic drives (environment) ─────────────────────────
    hp = checkpoint.get("homeostasis")
    if hp is not None and env is not None:
        env.energy = hp["energy"]
        env.affiliation = hp["affiliation"]
        env.turn = hp["turn"]
        env._recent_responses = deque(
            hp["recent_responses"],
            maxlen=env.cfg.repeat_tolerance + 1,
        )
        env.vocab = hp.get("vocab", {})


# ═══════════════════════════════════════════════════════════════════════
# 4.  CONVENIENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def checkpoint_exists(filepath: str = DEFAULT_SAVE_PATH) -> bool:
    """Return True if a checkpoint file exists at *filepath*."""
    return os.path.isfile(filepath)


def checkpoint_summary(filepath: str = DEFAULT_SAVE_PATH) -> str:
    """
    Load a checkpoint and return a human-readable one-line summary.

    Example
    -------
    >>> checkpoint_summary("checkpoints/agent_state.pkl")
    'v1 | 137 nodes | 16 states | 382 bigrams | E=95 A=4.9'
    """
    cp = load_agent_state(filepath)
    n_nodes = cp["memory_graph"]["num_nodes"]
    n_states = cp["inference"]["num_states"]
    n_bigrams = sum(
        len(dsts) for dsts in cp["policy"]["bigram"].values()
    )
    hp = cp.get("homeostasis")
    if hp:
        return (
            f"v{cp['version']} | {n_nodes} nodes | {n_states} states | "
            f"{n_bigrams} bigrams | E={hp['energy']} A={hp['affiliation']:.1f}"
        )
    return (
        f"v{cp['version']} | {n_nodes} nodes | {n_states} states | "
        f"{n_bigrams} bigrams"
    )
