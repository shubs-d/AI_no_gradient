#!/usr/bin/env python3
"""
chat_main.py
============
Interactive launcher for the gradient-free chatbot environment.

Features
────────
• **Automatic wake-up:**  If a checkpoint file exists, the agent's
  vocabulary, Dirichlet counts, memory graph, and homeostatic drives
  are restored — no blank-slate reset.
• **Graceful shutdown:**  Ctrl-C (or typing ``exit`` / ``quit``)
  triggers ``save_agent_state()``, printing a confirmation message
  before terminating.

Run:
    python3 chat_main.py
    python3 chat_main.py --save checkpoints/my_agent.pkl

Type 'exit' or 'quit' to stop.  Press Ctrl-C at any time for an
emergency save-and-exit.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Ensure project root is importable when launched from anywhere.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import Config
from environment.chatbot_env import ChatbotEnv, detokenise
from agent.core_agent import CognitiveAgent
from persistence import (
    save_agent_state,
    load_agent_state,
    restore_agent_state,
    checkpoint_exists,
    checkpoint_summary,
    DEFAULT_SAVE_PATH,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradient-Free Chatbot REPL")
    p.add_argument(
        "--save", type=str, default=DEFAULT_SAVE_PATH,
        help=f"Checkpoint file path (default: {DEFAULT_SAVE_PATH})",
    )
    p.add_argument(
        "--no-load", action="store_true",
        help="Start from blank slate even if a checkpoint exists",
    )
    return p.parse_args()


def _sleep(agent, env, save_path: str) -> None:
    """Save agent state and print a confirmation message."""
    abspath = save_agent_state(agent, env, filepath=save_path)
    diag = agent.get_diagnostics()
    print(
        f"\n  Going to sleep, memories saved."
        f"\n  [{diag['num_memory_nodes']} nodes | "
        f"{diag['bigram_entries']} bigrams → {abspath}]"
    )


def main() -> None:
    args = _parse_args()
    save_path: str = args.save

    cfg = Config()
    rng = np.random.default_rng(cfg.experiment.seed)

    env = ChatbotEnv(cfg.chat, rng)
    agent = CognitiveAgent(cfg, rng)

    # ── Wake-up: restore from checkpoint if available ─────────────────
    if not args.no_load and checkpoint_exists(save_path):
        try:
            cp = load_agent_state(save_path)
            restore_agent_state(agent, env, checkpoint=cp)
            print(f"\n  Waking up from sleep…  ({checkpoint_summary(save_path)})")
        except Exception as exc:
            print(f"\n  [warn] Could not load checkpoint: {exc}")
            print("  Starting from blank slate.\n")
    else:
        print()

    print("Gradient-Free Chatbot (Discrete Active Inference)")
    print("Type a message and press Enter. Type 'exit' to quit.")
    print("Press Ctrl-C at any time for an emergency save-and-exit.\n")

    try:
        while env.is_alive():
            try:
                user_text = input("You: ").strip()
            except EOFError:
                # Piped input exhausted — save and exit
                break

            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue

            obs = env.observe(user_text)
            response_tokens, vfe = agent.chat_observe(obs)
            info = env.act(response_tokens)

            response = detokenise(response_tokens)
            if not response:
                response = "..."

            print(f"Bot: {response}")
            print(
                f"   [vfe={vfe:.3f} | energy={info['energy']} | "
                f"affiliation={info['affiliation']:.1f}]"
            )

        if not env.is_alive():
            print("\nEnergy depleted.")

    except KeyboardInterrupt:
        # Ctrl-C: graceful emergency save
        pass

    # ── Graceful shutdown: always save on exit ────────────────────────
    _sleep(agent, env, save_path)


if __name__ == "__main__":
    main()
