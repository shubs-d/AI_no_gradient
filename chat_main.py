#!/usr/bin/env python3
"""
chat_main.py
============
Minimal interactive launcher for the gradient-free chatbot environment.

Run:
    python3 chat_main.py

Type 'exit' or 'quit' to stop.
"""

from __future__ import annotations

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


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.experiment.seed)

    env = ChatbotEnv(cfg.chat, rng)
    agent = CognitiveAgent(cfg, rng)

    print("\nGradient-Free Chatbot (Discrete Active Inference)")
    print("Type a message and press Enter. Type 'exit' to quit.\n")

    while env.is_alive():
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("\nSession ended.")
            return
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

    print("\nEnergy depleted. Session ended.")


if __name__ == "__main__":
    main()
