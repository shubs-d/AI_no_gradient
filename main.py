#!/usr/bin/env python3
"""
main.py
=======
Entry point for the gradient-free cognitive architecture benchmark.

Usage:
    python main.py              # run with default config
    python main.py --steps 5000 # override total steps
    python main.py --seed 123   # override random seed
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on sys.path so bare `import config` works
# regardless of working directory.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    Config,
    GridConfig,
    ChatConfig,
    PolicyConfig,
    InferenceConfig,
    MemoryConfig,
    TsetlinConfig,
    StructureConfig,
    ExperimentConfig,
)
from experiments.continual_learning_bench import run_benchmark, plot_results
from persistence import save_agent_state, load_agent_state, restore_agent_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gradient-free continual-learning benchmark"
    )
    p.add_argument("--steps", type=int, default=None,
                   help="Total environment steps (default: 3000)")
    p.add_argument("--regime-steps", type=int, default=None,
                   help="Steps per regime (default: 1000)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: 42)")
    p.add_argument("--output", type=str, default=None,
                   help="Plot output directory (default: results)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build config, overriding experiment params from CLI
    exp_kwargs = {}
    if args.steps is not None:
        exp_kwargs["total_steps"] = args.steps
    if args.regime_steps is not None:
        exp_kwargs["regime_steps"] = args.regime_steps
    if args.seed is not None:
        exp_kwargs["seed"] = args.seed
    if args.output is not None:
        exp_kwargs["plot_output_dir"] = args.output

    # If regime_steps changed but total_steps wasn't, adjust total
    if "regime_steps" in exp_kwargs and "total_steps" not in exp_kwargs:
        exp_kwargs["total_steps"] = exp_kwargs["regime_steps"] * 3

    cfg = Config(
        grid=GridConfig(),
        inference=InferenceConfig(),
        memory=MemoryConfig(),
        tsetlin=TsetlinConfig(),
        structure=StructureConfig(),
        experiment=ExperimentConfig(**exp_kwargs),
    )

    print()
    print("  Gradient-Free Cognitive Architecture — STRESS TEST")
    print("  Dark-Room BROKEN | BME OVERDRIVE | MDL OFF")
    print("  Active Inference × Tsetlin Machine × Spreading-Activation Memory")
    print(f"  Steps: {cfg.experiment.total_steps}  |  "
          f"Regime length: {cfg.experiment.regime_steps}  |  "
          f"Seed: {cfg.experiment.seed}")
    print()

    recorder = run_benchmark(cfg)
    plot_results(recorder, cfg)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
