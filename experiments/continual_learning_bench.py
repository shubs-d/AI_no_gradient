"""
continual_learning_bench.py
===========================
Experimental harness that runs the gradient-free cognitive agent through
three distinct environmental regimes (Tasks A → B → C) and records:

  1. **Adaptation latency**  — time (steps) for VFE to stabilise after
     each regime shift.
  2. **Retention / Catastrophic forgetting** — accuracy of predicting
     Task-A dynamics after learning Task C.
  3. **Structural complexity** — number of memory-graph nodes and active
     Tsetlin clauses over time, bounded by MDL pruning.
  4. **Computational cost** — wall-clock time and integer operations
     per step.

All plots are saved to ``config.experiment.plot_output_dir``.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless-safe backend
import matplotlib.pyplot as plt

from config import Config
from environment.non_stationary_grid import NonStationaryGrid
from agent.core_agent import CognitiveAgent


# ═══════════════════════════════════════════════════════════════════════
# Metric recorders
# ═══════════════════════════════════════════════════════════════════════

class MetricRecorder:
    """Accumulates per-step scalars and provides smoothed views."""

    def __init__(self) -> None:
        self.vfe: List[float] = []
        self.energy: List[float] = []
        self.num_nodes: List[int] = []
        self.num_clauses: List[int] = []
        self.available_clauses: List[int] = []   # total clause slots
        self.clause_saturation: List[float] = [] # active / available
        self.model_bits: List[float] = []
        self.lyapunov: List[float] = []
        self.wall_time: List[float] = []     # seconds per step
        self.int_ops: List[int] = []
        self.regime: List[int] = []
        self.consumed: List[int] = []        # cumulative resources consumed
        self.hazard_hits: List[int] = []     # cumulative hazard steps

    def record(
        self,
        vfe: float,
        info: dict,
        diag: dict,
        dt: float,
    ) -> None:
        self.vfe.append(vfe)
        self.energy.append(info.get("energy", 0))
        self.num_nodes.append(diag["num_memory_nodes"])
        self.num_clauses.append(diag["active_clauses"])
        self.available_clauses.append(diag.get("available_clauses", 0))
        self.clause_saturation.append(diag.get("clause_saturation", 0.0))
        self.model_bits.append(diag["model_bits"])
        self.lyapunov.append(diag["lyapunov_energy"])
        self.wall_time.append(dt)
        self.int_ops.append(diag["tsetlin_int_ops"])
        self.regime.append(info.get("regime", 0))

        prev_con = self.consumed[-1] if self.consumed else 0
        self.consumed.append(prev_con + int(info.get("consumed", False)))

        prev_haz = self.hazard_hits[-1] if self.hazard_hits else 0
        self.hazard_hits.append(prev_haz + int(info.get("hit_hazard", False)))

    @staticmethod
    def smooth(arr, window: int) -> np.ndarray:
        """Simple moving average."""
        a = np.asarray(arr, dtype=np.float64)
        if len(a) < window:
            return a
        kernel = np.ones(window) / window
        return np.convolve(a, kernel, mode="valid")


# ═══════════════════════════════════════════════════════════════════════
# Retention test (catastrophic forgetting measurement)
# ═══════════════════════════════════════════════════════════════════════

def measure_retention(
    agent: CognitiveAgent,
    env: NonStationaryGrid,
    steps: int = 100,
) -> float:
    """
    Replay the *current* regime for `steps` and measure how well the
    agent predicts observations (low VFE → good retention).

    Returns the mean VFE over the test episode.
    """
    vfe_sum = 0.0
    obs_view, obs_internal = env._observe()
    for _ in range(steps):
        obs_idx = env.dominant_obs_type(obs_view)
        _, vfe = agent.act(obs_view, obs_internal, obs_idx)
        action = agent._prev_action
        obs_view, obs_internal, _ = env.step(action)
        vfe_sum += vfe
    return vfe_sum / max(steps, 1)


# ═══════════════════════════════════════════════════════════════════════
# Main benchmark runner
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(cfg: Config) -> MetricRecorder:
    """
    Execute the full three-regime continual-learning experiment.

    Returns the populated MetricRecorder for downstream analysis.
    """
    rng = np.random.default_rng(cfg.experiment.seed)
    env = NonStationaryGrid(cfg.grid, rng)
    agent = CognitiveAgent(cfg, rng)
    recorder = MetricRecorder()

    total = cfg.experiment.total_steps
    regime_len = cfg.experiment.regime_steps

    obs_view, obs_internal = env.reset()

    print(f"{'='*60}")
    print(f"  Continual-Learning Benchmark  ({total} steps, 3 regimes)")
    print(f"{'='*60}")

    for t in range(1, total + 1):
        # ── Regime shift at boundaries ────────────────────────────────
        if t > 1 and (t - 1) % regime_len == 0:
            new_regime = env.trigger_regime_shift()
            print(f"\n  *** Regime shift → {['A','B','C'][new_regime]} "
                  f"at step {t} ***\n")

        # ── Agent step ────────────────────────────────────────────────
        obs_idx = env.dominant_obs_type(obs_view)

        t0 = time.perf_counter()
        action, vfe = agent.act(obs_view, obs_internal, obs_idx)
        dt = time.perf_counter() - t0

        # ── Environment step ──────────────────────────────────────────
        obs_view, obs_internal, info = env.step(action)

        # ── Record metrics ────────────────────────────────────────────
        diag = agent.get_diagnostics()
        recorder.record(vfe, info, diag, dt)

        # ── Progress logging ──────────────────────────────────────────
        if t % 200 == 0:
            w = min(50, t)
            avg_vfe = np.mean(recorder.vfe[-w:])
            avg_dt = np.mean(recorder.wall_time[-w:]) * 1000
            sat = diag.get("clause_saturation", 0.0)
            print(
                f"  step {t:>5d} | regime {['A','B','C'][info['regime']]} | "
                f"VFE {avg_vfe:7.2f} | energy {info['energy']:4d} | "
                f"nodes {diag['num_memory_nodes']:3d} | "
                f"clauses {diag['active_clauses']:4d}/"
                f"{diag.get('available_clauses',0):4d} "
                f"({sat:.0%}) | "
                f"dt {avg_dt:6.2f} ms"
            )

    # ── Post-experiment retention test on Task A ──────────────────────
    print(f"\n{'─'*60}")
    print("  Retention test: replaying Regime A dynamics …")
    # Reset env to regime A
    env.reset()
    env.regime = 0
    ret_vfe = measure_retention(agent, env, steps=100)
    print(f"  Regime-A retention VFE (lower = better): {ret_vfe:.3f}")
    print(f"{'─'*60}")

    return recorder


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_results(rec: MetricRecorder, cfg: Config) -> None:
    """Generate and save the six stress-test benchmark plots."""
    out = cfg.experiment.plot_output_dir
    os.makedirs(out, exist_ok=True)
    w = cfg.experiment.eval_window

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        "Controlled Destruction Stress Test  "
        "(Dark-Room broken, BME overdrive, MDL off)",
        fontsize=13, fontweight="bold",
    )

    # Regime shift markers (shared helper)
    regime_arr = np.array(rec.regime)
    shifts = np.where(np.diff(regime_arr) != 0)[0]

    def _vlines(ax):
        for s in shifts:
            ax.axvline(s, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    # ── 1. Adaptation latency (smoothed VFE) ─────────────────────────
    ax = axes[0, 0]
    vfe_smooth = MetricRecorder.smooth(rec.vfe, w)
    ax.plot(vfe_smooth, linewidth=0.8, color="steelblue")
    _vlines(ax)
    ax.set_title("1. Adaptation Latency (smoothed VFE)")
    ax.set_xlabel("Step")
    ax.set_ylabel("VFE")

    # ── 2. Retention / forgetting ─────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(rec.consumed, label="Resources consumed", color="green")
    ax.plot(rec.hazard_hits, label="Hazard hits", color="red")
    _vlines(ax)
    ax.set_title("2. Retention / Catastrophic Forgetting")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative count")
    ax.legend(fontsize=8)

    # ── 3. Structural complexity (nodes + clauses) ────────────────────
    ax = axes[1, 0]
    ax.plot(rec.num_nodes, label="Memory nodes", color="darkorange")
    ax.plot(rec.available_clauses, label="Available clauses",
            color="purple", alpha=0.5, linestyle="--")
    ax.plot(
        MetricRecorder.smooth(rec.num_clauses, w),
        label="Active clauses (smooth)",
        color="purple",
        alpha=0.9,
    )
    _vlines(ax)
    ax.set_title("3. Structural Explosion (MDL disabled)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # ── 4. Clause saturation ratio ────────────────────────────────────
    ax = axes[1, 1]
    sat_smooth = MetricRecorder.smooth(rec.clause_saturation, w)
    ax.plot(sat_smooth, linewidth=0.9, color="crimson")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.4,
               label="Full saturation")
    _vlines(ax)
    ax.set_title("4. Tsetlin Clause Saturation (Active / Available)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Saturation ratio")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8)

    # ── 5. Computational cost ─────────────────────────────────────────
    ax = axes[2, 0]
    dt_ms = np.array(rec.wall_time) * 1000
    dt_smooth = MetricRecorder.smooth(dt_ms, w)
    ax.plot(dt_smooth, linewidth=0.8, color="teal", label="Wall-clock (ms)")
    ax2 = ax.twinx()
    ops_smooth = MetricRecorder.smooth(rec.int_ops, w)
    ax2.plot(ops_smooth, linewidth=0.6, color="salmon", alpha=0.7,
             label="Cum. int ops")
    _vlines(ax)
    ax.set_title("5. Computational Cost")
    ax.set_xlabel("Step")
    ax.set_ylabel("ms / step")
    ax2.set_ylabel("Cumulative int ops")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # ── 6. Energy over time ───────────────────────────────────────────
    ax = axes[2, 1]
    energy_smooth = MetricRecorder.smooth(rec.energy, w)
    ax.plot(energy_smooth, linewidth=0.8, color="goldenrod")
    _vlines(ax)
    ax.set_title("6. Agent Energy (Dark-Room break validation)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out, "stress_test_results.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Plots saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
# Stand-alone entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    rec = run_benchmark(DEFAULT_CONFIG)
    plot_results(rec, DEFAULT_CONFIG)
