"""
config.py
=========
Central configuration for the gradient-free cognitive architecture.

All hyperparameters live here so that every module imports from a single
source of truth.  Nothing in this file performs computation; it is pure
declarative configuration.
"""

from dataclasses import dataclass, field
from typing import Tuple

# ──────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridConfig:
    """Parameters for the 2-D non-stationary grid world."""
    width: int = 15                      # Grid columns
    height: int = 15                     # Grid rows
    egocentric_radius: int = 2           # Radius of the 5×5 local view
    num_resources: int = 8               # Initial resource count
    num_hazards: int = 6                 # Initial hazard count
    num_obstacles: int = 12              # Static obstacles
    initial_energy: int = 100            # Agent starting energy
    energy_gain_resource: int = 20       # Energy from consuming a resource
    energy_loss_hazard: int = 30         # Energy penalty for stepping on hazard
    energy_cost_step: int = 1            # Energy cost per movement action
    regime_shift_interval: int = 200     # Steps between automatic regime shifts

# Cell types (integer codes for the observation matrix)
CELL_EMPTY: int    = 0
CELL_OBSTACLE: int = 1
CELL_RESOURCE: int = 2
CELL_HAZARD: int   = 3
CELL_AGENT: int    = 4

# Action codes  (Consume removed — agent auto-consumes on contact)
ACTION_UP: int      = 0
ACTION_DOWN: int    = 1
ACTION_LEFT: int    = 2
ACTION_RIGHT: int   = 3
NUM_ACTIONS: int    = 4

# ──────────────────────────────────────────────────────────────────────
# 2. ACTIVE INFERENCE
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InferenceConfig:
    """Parameters for the Dirichlet-count generative model."""
    dirichlet_prior: float = 1.0         # Symmetric Dirichlet prior (α₀)
    num_obs_symbols: int = 5             # Number of distinct observation types
    efe_horizon: int = 3                 # Planning depth for Expected Free Energy
    efe_epistemic_weight: float = 1.0    # Weight on epistemic (info-gain) term
    efe_pragmatic_weight: float = 4.0    # Weight on pragmatic (goal) — boosted
    preferred_obs: int = CELL_RESOURCE   # Observation the agent "wants" to see
    # ── Dirichlet forgetting  θ_{t+1} = ω·θ_t + η·χ_t ─────────────
    dirichlet_decay_omega: float = 0.95  # Forgetting rate ω
    dirichlet_learning_rate: float = 1.0 # Learning rate η for new evidence
    # ── Extreme preference (anti-Dark-Room) ────────────────────────
    consume_preference_boost: float = 10.0  # Inflated C-vector log-preference

# ──────────────────────────────────────────────────────────────────────
# 3. MEMORY GRAPH (Spreading Activation)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryConfig:
    """Parameters for the spreading-activation memory graph."""
    decay_lambda: float = 0.85           # Activation decay λ  (0 < λ < 1)
    stability_epsilon: float = 0.05      # Lyapunov margin ε
    max_activation: float = 10.0         # Clamp ceiling to avoid overflow
    initial_capacity: int = 64           # Pre-allocated node slots
    activation_threshold: float = 0.1    # Below this, node is "silent"
    edge_weight_increment: int = 1       # Hebbian co-activation bump (integer)
    edge_weight_max: int = 100           # Max raw edge counter

# ──────────────────────────────────────────────────────────────────────
# 4. TSETLIN MACHINE
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TsetlinConfig:
    """Parameters for the decentralised Tsetlin Machine logic engine."""
    num_clauses: int = 40                # Initial clauses per action class
    num_literals: int = 32               # Initial binary features from memory graph
    num_states: int = 100                # Tsetlin automaton states per side
    threshold: int = 15                  # Clause sum threshold T
    specificity: float = 3.0             # s  – controls inclusion probability
    type1_prob: float = 0.5              # P(Type-I feedback | positive clause)
    type2_prob: float = 0.5              # P(Type-II feedback | negative clause)
    clauses_per_literal: float = 1.25    # Dynamic scaling: C = ceil(L * ratio)

# ──────────────────────────────────────────────────────────────────────
# 5. STRUCTURE LEARNING (BME / BMR)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructureConfig:
    """Parameters for two-timescale expansion and MDL pruning."""
    vfe_expansion_tau: float = 0.05      # τ near-zero → BME OVERDRIVE
    rolling_window_T: int = 10           # Shorter window → faster triggering
    expansion_cooldown: int = 5          # Min steps between consecutive BME
    max_ai_states: int = 48              # Cap AI hidden-state dimension
    mdl_eval_interval: int = 100         # Steps between BMR pruning sweeps
    mdl_enabled: bool = False            # ★ MDL DISABLED for stress test
    min_node_visits: int = 3             # Prune nodes visited fewer times
    merge_similarity_thresh: float = 0.9 # Cosine-like overlap for node merging
    clause_usage_min: int = 5            # Prune clauses used fewer times

# ──────────────────────────────────────────────────────────────────────
# 6. EXPERIMENT
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters for the continual-learning benchmark harness."""
    total_steps: int = 3000              # Total environment steps
    regime_steps: int = 1000             # Steps per regime (A, B, C)
    eval_window: int = 50                # Sliding window for metric smoothing
    seed: int = 42                       # Reproducibility
    plot_output_dir: str = "results"     # Where to save figures


# ──────────────────────────────────────────────────────────────────────
# 7. AGGREGATE CONFIG  (single import for convenience)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tsetlin: TsetlinConfig = field(default_factory=TsetlinConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# Default global instance – importable everywhere
DEFAULT_CONFIG = Config()
