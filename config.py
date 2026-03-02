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
# 1a. ENVIRONMENT — Grid World (legacy, preserved for benchmarks)
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
# 1b. ENVIRONMENT — Chatbot  (replaces grid world for NLP pivot)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChatConfig:
    """Parameters for the conversational chatbot environment."""
    initial_energy: int = 100            # Agent starting energy
    initial_affiliation: float = 50.0    # Social bonding drive
    energy_cost_per_turn: int = 2        # Passive drain each turn
    energy_gain_positive: int = 15       # Energy from positive feedback
    energy_loss_negative: int = 10       # Energy penalty for negative feedback
    affiliation_gain: float = 5.0        # Affiliation bump from positive
    affiliation_decay: float = 0.98      # Per-turn multiplicative decay
    dark_room_penalty: float = 40.0      # Penalty for silence / repetition
    repeat_tolerance: int = 3            # Consecutive identical replies
    num_obs_buckets: int = 64            # Hash buckets for obs index
    top_k: int = 25                      # TopK nodes for comprehension
    activation_spike_strength: float = 2.0  # Injection strength per token
    cooccurrence_window: int = 3         # Bigram/trigram window for edges

# ──────────────────────────────────────────────────────────────────────
# 1c. POLICY LAYER — Sparse Bayesian Competitive Policy
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyConfig:
    """Parameters for the Dirichlet-based text generation policy."""
    dirichlet_prior: float = 1.0         # Symmetric Dirichlet for bigrams
    max_response_len: int = 18           # Hard cap on generated tokens
    eos_prior_boost: float = 0.9         # Lower EOS boost allows fuller replies
    policy_omega: float = 0.95           # Localised forgetting rate
    policy_eta: float = 1.0              # Learning rate for bigram counts
    feedback_strength: float = 2.0       # Type-I / Type-II adjustment

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
    vfe_expansion_tau: float = 4.0       # Higher τ avoids noisy over-expansion
    rolling_window_T: int = 30           # Smoother surprise integration window
    expansion_cooldown: int = 15         # Min steps between consecutive BME
    max_ai_states: int = 48              # Cap AI hidden-state dimension
    mdl_eval_interval: int = 100         # Steps between BMR pruning sweeps
    mdl_enabled: bool = True             # ★ MDL ENABLED for v3 scaling
    min_node_visits: int = 3             # Prune nodes visited fewer times
    merge_similarity_thresh: float = 0.9 # Cosine-like overlap for node merging
    clause_usage_min: int = 5            # Prune clauses used fewer times
    # ── MDL edge pruning (v3) ─────────────────────────────────────
    edge_mdl_interval: int = 200         # Steps between edge-MDL sweeps
    edge_min_count: int = 2              # Prune edges with count < this
    edge_vfe_benefit_thresh: float = 0.1 # Marginal VFE benefit to keep edge

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

# ──────────────────────────────────────────────────────────────────────
# 7a. GRAMMAR INDUCTION via Tsetlin Machine  (v3)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GrammarConfig:
    """Parameters for logic-based grammar induction."""
    num_syntactic_features: int = 12     # Binary features encoding POS/tense
    role_boost_subject: float = 2.0      # Activation boost for subject nodes
    role_boost_object: float = 1.5       # Activation boost for object nodes
    role_boost_predicate: float = 1.8    # Activation boost for predicate nodes
    triplet_edge_strength: int = 3       # Initial edge weight for triplet links


@dataclass(frozen=True)
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tsetlin: TsetlinConfig = field(default_factory=TsetlinConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    grammar: GrammarConfig = field(default_factory=GrammarConfig)


# Default global instance – importable everywhere
DEFAULT_CONFIG = Config()
