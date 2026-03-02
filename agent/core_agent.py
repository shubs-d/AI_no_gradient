"""
core_agent.py
=============
Main cognitive loop that binds together:

  1. Active Inference Engine   → belief updating, action selection
  2. Memory Graph              → spreading activation, binary features
  3. Tsetlin Machine           → logic-based policy learning
  4. Structure Learner         → BME expansion / BMR pruning

The agent follows a simple sense → infer → remember → decide → learn
cycle on every time-step.  No gradient computation, no floating-point
weight matrices — all learning is count-based or logic-based.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from config import Config, NUM_ACTIONS
from agent.active_inference import ActiveInferenceEngine
from agent.memory_graph import MemoryGraph
from agent.tsetlin_logic import TsetlinMachine
from agent.structure_learning import StructureLearner


class CognitiveAgent:
    """
    Gradient-free cognitive agent that combines discrete active inference,
    spreading-activation memory, and Tsetlin-machine logic.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

        # ── Module instantiation ──────────────────────────────────────
        # Start with a modest number of hidden states;
        # structure learning will grow it as needed.
        self._initial_states = 16

        self.inference = ActiveInferenceEngine(
            num_states=self._initial_states,
            cfg=cfg.inference,
        )
        self.memory = MemoryGraph(cfg.memory)
        self.tsetlin = TsetlinMachine(cfg.tsetlin, rng)
        self.structure = StructureLearner(cfg.structure)

        # Seed the memory graph with one node per observation symbol
        for i in range(cfg.inference.num_obs_symbols):
            self.memory.add_node(label=f"obs_{i}")

        # ── Tracking variables ────────────────────────────────────────
        self._prev_state: int = 0            # MAP state at t−1
        self._prev_action: int = 0           # action at t−1
        self._step: int = 0

    # ── Sense–Infer–Remember–Decide–Learn cycle ──────────────────────

    def act(
        self,
        ego_view: np.ndarray,
        internal: np.ndarray,
        obs_idx: int,
    ) -> Tuple[int, float]:
        """
        Full cognitive cycle.

        Parameters
        ----------
        ego_view : ndarray   5×5 egocentric grid view (informational).
        internal : ndarray   1-D internal state (energy etc.).
        obs_idx  : int       Dominant observation type in the view.

        Returns
        -------
        action : int          Chosen action.
        vfe    : float        Variational Free Energy (surprise).
        """
        self._step += 1

        # ── 1. SENSE: ground observation in memory graph ──────────────
        if obs_idx < self.memory.num_nodes:
            self.memory.inject_observation(obs_idx, strength=2.0)

        # Inject energy as activation on node 0 (a proxy)
        energy_strength = float(internal[0]) * 2.0
        if self.memory.num_nodes > 0:
            self.memory.inject_observation(0, strength=energy_strength)

        # ── 2. INFER: update beliefs using active inference ───────────
        vfe = self.inference.update_belief(obs_idx)

        # ── 3. REMEMBER: propagate activation through memory graph ───
        self.memory.step()

        # Hebbian edge strengthening between previous & current obs nodes
        cur_state = self.inference.most_likely_state()
        if self._step > 1 and obs_idx < self.memory.num_nodes:
            prev_obs_node = min(self._prev_state, self.memory.num_nodes - 1)
            self.memory.strengthen_edge(prev_obs_node, obs_idx)

        # ── 4. DECIDE: combine active-inference EFE + Tsetlin vote ───
        # Get binary features from the memory graph
        binary_features = self.memory.get_binary_features(
            self.cfg.tsetlin.num_literals
        )

        # Active-inference preferred action (epistemic + pragmatic)
        ai_action = self.inference.select_action()

        # Tsetlin machine vote
        tm_action = self.tsetlin.predict(binary_features)

        # Blend: use AI action primarily, but let TM override when its
        # vote confidence is high relative to the threshold
        votes = self.tsetlin.vote(binary_features)
        max_vote = int(votes.max())
        T = self.cfg.tsetlin.threshold

        if max_vote > T // 2:
            # TM is confident — use its suggestion
            action = tm_action
        else:
            # Fall back to active-inference EFE
            action = ai_action

        # Exploration: small ε-random chance to override
        if self.rng.random() < max(0.01, 0.1 * (vfe / 5.0)):
            action = int(self.rng.integers(NUM_ACTIONS))

        # ── 5. LEARN ─────────────────────────────────────────────────

        # 5a. Update Dirichlet counts (pure counting)
        self.inference.update_counts(
            prev_state=self._prev_state,
            action=self._prev_action,
            next_state=cur_state,
            obs_idx=obs_idx,
        )

        # 5b. Update Tsetlin Machine (VFE-driven feedback)
        self.tsetlin.update(binary_features, target_action=ai_action, vfe=vfe)

        # 5c. Structure learning
        self.structure.record_vfe(vfe)

        #   BME: expand if cumulative surprise is too high
        if self.structure.should_expand():
            new_node = self.memory.add_node(
                label=f"auto_{self.memory.num_nodes}"
            )
            # Expand AI state space only up to max_ai_states cap
            max_states = self.cfg.structure.max_ai_states
            if self.inference.num_states < max_states:
                target = min(
                    max(self.inference.num_states + 1,
                        self.memory.num_nodes),
                    max_states,
                )
                self.inference.expand_state_space(target)
            # ── Dynamically grow Tsetlin literal/clause pool ────────
            new_lit = max(
                self.tsetlin.current_num_literals,
                self.memory.num_nodes,
            )
            self.tsetlin.expand_literals(new_lit)
            self.structure.acknowledge_expansion()

        #   BMR: periodic MDL pruning
        if self.structure.should_prune():
            self.structure.mdl_prune_graph(
                self.memory, self.tsetlin, recent_vfe=vfe
            )

        # ── Bookkeeping ──────────────────────────────────────────────
        self._prev_state = cur_state
        self._prev_action = action

        return action, vfe

    # ── Diagnostics ───────────────────────────────────────────────────

    def get_diagnostics(self) -> Dict:
        """
        Return a snapshot of the agent's internal complexity metrics.
        """
        return {
            "num_memory_nodes": self.memory.num_nodes,
            "num_hidden_states": self.inference.num_states,
            "active_clauses": self.tsetlin.total_clauses_active(),
            "available_clauses": self.tsetlin.total_clauses_available(),
            "clause_saturation": self.tsetlin.clause_saturation_ratio(),
            "lyapunov_energy": self.memory.lyapunov_energy(),
            "model_bits": StructureLearner.model_description_length(
                self.memory, self.tsetlin
            ),
            "tsetlin_int_ops": self.tsetlin.int_ops,
        }
