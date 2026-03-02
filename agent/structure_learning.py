"""
structure_learning.py
=====================
Two-timescale structural adaptation:

  • **Fast timescale** — Tsetlin automata converge within the current
    graph topology  (handled by ``tsetlin_logic.py``).
  • **Slow timescale** — Bayesian Model Expansion (BME) adds nodes/edges
    when accumulated surprise warrants it; Bayesian Model Reduction (BMR)
    prunes dead-weight via the Minimum Description Length (MDL) principle.

NLP Chatbot Extension (v2)
──────────────────────────
  • **Unknown-word BME** — When the user inputs a completely unrecognised
    word, VFE spikes.  The structure learner detects this and dynamically
    allocates a new Node in the memory graph for the unknown word,
    linking it to the current conversational context.
  • **Localised forgetting** — The Dirichlet forgetting rate (ω = 0.95)
    is applied only to the active sub-graph so the agent can adapt to
    new topics without catastrophic forgetting of basic grammar.

Bayesian Model Expansion (BME)
──────────────────────────────
Expansion is NOT triggered on every high-surprise observation.  Instead
we maintain a *rolling VFE accumulator* over a window of length T.
A new memory-graph node (and corresponding hidden state in the
active-inference model) is allocated **only** when:

    cumulative_VFE  >  τ                                     … (BME)

This prevents naive over-expansion in transient noise.

Bayesian Model Reduction (BMR) via MDL
──────────────────────────────────────
Every ``mdl_eval_interval`` steps the pruner evaluates:

    ΔTotalCost = ΔModelBits + ΔErrorBits

• ΔModelBits  — change in description length of the model (number of
  literals in Tsetlin clauses, edges in the memory graph).
• ΔErrorBits  — change in prediction error (VFE) if the structure were
  simplified.

If removing a node, merging two nodes, or deleting an unused Tsetlin
clause *reduces* total bit-cost, the simplification is applied.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from config import StructureConfig


class StructureLearner:
    """
    Manages BME (expansion) and BMR (pruning) for the memory graph
    and active-inference state space.
    """

    def __init__(self, cfg: StructureConfig) -> None:
        self.cfg = cfg

        # ── Rolling VFE accumulator for BME ───────────────────────────
        self._vfe_window: deque[float] = deque(maxlen=cfg.rolling_window_T)

        # Step counter for periodic MDL sweeps
        self._step: int = 0
        # Cooldown counter to throttle expansion rate
        self._steps_since_expansion: int = cfg.expansion_cooldown
    # ── BME: should we expand? ────────────────────────────────────────

    def record_vfe(self, vfe: float) -> None:
        """Push the latest VFE into the rolling window."""
        self._vfe_window.append(vfe)
        self._step += 1
        self._steps_since_expansion += 1

    def should_expand(self) -> bool:
        """
        Returns True if cumulative VFE over the rolling window exceeds
        the expansion threshold τ.

        STRESS-TEST: τ lowered to near-zero and window shortened so
        expansion triggers aggressively during regime shifts.

            Σ_{t ∈ window} VFE(t)  >  τ          … (BME)
        """
        # Respect cooldown to avoid quadratic blowup
        if self._steps_since_expansion < self.cfg.expansion_cooldown:
            return False
        # Allow triggering even before the window is full—just need
        # at least 2 samples to avoid single-step noise.
        if len(self._vfe_window) < min(2, self.cfg.rolling_window_T):
            return False
        cumulative = sum(self._vfe_window)
        return cumulative > self.cfg.vfe_expansion_tau

    def acknowledge_expansion(self) -> None:
        """
        Called after a new node has been created so we reset the
        accumulator and cooldown to avoid immediately triggering again.
        """
        self._vfe_window.clear()
        self._steps_since_expansion = 0

    # ── BMR: periodic MDL pruning ─────────────────────────────────────

    def should_prune(self) -> bool:
        """
        True every ``mdl_eval_interval`` steps, BUT only if MDL is
        enabled.  ★ DISABLED in stress-test mode via config flag.
        """
        if not self.cfg.mdl_enabled:
            return False                         # MDL pruning OFF
        return (self._step > 0
                and self._step % self.cfg.mdl_eval_interval == 0)

    def mdl_prune_graph(
        self,
        memory_graph,      # MemoryGraph  (avoid circular import)
        tsetlin,           # TsetlinMachine
        recent_vfe: float, # current avg VFE (a proxy for error bits)
    ) -> dict:
        """
        Sweep the memory graph and Tsetlin clauses for MDL reductions.

        Returns a dict summarising what was pruned / merged.

        Algorithm
        ---------
        1.  **Unused-node removal:**  If a node has been visited fewer
            than ``min_node_visits`` times, its description cost exceeds
            its predictive benefit → remove it.
        2.  **Node merging:**  For every pair of nodes whose outgoing-edge
            patterns are more similar than ``merge_similarity_thresh``,
            merge them (reduces model bits, minimal error-bits increase).
        3.  **Clause pruning:**  Tsetlin clauses used fewer than
            ``clause_usage_min`` times are reset → saves model bits.
        """
        cfg = self.cfg
        report = {"nodes_removed": 0, "nodes_merged": 0, "clauses_reset": 0}

        # ── 1. Unused-node removal ───────────────────────────────────
        n = memory_graph.num_nodes
        for nid in range(n):
            if memory_graph.visit_counts[nid] < cfg.min_node_visits:
                # Model-bits saved ≈ edges incident on this node
                model_bits_saved = float(
                    np.sum(memory_graph._raw_counts[nid, :n] > 0)
                    + np.sum(memory_graph._raw_counts[:n, nid] > 0)
                )
                # Error-bits cost  ≈ 0 (node was barely used)
                if model_bits_saved > 0:
                    memory_graph.remove_node(nid)
                    report["nodes_removed"] += 1

        # ── 2. Node merging ──────────────────────────────────────────
        n = memory_graph.num_nodes
        merged: set = set()
        for i in range(n):
            if i in merged:
                continue
            for j in range(i + 1, n):
                if j in merged:
                    continue
                sim = memory_graph.node_similarity(i, j)
                if sim >= cfg.merge_similarity_thresh:
                    # Merge j into i
                    memory_graph.merge_nodes(keep=i, remove=j)
                    merged.add(j)
                    report["nodes_merged"] += 1

        # ── 3. Clause pruning ────────────────────────────────────────
        A = tsetlin.num_actions
        C = tsetlin.cfg.num_clauses
        for a in range(A):
            for c in range(C):
                if tsetlin.clause_usage[a, c] < cfg.clause_usage_min:
                    bit_cost = tsetlin.get_clause_bit_cost(a, c)
                    if bit_cost > 0:   # has included literals but never fires
                        tsetlin.reset_clause(a, c)
                        report["clauses_reset"] += 1

        return report

    # ── Complexity metrics ────────────────────────────────────────────

    # ── Unknown-word BME (NLP Chatbot v2) ─────────────────────────────

    def handle_unknown_words(
        self,
        unknown_words: List[str],
        memory_graph,                  # MemoryGraph  (avoid circular import)
        current_context_ids: List[int],
        inference_engine=None,         # ActiveInferenceEngine
    ) -> List[int]:
        """
        Bayesian Model Expansion triggered by unrecognised vocabulary.

        When the user inputs words that have no existing node in the
        memory graph, the VFE prediction error will spike.  This method
        handles the structural response:

        1.  Allocate a **new Node** in the memory graph for each
            unknown word.
        2.  Link the new node to the current conversational context
            (the last few active nodes) with initial co-occurrence edges.
        3.  If an Active Inference engine is provided, expand its state
            space to accommodate the new structure.

        Parameters
        ----------
        unknown_words : List[str]
            Words not yet in the memory graph.
        memory_graph : MemoryGraph
            The agent's spreading-activation memory graph.
        current_context_ids : List[int]
            Node IDs of the currently active conversational context
            (e.g., nodes from the most recent user tokens).
        inference_engine : ActiveInferenceEngine, optional
            If provided, state space is expanded in parallel.

        Returns
        -------
        new_node_ids : List[int]
            IDs of the newly allocated nodes.
        """
        new_ids: List[int] = []

        for word in unknown_words:
            # Respect cooldown to avoid quadratic blowup
            if self._steps_since_expansion < self.cfg.expansion_cooldown:
                break

            # Allocate new lexical node
            nid = memory_graph.get_or_create_word_node(word)
            new_ids.append(nid)

            # Link to current conversational context
            for ctx_id in current_context_ids[-5:]:
                # Bidirectional co-occurrence edges
                memory_graph.strengthen_edge(ctx_id, nid)
                memory_graph.strengthen_edge(nid, ctx_id)

            # Reset accumulator for this expansion event
            self._vfe_window.clear()
            self._steps_since_expansion = 0

        # Expand AI state space if needed
        if inference_engine is not None and new_ids:
            max_states = self.cfg.max_ai_states
            if inference_engine.num_states < max_states:
                target = min(
                    max(inference_engine.num_states + len(new_ids),
                        memory_graph.num_nodes),
                    max_states,
                )
                inference_engine.expand_state_space(target)

        return new_ids

    @staticmethod
    def model_description_length(memory_graph, tsetlin) -> float:
        """
        Total model description length in bits.

        model_bits  = graph_edge_bits + clause_bits

        graph_edge_bits = Σ_{(i,j) with count>0}  log2(max_count)
        clause_bits     = Σ_{a,c}  clause_bit_cost(a, c)
        """
        n = memory_graph.num_nodes
        raw = memory_graph._raw_counts[:n, :n]
        edge_count = int(np.sum(raw > 0))
        max_count = max(int(raw.max()), 1)
        graph_bits = edge_count * np.log2(max_count + 1)

        clause_bits = 0.0
        A = tsetlin.num_actions
        C = tsetlin.cfg.num_clauses
        for a in range(A):
            for c in range(C):
                clause_bits += tsetlin.get_clause_bit_cost(a, c)

        return graph_bits + clause_bits
