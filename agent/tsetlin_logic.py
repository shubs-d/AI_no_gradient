"""
tsetlin_logic.py
================
Decentralised Tsetlin Machine for learning state-action policies from
binary features produced by the spreading-activation memory graph.

v3 — Grammar Feature Literals
──────────────────────────────
The literal space is now partitioned into two regions:

    [ memory-graph binary features  |  syntactic context features ]
    [         L_graph bits          |   L_syntax bits (12 default) ]

The TM learns propositional rules that jointly condition on *what*
words are active (graph features) and *where in the sentence structure*
we are (syntactic features):

    IF cat_active ∧ has_subject ∧ ¬has_verb → vote for "verb" pathway
    IF ate_active ∧ has_verb ∧ ¬has_object → vote for "noun" pathway

The ``combine_features()`` utility concatenates graph + syntax vectors
into the unified input that ``_evaluate_all_clauses`` operates on.

Design principles
─────────────────
• **No gradients.**  All updates are integer increments/decrements on
  finite-state Tsetlin automata.
• **VFE-driven feedback.**  Type I / Type II feedback is gated by the
  active-inference prediction error (surprise), NOT by external reward.
• **Embarrassingly parallel.**  Clause evaluations are fully decoupled;
  the complexity is  O(C × L)  where C = clauses, L = literals.
• **Binary literals** come from ``MemoryGraph.get_binary_features()``
  concatenated with ``build_syntactic_features()`` from the policy layer.

Architecture (per action class)
───────────────────────────────
  C clauses, each with 2L Tsetlin automata  (L literals + L negated).
  Automaton state ∈ [1, 2N]  (N = num_states in config).
  States  ≤ N  → exclude literal.   States  > N  → include literal.

Clause polarity: even-index clauses are positive (+), odd are negative (−).
The output vote for action *a* is:

  v(a) = Σ_{j positive} clause_j(x) − Σ_{j negative} clause_j(x)

Action selection uses a softmax-free integer threshold comparison
(exactly how Tsetlin machines are designed to work).

Feedback
────────
• **Type I (reinforce):**  Applied to clauses whose action matches the
  VFE-selected "desired" action.  Strengthens inclusion of matching
  literals, weakens inclusion of non-matching ones.
• **Type II (penalise):**  Applied to clauses for competing actions.
  Promotes literal inclusion to increase discriminability.
"""

from __future__ import annotations

import numpy as np

from config import TsetlinConfig, NUM_ACTIONS

# Number of syntactic feature bits appended to graph features
NUM_SYNTAX_BITS: int = 12


class TsetlinMachine:
    """
    Multi-class Tsetlin Machine mapping binary features → action votes.

    Parameters
    ----------
    cfg : TsetlinConfig
        num_clauses, num_literals, num_states, threshold, specificity, etc.
    rng : numpy.random.Generator
        Seeded RNG for reproducibility.
    """

    def __init__(self, cfg: TsetlinConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.num_actions = NUM_ACTIONS
        C = cfg.num_clauses     # initial clauses PER action class
        L = cfg.num_literals    # initial binary features
        N = cfg.num_states      # automaton states per side
        A = self.num_actions

        # Current dimensions (may grow dynamically)
        self.current_num_clauses: int = C
        self.current_num_literals: int = L

        # ── Tsetlin automaton states ──────────────────────────────────
        # Shape: (num_actions, num_clauses, 2 * num_literals)
        # Axis-2:  first L entries = positive literal, next L = negated
        # Initialise to the "exclude" side  (state = N, just below threshold)
        self._ta = np.full((A, C, 2 * L), N, dtype=np.int32)

        # ── Clause usage counters (for MDL pruning) ───────────────────
        self.clause_usage = np.zeros((A, C), dtype=np.int64)

        # ── Integer ops counter (for compute benchmarking) ────────────
        self.int_ops: int = 0

    # ── Vectorised clause evaluation ─────────────────────────────────

    def _evaluate_all_clauses(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate ALL clauses across ALL actions in one vectorised pass.

        Returns
        -------
        fires : ndarray of shape (A, C), dtype int64
            1 where the clause fires on input *x*, 0 otherwise.
        """
        C = self.current_num_clauses
        N = self.cfg.num_states
        clause_L = self._ta.shape[2] // 2
        eff_L = min(len(x), clause_L)

        ta = self._ta[:, :C, :]                              # (A, C, 2·cL)

        inc_pos = ta[:, :, :eff_L] > N                       # (A, C, eL)
        inc_neg = ta[:, :, clause_L:clause_L + eff_L] > N    # (A, C, eL)

        x_eff = x[:eff_L]                                    # (eL,)

        # Unsatisfied ⇔ included AND input wrong
        pos_unsat = inc_pos & (x_eff == 0)                    # broadcast (eL,)
        neg_unsat = inc_neg & (x_eff == 1)

        fires = ~(pos_unsat.any(axis=2) | neg_unsat.any(axis=2))
        self.int_ops += self.num_actions * C * 2 * eff_L
        return fires.astype(np.int64)

    # ── Voting ────────────────────────────────────────────────────────

    def vote(self, x: np.ndarray) -> np.ndarray:
        """
        Compute integer votes for each action class (vectorised).

        v(a) = Σ_{even clauses} clause(x) − Σ_{odd clauses} clause(x)

        Returns
        -------
        votes : ndarray of shape (num_actions,), dtype int64
        """
        C = self.current_num_clauses
        T = self.cfg.threshold

        fires = self._evaluate_all_clauses(x)          # (A, C)

        # Polarity: even=+1, odd=−1
        pol = np.ones(C, dtype=np.int64)
        pol[1::2] = -1

        votes = (fires * pol[np.newaxis, :]).sum(axis=1)

        # Track clause usage
        self.clause_usage[:, :C] += fires

        np.clip(votes, -T, T, out=votes)
        return votes

    def predict(self, x: np.ndarray) -> int:
        """Return the action with the highest vote (ties broken randomly)."""
        votes = self.vote(x)
        max_v = votes.max()
        candidates = np.flatnonzero(votes == max_v)
        return int(self.rng.choice(candidates))

    # ── Feedback (VFE-driven, NOT reward-driven) ──────────────────────

    def update(self, x: np.ndarray, target_action: int, vfe: float) -> None:
        """
        Apply Type I / Type II feedback driven by VFE (vectorised).

        High VFE (surprise) → more aggressive updates (higher feedback
        probability).  Low VFE → exploit, barely perturb automata.
        """
        C = self.current_num_clauses
        L = self.current_num_literals
        N = self.cfg.num_states
        s = self.cfg.specificity

        clause_L = self._ta.shape[2] // 2
        eff_L = min(L, len(x), clause_L)

        # Feedback probability scales with surprise
        p_feedback = min(1.0, max(0.05, vfe / 10.0))

        # Evaluate all clauses once
        fires = self._evaluate_all_clauses(x)               # (A, C)

        # Pre-compute input features
        x_eff = x[:eff_L]                                    # (eff_L,)
        x1 = x_eff == 1                                      # bool (eff_L,)
        x0 = ~x1

        # Index ranges into clause automaton axis
        pos_r = np.arange(eff_L)
        neg_r = np.arange(clause_L, clause_L + eff_L)

        # Polarity per clause
        pol = np.ones(C, dtype=np.int64)
        pol[1::2] = -1

        for a in range(self.num_actions):
            ta = self._ta[a]                                 # (C_max, 2·L_max), VIEW

            # Per-clause random gate
            fb = self.rng.random(C) < p_feedback             # (C,) bool

            if a == target_action:
                self._type1_vec(
                    ta, fires[a], pol, fb, x1, x0,
                    N, eff_L, s, C, pos_r, neg_r,
                )
            else:
                self._type2_vec(
                    ta, fires[a], pol, fb, x1, x0,
                    N, eff_L, C, pos_r, neg_r,
                )

    # ── Vectorised Type I (reinforce target action) ───────────────────

    def _type1_vec(
        self, ta, fires, pol, fb, x1, x0,
        N, eff_L, s, C, pos_r, neg_r,
    ) -> None:
        # ---- Reward: clause fires AND positive polarity ----
        r_mask = fb & (fires == 1) & (pol == 1)
        nr = int(r_mask.sum())
        if nr > 0:
            r_idx = np.where(r_mask)[0]
            pos_ta = ta[np.ix_(r_idx, pos_r)].copy()         # (nr, eff_L)
            neg_ta = ta[np.ix_(r_idx, neg_r)].copy()

            rand1 = self.rng.random((nr, eff_L))
            rand2 = self.rng.random((nr, eff_L))

            # Positive literals: x==1 → include(++),  x==0 → exclude with prob 1/s
            pos_ta += (x1[np.newaxis, :] & (pos_ta < 2 * N)).astype(np.int32)
            pos_ta -= (x0[np.newaxis, :] & (rand1 < 1.0 / s) & (pos_ta > 1)).astype(np.int32)

            # Negated literals: x_neg==1 (= x0) → include(++), x_neg==0 (= x1) → exclude
            neg_ta += (x0[np.newaxis, :] & (neg_ta < 2 * N)).astype(np.int32)
            neg_ta -= (x1[np.newaxis, :] & (rand2 < 1.0 / s) & (neg_ta > 1)).astype(np.int32)

            ta[np.ix_(r_idx, pos_r)] = pos_ta
            ta[np.ix_(r_idx, neg_r)] = neg_ta
            self.int_ops += nr * eff_L * 4

        # ---- Penalty: clause off OR negative polarity ----
        p_mask = fb & ~((fires == 1) & (pol == 1))
        np_ = int(p_mask.sum())
        if np_ > 0:
            p_idx = np.where(p_mask)[0]
            pos_ta = ta[np.ix_(p_idx, pos_r)].copy()
            neg_ta = ta[np.ix_(p_idx, neg_r)].copy()

            rand1 = self.rng.random((np_, eff_L))
            rand2 = self.rng.random((np_, eff_L))

            pos_ta -= ((rand1 < 1.0 / s) & (pos_ta > 1)).astype(np.int32)
            neg_ta -= ((rand2 < 1.0 / s) & (neg_ta > 1)).astype(np.int32)

            ta[np.ix_(p_idx, pos_r)] = pos_ta
            ta[np.ix_(p_idx, neg_r)] = neg_ta
            self.int_ops += np_ * eff_L * 2

    # ── Vectorised Type II (penalise competing actions) ───────────────

    def _type2_vec(
        self, ta, fires, pol, fb, x1, x0,
        N, eff_L, C, pos_r, neg_r,
    ) -> None:
        t2_mask = fb & (fires == 1) & (pol == 1)
        nt2 = int(t2_mask.sum())
        if nt2 > 0:
            t2_idx = np.where(t2_mask)[0]
            pos_ta = ta[np.ix_(t2_idx, pos_r)].copy()
            neg_ta = ta[np.ix_(t2_idx, neg_r)].copy()

            # Push toward including literals that are FALSE ⇒ discriminate
            pos_ta += (x0[np.newaxis, :] & (pos_ta < 2 * N)).astype(np.int32)
            neg_ta += (x1[np.newaxis, :] & (neg_ta < 2 * N)).astype(np.int32)

            ta[np.ix_(t2_idx, pos_r)] = pos_ta
            ta[np.ix_(t2_idx, neg_r)] = neg_ta
            self.int_ops += nt2 * eff_L * 2

    # ── Pruning support ───────────────────────────────────────────────

    def get_clause_bit_cost(self, action: int, clause: int) -> float:
        """
        Approximate description length of a clause in bits.

        Counts the number of included literals (automaton state > N).
        Each included literal costs  log2(L)  bits to specify.
        """
        N = self.cfg.num_states
        L = self.current_num_literals
        if clause >= self.current_num_clauses:
            return 0.0
        included = int(np.sum(self._ta[action, clause] > N))
        if included == 0:
            return 0.0
        return included * np.log2(max(L, 2))

    def reset_clause(self, action: int, clause: int) -> None:
        """Reset a clause to the initial (exclude-all) state."""
        N = self.cfg.num_states
        self._ta[action, clause, :] = N
        self.clause_usage[action, clause] = 0

    def total_clauses_active(self) -> int:
        """Count how many clauses have at least one included literal."""
        N = self.cfg.num_states
        C = self.current_num_clauses
        return int(np.any(self._ta[:, :C, :] > N, axis=2).sum())

    def total_clauses_available(self) -> int:
        """Total clause slots across all action classes."""
        return self.num_actions * self.current_num_clauses

    def clause_saturation_ratio(self) -> float:
        """
        Ratio of active clauses to available clause slots.

        Values near 1.0 indicate the TM is structurally capped and
        needs more clauses to represent the growing feature space.
        """
        total = self.total_clauses_available()
        if total == 0:
            return 0.0
        return self.total_clauses_active() / total

    # ── Dynamic clause/literal expansion ──────────────────────────────

    def expand_literals(self, new_num_literals: int) -> None:
        """
        Grow the literal dimension when the memory graph adds nodes.

        New automata positions are initialised to N (exclude side).
        Clause count is also scaled via ``clauses_per_literal`` ratio.
        """
        old_L = self.current_num_literals
        if new_num_literals <= old_L:
            return

        N = self.cfg.num_states
        A = self.num_actions
        C_old = self.current_num_clauses
        delta_L = new_num_literals - old_L

        # Expand automaton array along literal axis
        # Current shape: (A, C_old, 2*old_L)
        # We need to insert delta_L columns after position old_L
        # (positive literals) and delta_L at the end (negated literals)
        old_ta = self._ta[:, :C_old, :]
        pos_part = old_ta[:, :, :old_L]                          # (A, C, old_L)
        neg_part = old_ta[:, :, old_L:]                          # (A, C, old_L)

        pad_pos = np.full((A, C_old, delta_L), N, dtype=np.int32)
        pad_neg = np.full((A, C_old, delta_L), N, dtype=np.int32)

        new_ta = np.concatenate(
            [pos_part, pad_pos, neg_part, pad_neg], axis=2
        )  # (A, C_old, 2*new_num_literals)

        self.current_num_literals = new_num_literals

        # ── Scale clause count proportionally ─────────────────────────
        target_C = max(
            C_old,
            int(np.ceil(new_num_literals * self.cfg.clauses_per_literal)),
        )
        # Make target_C even so polarity pairing is clean
        if target_C % 2 != 0:
            target_C += 1

        if target_C > C_old:
            delta_C = target_C - C_old
            pad_clauses = np.full(
                (A, delta_C, 2 * new_num_literals), N, dtype=np.int32
            )
            new_ta = np.concatenate([new_ta, pad_clauses], axis=1)

            # Expand usage counters
            pad_usage = np.zeros((A, delta_C), dtype=np.int64)
            self.clause_usage = np.concatenate(
                [self.clause_usage[:, :C_old], pad_usage], axis=1
            )
            self.current_num_clauses = target_C
        else:
            self.clause_usage = self.clause_usage[:, :C_old]

        self._ta = new_ta

    # ── Grammar-aware feature utilities ───────────────────────────────

    @staticmethod
    def combine_features(
        graph_features: np.ndarray,
        syntactic_features: np.ndarray,
    ) -> np.ndarray:
        """
        Concatenate memory-graph binary features with syntactic context
        features into a single input vector for clause evaluation.

        Parameters
        ----------
        graph_features : ndarray of shape (L_graph,), dtype int8
            Binary activation features from the spreading-activation
            graph (via ``MemoryGraph.get_binary_features()``).
        syntactic_features : ndarray of shape (L_syntax,), dtype int8
            Binary syntactic context features from
            ``build_syntactic_features()`` (default 12 bits).

        Returns
        -------
        combined : ndarray of shape (L_graph + L_syntax,), dtype int8
            Unified literal input for the TM.
        """
        return np.concatenate([graph_features, syntactic_features]).astype(np.int8)

    def vote_with_grammar(
        self,
        graph_features: np.ndarray,
        syntactic_features: np.ndarray,
    ) -> np.ndarray:
        """
        Compute action votes using combined graph + syntax features.

        This is the primary interface for grammar-aware action selection.
        The TM will learn clauses that condition on both *lexical context*
        (which words are active) and *syntactic context* (sentence
        structure so far).

        If the combined feature vector is longer than the current literal
        space, the TM is automatically expanded via ``expand_literals()``.

        Parameters
        ----------
        graph_features : ndarray of shape (L_graph,), int8
        syntactic_features : ndarray of shape (L_syntax,), int8

        Returns
        -------
        votes : ndarray of shape (num_actions,), int64
        """
        x = self.combine_features(graph_features, syntactic_features)
        if len(x) > self.current_num_literals:
            self.expand_literals(len(x))
        return self.vote(x)

    def update_with_grammar(
        self,
        graph_features: np.ndarray,
        syntactic_features: np.ndarray,
        target_action: int,
        vfe: float,
    ) -> None:
        """
        Apply VFE-driven feedback using combined graph + syntax features.

        Parameters
        ----------
        graph_features : ndarray of shape (L_graph,), int8
        syntactic_features : ndarray of shape (L_syntax,), int8
        target_action : int
        vfe : float
        """
        x = self.combine_features(graph_features, syntactic_features)
        if len(x) > self.current_num_literals:
            self.expand_literals(len(x))
        self.update(x, target_action, vfe)
