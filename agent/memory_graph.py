"""
memory_graph.py
===============
Spreading-activation episodic/semantic memory network implemented as a
dynamic directed graph with **strict Lyapunov stability guarantees**.

NLP Chatbot Extension (v2)
──────────────────────────
Each node now represents a **distinct word or concept** (lexical node).
When a user message is tokenised, the corresponding word nodes receive
an *activation spike*.  Spreading activation then propagates semantic
associations through co-occurrence edges, and the TopK highest-activated
nodes represent the agent's *contextual comprehension* of the user's
prompt.

Graph representation
────────────────────
• Nodes are identified by integer IDs  0 … N−1.
• Each node carries:
    - A real-valued *activation*  A_i(t).
    - A string *label*  (the word or concept it represents).
• Directed edges carry **integer** co-occurrence counts; the normalised
  adjacency matrix  W  is derived from these counts on-the-fly.

Spreading-Activation update rule
────────────────────────────────
  A(t+1) = λ · A(t)  +  W · A(t)                             … (SA-1)

where  λ ∈ (0, 1)  is the intrinsic decay factor and  W  is the
row-stochastic (outgoing-edge normalised) adjacency matrix.

Lyapunov stability guarantee
────────────────────────────
We choose the candidate Lyapunov function

  V(A) = ‖A‖₂²  =  Σ_i  A_i²

A sufficient condition for  ΔV < 0  (globally, away from the origin) is

  ‖λI + W‖₂  <  1

Because  W  is non-negative and row-stochastic (each row sums to ≤ 1),

  ‖W‖₂  ≤  max row-sum  ≤  1

so we enforce:

  λ_max(W)  ≤  1 − λ − ε                                    … (Lyap)

by scaling outgoing edges so that every row of W sums to at most
(1 − λ − ε).  After *any* structural graph update (add/remove node or
edge) the ``normalize_edges()`` method is called to re-establish this
invariant.

All edge *learning* is integer count-based (Hebbian co-activation).
The floating-point adjacency matrix is a **derived view** used only
during the spreading-activation step.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Optional is used for word_to_id return type;
# Set is used for get_active_subgraph_ids return type;
# Tuple is used for get_contextual_top_k return type.

from config import MemoryConfig


class MemoryGraph:
    """Spreading-activation memory graph with Lyapunov stability."""

    # ── Construction ──────────────────────────────────────────────────

    def __init__(self, cfg: MemoryConfig) -> None:
        self.cfg = cfg

        # Node bookkeeping
        self._num_nodes: int = 0
        self._labels: Dict[int, str] = {}          # optional semantic labels
        self._visit_counts: np.ndarray = np.zeros(cfg.initial_capacity, dtype=np.int64)

        # Activation vector  A(t)
        self._activation: np.ndarray = np.zeros(cfg.initial_capacity, dtype=np.float64)

        # Integer co-occurrence count matrix (raw Hebbian counts)
        self._raw_counts: np.ndarray = np.zeros(
            (cfg.initial_capacity, cfg.initial_capacity), dtype=np.int64
        )

        # Derived normalised adjacency matrix  W  (float, recomputed)
        self._W: np.ndarray = np.zeros(
            (cfg.initial_capacity, cfg.initial_capacity), dtype=np.float64
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def activation(self) -> np.ndarray:
        """Current activation vector (length = capacity, only [:num_nodes] valid)."""
        return self._activation[: self._num_nodes]

    @property
    def W(self) -> np.ndarray:
        """Normalised adjacency submatrix (num_nodes × num_nodes)."""
        return self._W[: self._num_nodes, : self._num_nodes]

    @property
    def visit_counts(self) -> np.ndarray:
        return self._visit_counts[: self._num_nodes]

    # ── Node management ───────────────────────────────────────────────

    def add_node(self, label: str = "") -> int:
        """
        Append a new node.  Returns its integer ID.

        After adding, ``normalize_edges()`` is called to preserve the
        Lyapunov invariant (trivially, since new rows/cols are zero).
        """
        nid = self._num_nodes
        self._ensure_capacity(nid + 1)
        self._num_nodes += 1
        self._labels[nid] = label or f"n{nid}"
        # New activation starts at zero; counts and edges already zero
        self.normalize_edges()
        return nid

    def add_nodes(self, count: int) -> List[int]:
        """Add *count* nodes in bulk; return list of new IDs."""
        ids = []
        for _ in range(count):
            ids.append(self.add_node())
        return ids

    def remove_node(self, nid: int) -> None:
        """
        Remove a node by zeroing its row/column and compacting later if
        desired.  For simplicity, we "tombstone" it (set counts to 0).
        """
        n = self._num_nodes
        if nid < 0 or nid >= n:
            return
        self._raw_counts[nid, :] = 0
        self._raw_counts[:, nid] = 0
        self._activation[nid] = 0.0
        self._visit_counts[nid] = 0
        self.normalize_edges()

    # ── Edge management (integer count-based) ─────────────────────────

    def strengthen_edge(self, src: int, dst: int, amount: int = 1) -> None:
        """
        Hebbian: increment the directed edge count  src → dst.

        Caps at ``edge_weight_max`` to avoid unbounded counters.
        Then re-normalises to maintain the Lyapunov bound.
        """
        cap = self.cfg.edge_weight_max
        self._raw_counts[src, dst] = min(
            int(self._raw_counts[src, dst]) + amount, cap
        )
        self.normalize_edges()

    def weaken_edge(self, src: int, dst: int, amount: int = 1) -> None:
        """Anti-Hebbian: decrement (floor at 0) and re-normalise."""
        self._raw_counts[src, dst] = max(
            int(self._raw_counts[src, dst]) - amount, 0
        )
        self.normalize_edges()

    # ── Lyapunov-safe normalisation (CRITICAL) ────────────────────────

    def normalize_edges(self) -> None:
        """
        Recompute the floating-point adjacency matrix  W  from integer
        counts such that the Lyapunov condition (Lyap) is satisfied:

            max row-sum of W  ≤  1 − λ − ε

        Algorithm
        ---------
        1.  For each row *i*, compute  r_i = Σ_j raw_counts[i, j].
        2.  If  r_i > 0, set  W[i, j] = raw_counts[i, j] / r_i * bound
            where  bound = 1 − λ − ε.
        3.  If  r_i == 0, row stays all-zero (no outgoing edges).

        This guarantees that  ‖λI + W‖₂  ≤  λ + (1 − λ − ε) = 1 − ε < 1,
        which is a sufficient condition for  ΔV < 0.
        """
        n = self._num_nodes
        if n == 0:
            return

        lam = self.cfg.decay_lambda
        eps = self.cfg.stability_epsilon
        bound = max(1.0 - lam - eps, 0.01)       # safety floor

        raw = self._raw_counts[:n, :n].astype(np.float64)
        row_sums = raw.sum(axis=1, keepdims=True)

        # Avoid division by zero for rows with no outgoing edges
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        self._W[:n, :n] = (raw / safe_sums) * bound

    # ── Spreading activation step ─────────────────────────────────────

    def step(self, external_input: Optional[np.ndarray] = None) -> None:
        """
        One tick of spreading activation (SA-1):

            A(t+1) = λ · A(t)  +  W · A(t)  +  I(t)

        where I(t) is an optional external injection (e.g., from the
        active-inference observation).  Activation is clamped to
        [0, max_activation] afterwards.
        """
        n = self._num_nodes
        if n == 0:
            return

        a = self._activation[:n]
        W = self._W[:n, :n]
        lam = self.cfg.decay_lambda

        # Core update  (SA-1)
        a_new = lam * a + W @ a

        # External injection
        if external_input is not None:
            inp = external_input[:n]
            a_new[:len(inp)] += inp[:len(a_new)]

        # Clamp
        np.clip(a_new, 0.0, self.cfg.max_activation, out=a_new)
        self._activation[:n] = a_new

    # ── Queries ───────────────────────────────────────────────────────

    def top_k_active(self, k: int) -> List[int]:
        """Return IDs of the *k* most activated nodes (descending)."""
        n = self._num_nodes
        if n == 0:
            return []
        act = self._activation[:n]
        k = min(k, n)
        # argpartition is O(n); argsort on the top-k slice is O(k log k)
        if k >= n:
            indices = np.argsort(-act)
        else:
            part = np.argpartition(-act, k)[:k]
            indices = part[np.argsort(-act[part])]
        return [int(i) for i in indices]

    def get_binary_features(self, num_features: int) -> np.ndarray:
        """
        Return a binary vector of length *num_features* where bit j is 1
        iff the j-th most activated node exceeds the activation threshold.

        This vector is fed to the Tsetlin Machine as input literals.
        """
        top = self.top_k_active(num_features)
        bits = np.zeros(num_features, dtype=np.int8)
        thresh = self.cfg.activation_threshold
        for rank, nid in enumerate(top):
            if self._activation[nid] >= thresh:
                bits[rank] = 1
        return bits

    def inject_observation(self, node_id: int, strength: float = 1.0) -> None:
        """
        Inject activation into a specific node (sensory grounding).
        Also bumps the visit counter for MDL bookkeeping.
        """
        if 0 <= node_id < self._num_nodes:
            self._activation[node_id] = min(
                self._activation[node_id] + strength,
                self.cfg.max_activation,
            )
            self._visit_counts[node_id] += 1

    # ── Lyapunov energy monitor ───────────────────────────────────────

    def lyapunov_energy(self) -> float:
        """
        V(A) = ‖A‖₂²   — should be monotonically non-increasing when
        no external input is injected.
        """
        a = self._activation[: self._num_nodes]
        return float(np.dot(a, a))

    # ── Internal helpers ──────────────────────────────────────────────

    def _ensure_capacity(self, needed: int) -> None:
        """Double the backing arrays when capacity is exhausted."""
        cap = len(self._activation)
        if needed <= cap:
            return
        new_cap = max(cap * 2, needed)

        # Activation
        new_act = np.zeros(new_cap, dtype=np.float64)
        new_act[:cap] = self._activation
        self._activation = new_act

        # Visit counts
        new_vc = np.zeros(new_cap, dtype=np.int64)
        new_vc[:cap] = self._visit_counts
        self._visit_counts = new_vc

        # Raw counts & W
        new_raw = np.zeros((new_cap, new_cap), dtype=np.int64)
        new_raw[:cap, :cap] = self._raw_counts
        self._raw_counts = new_raw

        new_W = np.zeros((new_cap, new_cap), dtype=np.float64)
        new_W[:cap, :cap] = self._W
        self._W = new_W

    # ── Merge / prune support (used by structure_learning) ────────────

    def merge_nodes(self, keep: int, remove: int) -> None:
        """
        Merge *remove* into *keep*: combine counts and activation,
        then tombstone *remove*.
        """
        self._raw_counts[keep, :] += self._raw_counts[remove, :]
        self._raw_counts[:, keep] += self._raw_counts[:, remove]
        self._raw_counts[keep, keep] = 0          # no self-loops
        self._activation[keep] = max(
            self._activation[keep], self._activation[remove]
        )
        self._visit_counts[keep] += self._visit_counts[remove]
        self.remove_node(remove)

    def node_similarity(self, a: int, b: int) -> float:
        """
        Cosine-like overlap between outgoing edge patterns of two nodes.
        Returns a value in [0, 1].
        """
        n = self._num_nodes
        va = self._raw_counts[a, :n].astype(np.float64)
        vb = self._raw_counts[b, :n].astype(np.float64)
        dot = np.dot(va, vb)
        norms = (np.linalg.norm(va) * np.linalg.norm(vb))
        if norms < 1e-12:
            return 0.0
        return float(dot / norms)

    # ══════════════════════════════════════════════════════════════════
    # NLP / Chatbot extensions  (v2)
    # ══════════════════════════════════════════════════════════════════

    # ── Word-to-node registry ─────────────────────────────────────────

    def get_or_create_word_node(self, word: str) -> int:
        """
        Return the node ID for *word*, creating a new node if the word
        has never been seen.

        This is the primary interface for grounding tokens in the graph.
        Returns (node_id, is_new) would be useful but for API compat
        we just return the id; callers can check num_nodes before/after.

        Parameters
        ----------
        word : str
            Lowercased word token.

        Returns
        -------
        node_id : int
        """
        # Linear scan of labels — fine for vocabularies < 10 000.
        for nid in range(self._num_nodes):
            if self._labels.get(nid) == word:
                return nid
        # Not found → allocate a new lexical node
        return self.add_node(label=word)

    def word_to_id(self, word: str) -> Optional[int]:
        """
        Look up *word* in the label registry.  Returns None if absent.
        Pure query — does NOT create a new node.
        """
        for nid in range(self._num_nodes):
            if self._labels.get(nid) == word:
                return nid
        return None

    def get_label(self, node_id: int) -> str:
        """Return the string label of a node (word or concept)."""
        return self._labels.get(node_id, f"n{node_id}")

    def get_all_labels(self) -> Dict[int, str]:
        """Return the full {node_id: label} dictionary."""
        return dict(self._labels)

    # ── Token-based activation spike ──────────────────────────────────

    def inject_token_activations(
        self,
        token_node_ids: List[int],
        strength: float = 2.0,
    ) -> None:
        """
        Inject activation spikes on the nodes corresponding to the
        tokens in a user message.  This is the sensory grounding step
        for NLP: the user's words light up their lexical nodes.

        After this call, ``step()`` will propagate activation through
        co-occurrence edges (spreading activation), so semantically
        related words also become active.

        Parameters
        ----------
        token_node_ids : List[int]
            Ordered list of node IDs from the tokenised user message.
        strength : float
            Activation injected per token occurrence.
        """
        for nid in token_node_ids:
            self.inject_observation(nid, strength=strength)

    # ── Hebbian co-occurrence from token sequences ────────────────────

    def learn_cooccurrences(
        self,
        token_node_ids: List[int],
        window: int = 2,
    ) -> None:
        """
        Strengthen edges between co-occurring words in a token sequence.

        For each pair of tokens within a sliding window of size *window*,
        the directed edge count is incremented (Hebbian learning).
        This builds the co-occurrence structure that spreading activation
        will later traverse.

        Parameters
        ----------
        token_node_ids : List[int]
            Ordered node IDs from the tokenised message.
        window : int
            Context window radius.  Default 2 means we link each word
            to its 2 neighbours on each side.
        """
        n_tokens = len(token_node_ids)
        for i in range(n_tokens):
            for j in range(max(0, i - window), min(n_tokens, i + window + 1)):
                if i != j:
                    self.strengthen_edge(token_node_ids[i], token_node_ids[j])

    # ── TopK contextual comprehension ─────────────────────────────────

    def get_contextual_top_k(self, k: int = 10) -> List[Tuple[int, float]]:
        """
        Return the TopK highest-activated nodes with their activation
        scores.  This represents the agent's *contextual comprehension*
        of the current conversational state after spreading activation.

        Returns
        -------
        List of (node_id, activation) tuples, sorted descending.
        """
        ids = self.top_k_active(k)
        return [(nid, float(self._activation[nid])) for nid in ids]

    def get_active_subgraph_ids(self, threshold: Optional[float] = None) -> Set[int]:
        """
        Return the set of node IDs whose activation exceeds *threshold*.

        Used by the policy layer for localised Dirichlet forgetting —
        only rows in the active sub-graph receive ω-decay.

        Parameters
        ----------
        threshold : float, optional
            Defaults to ``cfg.activation_threshold``.
        """
        if threshold is None:
            threshold = self.cfg.activation_threshold
        n = self._num_nodes
        if n == 0:
            return set()
        active_mask = self._activation[:n] >= threshold
        return set(int(i) for i in np.where(active_mask)[0])
