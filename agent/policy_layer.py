"""
policy_layer.py
===============
Sparse Bayesian Competitive Policy (SBCP) for natural-language generation
within the gradient-free Active Inference architecture.

No neural networks, no gradients — all computations are:
  • Dirichlet count tables  (transition probabilities between words),
  • Thompson sampling       (Bayesian action selection),
  • Integer / float ops     on sparse structures.

Architecture overview
─────────────────────
1.  **Input:** The TopK most-activated word nodes from the Lyapunov-stable
    spreading-activation memory graph.

2.  **Bigram transition model:**  A Dirichlet-categorical bigram table

        T[w_i, w_j]  ~ Dir(α)

    where T[w_i, w_j] counts how often word w_j followed word w_i.
    This is the agent's learned "grammar" — entirely count-based.

3.  **Thompson sampling:**  To select the next word, the policy draws
    a sample from the posterior Dirichlet for the current word's row:

        θ ~ Dir( T[w_current, :] )
        w_next = argmax θ   (among the TopK active candidates)

    This is the *Sparse Bayesian Competitive* step: only the TopK
    activated nodes compete, making generation O(K) per token rather
    than O(V) over the full vocabulary.

4.  **Sentence termination:**  A special <EOS> node competes with word
    nodes.  When <EOS> wins the Thompson sample, generation stops.
    The anti-dark-room penalty in the environment ensures the agent
    doesn't learn to always select <EOS> immediately.

5.  **Caregiver feedback integration:**
    - Positive feedback (Type I) → *strengthen* the Dirichlet counts
      for every bigram transition in the agent's last response.
    - Negative feedback (Type II) → *weaken* those counts.

6.  **Dirichlet forgetting:**
        T_{t+1}[w_i, :] = ω · T_t[w_i, :] + η · evidence
    with ω applied *only* to the active sub-graph rows (localised
    forgetting) so that basic grammar isn't catastrophically erased
    when the topic changes.

Complexity
──────────
  Per-token generation: O(K)  where K = TopK active nodes.
  Per-turn feedback:    O(R)  where R = response length.
  Per-turn forgetting:  O(K²) over the active sub-graph only.
  No matrix inversions, no backpropagation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

EOS_TOKEN: str = "<EOS>"       # End-of-sentence marker
BOS_TOKEN: str = "<BOS>"       # Beginning-of-sentence marker
UNK_TOKEN: str = "<UNK>"       # Placeholder for truly unknown nodes


class SBCPolicy:
    """
    Sparse Bayesian Competitive Policy — gradient-free sentence generator.

    Maintains a Dirichlet bigram transition table over the memory graph's
    word nodes and uses Thompson sampling restricted to the TopK active
    set to generate responses token-by-token.

    Parameters
    ----------
    dirichlet_prior : float
        Symmetric Dirichlet prior α₀ for each bigram cell.
    max_response_len : int
        Hard cap on generated tokens per turn (prevents runaway loops).
    eos_prior_boost : float
        Extra prior mass on the <EOS> transition from every word,
        encouraging the agent to occasionally stop speaking.  Without
        this, the agent might generate infinite streams.
    omega : float
        Dirichlet forgetting rate (localised to active sub-graph).
    eta : float
        Learning rate for Dirichlet count updates.
    feedback_strength : float
        Multiplier on Type-I / Type-II feedback count adjustments.
    """

    def __init__(
        self,
        dirichlet_prior: float = 1.0,
        max_response_len: int = 30,
        eos_prior_boost: float = 2.0,
        omega: float = 0.95,
        eta: float = 1.0,
        feedback_strength: float = 2.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.dirichlet_prior = dirichlet_prior
        self.max_response_len = max_response_len
        self.eos_prior_boost = eos_prior_boost
        self.omega = omega
        self.eta = eta
        self.feedback_strength = feedback_strength
        self.rng = rng or np.random.default_rng()

        # ── Bigram transition table ───────────────────────────────────
        # Sparse representation: Dict[(src_node_id, dst_node_id)] → float count
        # Using a dict-of-dicts for O(1) lookup per active pair.
        #   _bigram[src_id][dst_id] = Dirichlet concentration count
        self._bigram: Dict[int, Dict[int, float]] = {}

        # ── Special node IDs ──────────────────────────────────────────
        # These are assigned when the memory graph registers them.
        self.eos_id: int = -1       # set externally after graph init
        self.bos_id: int = -1       # set externally after graph init

        # ── Last generated sequence (for feedback reinforcement) ──────
        self._last_bigrams: List[Tuple[int, int]] = []

    # ── Bigram access helpers ─────────────────────────────────────────

    def _get_count(self, src: int, dst: int) -> float:
        """Return the Dirichlet count for transition src → dst."""
        return self._bigram.get(src, {}).get(dst, self.dirichlet_prior)

    def _set_count(self, src: int, dst: int, value: float) -> None:
        """Set the Dirichlet count for transition src → dst."""
        if src not in self._bigram:
            self._bigram[src] = {}
        self._bigram[src][dst] = max(value, self.dirichlet_prior * 0.1)

    def _ensure_row(self, src: int, candidates: List[int]) -> np.ndarray:
        """
        Build the Dirichlet concentration vector for *src* restricted to
        *candidates*.

        Returns
        -------
        alpha : ndarray of shape (len(candidates),)
            Dirichlet concentrations for each candidate destination.
        """
        alpha = np.array(
            [self._get_count(src, c) for c in candidates],
            dtype=np.float64,
        )
        return alpha

    # ── Core generation: Thompson sampling over active sub-graph ──────

    def generate(
        self,
        top_k_ids: List[int],
        node_labels: Dict[int, str],
        activation_scores: Optional[Dict[int, float]] = None,
    ) -> List[str]:
        """
        Generate a response token sequence using Dirichlet Thompson
        sampling restricted to the TopK active memory-graph nodes.

        Algorithm
        ---------
        1.  Start from <BOS>.
        2.  At each position, build the candidate set = TopK ∪ {<EOS>}.
        3.  Construct the Dirichlet concentration vector α for the
            current word over the candidates.
        4.  Draw  θ ~ Dir(α)  — a Thompson sample.
        5.  Optionally weight θ by activation scores (nodes with higher
            spreading-activation contribute more).
        6.  Select  w_next = argmax(θ · activation_weight).
        7.  If w_next == <EOS>  or  length ≥ max_response_len, stop.
        8.  Otherwise append w_next and repeat.

        Parameters
        ----------
        top_k_ids : List[int]
            Node IDs of the K most-activated word nodes from the
            memory graph (post spreading-activation).
        node_labels : Dict[int, str]
            Mapping from node ID → word string (for all graph nodes).
        activation_scores : Dict[int, float], optional
            Mapping from node ID → activation level.  If provided,
            Thompson-sampled probabilities are scaled by activation.

        Returns
        -------
        tokens : List[str]
            Generated word tokens (excluding <BOS>/<EOS> markers).
        """
        if not top_k_ids:
            # No activated nodes → agent has nothing to say.
            # (The environment will penalise this via dark-room.)
            return []

        # Build candidate set: TopK word nodes + <EOS>
        candidates = list(top_k_ids)
        if self.eos_id >= 0 and self.eos_id not in candidates:
            candidates.append(self.eos_id)

        tokens: List[str] = []
        self._last_bigrams = []
        current_id = self.bos_id if self.bos_id >= 0 else candidates[0]

        for _ in range(self.max_response_len):
            # ── Step 3: Dirichlet concentration vector ────────────────
            alpha = self._ensure_row(current_id, candidates)

            # Boost <EOS> prior to prevent infinite generation
            if self.eos_id >= 0 and self.eos_id in candidates:
                eos_pos = candidates.index(self.eos_id)
                alpha[eos_pos] += self.eos_prior_boost

            # ── Step 4: Thompson sample  θ ~ Dir(α) ──────────────────
            # Gamma-based sampling (standard Dirichlet procedure):
            #   g_i ~ Gamma(α_i, 1);   θ_i = g_i / Σ g
            # All discrete, no gradients.
            gamma_samples = np.array([
                self.rng.gamma(a) if a > 0 else 0.0
                for a in alpha
            ])
            total = gamma_samples.sum()
            if total < 1e-30:
                theta = np.ones(len(candidates)) / len(candidates)
            else:
                theta = gamma_samples / total

            # ── Step 5: Modulate by activation scores ─────────────────
            if activation_scores is not None:
                act_weights = np.array([
                    activation_scores.get(c, 0.01) for c in candidates
                ])
                # Softmax-free: just multiply and re-normalise
                theta *= act_weights
                w_sum = theta.sum()
                if w_sum > 1e-30:
                    theta /= w_sum

            # ── Step 6: Select next word ──────────────────────────────
            next_idx = int(np.argmax(theta))
            next_id = candidates[next_idx]

            # ── Step 7: Check for <EOS> ───────────────────────────────
            if next_id == self.eos_id:
                # Only accept EOS if we've generated at least 1 token
                # (anti-dark-room: don't let the agent go silent easily)
                if len(tokens) > 0:
                    break
                else:
                    # Force the agent to pick the second-best candidate
                    theta[next_idx] = 0.0
                    if theta.sum() < 1e-30:
                        break  # truly no options
                    theta /= theta.sum()
                    next_idx = int(np.argmax(theta))
                    next_id = candidates[next_idx]
                    if next_id == self.eos_id:
                        break

            # ── Step 8: Append token ──────────────────────────────────
            label = node_labels.get(next_id, UNK_TOKEN)
            # Skip special tokens in output
            if label not in (EOS_TOKEN, BOS_TOKEN, UNK_TOKEN):
                tokens.append(label)

            # Record bigram for later feedback
            self._last_bigrams.append((current_id, next_id))
            current_id = next_id

        return tokens

    # ── Caregiver feedback: Type-I (reinforce) / Type-II (punish) ─────

    def apply_feedback(self, sentiment: int) -> None:
        """
        Adjust Dirichlet counts for the last generated bigram sequence
        based on caregiver sentiment.

        Parameters
        ----------
        sentiment : int
            +1 → Type-I feedback (strengthen transitions).
            −1 → Type-II feedback (weaken transitions).
             0 → no adjustment.

        Mathematical effect
        ───────────────────
        For each bigram (w_i, w_j) in the last response:

            Type I:   T[w_i, w_j] += feedback_strength
            Type II:  T[w_i, w_j] -= feedback_strength  (floored at prior/10)

        This directly modifies the Dirichlet concentrations, shifting
        the posterior toward or away from the reinforced sequence.
        """
        if sentiment == 0 or not self._last_bigrams:
            return

        delta = self.feedback_strength * sentiment

        for src, dst in self._last_bigrams:
            old = self._get_count(src, dst)
            new_val = old + delta
            self._set_count(src, dst, new_val)

    # ── Localised Dirichlet forgetting ────────────────────────────────

    def apply_localised_forgetting(self, active_node_ids: Set[int]) -> None:
        """
        Apply Dirichlet decay  T_{t+1} = ω · T_t  **only** to rows
        corresponding to the currently active sub-graph nodes.

        This prevents catastrophic forgetting of basic grammar (e.g.,
        common word transitions) while allowing topic-specific counts
        to decay when the conversation shifts.

        Parameters
        ----------
        active_node_ids : Set[int]
            The set of node IDs currently in the active spreading-
            activation neighbourhood.

        Mathematical guarantee
        ─────────────────────
        Because ω < 1, repeatedly-active rows converge toward the prior
        α₀, ensuring the agent can re-learn transitions for new topics.
        Inactive rows retain their counts indefinitely, preserving
        long-term grammatical knowledge.
        """
        omega = self.omega
        prior = self.dirichlet_prior

        for src in active_node_ids:
            if src not in self._bigram:
                continue
            row = self._bigram[src]
            for dst in list(row.keys()):
                row[dst] = max(omega * row[dst], prior * 0.1)

    # ── Observation of user input (update bigram counts) ──────────────

    def observe_user_tokens(
        self,
        token_node_ids: List[int],
    ) -> None:
        """
        Update the bigram transition table from the user's token sequence.

        This is how the agent learns language structure: by counting
        bigram transitions in observed text.

        Parameters
        ----------
        token_node_ids : List[int]
            Sequence of memory-graph node IDs corresponding to the
            tokenised user message (in order).

        Update rule
        ───────────
        For each consecutive pair (w_i, w_j):
            T[w_i, w_j] += η
        """
        if len(token_node_ids) < 2:
            return

        for i in range(len(token_node_ids) - 1):
            src = token_node_ids[i]
            dst = token_node_ids[i + 1]
            old = self._get_count(src, dst)
            self._set_count(src, dst, old + self.eta)

        # Also learn transition from <BOS> → first token
        if self.bos_id >= 0 and token_node_ids:
            first = token_node_ids[0]
            old = self._get_count(self.bos_id, first)
            self._set_count(self.bos_id, first, old + self.eta)

        # And from last token → <EOS>
        if self.eos_id >= 0 and token_node_ids:
            last = token_node_ids[-1]
            old = self._get_count(last, self.eos_id)
            self._set_count(last, self.eos_id, old + self.eta * 0.5)

    # ── Diagnostics ───────────────────────────────────────────────────

    def total_bigram_entries(self) -> int:
        """Total number of non-default bigram entries stored."""
        return sum(len(row) for row in self._bigram.values())

    def get_top_transitions(
        self, src: int, k: int = 5
    ) -> List[Tuple[int, float]]:
        """Return the top-k most probable transitions from src node."""
        if src not in self._bigram:
            return []
        row = self._bigram[src]
        sorted_items = sorted(row.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]
