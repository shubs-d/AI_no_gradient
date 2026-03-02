"""
policy_layer.py
===============
Sparse Bayesian Competitive Policy (SBCP) for natural-language generation
within the gradient-free Active Inference architecture.

No neural networks, no gradients — all computations are:
  • Dirichlet count tables  (transition probabilities between words),
  • Thompson sampling       (Bayesian action selection),
  • Integer / float ops     on sparse structures.

v3 — Logic-Based Grammar Induction
───────────────────────────────────
The policy layer now extracts **syntactic context features** from the
tokens generated so far and feeds them to the Tsetlin Machine as
auxiliary input literals.  The TM's Include/Exclude decisions for these
grammatical features bias word selection toward syntactically coherent
continuations:

    Syntactic features  (binary, 12-bit default):
      bit 0  has_subject       — ≥1 noun/pronoun preceded the verb
      bit 1  has_verb          — ≥1 verb token generated so far
      bit 2  has_object        — ≥1 noun/pronoun after the verb
      bit 3  sentence_start    — current position is first token
      bit 4  after_determiner  — previous token was a/the/this/…
      bit 5  after_preposition — previous token was in/on/at/…
      bit 6  after_verb        — previous token was a verb
      bit 7  after_noun        — previous token was a noun
      bit 8  is_plural_ctx     — last noun ended in 's'/'es'
      bit 9  past_tense_ctx    — last verb ended in 'ed'
      bit 10 progressive_ctx   — last verb ended in 'ing'
      bit 11 position_late     — generated ≥ 3 tokens already

These features let the TM learn propositional grammar rules like:
    IF has_subject ∧ ¬has_verb ∧ after_noun → favour VERBS
    IF has_verb ∧ ¬has_object → favour NOUNS/PRONOUNS

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

4.  **Grammar-biased selection:**  Before argmax, θ is modulated by
    the Tsetlin Machine's output for the current syntactic context
    features, boosting candidates whose grammatical role matches the
    TM's Include decision.

5.  **Sentence termination:**  A special <EOS> node competes with word
    nodes.  When <EOS> wins the Thompson sample, generation stops.
    The anti-dark-room penalty in the environment ensures the agent
    doesn't learn to always select <EOS> immediately.

6.  **Caregiver feedback integration:**
    - Positive feedback (Type I) → *strengthen* the Dirichlet counts
      for every bigram transition in the agent's last response.
    - Negative feedback (Type II) → *weaken* those counts.

7.  **Dirichlet forgetting:**
        T_{t+1}[w_i, :] = ω · T_t[w_i, :] + η · evidence
    with ω applied *only* to the active sub-graph rows (localised
    forgetting) so that basic grammar isn't catastrophically erased
    when the topic changes.

Complexity
──────────
  Per-token generation: O(K)  where K = TopK active nodes.
  Per-turn feedback:    O(R)  where R = response length.
  Per-turn forgetting:  O(K²) over the active sub-graph only.
  Grammar features:     O(1) per token (fixed 12-bit vector).
  No matrix inversions, no backpropagation.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from agent.memory_graph import (
    _AUXILIARIES,
    _DETERMINERS,
    _PREPOSITIONS,
    _PRONOUNS,
    _is_likely_noun,
    _is_likely_verb,
)
from agent.dirichlet_diagnostics import (
    precision_scaled_alpha,
    dcm_predictive,
    rescale_concentration,
    normalised_entropy_ratio,
    expected_thompson_gap,
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

EOS_TOKEN: str = "<EOS>"       # End-of-sentence marker
BOS_TOKEN: str = "<BOS>"       # Beginning-of-sentence marker
UNK_TOKEN: str = "<UNK>"       # Placeholder for truly unknown nodes

# ── Refractory Period ─────────────────────────────────────────────────
REFRACTORY_WINDOW: int        = 3      # Track last 3 tokens
REFRACTORY_SUPPRESSION: float = 0.01   # Multiply prob by 1% if refractory

# ── Syntactic Feature Indices ─────────────────────────────────────────
NUM_SYNTACTIC_FEATURES: int = 12
SF_HAS_SUBJECT:      int = 0
SF_HAS_VERB:         int = 1
SF_HAS_OBJECT:       int = 2
SF_SENTENCE_START:   int = 3
SF_AFTER_DETERMINER: int = 4
SF_AFTER_PREPOSITION:int = 5
SF_AFTER_VERB:       int = 6
SF_AFTER_NOUN:       int = 7
SF_IS_PLURAL_CTX:    int = 8
SF_PAST_TENSE_CTX:   int = 9
SF_PROGRESSIVE_CTX:  int = 10
SF_POSITION_LATE:    int = 11


def build_syntactic_features(
    tokens_so_far: List[str],
) -> np.ndarray:
    """
    Construct a binary syntactic feature vector from the tokens
    generated so far in the current response.

    This is a pure function (no side effects, no external state)
    that maps a partial token sequence to a 12-bit representation
    of its grammatical context.

    Parameters
    ----------
    tokens_so_far : List[str]
        Lowercased tokens generated so far in this response turn.

    Returns
    -------
    features : ndarray of shape (NUM_SYNTACTIC_FEATURES,), dtype int8
        Binary feature vector (0 or 1 per slot).
    """
    feat = np.zeros(NUM_SYNTACTIC_FEATURES, dtype=np.int8)

    if not tokens_so_far:
        feat[SF_SENTENCE_START] = 1
        return feat

    # Track SVO state by scanning tokens
    found_verb = False
    found_subject_before_verb = False
    found_object_after_verb = False
    last_noun: Optional[str] = None
    last_verb: Optional[str] = None

    for w in tokens_so_far:
        wl = w.lower()
        if not found_verb:
            if wl in _PRONOUNS or _is_likely_noun(wl):
                found_subject_before_verb = True
                last_noun = wl
            if _is_likely_verb(wl):
                found_verb = True
                last_verb = wl
        else:
            # After verb — look for object
            if wl in _PRONOUNS or _is_likely_noun(wl):
                found_object_after_verb = True
                last_noun = wl
            if _is_likely_verb(wl):
                last_verb = wl

    feat[SF_HAS_SUBJECT] = int(found_subject_before_verb)
    feat[SF_HAS_VERB] = int(found_verb)
    feat[SF_HAS_OBJECT] = int(found_object_after_verb)

    # Previous-token features
    prev = tokens_so_far[-1].lower()
    feat[SF_AFTER_DETERMINER] = int(prev in _DETERMINERS)
    feat[SF_AFTER_PREPOSITION] = int(prev in _PREPOSITIONS)
    feat[SF_AFTER_VERB] = int(_is_likely_verb(prev))
    feat[SF_AFTER_NOUN] = int(prev not in _DETERMINERS
                              and prev not in _PREPOSITIONS
                              and not _is_likely_verb(prev)
                              and prev not in _AUXILIARIES
                              and _is_likely_noun(prev))

    # Morphological context
    if last_noun and last_noun.endswith(("s", "es")):
        feat[SF_IS_PLURAL_CTX] = 1
    if last_verb and last_verb.endswith("ed"):
        feat[SF_PAST_TENSE_CTX] = 1
    if last_verb and last_verb.endswith("ing"):
        feat[SF_PROGRESSIVE_CTX] = 1

    # Position feature
    if len(tokens_so_far) >= 3:
        feat[SF_POSITION_LATE] = 1

    return feat


def grammar_role_boost(
    word: str,
    syntactic_features: np.ndarray,
    role_boost_subject: float = 2.0,
    role_boost_predicate: float = 1.8,
    role_boost_object: float = 1.5,
) -> float:
    """
    Compute a multiplicative grammar boost for a candidate word
    based on the current syntactic context.

    Logic rules (manually-engineered priors, later refined by TM):
      - If context needs a verb (has_subject, no verb yet) → boost verbs
      - If context needs an object (has verb, no object) → boost nouns
      - If after determiner → boost nouns
      - If after preposition → boost nouns
      - If sentence start → boost determiners/pronouns/nouns slightly

    Parameters
    ----------
    word : str
        Candidate word (lowercased).
    syntactic_features : ndarray (NUM_SYNTACTIC_FEATURES,)
        Current syntactic context features.
    role_boost_subject, role_boost_predicate, role_boost_object : float
        Multiplicative boosts for grammatical role matching.

    Returns
    -------
    boost : float ≥ 1.0
        Multiplicative boost (1.0 = no change).
    """
    boost = 1.0
    wl = word.lower()

    has_subj = bool(syntactic_features[SF_HAS_SUBJECT])
    has_verb = bool(syntactic_features[SF_HAS_VERB])
    has_obj = bool(syntactic_features[SF_HAS_OBJECT])
    after_det = bool(syntactic_features[SF_AFTER_DETERMINER])
    after_prep = bool(syntactic_features[SF_AFTER_PREPOSITION])
    sent_start = bool(syntactic_features[SF_SENTENCE_START])

    is_verb = _is_likely_verb(wl)
    is_noun_or_pron = (wl in _PRONOUNS or _is_likely_noun(wl))
    is_det = (wl in _DETERMINERS)

    # Rule 1: Need a verb → boost verbs
    if has_subj and not has_verb and is_verb:
        boost *= role_boost_predicate

    # Rule 2: Need an object → boost nouns/pronouns
    if has_verb and not has_obj and is_noun_or_pron:
        boost *= role_boost_object

    # Rule 3: After determiner → boost nouns strongly
    if after_det and is_noun_or_pron:
        boost *= role_boost_subject

    # Rule 4: After preposition → boost nouns
    if after_prep and is_noun_or_pron:
        boost *= role_boost_object

    # Rule 5: Sentence start → slightly boost determiners & pronouns
    if sent_start and (is_det or wl in _PRONOUNS):
        boost *= 1.3

    return boost


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
        dirichlet_prior: float = 0.2,
        max_response_len: int = 30,
        eos_prior_boost: float = 0.3,
        omega: float = 0.95,
        eta: float = 1.0,
        feedback_strength: float = 3.0,
        target_mass: float = 5.0,
        precision_beta: float = 1.5,
        use_dcm: bool = True,
        clause_boost_scale: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.dirichlet_prior = dirichlet_prior
        self.max_response_len = max_response_len
        self.eos_prior_boost = eos_prior_boost
        self.omega = omega
        self.eta = eta
        self.feedback_strength = feedback_strength
        self.target_mass = target_mass
        self.precision_beta = precision_beta
        self.use_dcm = use_dcm
        self.clause_boost_scale = clause_boost_scale
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

        # ── Refractory memory (Inhibition of Return) ─────────────────
        # A sliding window of the last REFRACTORY_WINDOW node IDs that
        # were selected during generation.  Any node still in this
        # window receives a ×0.01 suppression on its Thompson-sampled
        # probability, mimicking a synaptic refractory period.
        self._refractory: deque = deque(maxlen=REFRACTORY_WINDOW)

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
        grammar_boost: bool = True,
        role_boost_subject: float = 2.0,
        role_boost_predicate: float = 1.8,
        role_boost_object: float = 1.5,
        clause_votes: Optional[np.ndarray] = None,
        clause_threshold: int = 15,
    ) -> List[str]:
        """
        Generate a response via Bayesian-consistent pre-sampling modulation.

        Revised algorithm (v4 — systems-level fix):
        ─────────────────────────────────────────────
        1. Start from <BOS>.
        2. Build candidate set = TopK ∪ {<EOS>}.
        3. Construct raw Dirichlet concentration α from bigram table.
        4. Compute combined boost vector g_i:
             g_i = grammar_boost_i · clause_boost_i · activation_i
        5. Apply precision-scaled rescaling (ALL modulations pre-sampling):
             α'_i = (α_i / α_0) · M_target · g_i · β
        6. EITHER:
           (a) Thompson sample:  θ ~ Dir(α'),  w = argmax θ
           (b) DCM (Pólya urn):  P(w) = (α'_w + n_w) / (α'_0 + n_total)
               where n_w counts intra-utterance selections of w
        7. Apply refractory suppression (post-selection filter).
        8. If w == <EOS> or length ≥ max_response_len, stop.

        Parameters
        ----------
        top_k_ids : List[int]
            Node IDs of the K most-activated word nodes.
        node_labels : Dict[int, str]
            Mapping from node ID → word string.
        activation_scores : Dict[int, float], optional
            Node ID → activation level.
        grammar_boost : bool
            Apply syntactic grammar-role boosting.
        clause_votes : ndarray (K,), optional
            Per-candidate Tsetlin Machine vote (integer, in [-T, T]).
            If provided, converted to exp(v/T) boost on α.
        clause_threshold : int
            Tsetlin threshold T for normalising clause votes.

        Returns
        -------
        tokens : List[str]
        """
        if not top_k_ids:
            return []

        # Build candidate set: TopK word nodes + <EOS>
        candidates = list(top_k_ids)
        if self.eos_id >= 0 and self.eos_id not in candidates:
            candidates.append(self.eos_id)

        K = len(candidates)

        # ── Pre-compute candidate → index mapping ────────────────────
        cid_to_idx = {cid: i for i, cid in enumerate(candidates)}

        tokens: List[str] = []
        self._last_bigrams = []
        current_id = self.bos_id if self.bos_id >= 0 else candidates[0]

        # ── DCM intra-utterance counts (Pólya urn) ───────────────────
        utterance_counts = np.zeros(K, dtype=np.float64)

        # ── Last diagnostic snapshot ─────────────────────────────────
        self._last_entropy_ratio: float = 1.0
        self._last_thompson_gap: float = 0.0

        for step in range(self.max_response_len):
            # ── Step 3: Raw Dirichlet concentrations from bigram table ─
            raw_alpha = self._ensure_row(current_id, candidates)

            # Boost <EOS> prior
            if self.eos_id >= 0 and self.eos_id in cid_to_idx:
                eos_pos = cid_to_idx[self.eos_id]
                raw_alpha[eos_pos] += self.eos_prior_boost

            # ── Step 4a: Grammar boost vector ────────────────────────
            grammar_g = np.ones(K, dtype=np.float64)
            if grammar_boost:
                syn_feat = build_syntactic_features(tokens)
                for gi, cid in enumerate(candidates):
                    if cid == self.eos_id:
                        continue
                    label_g = node_labels.get(cid, "")
                    if label_g and label_g not in (EOS_TOKEN, BOS_TOKEN, UNK_TOKEN):
                        grammar_g[gi] = grammar_role_boost(
                            label_g, syn_feat,
                            role_boost_subject=role_boost_subject,
                            role_boost_predicate=role_boost_predicate,
                            role_boost_object=role_boost_object,
                        )

            # ── Step 4b: Clause boost vector ─────────────────────────
            clause_g = np.ones(K, dtype=np.float64)
            if clause_votes is not None and len(clause_votes) >= K:
                # Convert integer votes [-T, T] to multiplicative boost
                # exp(v_k / T) maps to [e^{-1}, e^{+1}] ≈ [0.37, 2.72]
                T = max(clause_threshold, 1)
                scaled = np.clip(clause_votes[:K], -T, T).astype(np.float64) / T
                clause_g = np.exp(scaled * self.clause_boost_scale)

            # ── Step 4c: Activation weight vector ────────────────────
            act_w = None
            if activation_scores is not None:
                act_w = np.array([
                    activation_scores.get(c, 0.01) for c in candidates
                ], dtype=np.float64)

            # ── Step 5: Precision-scaled pre-sampling modulation ─────
            alpha_final = precision_scaled_alpha(
                raw_alpha=raw_alpha,
                target_mass=self.target_mass,
                grammar_boost=grammar_g,
                clause_boost=clause_g,
                precision_beta=self.precision_beta,
                activation_weights=act_w,
                floor=0.01,
            )

            # ── Step 6: Sample next token ────────────────────────────
            if self.use_dcm:
                # DCM (Pólya urn): deterministic predictive + noise
                probs = dcm_predictive(alpha_final, utterance_counts)
            else:
                # Thompson sampling: θ ~ Dir(α'), w = argmax θ
                gamma_samples = self.rng.standard_gamma(alpha_final)
                total = gamma_samples.sum()
                if total < 1e-30:
                    probs = np.ones(K) / K
                else:
                    probs = gamma_samples / total

            # ── Step 6b: Refractory suppression ──────────────────────
            for ri, cid in enumerate(candidates):
                if cid in self._refractory:
                    probs[ri] *= REFRACTORY_SUPPRESSION
            r_sum = probs.sum()
            if r_sum > 1e-30:
                probs /= r_sum

            # ── Step 6c: Select ──────────────────────────────────────
            if self.use_dcm:
                # Under DCM, add stochastic exploration via sampling
                # from the predictive probabilities (not just argmax)
                next_idx = int(self.rng.choice(K, p=probs))
            else:
                next_idx = int(np.argmax(probs))

            next_id = candidates[next_idx]

            # ── Step 7: Check for <EOS> ──────────────────────────────
            if next_id == self.eos_id:
                if len(tokens) > 0:
                    break
                else:
                    # Force the agent to pick something else
                    probs[next_idx] = 0.0
                    r_sum = probs.sum()
                    if r_sum < 1e-30:
                        break
                    probs /= r_sum
                    if self.use_dcm:
                        next_idx = int(self.rng.choice(K, p=probs))
                    else:
                        next_idx = int(np.argmax(probs))
                    next_id = candidates[next_idx]
                    if next_id == self.eos_id:
                        break

            # ── Step 8: Append token ─────────────────────────────────
            label = node_labels.get(next_id, UNK_TOKEN)
            if label not in (EOS_TOKEN, BOS_TOKEN, UNK_TOKEN):
                tokens.append(label)

            self._last_bigrams.append((current_id, next_id))

            # Update DCM intra-utterance counts (Pólya urn memory)
            utterance_counts[next_idx] += 1.0

            # Push into refractory window
            self._refractory.append(next_id)

            current_id = next_id

        # ── Post-generation diagnostics ──────────────────────────────
        if K > 1:
            final_alpha = precision_scaled_alpha(
                raw_alpha=self._ensure_row(current_id, candidates),
                target_mass=self.target_mass,
                grammar_boost=np.ones(K),
                clause_boost=np.ones(K),
                precision_beta=self.precision_beta,
                floor=0.01,
            )
            self._last_entropy_ratio = normalised_entropy_ratio(final_alpha)
            self._last_thompson_gap = expected_thompson_gap(final_alpha)

        self._refractory.clear()
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

    # ── Grammar feature accessors ─────────────────────────────────────

    def get_syntactic_features_for_context(
        self,
        tokens_so_far: List[str],
    ) -> np.ndarray:
        """
        Build the syntactic feature vector for an arbitrary token
        context.  Thin wrapper around module-level ``build_syntactic_features``.

        Parameters
        ----------
        tokens_so_far : List[str]

        Returns
        -------
        ndarray of shape (NUM_SYNTACTIC_FEATURES,), dtype int8
        """
        return build_syntactic_features(tokens_so_far)
