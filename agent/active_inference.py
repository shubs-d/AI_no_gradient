"""
active_inference.py
===================
Discrete Active Inference engine built entirely on Dirichlet concentration
counts – no gradient descent, no floating-point weight matrices.

Generative model
────────────────
The agent maintains two categorical-Dirichlet tables:

  B[s, a, s']  –  transition counts   P(s' | s, a)
  A[s, o]      –  likelihood counts    P(o | s)

Expected log-probabilities are computed via the digamma (ψ) function:

  E[ln A] = ψ(a) − ψ(a₀)   where a₀ = Σ_j a_j        … (Eq. 1)

Dirichlet Forgetting  (STRESS-TEST)
───────────────────
To prevent count contamination from stale regimes:

  θ_{t+1} = ω · θ_t  +  η · χ_t                          … (Forget)

where ω = 0.95 (decay) and η = 1.0 (learning rate).

Preference Prior C  (anti-Dark-Room)
────────────────────
The EFE pragmatic term uses an inflated log-preference vector C(o)
with an extreme boost for CELL_RESOURCE, mathematically compelling
the agent to seek resource states over hiding.

Variational Free Energy (VFE)
─────────────────────────────
  F  =  E_q[ ln q(s) − ln P(o, s) ]
     ≈  KL[ q(s) ‖ P(s) ] − E_q[ ln P(o | s) ]

Because q(s) is updated by Bayesian belief updating (count increments),
VFE reduces to the *prediction error* (surprise) at each step.

Expected Free Energy (EFE)
──────────────────────────
  G(π) = Σ_τ  [ −epistemic − pragmatic ]
  epistemic   = H[ P(o | s) ]            (expected information gain)
  pragmatic   = E_q[ ln P(o | preferred) ] (goal-seeking term)

All arrays are integer counts updated via the forgetting equation.
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np
from scipy.special import digamma

from config import InferenceConfig, NUM_ACTIONS, CELL_RESOURCE


class ActiveInferenceEngine:
    """Discrete active-inference module using Dirichlet count tables."""

    # ── Construction ──────────────────────────────────────────────────

    def __init__(self, num_states: int, cfg: InferenceConfig) -> None:
        """
        Parameters
        ----------
        num_states : int
            Number of *hidden* states the agent distinguishes.
            This can grow at runtime via structure learning.
        cfg : InferenceConfig
            Hyper-parameters (prior, horizon, weights …).
        """
        self.cfg = cfg
        self.num_states = num_states
        self.num_obs = cfg.num_obs_symbols
        self.num_actions = NUM_ACTIONS

        # ── Dirichlet count tables (integer + prior) ──────────────
        # A[s, o] : likelihood counts  →  P(o | s)
        self._A = np.full(
            (num_states, self.num_obs),
            cfg.dirichlet_prior,
            dtype=np.float64,          # counts are logically integers;
        )                               # we keep float for digamma convenience

        # B[s, a, s'] : transition counts →  P(s' | s, a)
        self._B = np.full(
            (num_states, self.num_actions, num_states),
            cfg.dirichlet_prior,
            dtype=np.float64,
        )

        # Current posterior belief over hidden states  q(s)
        self._qs = np.ones(num_states, dtype=np.float64) / num_states

        # Running VFE for the structure-learning module to inspect
        self.last_vfe: float = 0.0

        # ── Preference prior C(o):  inflated log-preference vector ───
        # C[o] = 0 for non-preferred obs, large positive for resource.
        # This enters EFE pragmatic = Σ_{s'} q(s') Σ_o P(o|s') C(o).
        self._C = np.zeros(self.num_obs, dtype=np.float64)
        self._C[cfg.preferred_obs] = cfg.consume_preference_boost

    # ── Properties for external read access ───────────────────────────

    @property
    def A(self) -> np.ndarray:
        """Likelihood count matrix  A[s, o]."""
        return self._A

    @property
    def B(self) -> np.ndarray:
        """Transition count tensor  B[s, a, s']."""
        return self._B

    @property
    def belief(self) -> np.ndarray:
        """Current posterior  q(s)."""
        return self._qs

    # ── Dirichlet expected log-probabilities (Eq. 1) ──────────────────

    @staticmethod
    def _expected_log(counts: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Compute  E[ln Cat] = ψ(a_j) − ψ(Σ_j a_j)  along *axis*.

        Parameters
        ----------
        counts : ndarray
            Dirichlet concentration parameters (≥ prior > 0).
        axis : int
            Summation axis for a₀.

        Returns
        -------
        ndarray of same shape as *counts*.
        """
        a0 = counts.sum(axis=axis, keepdims=True)
        return digamma(counts) - digamma(a0)

    # ── Belief update (state estimation) ──────────────────────────────

    def update_belief(self, obs_idx: int) -> float:
        """
        Bayesian belief update given a new observation.

        1.  Compute likelihood  ln P(o | s)  from Dirichlet counts.
        2.  Combine with prior  ln q(s)  (previous posterior).
        3.  Normalise to obtain updated  q(s).
        4.  Return scalar VFE (negative log-evidence ≈ surprise).

        Parameters
        ----------
        obs_idx : int
            Index of the observed cell type.

        Returns
        -------
        vfe : float
            Variational Free Energy for this time-step.
        """
        # Expected log-likelihood  E[ln P(o|s)]  for every state
        ln_A = self._expected_log(self._A, axis=1)          # shape (S, O)
        ln_lik = ln_A[:, obs_idx]                            # shape (S,)

        # Log prior = current posterior (message passing)
        ln_prior = np.log(self._qs + 1e-16)

        # Unnormalised log posterior
        ln_post = ln_lik + ln_prior
        ln_post -= ln_post.max()                             # numerical stability
        qs_new = np.exp(ln_post)
        qs_new /= qs_new.sum()

        # ── VFE = KL[q‖prior] − E_q[ln P(o|s)]  ≈  surprise ────────
        kl = np.sum(qs_new * (np.log(qs_new + 1e-16) - ln_prior))
        expected_ll = np.dot(qs_new, ln_lik)
        vfe = float(kl - expected_ll)

        self._qs = qs_new
        self.last_vfe = vfe
        return vfe

    # ── Count updates (learning by counting) ──────────────────────────

    def update_counts(
        self,
        prev_state: int,
        action: int,
        next_state: int,
        obs_idx: int,
    ) -> None:
        """
        Update Dirichlet counts with **forgetting** and
        **arousal-modulated plasticity**:

            θ_{t+1}  =  ω · θ_t  +  η_eff · χ_t

        where χ_t is a one-hot evidence vector for the observed
        transition, ω is the decay rate, and η_eff is the *dynamic*
        learning rate.

        Arousal-modulated plasticity (Psi-theory inspired)
        ──────────────────────────────────────────────────
        In biological cognition, high arousal / surprise amplifies
        synaptic plasticity (Yerkes-Dodson law).  We model this by
        scaling η with the current VFE prediction error:

            η_eff = η_base · (1 + max(0, VFE − τ))

        where τ = 3.0 is the surprise threshold.  When VFE > 3.0 the
        agent's learning rate spikes, forcing rapid incorporation of
        novel syntactic structures.  When VFE is low (agent predicts
        well), η_eff ≈ η_base and learning is gentle.

        No gradients — just a multiplicative scalar on integer counts.
        """
        omega = self.cfg.dirichlet_decay_omega    # 0.95
        eta_base = self.cfg.dirichlet_learning_rate  # 1.0
        prior = self.cfg.dirichlet_prior

        # ── Arousal-modulated learning rate ───────────────────────────
        # Surprise threshold above which plasticity spikes.
        AROUSAL_THRESHOLD = 3.0
        # Clamp the multiplier to avoid runaway counts on extreme VFE.
        MAX_AROUSAL_MULT  = 5.0
        arousal_boost = min(
            max(0.0, self.last_vfe - AROUSAL_THRESHOLD),
            MAX_AROUSAL_MULT - 1.0,
        )
        eta = eta_base * (1.0 + arousal_boost)

        # ── Decay ALL counts toward the prior, then inject evidence ──
        # B[prev_state, action, :] *= ω  then += η at next_state
        self._B[prev_state, action, :] *= omega
        self._B[prev_state, action, next_state] += eta
        # Floor at prior to avoid degenerate zero-counts
        np.maximum(self._B[prev_state, action, :], prior,
                   out=self._B[prev_state, action, :])

        # A[next_state, :] *= ω  then += η at obs_idx
        self._A[next_state, :] *= omega
        self._A[next_state, obs_idx] += eta
        np.maximum(self._A[next_state, :], prior,
                   out=self._A[next_state, :])

    # ── Action selection via Expected Free Energy (EFE) ───────────────

    def select_action(self) -> int:
        """
        One-step Expected Free Energy action selection.

        G(a) = − epistemic(a) − pragmatic(a)

        epistemic(a) = Σ_s' q(s') H[P(o | s')]      (info-gain proxy)
        pragmatic(a) = Σ_s' q(s') E[ln P(o_pref | s')]

        Returns the action that **minimises** G  (i.e., maximises
        negative free energy = epistemic value + pragmatic value).
        """
        cfg = self.cfg
        G = np.zeros(self.num_actions, dtype=np.float64)

        # Transition probabilities  P(s'|s,a)  from Dirichlet counts
        ln_B = self._expected_log(self._B, axis=2)     # (S, A, S')

        # Expected log-likelihood matrix  E[ln P(o|s)]
        ln_A = self._expected_log(self._A, axis=1)     # (S, O)

        for a in range(self.num_actions):
            # Expected next-state distribution under current belief
            # q(s') = Σ_s q(s) P(s'|s,a)
            transition_probs = np.exp(ln_B[:, a, :])            # (S, S')
            transition_probs /= transition_probs.sum(axis=1, keepdims=True) + 1e-16
            qs_next = self._qs @ transition_probs                # (S',)
            qs_next /= qs_next.sum() + 1e-16

            # ── Epistemic value: expected entropy of observations ────
            #    H[P(o|s')] for each s', weighted by q(s')
            A_probs = self._A / self._A.sum(axis=1, keepdims=True)
            entropy_per_state = -np.sum(
                A_probs * np.log(A_probs + 1e-16), axis=1
            )                                                     # (S,)
            epistemic = float(np.dot(qs_next, entropy_per_state))

            # ── Pragmatic value: E_{q(s')}[ Σ_o P(o|s') C(o) ] ────
            # Uses the inflated preference vector C instead of just
            # a single preferred-obs log-probability.
            pragmatic_per_state = A_probs @ self._C               # (S,)
            pragmatic = float(np.dot(qs_next, pragmatic_per_state))

            G[a] = -(cfg.efe_epistemic_weight * epistemic
                     + cfg.efe_pragmatic_weight * pragmatic)

        # Select action that *minimises* G  (argmin of neg values = argmax value)
        best_action = int(np.argmin(G))
        return best_action

    # ── Structural growth helpers ─────────────────────────────────────

    def expand_state_space(self, new_num_states: int) -> None:
        """
        Grow Dirichlet count tables when structure learning adds nodes.

        New rows / slices are initialised to the symmetric prior so they
        carry no empirical information until the agent visits them.
        """
        old = self.num_states
        if new_num_states <= old:
            return

        delta = new_num_states - old
        prior = self.cfg.dirichlet_prior

        # Expand A  (S, O)
        new_A = np.full((delta, self.num_obs), prior, dtype=np.float64)
        self._A = np.concatenate([self._A, new_A], axis=0)

        # Expand B  (S, A, S')  along both state axes
        # First add new "from" states
        new_B_rows = np.full(
            (delta, self.num_actions, old), prior, dtype=np.float64
        )
        self._B = np.concatenate([self._B, new_B_rows], axis=0)
        # Then add new "to" states for ALL rows
        new_B_cols = np.full(
            (new_num_states, self.num_actions, delta), prior, dtype=np.float64
        )
        self._B = np.concatenate([self._B, new_B_cols], axis=2)

        # Expand belief vector
        new_qs = np.zeros(delta, dtype=np.float64)
        self._qs = np.concatenate([self._qs, new_qs])
        self._qs /= self._qs.sum() + 1e-16

        self.num_states = new_num_states

    def shrink_state_space(self, keep_mask: np.ndarray) -> None:
        """
        Remove states flagged False in *keep_mask* (BMR pruning).

        Parameters
        ----------
        keep_mask : bool array of shape (num_states,)
        """
        self._A = self._A[keep_mask]
        self._B = self._B[keep_mask][:, :, keep_mask]
        self._qs = self._qs[keep_mask]
        self._qs /= self._qs.sum() + 1e-16
        self.num_states = int(keep_mask.sum())

    # ── Utilities ─────────────────────────────────────────────────────

    def get_surprise(self, obs_idx: int) -> float:
        """
        Point-wise surprise  −ln P(o)  under the *current* generative model,
        marginalised over states.

        Useful for the structure-learning trigger.
        """
        # P(o) ≈ Σ_s q(s) P(o|s)
        A_probs = self._A / self._A.sum(axis=1, keepdims=True)
        p_o = float(np.dot(self._qs, A_probs[:, obs_idx]))
        return -np.log(p_o + 1e-16)

    def most_likely_state(self) -> int:
        """MAP estimate of the current hidden state."""
        return int(np.argmax(self._qs))

    # ═══════════════════════════════════════════════════════════════════
    # NLP / Chatbot extensions  (v2)
    # ═══════════════════════════════════════════════════════════════════

    # ── Caregiver feedback (Social Active Inference) ─────────────────

    def apply_social_feedback(
        self,
        sentiment: int,
        current_state: int,
        obs_idx: int,
        strength: float = 2.0,
    ) -> None:
        """
        Integrate caregiver feedback into the Dirichlet count tables.

        This implements the *Social Active Inference* loop:
        - Positive sentiment (“Good”, “Yes”) → Type-I reinforcement:
          strengthen the A[state, obs] count so the agent learns that
          producing this type of response in this state is desirable.
        - Negative sentiment (“No”, “Bad”) → Type-II correction:
          mildly weaken the A[state, obs] count.

        Parameters
        ----------
        sentiment : int
            +1 = positive, −1 = negative, 0 = neutral (no-op).
        current_state : int
            MAP hidden state at the time of feedback.
        obs_idx : int
            Observation index at the time of feedback.
        strength : float
            Magnitude of the count adjustment.
        """
        if sentiment == 0:
            return
        if current_state < 0 or current_state >= self.num_states:
            return
        obs_idx = obs_idx % self.num_obs

        if sentiment > 0:
            # Type I: reinforce  A[state, obs]
            self._A[current_state, obs_idx] += strength
        else:
            # Type II: weaken (floored at prior)
            prior = self.cfg.dirichlet_prior
            self._A[current_state, obs_idx] = max(
                self._A[current_state, obs_idx] - strength * 0.5,
                prior,
            )

    # ── Localised Dirichlet forgetting ───────────────────────────────

    def apply_localised_forgetting(
        self,
        active_states: Optional[Set[int]] = None,
    ) -> None:
        """
        Apply Dirichlet decay ω **only** to the rows of A and B that
        correspond to the currently active hidden states.

        This prevents catastrophic forgetting of grammatical knowledge
        encoded in inactive states while allowing topic-specific counts
        to decay when the conversation shifts.

        Mathematical formulation
        ───────────────────────
        For each active state s ∈ active_states:
            A[s, :] ← ω · A[s, :]
            B[s, a, :] ← ω · B[s, a, :]   ∀ a

        Inactive states retain their full count tables.

        Parameters
        ----------
        active_states : Set[int], optional
            Hidden-state indices to decay.  If None, decay ALL
            (falls back to the original global forgetting behaviour).
        """
        omega = self.cfg.dirichlet_decay_omega
        prior = self.cfg.dirichlet_prior

        if active_states is None:
            # Global decay (fallback)
            states_to_decay = range(self.num_states)
        else:
            states_to_decay = [
                s for s in active_states if 0 <= s < self.num_states
            ]

        for s in states_to_decay:
            # Decay A[s, :]
            self._A[s, :] *= omega
            np.maximum(self._A[s, :], prior, out=self._A[s, :])
            # Decay B[s, a, :] for all actions
            for a in range(self.num_actions):
                self._B[s, a, :] *= omega
                np.maximum(self._B[s, a, :], prior, out=self._B[s, a, :])

    # ── Unknown-word surprise detection ─────────────────────────────

    def compute_novelty_surprise(self, obs_idx: int) -> float:
        """
        Compute the VFE prediction error for a single observation.

        When the user inputs a completely unrecognised word, the hash-
        based obs_idx will map to a rarely-seen observation bucket,
        causing a large surprise spike.  This value is consumed by the
        structure learner to decide whether BME should fire.

        Returns
        -------
        surprise : float
            − ln P(o)  marginalised over states.
        """
        return self.get_surprise(obs_idx)

    # ── Dynamically grow obs space for chatbot mode ─────────────────

    def expand_obs_space(self, new_num_obs: int) -> None:
        """
        Grow the A matrix when the observation space increases
        (e.g., more hash buckets needed for a larger vocabulary).

        New columns are initialised to the symmetric prior.

        Parameters
        ----------
        new_num_obs : int
            Target observation dimension.
        """
        if new_num_obs <= self.num_obs:
            return
        delta = new_num_obs - self.num_obs
        prior = self.cfg.dirichlet_prior
        new_cols = np.full(
            (self.num_states, delta), prior, dtype=np.float64
        )
        self._A = np.concatenate([self._A, new_cols], axis=1)

        # Extend preference vector C with zeros (neutral)
        new_C = np.zeros(delta, dtype=np.float64)
        self._C = np.concatenate([self._C, new_C])

        self.num_obs = new_num_obs
