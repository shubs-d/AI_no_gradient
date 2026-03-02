"""
dirichlet_diagnostics.py
========================
Mathematically grounded diagnostic metrics for the Dirichlet-based
policy and inference system.  All formulas derived from the Dirichlet
distribution's known analytical properties — no heuristics.

Key metrics:
  (1) Expected policy entropy  H(α)
  (2) Normalised entropy ratio  ρ_H ∈ [0, 1]
  (3) Attractor reachability condition
  (4) Effective sample size  n_eff
  (5) Concentration scaling diagnostic
  (6) Phase instability checklist (automated)
  (7) Safe concentration rescaling transform

All functions are pure (no side effects), operate on numpy arrays,
and require no gradients.

References
──────────
  • Minka (2000), "Estimating a Dirichlet distribution"
  • Thompson (1933), "On the likelihood that one unknown probability
    exceeds another in view of the evidence of two samples"
  • Johnson, Kotz & Balakrishnan, "Continuous Multivariate Distributions"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import digamma, gammaln


# ═══════════════════════════════════════════════════════════════════════
# 1. Expected Entropy of Categorical Sampled from Dirichlet
# ═══════════════════════════════════════════════════════════════════════

def expected_categorical_entropy(alpha: np.ndarray) -> float:
    """
    Compute  E_{θ ~ Dir(α)}[ H[Cat(θ)] ]  analytically.

    Formula (exact):
        E[H] = -Σ_i  (α_i / α_0) · (ψ(α_i + 1) - ψ(α_0 + 1))

    where α_0 = Σ_i α_i  and  ψ = digamma.

    Parameters
    ----------
    alpha : ndarray (K,)
        Dirichlet concentration parameters (all > 0).

    Returns
    -------
    entropy : float
        Expected entropy in nats.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha_0 = alpha.sum()
    if alpha_0 < 1e-12:
        return 0.0
    proportions = alpha / alpha_0
    return float(-np.sum(
        proportions * (digamma(alpha + 1.0) - digamma(alpha_0 + 1.0))
    ))


def normalised_entropy_ratio(alpha: np.ndarray) -> float:
    """
    ρ_H = E[H(α)] / ln(K)  ∈ [0, 1]

    Values:
      > 0.85  → overdispersed (near-uniform sampling)
      0.4–0.85 → healthy exploration–exploitation balance
      0.15–0.4 → converging (near-attractor)
      < 0.15  → collapsed (degenerate)

    Parameters
    ----------
    alpha : ndarray (K,)

    Returns
    -------
    rho : float in [0, 1]
    """
    K = len(alpha)
    if K <= 1:
        return 0.0
    h = expected_categorical_entropy(alpha)
    return float(h / np.log(K))


# ═══════════════════════════════════════════════════════════════════════
# 2. Thompson Sampling Attractor Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def attractor_threshold(K: int, alpha_0: float) -> float:
    """
    Minimum concentration ratio  α_k / α_0  for 90% attractor
    stability under Thompson sampling with argmax.

    Derived from the Gumbel approximation to the max of K−1
    Beta-distributed random variables:

        r_k > 0.5 + √(2 ln(K−1)) / (2√(α_0 + 1))

    Parameters
    ----------
    K : int
        Number of competing actions/candidates.
    alpha_0 : float
        Total Dirichlet concentration.

    Returns
    -------
    threshold : float
        Minimum α_k / α_0 ratio for 90%-decisive selection.
    """
    if K <= 1:
        return 0.0
    return 0.5 + np.sqrt(2.0 * np.log(max(K - 1, 1))) / (2.0 * np.sqrt(alpha_0 + 1.0))


def check_attractor_reachable(
    alpha: np.ndarray,
    eta: float,
    omega: float,
    alpha_prior: float,
) -> Dict[str, float]:
    """
    Determine whether a stable attractor CAN form given the learning
    dynamics  θ_{t+1} = ω·θ_t + η·evidence.

    Equilibrium count for a consistently-observed transition:
        α_eq = η / (1 − ω) + α_prior

    Equilibrium total:
        α_0_eq = α_eq + (K−1) · α_prior

    Returns diagnostics dict.
    """
    K = len(alpha)
    alpha_0 = float(alpha.sum())
    alpha_eq = eta / (1.0 - omega) + alpha_prior
    alpha_0_eq = alpha_eq + (K - 1) * alpha_prior
    eq_ratio = alpha_eq / alpha_0_eq
    threshold = attractor_threshold(K, alpha_0_eq)

    return {
        "K": K,
        "alpha_0_current": alpha_0,
        "alpha_equilibrium": alpha_eq,
        "alpha_0_equilibrium": alpha_0_eq,
        "equilibrium_ratio": eq_ratio,
        "attractor_threshold": threshold,
        "attractor_reachable": eq_ratio > threshold,
        "observations_to_attractor": _obs_to_attractor(
            K, alpha_prior, eta, omega, threshold
        ),
    }


def _obs_to_attractor(
    K: int,
    alpha_prior: float,
    eta: float,
    omega: float,
    threshold: float,
) -> float:
    """
    Estimate the number of consistent observations needed to reach
    the attractor threshold, accounting for exponential decay.

    After n observations:
        α_n = α_prior + η · Σ_{t=0}^{n-1} ω^{n-1-t} = α_prior + η · (1 − ω^n)/(1 − ω)
        α_0_n = α_n + (K−1)·α_prior

    Solve  α_n / α_0_n > threshold  for n.
    """
    for n in range(1, 1000):
        alpha_n = alpha_prior + eta * (1.0 - omega**n) / (1.0 - omega)
        alpha_0_n = alpha_n + (K - 1) * alpha_prior
        if alpha_n / alpha_0_n > threshold:
            return float(n)
    return float("inf")


# ═══════════════════════════════════════════════════════════════════════
# 3. Effective Sample Size
# ═══════════════════════════════════════════════════════════════════════

def effective_sample_size(alpha: np.ndarray, alpha_prior: float) -> float:
    """
    Effective number of observations beyond the prior.

        n_eff = α_0 − K · α_prior

    Negative values indicate the system has decayed below prior.
    """
    return float(alpha.sum() - len(alpha) * alpha_prior)


# ═══════════════════════════════════════════════════════════════════════
# 4. Component-wise Variance Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def dirichlet_component_variance(alpha: np.ndarray) -> np.ndarray:
    """
    Var(θ_i) = α_i(α_0 − α_i) / (α_0²(α_0 + 1))

    Returns per-component variance array.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha_0 = alpha.sum()
    if alpha_0 < 1e-12:
        return np.zeros_like(alpha)
    return alpha * (alpha_0 - alpha) / (alpha_0**2 * (alpha_0 + 1.0))


def expected_thompson_gap(alpha: np.ndarray) -> float:
    """
    Expected gap between the highest and second-highest Thompson
    sample:  E[θ_{(1)} − θ_{(2)}].

    Approximated via order statistics of the marginal Beta
    distributions  θ_i ~ Beta(α_i, α_0 − α_i).

    A small gap indicates that argmax selection is dominated by
    noise rather than learned structure.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    K = len(alpha)
    alpha_0 = alpha.sum()
    if K <= 1 or alpha_0 < 1e-12:
        return 0.0

    means = alpha / alpha_0
    stds = np.sqrt(dirichlet_component_variance(alpha))

    # Gumbel approximation for the gap between max and second-max
    # of K quasi-independent samples with given means and stds
    # Gap ≈ max_std / √(2 ln K)  for similar means
    max_std = np.max(stds)
    gap = max_std / np.sqrt(max(2.0 * np.log(K), 1.0))
    return float(gap)


# ═══════════════════════════════════════════════════════════════════════
# 5. Safe Concentration Rescaling
# ═══════════════════════════════════════════════════════════════════════

def rescale_concentration(
    alpha: np.ndarray,
    target_mass: float,
    floor: float = 0.01,
) -> np.ndarray:
    """
    Rescale Dirichlet concentrations to a target total mass while
    preserving the relative proportions:

        α'_i = (α_i / α_0) · M_target

    with a per-component floor to avoid degeneracy.

    Parameters
    ----------
    alpha : ndarray (K,)
        Current concentration parameters.
    target_mass : float
        Desired Σ α'_i.
    floor : float
        Minimum per-component value (numerical stability).

    Returns
    -------
    alpha_rescaled : ndarray (K,)
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha_0 = alpha.sum()
    if alpha_0 < 1e-12:
        return np.full_like(alpha, target_mass / len(alpha))
    alpha_new = (alpha / alpha_0) * target_mass
    np.maximum(alpha_new, floor, out=alpha_new)
    return alpha_new


# ═══════════════════════════════════════════════════════════════════════
# 6. Phase Instability Checklist (Automated)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PhaseReport:
    """Structured diagnostic report for Dirichlet policy health."""
    entropy_ratio: float
    expected_entropy: float
    max_entropy: float
    effective_n: float
    thompson_gap: float
    attractor_reachable: bool
    observations_to_attractor: float
    dominant_ratio: float
    alpha_0: float
    diagnosis: str
    interventions: List[str]


def full_phase_diagnostic(
    alpha: np.ndarray,
    eta: float = 1.0,
    omega: float = 0.95,
    alpha_prior: float = 0.2,
) -> PhaseReport:
    """
    Run the complete phase instability diagnostic checklist.

    Parameters
    ----------
    alpha : ndarray (K,)
        Current Dirichlet concentrations for the active candidate set.
    eta : float
        Learning rate.
    omega : float
        Dirichlet decay rate.
    alpha_prior : float
        Per-component prior.

    Returns
    -------
    PhaseReport with diagnosis and recommended interventions.
    """
    K = len(alpha)
    alpha_0 = float(alpha.sum())
    rho = normalised_entropy_ratio(alpha)
    h = expected_categorical_entropy(alpha)
    h_max = float(np.log(max(K, 2)))
    n_eff = effective_sample_size(alpha, alpha_prior)
    gap = expected_thompson_gap(alpha)

    att = check_attractor_reachable(alpha, eta, omega, alpha_prior)
    dominant_idx = int(np.argmax(alpha))
    dominant_ratio = float(alpha[dominant_idx] / alpha_0) if alpha_0 > 0 else 0.0

    # ── Diagnosis ─────────────────────────────────────────────────
    interventions: List[str] = []

    if rho > 0.85:
        diagnosis = "OVERDISPERSED"
        interventions.append(
            f"Rescale α_0 from {alpha_0:.1f} to target_mass=5.0 "
            f"(ratio would drop from {rho:.3f} to ~0.74)"
        )
    elif rho > 0.4:
        diagnosis = "HEALTHY"
    elif rho > 0.15:
        diagnosis = "CONVERGING"
        interventions.append("Monitor for collapse; maintain exploration ε ≥ 0.05")
    else:
        diagnosis = "COLLAPSED"
        interventions.append(
            f"Inject entropy: reset α to uniform prior (α_0 = K·α_prior = "
            f"{K * alpha_prior:.1f})"
        )

    if not att["attractor_reachable"]:
        interventions.append(
            f"Attractor UNREACHABLE: eq_ratio={att['equilibrium_ratio']:.3f} "
            f"< threshold={att['attractor_threshold']:.3f}. "
            f"Reduce α_prior or increase η."
        )

    if gap < 0.01:
        interventions.append(
            f"Thompson gap={gap:.4f} is negligible — argmax selection "
            f"dominated by noise. Increase precision β or decrease α_0."
        )

    if n_eff < 0:
        interventions.append(
            f"Negative effective sample size ({n_eff:.1f}): "
            f"decay is erasing evidence faster than it accumulates."
        )

    return PhaseReport(
        entropy_ratio=rho,
        expected_entropy=h,
        max_entropy=h_max,
        effective_n=n_eff,
        thompson_gap=gap,
        attractor_reachable=att["attractor_reachable"],
        observations_to_attractor=att["observations_to_attractor"],
        dominant_ratio=dominant_ratio,
        alpha_0=alpha_0,
        diagnosis=diagnosis,
        interventions=interventions,
    )


# ═══════════════════════════════════════════════════════════════════════
# 7. DCM (Pólya Urn) Predictive Probability
# ═══════════════════════════════════════════════════════════════════════

def dcm_predictive(
    alpha: np.ndarray,
    counts_this_utterance: np.ndarray,
) -> np.ndarray:
    """
    Dirichlet-Compound-Multinomial (Pólya urn) predictive probability
    for the next token, given Dirichlet prior α and intra-utterance
    counts accumulated so far.

        P(x_{t+1} = k | x_{1:t}, α) = (α_k + n_k) / (α_0 + n_total)

    This provides *burstiness*: selecting word k once increases its
    probability on the next draw, naturally modelling topical coherence.

    Parameters
    ----------
    alpha : ndarray (K,)
        Dirichlet concentration (prior + learned counts, rescaled).
    counts_this_utterance : ndarray (K,)
        How many times each candidate was selected in the current
        utterance so far (intra-utterance counts).

    Returns
    -------
    probs : ndarray (K,)
        Predictive probability distribution over candidates.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = np.asarray(counts_this_utterance, dtype=np.float64)
    numerator = alpha + n
    denominator = numerator.sum()
    if denominator < 1e-12:
        return np.ones(len(alpha)) / len(alpha)
    return numerator / denominator


def dcm_log_marginal(
    alpha: np.ndarray,
    counts: np.ndarray,
) -> float:
    """
    Log marginal likelihood of observed counts under DCM:

      ln P(n | α) = ln Γ(α_0) − ln Γ(α_0 + N) + Σ_k [ln Γ(α_k + n_k) − ln Γ(α_k)]

    where N = Σ_k n_k.  Useful for model comparison / MDL scoring.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    alpha_0 = alpha.sum()
    N = counts.sum()
    return float(
        gammaln(alpha_0) - gammaln(alpha_0 + N)
        + np.sum(gammaln(alpha + counts) - gammaln(alpha))
    )


# ═══════════════════════════════════════════════════════════════════════
# 8. Precision-Scaled Sampling (Revised Equation)
# ═══════════════════════════════════════════════════════════════════════

def precision_scaled_alpha(
    raw_alpha: np.ndarray,
    target_mass: float,
    grammar_boost: np.ndarray,
    clause_boost: np.ndarray,
    precision_beta: float,
    activation_weights: Optional[np.ndarray] = None,
    floor: float = 0.01,
) -> np.ndarray:
    """
    Compute the final Dirichlet concentration vector with all
    modulations applied BEFORE sampling (Bayesian-consistent).

    Revised sampling equation:

        α'_i = (α_i / α_0) · M_target · g_i^grammar · g_i^clause · β

    Optionally modulated by activation weights (multiplicative).

    Parameters
    ----------
    raw_alpha : ndarray (K,)
        Raw Dirichlet counts from the bigram table.
    target_mass : float
        Target total concentration mass M_target.
    grammar_boost : ndarray (K,)
        Per-candidate grammar role boost (≥ 1.0).
    clause_boost : ndarray (K,)
        Per-candidate clause-derived boost  exp(v_k / T).
    precision_beta : float
        Global precision multiplier β.
    activation_weights : ndarray (K,), optional
        Spreading-activation weights (> 0).
    floor : float
        Minimum per-component value.

    Returns
    -------
    alpha_final : ndarray (K,)
        Ready for  θ ~ Dir(α_final)  or  DCM predictive.
    """
    alpha = np.asarray(raw_alpha, dtype=np.float64)
    alpha_0 = alpha.sum()
    if alpha_0 < 1e-12:
        alpha_0 = 1.0

    # Step 1: Rescale to target mass
    alpha_scaled = (alpha / alpha_0) * target_mass

    # Step 2: Apply grammar boost (multiplicative, pre-sampling)
    alpha_scaled *= np.asarray(grammar_boost, dtype=np.float64)

    # Step 3: Apply clause boost (multiplicative, pre-sampling)
    alpha_scaled *= np.asarray(clause_boost, dtype=np.float64)

    # Step 4: Apply activation weights if provided
    if activation_weights is not None:
        act = np.asarray(activation_weights, dtype=np.float64)
        act = np.maximum(act, 0.01)  # floor activations
        alpha_scaled *= act

    # Step 5: Apply global precision
    alpha_scaled *= precision_beta

    # Step 6: Floor
    np.maximum(alpha_scaled, floor, out=alpha_scaled)

    return alpha_scaled
