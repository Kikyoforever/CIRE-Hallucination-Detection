"""Core CIRE scoring for precomputed answer-token representations.

Inputs are pre-extracted and TOKEN-ALIGNED hidden-state matrices of one response
under (1) response-only and (2) question-conditioned encodings. Token generation
log probabilities must be from p(r_i | question, r_{<i}), not response-only logits.

This lightweight implementation does not tokenize text, call an LLM, infer a
hallucination label, or reproduce the full benchmark numbers in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIREScores:
    """CIRE and its components; these are ranking signals, not calibrated odds."""

    ifl: float
    normalized_output_probability: float
    groundedness_score: float
    hallucination_risk_score: float


def _aligned_matrices(
    answer_only: np.ndarray, conditioned_answer: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(answer_only, dtype=np.float64)
    c = np.asarray(conditioned_answer, dtype=np.float64)
    if a.ndim != 2 or c.ndim != 2 or a.shape != c.shape:
        raise ValueError("Both answer matrices must have exactly the same shape (tokens, hidden_dim)")
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("At least one answer token and one hidden dimension are required")
    if not np.isfinite(a).all() or not np.isfinite(c).all():
        raise ValueError("Hidden-state matrices must contain only finite values")
    return a, c


def relative_frobenius_shift(
    answer_only: np.ndarray, conditioned_answer: np.ndarray
) -> float:
    """Compute the relative Frobenius shift (IFL); require a nonzero denominator."""
    a, c = _aligned_matrices(answer_only, conditioned_answer)
    denominator = float(np.linalg.norm(a, ord="fro"))
    if denominator == 0.0 or not np.isfinite(denominator):
        raise ValueError("The response-only Frobenius norm must be finite and nonzero")
    numerator = float(np.linalg.norm(a - c, ord="fro"))
    if not np.isfinite(numerator):
        raise ValueError("The relative shift overflows; check embedding scaling")
    return numerator / denominator


def geometric_mean_token_probability(token_log_probs: np.ndarray) -> float:
    """P = exp(mean(log p(r_i | S,r_<i))) (Eq. 6).

    The input must be the log probability of each actual generated response token
    under question-conditioned decoding. Exclude prompt, EOS and padding tokens.
    """
    v = np.asarray(token_log_probs, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all():
        raise ValueError("token_log_probs must be a nonempty finite one-dimensional array")
    if np.any(v > 1e-12):
        raise ValueError("Log probabilities cannot be positive")
    # Clamp only a negligible *positive* floating-point roundoff at zero.
    result = float(np.exp(np.mean(np.minimum(v, 0.0))))
    if not (0.0 <= result <= 1.0):
        raise ArithmeticError("Normalized generation probability is outside [0, 1]")
    return result


def hallucination_risk(groundedness_score: float) -> float:
    """Orient CIRE to hallucination risk for ranking/AUROC: lower CIRE -> higher risk.

    Negative CIRE is an uncalibrated ranking score, NOT a hallucination
    probability. The benchmark labeling/threshold protocol is not included.
    """
    s = float(groundedness_score)
    if not np.isfinite(s) or s < 0:
        raise ValueError("groundedness_score must be finite and nonnegative")
    return -s


def cire_scores(
    answer_only: np.ndarray,
    conditioned_answer: np.ndarray,
    *,
    token_log_probs: np.ndarray,
) -> CIREScores:
    """Return IFL, P, IFL*P, and risk=-IFL*P.

    Inputs MUST describe the same response token sequence in both contexts;
    call assert_matching_response_tokens separately when token IDs are known.
    """
    a, c = _aligned_matrices(answer_only, conditioned_answer)
    logps = np.asarray(token_log_probs, dtype=np.float64)
    if logps.ndim != 1 or logps.shape[0] != a.shape[0]:
        raise ValueError("Exactly one generated-token log probability is required per response row")
    ifl = relative_frobenius_shift(a, c)
    p = geometric_mean_token_probability(logps)
    groundedness = float(ifl * p)
    if not np.isfinite(groundedness):
        raise ValueError("CIRE score is not finite")
    return CIREScores(
        ifl=ifl,
        normalized_output_probability=p,
        groundedness_score=groundedness,
        hallucination_risk_score=hallucination_risk(groundedness),
    )
