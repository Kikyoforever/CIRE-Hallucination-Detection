"""Independent, equation-level reference for CIRE single-response scoring."""

from .alignment import assert_matching_response_tokens, response_hidden_block
from .metrics import (
    CIREScores,
    cire_scores,
    geometric_mean_token_probability,
    hallucination_risk,
    relative_frobenius_shift,
)

__all__ = [
    "assert_matching_response_tokens",
    "response_hidden_block",
    "CIREScores",
    "cire_scores",
    "geometric_mean_token_probability",
    "hallucination_risk",
    "relative_frobenius_shift",
]
