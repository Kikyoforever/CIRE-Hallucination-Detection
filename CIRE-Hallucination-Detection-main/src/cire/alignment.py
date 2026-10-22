"""Explicit answer-token positioning and strict alignment contracts.

CIRE compares the *same response tokens* in answer-only and question-conditioned
forward passes. The caller must locate corresponding tokens after tokenization.
No tokenization heuristics, silent sequence cropping, or prompt parsing is done here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def response_hidden_block(
    hidden_states: np.ndarray, *, answer_start: int, answer_length: int
) -> np.ndarray:
    """Extract one contiguous answer-token block from a token-by-hidden matrix.

    The indices refer to the actual tokenized input of the corresponding pass,
    not character positions. Special tokens must be excluded by the caller.
    """
    matrix = np.asarray(hidden_states, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("hidden_states must be a finite 2D token-by-hidden array")
    if not isinstance(answer_start, int) or not isinstance(answer_length, int):
        raise TypeError("answer_start and answer_length must be Python integers")
    if answer_start < 0 or answer_length <= 0 or answer_start + answer_length > matrix.shape[0]:
        raise ValueError("answer span is empty or outside hidden_states")
    return np.ascontiguousarray(matrix[answer_start : answer_start + answer_length])


def assert_matching_response_tokens(
    answer_only_ids: Sequence[int], conditioned_answer_ids: Sequence[int]
) -> None:
    """Raise on token-ID mismatch; the formula requires tokenwise correspondence.

    Different tokenizers/templates can tokenize the same text differently. This
    minimal reference deliberately DOES NOT silently crop or align via offsets.
    """
    a = list(answer_only_ids)
    b = list(conditioned_answer_ids)
    if not a or len(a) != len(b) or a != b:
        raise ValueError(
            "Token IDs for answer-only and conditioned answer are not identical; "
            "resolve the tokenization and response spans before scoring"
        )
