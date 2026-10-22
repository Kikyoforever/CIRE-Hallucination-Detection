"""Synthetic CIRE example; run `python -m examples.minimal_example` from repo root.

There is no model download, prompt generation, factuality judgment or real-data
metric evaluation. The values below are mathematical checks only.
"""

import numpy as np

from src.cire import cire_scores, assert_matching_response_tokens


def main() -> None:
    # Two tokens from the SAME answer under answer-only vs question-conditioned
    # encodings, already strictly aligned and excluding special/prompt tokens.
    answer_only = np.array([[1.0, 0.0], [0.0, 1.0]])
    question_conditioned = np.array([[2.0, 0.0], [0.0, 1.0]])
    assert_matching_response_tokens([12, 24], [12, 24])

    # For illustration, the generated tokens had probabilities 0.5 and 0.8
    # under the QUESTION-CONDITIONED decoding distribution.
    log_probs = np.log(np.array([0.5, 0.8]))
    scores = cire_scores(answer_only, question_conditioned, token_log_probs=log_probs)
    print(f"IFL relative Frobenius shift: {scores.ifl:.6f}")
    print(f"P normalized output probability: {scores.normalized_output_probability:.6f}")
    print(f"CIRE groundedness score (IFL * P): {scores.groundedness_score:.6f}")
    print(f"Hallucination-risk ranking score (-CIRE): {scores.hallucination_risk_score:.6f}")


if __name__ == "__main__":
    main()
