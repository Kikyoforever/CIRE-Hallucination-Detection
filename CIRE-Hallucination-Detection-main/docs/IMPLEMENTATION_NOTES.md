# CIRE core scoring — implementation notes

This repository contains the **core CIRE score**, implemented with NumPy:

- **IFL:** relative Frobenius difference between two answer-token hidden-state matrices.
- **P:** geometric mean of the generated tokens' question-conditioned probabilities.
- **CIRE:** `IFL * P`, with `-CIRE` available for hallucination-risk ranking.

## Inputs

Callers must provide two **token-aligned** hidden-state matrices of the same shape (`answer_tokens × hidden_size`) and one question-conditioned token log probability per answer token. The scorer validates numerical inputs and matrix shapes. Token-ID alignment can be checked with the helpers in `src/cire/alignment.py`.

The score is a **ranking signal**, not a verified factuality judgment or calibrated probability.

## Availability

The present release covers **core scoring, a minimal example, and tests**. It does not yet include model generation, hidden-state extraction, dataset evaluation, or the complete experimental pipeline.

**Full experimental code coming soon.**
