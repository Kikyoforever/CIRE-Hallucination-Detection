# CIRE: Contextualized Influence on Response Embeddings

**Grounded or Fabricated? Unsupervised Detection of LLM Hallucinations via Contextualized Influence on Response Embeddings**

**NeurIPS 2026 · Core Code Available · Full Experimental Code Coming Soon**

## Overview

CIRE is a training-free, single-response method for ranking hallucination risk. It compares the hidden states of the same answer tokens with and without the question context, then weights the difference by the answer's normalized generation probability.

This repository provides a lightweight **NumPy implementation of the core CIRE scoring formula**, a minimal example, and unit tests. **The full experimental code is being prepared and will be released later.**

## Core score

Given answer-token hidden-state matrices `X_R` (answer only) and `X_SR` (answer under the question context):

```math
\mathrm{IFL} = \frac{\|X_R - X_{SR}\|_F}{\|X_R\|_F},
\qquad
P = \exp\left(\frac{1}{m}\sum_{i=1}^{m}\log p(r_i\mid S,r_{<i})\right),
```

```math
\mathrm{CIRE} = \mathrm{IFL}\cdot P.
```

Higher CIRE scores indicate lower empirical hallucination risk on average, **not guaranteed factual correctness**. For risk-oriented ranking, the code also returns `-CIRE`; this is not a calibrated probability.

## Quick start

Python 3.10+:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m examples.minimal_example
```

A minimal example (no model download or GPU required):

```python
import numpy as np
from src.cire import cire_scores

# Precomputed hidden states for the same two answer tokens.
answer_only = np.array([[1.0, 0.0], [0.0, 1.0]])
question_conditioned = np.array([[2.0, 0.0], [0.0, 1.0]])

# Log probabilities of the generated tokens conditioned on the question.
token_log_probs = np.log([0.5, 0.8])

scores = cire_scores(
    answer_only,
    question_conditioned,
    token_log_probs=token_log_probs,
)
print(scores.ifl, scores.normalized_output_probability)
print(scores.groundedness_score, scores.hallucination_risk_score)
```

**Inputs:** The caller supplies matching, token-aligned answer-only and question-conditioned hidden-state matrices of shape `m × d`, plus one question-conditioned log probability per answer token. Model loading, hidden-state extraction, and benchmark reproduction are **not yet included**.

## Repository layout

```text
src/cire/                 Core scoring and token-alignment helpers
examples/minimal_example.py
                          Small synthetic example
tests/test_core.py        Unit tests
requirements.txt          Python dependencies
docs/IMPLEMENTATION_NOTES.md
                          Brief usage notes
```

## Release status

**Core scoring code: available. Full experimental code: coming soon.**
