"""Quantitative evaluation contracts (Classification, Korean ROUGE, Judge Agreement)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    """Compute accuracy, positive-class precision/recall/F1, and confusion matrix."""
    raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")


def evaluate_korean_rouge(
    references: list[str],
    predictions: list[str],
    tokenizer: str = "space",
) -> dict[str, float]:
    """Compute Korean ROUGE-1 and ROUGE-L using custom 'space' or 'okt' tokenizers."""
    raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")


def evaluate_judge_agreement(
    y_human: Sequence[int],
    y_judge: Sequence[int],
) -> dict[str, float]:
    """Compute accuracy and Cohen's Kappa between Human Gold and LLM Silver labels."""
    raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")
