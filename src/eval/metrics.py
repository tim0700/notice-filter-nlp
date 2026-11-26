"""Quantitative evaluation metrics (Classification, Korean ROUGE, Judge Agreement)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_SPACE_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]+")
_PUNCT_ONLY_RE = re.compile(r"[^\w가-힣]+")


class _SpaceTokenizer:
    """Whitespace/word tokenizer preserving Korean Hangul, ASCII words, and digits."""

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return _SPACE_TOKEN_RE.findall(str(text).lower())


class _OktTokenizer:
    """Morphological Korean tokenizer backed by KoNLPy Okt."""

    def __init__(self) -> None:
        from konlpy.tag import Okt

        self._okt = Okt()

    def tokenize(self, text: str) -> list[str]:
        if not text or not str(text).strip():
            return []
        morphs = self._okt.morphs(str(text))
        filtered = [tok for tok in morphs if tok.strip() and not _PUNCT_ONLY_RE.fullmatch(tok)]
        return filtered if filtered else [tok for tok in morphs if tok.strip()]


def _validate_binary_pair(
    left: Sequence[int],
    right: Sequence[int],
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two equal-length non-empty integer label sequences."""
    if len(left) != len(right):
        raise ValueError(
            f"{left_name} and {right_name} length mismatch: {len(left)} vs {len(right)}"
        )
    if len(left) == 0:
        raise ValueError(f"{left_name} and {right_name} must not be empty.")
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    """Compute accuracy, positive-class precision/recall/F1, and confusion matrix."""
    yt, yp = _validate_binary_pair(y_true, y_pred, "y_true", "y_pred")
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "confusion_matrix": [[int(cell) for cell in row] for row in cm.tolist()],
    }


def evaluate_korean_rouge(
    references: list[str],
    predictions: list[str],
    tokenizer: str = "space",
) -> dict[str, float]:
    """Compute Korean ROUGE-1 and ROUGE-L using custom 'space' or 'okt' tokenizers."""
    if len(references) != len(predictions):
        raise ValueError(
            f"references and predictions length mismatch: "
            f"{len(references)} vs {len(predictions)}"
        )
    if not references:
        return {"rouge1": 0.0, "rougeL": 0.0}

    mode = tokenizer.strip().lower()
    if mode == "space":
        tok: _SpaceTokenizer | _OktTokenizer = _SpaceTokenizer()
    elif mode == "okt":
        tok = _OktTokenizer()
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer!r}. Expected 'space' or 'okt'.")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=tok)
    rouge1_scores: list[float] = []
    rouge_l_scores: list[float] = []

    for ref, pred in zip(references, predictions):
        scores = scorer.score(str(ref or ""), str(pred or ""))
        rouge1_scores.append(float(scores["rouge1"].fmeasure))
        rouge_l_scores.append(float(scores["rougeL"].fmeasure))

    return {
        "rouge1": float(np.mean(rouge1_scores)),
        "rougeL": float(np.mean(rouge_l_scores)),
    }


def evaluate_judge_agreement(
    y_human: Sequence[int],
    y_judge: Sequence[int],
) -> dict[str, float]:
    """Compute accuracy and Cohen's Kappa between Human Gold and LLM Silver labels."""
    yh, yj = _validate_binary_pair(y_human, y_judge, "y_human", "y_judge")
    acc = float(accuracy_score(yh, yj))

    if len(np.unique(yh)) == 1 and np.array_equal(yh, yj):
        kappa = 1.0
    else:
        kappa = float(cohen_kappa_score(yh, yj, labels=[0, 1]))
        if math.isnan(kappa):
            kappa = 0.0

    return {
        "accuracy": acc,
        "cohen_kappa": kappa,
    }
