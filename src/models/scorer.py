"""Notice relevance scoring contracts for Cold Start and feedback-driven experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class NoticeScorer(ABC):
    """Base contract for scoring and thresholding candidate notices."""

    @abstractmethod
    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        """Return relevance scores in [0, 1] or cosine range for each row in notices_df."""

    def predict(self, notices_df: pd.DataFrame, threshold: float = 0.45) -> np.ndarray:
        """Convert continuous relevance scores into binary {0, 1} predictions."""
        scores = np.asarray(self.score(notices_df), dtype=float)
        return (scores >= threshold).astype(int)


class CosineProfileScorer(NoticeScorer):
    """Zero-shot Cold Start scorer using cosine similarity to profile queries."""

    def __init__(self, embedder: object, profile_queries: list[str]) -> None:
        self.embedder = embedder
        self.profile_queries = profile_queries

    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")


class RocchioScorer(NoticeScorer):
    """Few-shot feedback scorer shifting profile vector via Rocchio update."""

    def __init__(
        self,
        embedder: object,
        profile_queries: list[str],
        alpha: float = 0.5,
        beta: float = 0.25,
    ) -> None:
        self.embedder = embedder
        self.profile_queries = profile_queries
        self.alpha = alpha
        self.beta = beta

    def fit(self, feedback_texts: list[str], feedback_labels: list[int]) -> RocchioScorer:
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")

    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")


class EmbeddingLogRegScorer(NoticeScorer):
    """Supervised classifier over dense sentence embeddings with LogisticRegression."""

    def __init__(self, embedder: object, c_value: float = 1.0) -> None:
        self.embedder = embedder
        self.c_value = c_value

    def fit(self, train_texts: list[str], train_labels: list[int]) -> EmbeddingLogRegScorer:
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")

    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")


class TfidfLogRegScorer(NoticeScorer):
    """Lecture baseline classifier using TF-IDF features and LogisticRegression."""

    def __init__(self, c_value: float = 1.0) -> None:
        self.c_value = c_value

    def fit(self, train_texts: list[str], train_labels: list[int]) -> TfidfLogRegScorer:
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")

    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")
