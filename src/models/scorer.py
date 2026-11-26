"""Notice relevance scoring models for Cold Start and feedback-driven experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _clean_field(value: Any) -> str:
    """Convert a DataFrame cell value to a stripped string, treating NaN/None as empty."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def extract_texts_from_df(notices_df: pd.DataFrame, text_mode: str = "meta_title") -> list[str]:
    """Compose model input texts from a notices DataFrame without importing src.preprocess."""
    if text_mode not in {"title", "meta_title", "full"}:
        raise ValueError(f"Unsupported text_mode: {text_mode!r}")
    if notices_df.empty:
        return []

    texts: list[str] = []
    for row in notices_df.to_dict(orient="records"):
        title = _clean_field(row.get("title"))
        if text_mode == "title":
            texts.append(title)
            continue

        category = _clean_field(row.get("category"))
        meta_tags = _clean_field(row.get("meta_tags")).replace("|", " ")
        if text_mode == "meta_title":
            parts = [part for part in (category, meta_tags, title) if part]
        else:
            content = _clean_field(row.get("content"))
            parts = [part for part in (category, meta_tags, title, content) if part]
        texts.append(" ".join(parts))
    return texts


def _l2_normalize_2d(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D array safely."""
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)


def _l2_normalize_1d(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D vector safely."""
    arr = np.asarray(vector, dtype=float).ravel()
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else np.zeros_like(arr)


def _compute_centroid(embedder: Any, texts: list[str]) -> np.ndarray:
    """Encode texts and return their L2-normalized mean embedding vector."""
    if not texts:
        raise ValueError("Cannot compute centroid from an empty list of texts.")
    vectors = _l2_normalize_2d(embedder.encode(list(texts), normalize=True))
    return _l2_normalize_1d(vectors.mean(axis=0))


def _validate_labeled_texts(texts: list[str], labels: list[int]) -> tuple[list[str], np.ndarray]:
    """Validate parallel text/label training sequences for supervised scorers."""
    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels length mismatch: {len(texts)} vs {len(labels)}"
        )
    if not texts:
        raise ValueError("Training data must not be empty.")
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("At least two distinct classes (0 and 1) are required to fit.")
    return [("" if t is None else str(t)) for t in texts], y


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

    def __init__(
        self,
        embedder: Any,
        profile_queries: list[str],
        negative_queries: list[str] | None = None,
        text_mode: str = "meta_title",
        neg_weight: float = 0.25,
    ) -> None:
        if not profile_queries:
            raise ValueError("profile_queries must not be empty.")
        self.embedder = embedder
        self.profile_queries = list(profile_queries)
        self.negative_queries = list(negative_queries) if negative_queries else []
        self.text_mode = text_mode
        self.neg_weight = float(neg_weight)
        self._pos_centroid: np.ndarray | None = None
        self._neg_centroid: np.ndarray | None = None

    def _ensure_centroids(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self._pos_centroid is None:
            self._pos_centroid = _compute_centroid(self.embedder, self.profile_queries)
        if self.negative_queries and self._neg_centroid is None:
            self._neg_centroid = _compute_centroid(self.embedder, self.negative_queries)
        return self._pos_centroid, self._neg_centroid

    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        """Score notices via cosine similarity to positive profile (minus negative penalty)."""
        if notices_df.empty:
            return np.empty(0, dtype=float)
        pos_centroid, neg_centroid = self._ensure_centroids()
        texts = extract_texts_from_df(notices_df, text_mode=self.text_mode)
        doc_vecs = _l2_normalize_2d(self.embedder.encode(texts, normalize=True))
        scores = doc_vecs @ pos_centroid
        if neg_centroid is not None:
            neg_sim = doc_vecs @ neg_centroid
            scores = scores - self.neg_weight * np.maximum(neg_sim, 0.0)
        return np.asarray(scores, dtype=float)


class RocchioScorer(NoticeScorer):
    """Few-shot feedback scorer shifting profile vector via Rocchio update."""

    def __init__(
        self,
        embedder: Any,
        profile_queries: list[str],
        alpha: float = 0.5,
        beta: float = 0.25,
        text_mode: str = "meta_title",
    ) -> None:
        if not profile_queries:
            raise ValueError("profile_queries must not be empty.")
        self.embedder = embedder
        self.profile_queries = list(profile_queries)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.text_mode = text_mode
        self._profile_vector: np.ndarray | None = None

    def fit(self, feedback_texts: list[str], feedback_labels: list[int]) -> RocchioScorer:
        """Update the profile vector using labeled positive and negative feedback texts."""
        if len(feedback_texts) != len(feedback_labels):
            raise ValueError(
                f"feedback_texts and feedback_labels length mismatch: "
                f"{len(feedback_texts)} vs {len(feedback_labels)}"
            )
        v_new = _compute_centroid(self.embedder, self.profile_queries).copy()
        pos_texts = [str(t) for t, y in zip(feedback_texts, feedback_labels) if int(y) == 1]
        neg_texts = [str(t) for t, y in zip(feedback_texts, feedback_labels) if int(y) == 0]

        if pos_texts:
            v_new = v_new + self.alpha * _compute_centroid(self.embedder, pos_texts)
        if neg_texts:
            v_new = v_new - self.beta * _compute_centroid(self.embedder, neg_texts)

        self._profile_vector = _l2_normalize_1d(v_new)
        return self

    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        """Score notices via cosine similarity to the Rocchio-updated profile vector."""
        if notices_df.empty:
            return np.empty(0, dtype=float)
        if self._profile_vector is None:
            self._profile_vector = _compute_centroid(self.embedder, self.profile_queries)
        texts = extract_texts_from_df(notices_df, text_mode=self.text_mode)
        doc_vecs = _l2_normalize_2d(self.embedder.encode(texts, normalize=True))
        return np.asarray(doc_vecs @ self._profile_vector, dtype=float)


class EmbeddingLogRegScorer(NoticeScorer):
    """Supervised classifier over dense sentence embeddings with LogisticRegression."""

    def __init__(
        self,
        embedder: Any,
        c_value: float = 1.0,
        text_mode: str = "meta_title",
    ) -> None:
        self.embedder = embedder
        self.c_value = float(c_value)
        self.text_mode = text_mode
        self._clf: LogisticRegression | None = None

    def fit(self, train_texts: list[str], train_labels: list[int]) -> EmbeddingLogRegScorer:
        """Fit balanced LogisticRegression on normalized sentence embeddings."""
        clean_texts, y = _validate_labeled_texts(train_texts, train_labels)
        X = _l2_normalize_2d(self.embedder.encode(clean_texts, normalize=True))
        self._clf = LogisticRegression(
            C=self.c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self._clf.fit(X, y)
        return self

    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        """Return positive-class probability estimates for notices_df."""
        if self._clf is None:
            raise RuntimeError("EmbeddingLogRegScorer must be fitted before calling score().")
        if notices_df.empty:
            return np.empty(0, dtype=float)
        texts = extract_texts_from_df(notices_df, text_mode=self.text_mode)
        X = _l2_normalize_2d(self.embedder.encode(texts, normalize=True))
        return np.asarray(self._clf.predict_proba(X)[:, 1], dtype=float)


class TfidfLogRegScorer(NoticeScorer):
    """Lecture baseline classifier using TF-IDF features and LogisticRegression."""

    def __init__(self, c_value: float = 1.0, text_mode: str = "meta_title") -> None:
        self.c_value = float(c_value)
        self.text_mode = text_mode
        self._vectorizer: TfidfVectorizer | None = None
        self._clf: LogisticRegression | None = None

    def fit(self, train_texts: list[str], train_labels: list[int]) -> TfidfLogRegScorer:
        """Fit TF-IDF vectorizer and balanced LogisticRegression classifier."""
        clean_texts, y = _validate_labeled_texts(train_texts, train_labels)
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self._vectorizer.fit_transform(clean_texts)
        self._clf = LogisticRegression(
            C=self.c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self._clf.fit(X, y)
        return self

    def score(self, notices_df: pd.DataFrame) -> np.ndarray:
        """Return positive-class probability estimates for notices_df."""
        if self._clf is None or self._vectorizer is None:
            raise RuntimeError("TfidfLogRegScorer must be fitted before calling score().")
        if notices_df.empty:
            return np.empty(0, dtype=float)
        texts = extract_texts_from_df(notices_df, text_mode=self.text_mode)
        X = self._vectorizer.transform(texts)
        return np.asarray(self._clf.predict_proba(X)[:, 1], dtype=float)
