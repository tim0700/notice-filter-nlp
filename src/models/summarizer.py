"""TextRank extractive summarizer for multi-sentence Korean notices."""

from __future__ import annotations

import re

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\d+\.)")
_WHITESPACE_RE = re.compile(r"\s+")


class TextRankSummarizer:
    """Extract top-K salient sentences via TF-IDF similarity graph and PageRank."""

    def __init__(self, top_k: int = 3, min_sentences: int = 5) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if min_sentences < 1:
            raise ValueError(f"min_sentences must be >= 1, got {min_sentences}")
        self.top_k = top_k
        self.min_sentences = min_sentences

    def split_sentences(self, text: str) -> list[str]:
        """Split Korean notice body text into ordered candidate sentences."""
        if not text or not str(text).strip():
            return []

        sentences: list[str] = []
        for raw_line in str(text).splitlines():
            line = _WHITESPACE_RE.sub(" ", raw_line).strip()
            if not line:
                continue
            for segment in _SENTENCE_SPLIT_RE.split(line):
                cleaned = segment.strip()
                # Avoid splitting on standalone list indices like "1."
                if cleaned and not re.fullmatch(r"[\d.\-•·*]+", cleaned):
                    sentences.append(cleaned)
        return sentences

    def _rank_sentences(self, sentences: list[str]) -> dict[int, float]:
        """Rank sentence indices with TF-IDF cosine graph PageRank and uniform fallback."""
        n = len(sentences)
        uniform = {idx: 1.0 / n for idx in range(n)}
        if n <= 1:
            return uniform

        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(sentences)
        except ValueError:
            return uniform

        sim_matrix = np.asarray((tfidf @ tfidf.T).toarray(), dtype=float)
        np.fill_diagonal(sim_matrix, 0.0)
        sim_matrix = np.maximum(sim_matrix, 0.0)

        if float(sim_matrix.sum()) <= 0.0:
            return uniform

        graph = nx.from_numpy_array(sim_matrix)
        try:
            scores = nx.pagerank(graph, weight="weight")
            return {int(k): float(v) for k, v in scores.items()}
        except (nx.PowerIterationFailedConvergence, ZeroDivisionError, ValueError):
            return uniform

    def summarize(self, text: str) -> str:
        """Return up to top_k salient sentences preserving original sentence order."""
        sentences = self.split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) <= self.min_sentences or len(sentences) <= self.top_k:
            return " ".join(sentences)

        scores = self._rank_sentences(sentences)
        ranked_indices = sorted(
            range(len(sentences)),
            key=lambda idx: (-scores.get(idx, 0.0), idx),
        )
        selected_indices = sorted(ranked_indices[: self.top_k])
        return " ".join(sentences[idx] for idx in selected_indices)
