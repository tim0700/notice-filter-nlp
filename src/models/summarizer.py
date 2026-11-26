"""TextRank extractive summarizer contract for multi-sentence notices."""

from __future__ import annotations


class TextRankSummarizer:
    """Extract top-K salient sentences via TF-IDF similarity graph and PageRank."""

    def __init__(self, top_k: int = 3, min_sentences: int = 5) -> None:
        self.top_k = top_k
        self.min_sentences = min_sentences

    def split_sentences(self, text: str) -> list[str]:
        """Split Korean notice body text into ordered candidate sentences."""
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")

    def summarize(self, text: str) -> str:
        """Return a 2-3 sentence summary preserving original sentence order."""
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")
