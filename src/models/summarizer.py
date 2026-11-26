"""TextRank extractive summarizer for multi-sentence Korean notices."""

from __future__ import annotations

import html
import re

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\d+\.)")
_WHITESPACE_RE = re.compile(r"\s+")

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_DECORATIVE_SYMBOL_RE = re.compile(r"[💭★☆]+|\*{2,}")
_STRUCTURAL_BULLET_RE = re.compile(r"\s*([■□▪▶▷◆◇❍○●•ㆍ※①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])\s*")
_BRACKET_SECTION_RE = re.compile(r"\s*(\[[^\]\n]{2,15}\])\s*")
_INLINE_PAREN_LIST_RE = re.compile(r"(?<!\d)\s+(?=(?:\d{1,2}\)|[가나다라마바사]\.)\s)")
_INLINE_DOT_LIST_RE = re.compile(r"(?<!\d\.)(?<!\d)\s+(?=\d{1,2}\.\s+[가-힣A-Za-z])")
_PURE_INDEX_RE = re.compile(r"^(?:\d{1,2}[.)]?|[가나다라마바사]\.|[\d.\-•·*※❍○●]+)$")
_SECTION_HEADER_ONLY_RE = re.compile(r"^\[([^\]\n]{2,15})\]$")
_DOC_SECTION_HEADER_RE = re.compile(
    r"^(?:제출\s*서류|구비\s*서류|신청\s*서류|첨부\s*서류|유의\s*사항|참고\s*사항|문의\s*처|문의|연락처)$"
)

_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"제출\s*서류|구비\s*서류|각\s*\d+\s*부|\d+\s*부\b|"
    r"성적\s*증명서|재학\s*증명서|납부\s*확인서|자격\s*확인|자격득실\s*확인서|"
    r"주민등록\s*등본|가족관계\s*증명서|개인정보.{0,8}동의서|소정\s*양식|건강보험증\s*사본|"
    r"별첨|첨부\s*파일|붙임\s*\d*|"
    r"문의\s*[:：]|문의처|담당자\s*[:：]|연락처\s*[:：]|\b0\d{1,2}-\d{3,4}-\d{4}\b|"
    r"참고하시기\s*바랍니다|협조하여\s*주시기\s*바랍니다|"
    r"많은\s*관심\s*부탁|많은\s*참여\s*바랍니다|별도\s*문의를\s*하지\s*않도록"
    r")"
)
_DOC_SECTION_ITEM_RE = re.compile(r"(?:서류|증명서|확인서|동의서|사본|양식|1부|제출)")
_KEY_INFO_RE = re.compile(
    r"(?:대상|자격|지원|장학|혜택|시상|모집|선발|일시|일정|기간|접수|신청|장소|전형|직무|금액|만원)"
)


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


class PreprocessedTextRankSummarizer(TextRankSummarizer):
    """TextRank summarizer enhanced with notice-structure splitting, boilerplate filtering, and key-info boosting."""

    def __init__(
        self,
        top_k: int = 3,
        min_sentences: int = 5,
        lead_boost: float = 0.25,
        keyword_boost: float = 0.20,
        filter_boilerplate: bool = True,
    ) -> None:
        super().__init__(top_k=top_k, min_sentences=min_sentences)
        self.lead_boost = float(lead_boost)
        self.keyword_boost = float(keyword_boost)
        self.filter_boilerplate = bool(filter_boilerplate)

    @staticmethod
    def _clean_segment(segment: str) -> str:
        """Normalize spacing and trim dangling list symbols or unbalanced edge parentheses."""
        cleaned = _WHITESPACE_RE.sub(" ", segment).strip(" -•·ㆍ■□▪▶▷◆◇❍○●※")
        while cleaned.endswith(")") and cleaned.count(")") > cleaned.count("("):
            cleaned = cleaned[:-1].rstrip()
        while cleaned.startswith("(") and cleaned.count("(") > cleaned.count(")"):
            cleaned = cleaned[1:].lstrip()
        return cleaned.strip(" -•·ㆍ")

    @staticmethod
    def _is_boilerplate(sentence: str, in_doc_section: bool = False) -> bool:
        """Return True when a sentence is administrative checklist or footer boilerplate."""
        if _BOILERPLATE_RE.search(sentence):
            return True
        if in_doc_section and _DOC_SECTION_ITEM_RE.search(sentence):
            return True
        return False

    def preprocess_and_split(self, text: str) -> list[tuple[str, bool]]:
        """Normalize raw notice text and split into (sentence, is_boilerplate) pairs."""
        if not text or not str(text).strip():
            return []

        raw = str(text)
        raw = _SCRIPT_STYLE_RE.sub(" ", raw)
        raw = html.unescape(raw)
        raw = _HTML_TAG_RE.sub(" ", raw)
        raw = raw.replace("\xa0", " ").replace("\u200b", " ").replace("\ufeff", " ")
        raw = _URL_RE.sub(" ", raw)
        raw = _EMAIL_RE.sub(" ", raw)
        raw = _DECORATIVE_SYMBOL_RE.sub(" ", raw)

        # Insert newline boundaries around notice structural bullets, section headers, and inline indices.
        raw = _BRACKET_SECTION_RE.sub(r"\n\1\n", raw)
        raw = _STRUCTURAL_BULLET_RE.sub(r"\n\1 ", raw)
        raw = _INLINE_PAREN_LIST_RE.sub("\n", raw)
        raw = _INLINE_DOT_LIST_RE.sub("\n", raw)

        results: list[tuple[str, bool]] = []
        in_doc_section = False

        for raw_line in raw.splitlines():
            line = _WHITESPACE_RE.sub(" ", raw_line).strip()
            if not line:
                continue

            header_match = _SECTION_HEADER_ONLY_RE.match(line)
            if header_match:
                header_name = header_match.group(1).strip()
                in_doc_section = bool(_DOC_SECTION_HEADER_RE.match(header_name))
                continue

            for segment in _SENTENCE_SPLIT_RE.split(line):
                cleaned = self._clean_segment(segment)
                if not cleaned or len(cleaned) < 8 or _PURE_INDEX_RE.match(cleaned):
                    continue
                is_bp = self._is_boilerplate(cleaned, in_doc_section=in_doc_section)
                results.append((cleaned, is_bp))

        return results

    def extract_candidates(self, text: str) -> list[str]:
        """Return cleaned non-boilerplate sentences with fallback to all cleaned sentences."""
        tagged = self.preprocess_and_split(text)
        if not tagged:
            return []

        all_cleaned = [sent for sent, _ in tagged]
        if not self.filter_boilerplate:
            return all_cleaned

        filtered = [sent for sent, is_bp in tagged if not is_bp]
        if len(filtered) >= self.top_k:
            return filtered
        return all_cleaned

    def summarize(self, text: str) -> str:
        """Summarize notice text after structure-aware cleaning, boilerplate filtering, and key-info boosting."""
        candidates = self.extract_candidates(text)
        if not candidates:
            return ""
        if len(candidates) <= self.min_sentences or len(candidates) <= self.top_k:
            return " ".join(candidates)

        base_scores = self._rank_sentences(candidates)
        final_scores: dict[int, float] = {}
        for idx, sentence in enumerate(candidates):
            lead_bonus = (
                self.lead_boost
                if idx == 0
                else (self.lead_boost * 0.5 if idx == 1 else 0.0)
            )
            kw_bonus = self.keyword_boost if _KEY_INFO_RE.search(sentence) else 0.0
            final_scores[idx] = base_scores.get(idx, 0.0) * (1.0 + lead_bonus + kw_bonus)

        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda idx: (-final_scores.get(idx, 0.0), idx),
        )
        selected_indices = sorted(ranked_indices[: self.top_k])
        return " ".join(candidates[idx] for idx in selected_indices)
