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
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_EMPTY_PAREN_RE = re.compile(r"\(\s*\)|\[\s*\]")
_DECORATIVE_SYMBOL_RE = re.compile(r"[💭★☆]+|\*{2,}")
_STRUCTURAL_BULLET_RE = re.compile(r"\s*([■□▪▶▷◆◇❍○●•ㆍ※①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])\s*")
_BRACKET_SECTION_RE = re.compile(r"\s*(\[[^\]\n]{2,15}\])\s*")
_INLINE_PAREN_LIST_RE = re.compile(r"(?<!\d)\s+(?=(?:\d{1,2}\)|[가나다라마바사]\.)\s)")
_INLINE_DOT_LIST_RE = re.compile(r"(?<!\d\.)(?<!\d)\s+(?=\d{1,2}\.\s+[가-힣A-Za-z])")
_INLINE_STAR_BULLET_RE = re.compile(r"(?<!\*)\s+\*\s+(?=[가-힣A-Za-z0-9])")
_INLINE_DASH_LABEL_RE = re.compile(r"\s+-\s+(?=[가-힣A-Za-z][가-힣A-Za-z0-9\s/·]{0,16}\s*[:：])")
_INLINE_SLOT_LABEL_RE = re.compile(
    r"\s+(?=(?:일\s*시|모집\s*기간|접수\s*기간|신청\s*기간|운영\s*일시|장\s*소|운영\s*장소|대\s*상|모집\s*대상|지원\s*대상|모집\s*인원|교육\s*비|지원\s*금액|시상\s*규모)\s*[:：])"
)
_PURE_INDEX_RE = re.compile(r"^(?:\d{1,2}[.)]?|[가나다라마바사]\.|[\d.\-•·*※❍○●]+)$")
_SECTION_HEADER_ONLY_RE = re.compile(r"^\[([^\]\n]{2,15})\]$")
_DOC_SECTION_HEADER_RE = re.compile(
    r"^(?:제출\s*서류|구비\s*서류|신청\s*서류|첨부\s*서류|유의\s*사항|참고\s*사항|문의\s*처|문의|연락처)$"
)

_TRAILING_CONTACT_RE = re.compile(
    r"(?:주최\s*[:：].*?(?=문의)|문의\s*[:：]|문의처\s*[:：]?|문\s*의\s*[:：]|담당자\s*[:：]|연락처\s*[:：]|☎|📞).*$"
)
_PHONE_NUMBER_RE = re.compile(r"\(?\b0\d{1,2}[-.\s]\d{3,4}[-.\s]\d{4}\b\)?")
_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"제출\s*서류|구비\s*서류|각\s*\d+\s*부|\d+\s*부\b|"
    r"성적\s*증명서|재학\s*증명서|납부\s*확인서|자격\s*확인|자격득실\s*확인서|"
    r"주민등록\s*등본|가족관계\s*증명서|개인정보.{0,8}동의서|소정\s*양식|건강보험증\s*사본|"
    r"별첨|첨부\s*파일|붙임\s*\d*|"
    r"참고하시기\s*바랍니다|협조하여\s*주시기\s*바랍니다|"
    r"많은\s*관심\s*부탁|많은\s*참여\s*바랍니다|별도\s*문의를\s*하지\s*않도록|"
    r"해당\s*사업은\s*[^.\n]*플랫폼이\s*아닌|플랫폼에서\s*신청하신\s*건은\s*접수되지\s*않습니다|"
    r"참가\s*공고문\s*및\s*신청서를\s*꼭\s*정독|자세한\s*내용은\s*첨부\s*파일"
    r")"
)
_DOC_SECTION_ITEM_RE = re.compile(r"(?:서류|증명서|확인서|동의서|사본|양식|1부|제출)")
_KEY_INFO_RE = re.compile(
    r"(?:대상|자격|지원|장학|혜택|시상|모집|선발|일시|일정|기간|접수|신청|장소|전형|직무|금액|만원)"
)

_SLOT_SCHEDULE_RE = re.compile(
    r"^(?:모집\s*기간|신청\s*기간|접수\s*기간|제출\s*기한|마감\s*기한|사전\s*예약\s*기간|운영\s*일시|행사\s*일시|일\s*시|활동\s*기간|사업\s*기간|전형\s*일정|면접\s*일|접수\s*일정|일\s*정)(?:\s*[:：]\s*|\s*(?:은|는)\s+|\s+(?=(?:\d{4}|‘?\d{2}|\d{1,2}[./월]|~)))(.+)$"
)
_SLOT_TARGET_RE = re.compile(
    r"^(?:모집\s*대상|지원\s*대상|신청\s*대상|참여\s*대상|참가\s*대상|모집\s*인원|지원\s*자격|신청\s*자격|대상자|대\s*상)(?:\s*[:：]\s*|\s*(?:은|는)\s+)(.+)$"
)
_SLOT_BENEFIT_RE = re.compile(
    r"^(?:시상\s*규모|총\s*시상금|지원\s*규모|지원\s*내용|지원\s*금액|지원\s*혜택|참가자\s*혜택|장학\s*금액|지\s*원|혜\s*택|교육\s*비|참가\s*비|급\s*여)\s*[:：]\s*(.+)$"
)
_SLOT_BENEFIT_FALLBACK_RE = re.compile(
    r"(?:총\s*시상금\s*[^.!\n]+|최대\s*\d[\d,]*\s*(?:만\s*원|만원|원)[^.!\n]{0,40}|1인당\s*[^.!\n]*(?:만\s*원|만원|원)[^.!\n]{0,40}|전액\s*무료)"
)
_SLOT_LOCATION_RE = re.compile(
    r"^(?:운영\s*장소|개최\s*장소|교육\s*장소|행사\s*장소|대여\s*장소|장\s*소|근무\s*지)\s*[:：]\s*(.+)$"
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
        cleaned = _EMPTY_PAREN_RE.sub("", segment)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip(" -•·ㆍ*#■□▪▶▷◆◇❍○●※")
        while cleaned.endswith(")") and cleaned.count(")") > cleaned.count("("):
            cleaned = cleaned[:-1].rstrip()
        while cleaned.startswith("(") and cleaned.count("(") > cleaned.count(")"):
            cleaned = cleaned[1:].lstrip()
        return cleaned.strip(" -•·ㆍ*#")

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
        raw = _EMPTY_PAREN_RE.sub("", raw)
        raw = _DECORATIVE_SYMBOL_RE.sub(" ", raw)

        # Insert newline boundaries around notice structural bullets, section headers, and inline indices.
        raw = _BRACKET_SECTION_RE.sub(r"\n\1\n", raw)
        raw = _STRUCTURAL_BULLET_RE.sub(r"\n\1 ", raw)
        raw = _INLINE_PAREN_LIST_RE.sub("\n", raw)
        raw = _INLINE_DOT_LIST_RE.sub("\n", raw)
        raw = _INLINE_STAR_BULLET_RE.sub("\n", raw)
        raw = _INLINE_DASH_LABEL_RE.sub("\n", raw)
        raw = _INLINE_SLOT_LABEL_RE.sub("\n", raw)

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
                stripped_contact = _TRAILING_CONTACT_RE.sub(" ", segment)
                stripped_contact = _PHONE_NUMBER_RE.sub(" ", stripped_contact)
                cleaned = self._clean_segment(stripped_contact)
                if not cleaned or len(cleaned) < 8 or _PURE_INDEX_RE.match(cleaned):
                    continue
                is_bp = self._is_boilerplate(cleaned, in_doc_section=in_doc_section)
                results.append((cleaned, is_bp))

        return results

    @staticmethod
    def _deduplicate_sentences(sentences: list[str]) -> list[str]:
        """Remove exact normalized duplicate lines while preserving first-seen order."""
        seen: set[str] = set()
        unique: list[str] = []
        for sent in sentences:
            norm = _WHITESPACE_RE.sub(" ", sent).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(sent)
        return unique

    def extract_candidates(self, text: str) -> list[str]:
        """Return deduplicated, cleaned non-boilerplate sentences with fallback only when none remain."""
        tagged = self.preprocess_and_split(text)
        if not tagged:
            return []

        all_cleaned = self._deduplicate_sentences([sent for sent, _ in tagged])
        if not self.filter_boilerplate:
            return all_cleaned

        filtered = self._deduplicate_sentences([sent for sent, is_bp in tagged if not is_bp])
        if filtered:
            return filtered
        return all_cleaned

    def _compute_boost_factors(self, candidates: list[str]) -> dict[int, float]:
        """Compute lead-position + key-info multiplicative boost factors."""
        factors: dict[int, float] = {}
        for idx, sentence in enumerate(candidates):
            lead_bonus = (
                self.lead_boost
                if idx == 0
                else (self.lead_boost * 0.5 if idx == 1 else 0.0)
            )
            kw_bonus = self.keyword_boost if _KEY_INFO_RE.search(sentence) else 0.0
            factors[idx] = 1.0 + lead_bonus + kw_bonus
        return factors

    def summarize(self, text: str) -> str:
        """Summarize notice text after structure-aware cleaning, boilerplate filtering, and key-info boosting."""
        candidates = self.extract_candidates(text)
        if not candidates:
            return ""
        if len(candidates) <= self.min_sentences or len(candidates) <= self.top_k:
            return " ".join(candidates)

        base_scores = self._rank_sentences(candidates)
        factors = self._compute_boost_factors(candidates)
        final_scores = {
            idx: base_scores.get(idx, 0.0) * factors[idx]
            for idx in range(len(candidates))
        }

        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda idx: (-final_scores.get(idx, 0.0), idx),
        )
        selected_indices = sorted(ranked_indices[: self.top_k])
        return " ".join(candidates[idx] for idx in selected_indices)


class SemanticMMRTextRankSummarizer(PreprocessedTextRankSummarizer):
    """Embedding-based Semantic TextRank with optional title guidance and MMR redundancy control."""

    def __init__(
        self,
        embedder: object | None = None,
        top_k: int = 3,
        min_sentences: int = 5,
        lead_boost: float = 0.25,
        keyword_boost: float = 0.20,
        title_weight: float = 0.25,
        mmr_lambda: float = 0.70,
        filter_boilerplate: bool = True,
    ) -> None:
        super().__init__(
            top_k=top_k,
            min_sentences=min_sentences,
            lead_boost=lead_boost,
            keyword_boost=keyword_boost,
            filter_boilerplate=filter_boilerplate,
        )
        if not 0.0 <= float(title_weight) <= 1.0:
            raise ValueError(f"title_weight must be in [0, 1], got {title_weight}")
        if not 0.0 <= float(mmr_lambda) <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {mmr_lambda}")
        self._embedder = embedder
        self.title_weight = float(title_weight)
        self.mmr_lambda = float(mmr_lambda)

    def _get_embedder(self) -> object:
        """Return injected embedder or lazily construct default SentenceEmbedder."""
        if self._embedder is None:
            from src.models.embedder import SentenceEmbedder

            self._embedder = SentenceEmbedder()
        return self._embedder

    def summarize(self, text: str, title: str = "") -> str:
        """Summarize notice text via embedding PageRank, title similarity, and MMR selection."""
        candidates = self.extract_candidates(text)
        if not candidates:
            return ""
        if len(candidates) <= self.min_sentences or len(candidates) <= self.top_k:
            return " ".join(candidates)

        embedder = self._get_embedder()
        cand_vecs = np.asarray(embedder.encode(candidates, normalize=True), dtype=np.float32)  # type: ignore[attr-defined]
        pair_sim = np.clip(cand_vecs @ cand_vecs.T, 0.0, 1.0)
        graph_sim = pair_sim.copy()
        np.fill_diagonal(graph_sim, 0.0)

        n = len(candidates)
        pr_arr = np.full(n, 1.0 / n, dtype=float)
        if float(graph_sim.sum()) > 0.0:
            graph = nx.from_numpy_array(graph_sim)
            try:
                pr_dict = nx.pagerank(graph, weight="weight")
                pr_arr = np.array([float(pr_dict.get(i, 0.0)) for i in range(n)], dtype=float)
            except (nx.PowerIterationFailedConvergence, ZeroDivisionError, ValueError):
                pass

        max_pr = float(pr_arr.max())
        pr_norm = pr_arr / max_pr if max_pr > 0.0 else np.ones(n, dtype=float)

        clean_title = str(title or "").strip()
        if clean_title and self.title_weight > 0.0:
            title_vec = np.asarray(embedder.encode([clean_title], normalize=True), dtype=np.float32)[0]  # type: ignore[attr-defined]
            title_sim = np.clip(cand_vecs @ title_vec, 0.0, 1.0)
            base_rel = (1.0 - self.title_weight) * pr_norm + self.title_weight * title_sim
        else:
            base_rel = pr_norm

        factors = self._compute_boost_factors(candidates)
        boosted = np.array([base_rel[i] * factors[i] for i in range(n)], dtype=float)
        max_rel = float(boosted.max())
        rel_norm = boosted / max_rel if max_rel > 0.0 else np.ones(n, dtype=float)

        selected: list[int] = []
        remaining = list(range(n))
        k_pick = min(self.top_k, n)
        for _ in range(k_pick):
            if not selected:
                best_idx = max(remaining, key=lambda idx: (rel_norm[idx], -idx))
            else:
                best_idx = max(
                    remaining,
                    key=lambda idx: (
                        self.mmr_lambda * rel_norm[idx]
                        - (1.0 - self.mmr_lambda) * max(float(pair_sim[idx, j]) for j in selected),
                        -idx,
                    ),
                )
            selected.append(best_idx)
            remaining.remove(best_idx)

        return " ".join(candidates[idx] for idx in sorted(selected))


def _truncate_slot(value: str, max_len: int = 75) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", str(value or "")).strip(" -•·ㆍ")
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[: max_len - 3].rstrip()}..."


def extract_notice_slots(text: str, deadline: str = "") -> dict[str, str]:
    """Extract structured key-info slots (schedule, target, benefit, location) from notice text."""
    slots: dict[str, str] = {}
    splitter = PreprocessedTextRankSummarizer(filter_boilerplate=False)
    segments = splitter.preprocess_and_split(text)

    for seg, _ in segments:
        if "schedule" not in slots:
            m = _SLOT_SCHEDULE_RE.match(seg)
            if m and m.group(1).strip():
                slots["schedule"] = _truncate_slot(m.group(1))
                continue
        if "target" not in slots:
            m = _SLOT_TARGET_RE.match(seg)
            if m and m.group(1).strip():
                slots["target"] = _truncate_slot(m.group(1))
                continue
        if "benefit" not in slots:
            m = _SLOT_BENEFIT_RE.match(seg)
            if m and m.group(1).strip():
                slots["benefit"] = _truncate_slot(m.group(1))
                continue
        if "location" not in slots:
            m = _SLOT_LOCATION_RE.match(seg)
            if m and m.group(1).strip():
                slots["location"] = _truncate_slot(m.group(1))
                continue

    if "benefit" not in slots:
        for seg, _ in segments:
            fb = _SLOT_BENEFIT_FALLBACK_RE.search(seg)
            if fb:
                slots["benefit"] = _truncate_slot(fb.group(0))
                break

    if "schedule" not in slots and str(deadline or "").strip():
        slots["schedule"] = _truncate_slot(str(deadline).strip())

    return slots
