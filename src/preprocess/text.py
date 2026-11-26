"""Contract interfaces for Korean notice preprocessing (implemented in Phase 1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.schema import NoticeRecord

DEFAULT_STOPWORDS: set[str] = {
    "안내",
    "공고",
    "모집",
    "신청",
    "대상",
    "관련",
    "진행",
    "예정",
    "확인",
    "참고",
    "바랍니다",
    "경희대학교",
    "경희대",
}


def clean_text(text: str) -> str:
    """Normalize HTML artifacts, special symbols, and redundant whitespace."""
    raise NotImplementedError("Implemented in Phase 1 (wt-preprocess)")


def tokenize_okt(
    text: str,
    mode: str = "nouns",
    stopwords: set[str] | None = None,
) -> list[str]:
    """Tokenize Korean text using KoNLPy Okt ('nouns' or 'morphs') with stopword filtering."""
    raise NotImplementedError("Implemented in Phase 1 (wt-preprocess)")


def compose_notice_text(
    record: Mapping[str, Any] | NoticeRecord,
    mode: str = "meta_title",
) -> str:
    """Compose model input string from a notice record ('title', 'meta_title', or 'full')."""
    data = record.to_dict() if isinstance(record, NoticeRecord) else record
    title = str(data.get("title", "") or "").strip()
    category = str(data.get("category", "") or "").strip()
    meta_tags = str(data.get("meta_tags", "") or "").strip().replace("|", " ")
    content = str(data.get("content", "") or "").strip()

    if mode == "title":
        return title
    if mode == "meta_title":
        parts = [part for part in (category, meta_tags, title) if part]
        return " ".join(parts)
    if mode == "full":
        parts = [part for part in (category, meta_tags, title, content) if part]
        return " ".join(parts)
    raise ValueError(f"Unsupported compose mode: {mode!r}")


def extract_tfidf_keywords(
    texts: list[str],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Extract top-K corpus keywords ranked by TF-IDF score."""
    raise NotImplementedError("Implemented in Phase 1 (wt-preprocess)")
