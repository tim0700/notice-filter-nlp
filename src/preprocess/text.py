"""Korean text preprocessing, Okt morphological tokenization, and TF-IDF keywords."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess.stopwords import DEFAULT_STOPWORDS, load_stopwords
from src.schema import NoticeRecord

if TYPE_CHECKING:
    from konlpy.tag import Okt

IT_ACRONYMS: frozenset[str] = frozenset(
    {"AI", "SW", "IT", "ML", "DL", "QA", "UI", "UX", "DB", "C", "R"}
)
SINGLE_CHAR_IT_ACRONYMS: frozenset[str] = frozenset({"C", "R"})

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_DECORATIVE_BRACKET_RE = re.compile(r"[\[\]【】「」『』〈〉《》{}]")
_REPEATED_OPEN_PAREN_RE = re.compile(r"\(+")
_REPEATED_CLOSE_PAREN_RE = re.compile(r"\)+")
_EMPTY_PAREN_RE = re.compile(r"\(\s*\)")
_ALLOWED_CHARS_RE = re.compile(
    r'''[^\uAC00-\uD7A3\u3131-\u318EA-Za-z0-9\s.,!?:;\-/~%'"()·+&#]'''
)
_WHITESPACE_RE = re.compile(r"\s+")

_OKT_INSTANCE: Okt | None = None


def _get_okt() -> Okt:
    """Return a lazily initialized singleton KoNLPy Okt instance."""
    global _OKT_INSTANCE
    if _OKT_INSTANCE is None:
        from konlpy.tag import Okt

        _OKT_INSTANCE = Okt()
    return _OKT_INSTANCE


def _safe_str(value: Any) -> str:
    """Convert field value to stripped string while treating None and NaN as empty."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _identity(tokens: list[str]) -> list[str]:
    """Pass pre-tokenized token lists directly into scikit-learn vectorizers."""
    return tokens


def _is_hangul_char(ch: str) -> bool:
    """Return True if a character is a Hangul syllable or Jamo."""
    return ("\uAC00" <= ch <= "\uD7A3") or ("\u3131" <= ch <= "\u318E")


def _is_valid_token(token: str, stopwords: set[str]) -> bool:
    """Filter stopwords, punctuation, 1-char Hangul noise, and non-IT 1-char ASCII."""
    stripped = token.strip()
    if not stripped:
        return False
    if stripped in stopwords or stripped.lower() in stopwords:
        return False
    if not any(ch.isalnum() for ch in stripped):
        return False
    if len(stripped) == 1:
        if _is_hangul_char(stripped):
            return False
        return stripped in SINGLE_CHAR_IT_ACRONYMS
    return True


def clean_text(text: str) -> str:
    """Normalize HTML artifacts, URLs, emails, bracket markers, and whitespace."""
    raw = _safe_str(text)
    if not raw.strip():
        return ""

    cleaned = _SCRIPT_STYLE_RE.sub(" ", raw)
    cleaned = html.unescape(cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\xa0", " ").replace("\u200b", " ").replace("\ufeff", " ")
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _EMAIL_RE.sub(" ", cleaned)
    cleaned = _DECORATIVE_BRACKET_RE.sub(" ", cleaned)
    cleaned = _REPEATED_OPEN_PAREN_RE.sub("(", cleaned)
    cleaned = _REPEATED_CLOSE_PAREN_RE.sub(")", cleaned)
    cleaned = _EMPTY_PAREN_RE.sub(" ", cleaned)
    cleaned = _ALLOWED_CHARS_RE.sub(" ", cleaned)
    cleaned = _EMPTY_PAREN_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def tokenize_okt(
    text: str,
    mode: str = "nouns",
    stopwords: set[str] | None = None,
) -> list[str]:
    """Tokenize Korean text using KoNLPy Okt ('nouns' or 'morphs') with stopword filtering."""
    if mode not in {"nouns", "morphs"}:
        raise ValueError(f"Unsupported tokenization mode: {mode!r}")

    normalized = _safe_str(text).strip()
    if not normalized:
        return []

    active_stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
    okt = _get_okt()
    pos_tokens = okt.pos(normalized)

    if mode == "nouns":
        candidates = (tok for tok, tag in pos_tokens if tag in {"Noun", "Alpha"})
    else:
        candidates = (tok for tok, tag in pos_tokens if tag != "Punctuation")

    return [tok for tok in candidates if _is_valid_token(tok, active_stopwords)]


def compose_notice_text(
    record: Mapping[str, Any] | NoticeRecord,
    mode: str = "meta_title",
) -> str:
    """Compose cleaned model input string from a notice record ('title', 'meta_title', or 'full')."""
    data = record.to_dict() if isinstance(record, NoticeRecord) else record
    title = clean_text(_safe_str(data.get("title", "")))
    category = clean_text(_safe_str(data.get("category", "")))
    meta_tags = clean_text(_safe_str(data.get("meta_tags", "")).replace("|", " "))
    content = clean_text(_safe_str(data.get("content", "")))

    if mode == "title":
        return title
    if mode == "meta_title":
        parts = [part for part in (category, meta_tags, title) if part]
        return " ".join(parts)
    if mode == "full":
        parts = [part for part in (category, meta_tags, title, content) if part]
        return " ".join(parts)
    raise ValueError(f"Unsupported compose mode: {mode!r}")


def extract_okt_tfidf_keywords(
    texts: Sequence[str],
    top_k: int = 20,
    stopwords: set[str] | None = None,
    mode: str = "nouns",
) -> list[tuple[str, float]]:
    """Extract top-K corpus keywords using clean_text, Okt tokenization, and TF-IDF."""
    if top_k <= 0 or not texts:
        return []

    active_stopwords = load_stopwords() if stopwords is None else stopwords
    tokenized_docs = [
        tokenize_okt(clean_text(text), mode=mode, stopwords=active_stopwords) for text in texts
    ]
    non_empty_docs = [doc for doc in tokenized_docs if doc]
    if not non_empty_docs:
        return []

    vectorizer = TfidfVectorizer(
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
        lowercase=False,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(non_empty_docs)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) == 0:
        return []

    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    scored = [
        (str(term), float(score))
        for term, score in zip(feature_names, mean_scores, strict=True)
        if float(score) > 0.0
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


def extract_tfidf_keywords(
    texts: list[str],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Extract top-K corpus keywords ranked by TF-IDF score."""
    return extract_okt_tfidf_keywords(texts=texts, top_k=top_k)
