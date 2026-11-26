"""Offline unit tests for Korean notice preprocessing, Okt tokenization, TF-IDF, and WordCloud."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import find_repo_root
from src.preprocess import (
    DEFAULT_STOPWORDS,
    clean_text,
    compose_notice_text,
    extract_okt_tfidf_keywords,
    extract_tfidf_keywords,
    find_korean_font_path,
    generate_wordcloud,
    load_stopwords,
    tokenize_okt,
)
from src.preprocess import wordcloud_viz
from src.preprocess.text import _get_okt
from src.schema import NoticeRecord


def test_stopwords_defaults_and_load_extra() -> None:
    assert isinstance(DEFAULT_STOPWORDS, set)
    assert {"안내", "공고", "모집", "신청", "경희대학교"}.issubset(DEFAULT_STOPWORDS)

    loaded = load_stopwords()
    assert loaded == DEFAULT_STOPWORDS
    assert loaded is not DEFAULT_STOPWORDS

    extended = load_stopwords(extra=["커스텀불용어", "  ", ""])
    assert "커스텀불용어" in extended
    assert "" not in extended
    assert "커스텀불용어" not in DEFAULT_STOPWORDS


def test_clean_text_html_urls_emails_brackets_and_punctuation() -> None:
    raw = (
        "<script>alert('x')</script>"
        "<p>[필독] &lt;b&gt;소프트웨어융합학과&lt;/b&gt;&nbsp;AI·보안 특강! "
        "자세한 내용은 https://swcon.khu.ac.kr/notice?id=100 또는 "
        "www.example.com 참고, 문의: admin@khu.ac.kr ★※▶ () [] ((중요))</p>"
    )
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "alert" not in cleaned
    assert "https://" not in cleaned
    assert "www.example.com" not in cleaned
    assert "admin@khu.ac.kr" not in cleaned
    assert "[" not in cleaned and "]" not in cleaned
    assert "★" not in cleaned and "※" not in cleaned and "▶" not in cleaned
    assert "필독" in cleaned
    assert "소프트웨어융합학과" in cleaned
    assert "AI·보안 특강!" in cleaned
    assert "(중요)" in cleaned

    # Preserve digits, C++, parentheses with content, and multi-sentence punctuation.
    tech_text = "C++ 및 안랩(AhnLab) 2026년 19~34세 지원율 50% 달성. 신청하세요!"
    assert clean_text(tech_text) == tech_text
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""


def test_compose_notice_text_modes_and_cleaning() -> None:
    notice = NoticeRecord(
        id="n_001",
        source="swcon_career",
        category="특강/AI",
        title="[특강] 생성형 AI &amp; 게임 개발 <br> https://example.com",
        meta_tags="국제캠퍼스|AI|SW",
        posted_at="2026-11-26",
        url="https://example.com/1",
        content="<p>실무 특강 내용입니다. 문의: help@khu.ac.kr</p>",
    )

    assert compose_notice_text(notice, mode="title") == "특강 생성형 AI & 게임 개발"
    assert (
        compose_notice_text(notice, mode="meta_title")
        == "특강/AI 국제캠퍼스 AI SW 특강 생성형 AI & 게임 개발"
    )
    assert (
        compose_notice_text(notice, mode="full")
        == "특강/AI 국제캠퍼스 AI SW 특강 생성형 AI & 게임 개발 실무 특강 내용입니다. 문의:"
    )

    row_with_nan = {
        "category": "일반채용",
        "meta_tags": "IT|백엔드",
        "title": "[네이버] 신입 채용",
        "content": float("nan"),
    }
    assert compose_notice_text(row_with_nan, mode="full") == "일반채용 IT 백엔드 네이버 신입 채용"

    with pytest.raises(ValueError, match="Unsupported compose mode"):
        compose_notice_text(notice, mode="invalid")


def test_tokenize_okt_real_nouns_morphs_and_acronyms() -> None:
    assert _get_okt() is _get_okt()
    assert tokenize_okt("") == []
    assert tokenize_okt("   ") == []

    text = (
        "소프트웨어융합학과 학생들을 위한 AI, SW, IT, ML, DL, QA, UI, UX, DB, C, R "
        "기반 실무 특강 및 프로젝트 안내입니다. 것 수 들 A X z"
    )
    noun_tokens = tokenize_okt(text, mode="nouns")
    for expected in ("소프트웨어", "특강", "프로젝트", "AI", "SW", "IT", "ML", "DL", "QA", "UI", "UX", "DB", "C", "R"):
        assert expected in noun_tokens
    for noise in ("안내", "학생", "것", "수", "들", "A", "X", "z"):
        assert noise not in noun_tokens

    morph_tokens = tokenize_okt("2026년 AI 백엔드 개발을 함께 공부합니다!", mode="morphs")
    assert "AI" in morph_tokens
    assert "백엔드" in morph_tokens
    assert "개발" in morph_tokens
    assert "!" not in morph_tokens
    assert "을" not in morph_tokens
    assert "함께" not in morph_tokens

    # Empty stopword set disables domain stopword removal while still dropping 1-char Hangul noise.
    no_stop_tokens = tokenize_okt("안내 공고 것 수", mode="nouns", stopwords=set())
    assert no_stop_tokens == ["안내", "공고"]

    with pytest.raises(ValueError, match="Unsupported tokenization mode"):
        tokenize_okt("테스트", mode="phrases")


def test_extract_tfidf_keywords_empty_and_sample_notices() -> None:
    assert extract_tfidf_keywords([]) == []
    assert extract_tfidf_keywords(["", "   ", "https://example.com 안내 공고"]) == []
    assert extract_tfidf_keywords(["소프트웨어 개발"], top_k=0) == []

    sample_path = find_repo_root() / "data" / "sample_notices.csv"
    df = pd.read_csv(sample_path, encoding="utf-8")
    corpus = [compose_notice_text(row.to_dict(), mode="full") for _, row in df.iterrows()]

    keywords = extract_tfidf_keywords(corpus, top_k=15)
    assert 0 < len(keywords) <= 15
    assert all(isinstance(term, str) and isinstance(score, float) and score > 0.0 for term, score in keywords)

    scores = [score for _, score in keywords]
    assert scores == sorted(scores, reverse=True)

    terms = {term for term, _ in keywords}
    assert terms.isdisjoint({"안내", "공고", "모집"})
    assert len(terms & {"소프트웨어", "AI", "채용", "청년", "클라우드", "게임", "보안", "성남시", "특강", "신입"}) >= 3

    custom_keywords = extract_okt_tfidf_keywords(
        corpus,
        top_k=5,
        stopwords=load_stopwords(extra=["청년", "채용"]),
    )
    assert 0 < len(custom_keywords) <= 5
    custom_terms = {term for term, _ in custom_keywords}
    assert "청년" not in custom_terms
    assert "채용" not in custom_terms


def test_wordcloud_font_discovery_and_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    font_path = find_korean_font_path()
    assert font_path is not None
    assert Path(font_path).is_file()

    # Exercise koreanize_matplotlib fallback when OS font candidate paths are absent.
    monkeypatch.setattr(
        wordcloud_viz,
        "_CANDIDATE_FONT_PATHS",
        (tmp_path / "missing_malgun.ttf",),
    )
    fallback_font = find_korean_font_path()
    assert fallback_font is not None
    assert Path(fallback_font).is_file()

    sample_path = find_repo_root() / "data" / "sample_notices.csv"
    df = pd.read_csv(sample_path, encoding="utf-8")
    texts = [compose_notice_text(row.to_dict(), mode="full") for _, row in df.iterrows()]

    out_file = tmp_path / "nested" / "notices_wc.png"
    result_path = generate_wordcloud(texts, out_file, width=400, height=240)
    assert result_path == out_file
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000
    assert out_file.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    with pytest.raises(ValueError, match="No valid tokens"):
        generate_wordcloud(["", "https://example.com 안내 공고"], tmp_path / "empty.png")
