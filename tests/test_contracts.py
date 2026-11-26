"""Phase 0 contract tests verifying schema, config, sample data, and module interfaces."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.adapters.discord import NoticeDiscordClient, append_feedback, format_notice_message
from src.collectors import COLLECTOR_REGISTRY, BaseCollector
from src.config import (
    CANONICAL_COLUMNS,
    FEEDBACK_COLUMNS,
    find_repo_root,
    find_shared_env_path,
    get_device,
    load_profile,
)
from src.eval import (
    evaluate_classification,
    evaluate_judge_agreement,
    evaluate_korean_rouge,
)
from src.models import (
    CosineProfileScorer,
    EmbeddingLogRegScorer,
    NoticeScorer,
    RocchioScorer,
    SentenceEmbedder,
    TextRankSummarizer,
    TfidfLogRegScorer,
)
from src.preprocess import (
    DEFAULT_STOPWORDS,
    clean_text,
    compose_notice_text,
    extract_tfidf_keywords,
    tokenize_okt,
)
from src.schema import (
    FeedbackRecord,
    NoticeRecord,
    normalize_notices_df,
    validate_notices_df,
)


def test_profile_yaml_contract() -> None:
    profile = load_profile()
    assert set(["user", "interests", "judge_rubric", "models", "sources"]).issubset(profile)
    assert profile["user"]["major"] == "소프트웨어융합학과"
    assert profile["models"]["baseline_embedding"] == "jhgan/ko-sroberta-multitask"
    assert profile["models"]["modern_embedding"]
    required_sources = [s for s in profile["sources"] if s.get("required")]
    assert len(required_sources) == 6
    assert len(profile["interests"]["positive_queries"]) >= 3
    assert len(profile["interests"]["negative_queries"]) >= 3


def test_sample_notices_csv_contract() -> None:
    sample_path = find_repo_root() / "data" / "sample_notices.csv"
    df = pd.read_csv(sample_path, encoding="utf-8")
    assert len(df) == 20
    assert list(df.columns) == CANONICAL_COLUMNS
    validate_notices_df(df)

    sources = set(df["source"].unique())
    expected_mandatory = {
        "swcon_academic",
        "swcon_career",
        "goodjob_extracurricular",
        "goodjob_general",
        "goodjob_recommend",
        "seongnam_youth",
    }
    assert expected_mandatory.issubset(sources)

    multi_sentence_count = sum(
        1
        for text in df["content"].fillna("")
        if len([seg for seg in str(text).split(".") if seg.strip()]) >= 5
    )
    assert multi_sentence_count >= 5


def test_notice_and_feedback_records(tmp_path: Path) -> None:
    notice = NoticeRecord(
        id="test_001",
        source="swcon_academic",
        category="학사",
        title="테스트 공지",
        meta_tags="국제캠퍼스|4학년",
        posted_at="2026-11-26",
        url="https://example.com/1",
    )
    assert list(notice.to_dict().keys()) == CANONICAL_COLUMNS
    assert compose_notice_text(notice, mode="meta_title") == "학사 국제캠퍼스 4학년 테스트 공지"

    feedback = FeedbackRecord(
        notice_id="test_001",
        label=1,
        source="discord",
        timestamp="2026-11-26T18:30:00",
        comment="관심 공지",
    )
    assert list(feedback.to_dict().keys()) == FEEDBACK_COLUMNS

    feedback_csv = tmp_path / "feedback.csv"
    append_feedback(feedback_csv, feedback)
    append_feedback(feedback_csv, FeedbackRecord("test_002", 0, "discord", "2026-11-26T18:31:00"))
    saved = pd.read_csv(feedback_csv, encoding="utf-8")
    assert list(saved.columns) == FEEDBACK_COLUMNS
    assert len(saved) == 2

    with pytest.raises(ValueError):
        FeedbackRecord(notice_id="bad", label=2, source="discord", timestamp="2026-11-26")


def test_validate_notices_df_rejects_duplicates() -> None:
    df = pd.DataFrame(
        [
            {
                "id": "dup_1",
                "source": "swcon_academic",
                "category": "학사",
                "title": "A",
                "meta_tags": "",
                "content": "",
                "posted_at": "2026-11-26",
                "deadline": "",
                "url": "https://example.com/a",
            },
            {
                "id": "dup_1",
                "source": "swcon_academic",
                "category": "학사",
                "title": "B",
                "meta_tags": "",
                "content": "",
                "posted_at": "2026-11-26",
                "deadline": "",
                "url": "https://example.com/b",
            },
        ]
    )
    normalized = normalize_notices_df(df)
    assert list(normalized.columns) == CANONICAL_COLUMNS
    with pytest.raises(ValueError, match="Duplicate"):
        validate_notices_df(normalized)


def test_device_and_env_resolution() -> None:
    assert get_device() in {"cuda", "cpu"}
    assert get_device("cpu") == "cpu"
    env_path = find_shared_env_path()
    assert env_path is None or env_path.name == ".env"


def test_phase1_interface_signatures() -> None:
    assert issubclass(CosineProfileScorer, NoticeScorer)
    assert issubclass(RocchioScorer, NoticeScorer)
    assert issubclass(EmbeddingLogRegScorer, NoticeScorer)
    assert issubclass(TfidfLogRegScorer, NoticeScorer)
    assert isinstance(COLLECTOR_REGISTRY, dict)
    assert hasattr(BaseCollector, "check_robots")
    assert hasattr(BaseCollector, "fetch")
    assert isinstance(DEFAULT_STOPWORDS, set)

    assert list(inspect.signature(clean_text).parameters) == ["text"]
    assert list(inspect.signature(tokenize_okt).parameters) == ["text", "mode", "stopwords"]
    assert list(inspect.signature(extract_tfidf_keywords).parameters) == ["texts", "top_k"]
    assert list(inspect.signature(SentenceEmbedder.__init__).parameters) == [
        "self",
        "model_name",
        "device",
        "cache_dir",
    ]
    assert list(inspect.signature(TextRankSummarizer.__init__).parameters) == [
        "self",
        "top_k",
        "min_sentences",
    ]
    assert list(inspect.signature(evaluate_classification).parameters) == ["y_true", "y_pred"]
    assert list(inspect.signature(evaluate_korean_rouge).parameters) == [
        "references",
        "predictions",
        "tokenizer",
    ]
    assert list(inspect.signature(evaluate_judge_agreement).parameters) == ["y_human", "y_judge"]

    notice = NoticeRecord(
        id="n1",
        source="seongnam_youth",
        category="청년복지",
        title="청년 지원금",
        meta_tags="성남시",
        posted_at="2026-11-26",
        url="https://example.com/n1",
    )
    payload = format_notice_message(notice, score=0.88, summary="핵심 요약")
    assert payload["title"] == "청년 지원금"
    assert len(payload["fields"]) == 3
    client = NoticeDiscordClient(channel_id=123456)
    assert client.channel_id == 123456
