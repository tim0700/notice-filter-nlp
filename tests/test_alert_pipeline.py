"""Offline unit tests for periodic alert pipeline, feedback adaptation, and memory cleanup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from scripts.run_alert_pipeline import (
    build_training_dataset,
    load_sent_ids,
    mark_sent_ids,
    run_alert_cycle,
    seed_sent_ids_from_csv,
)
from src.adapters.discord import append_feedback
from src.config import CANONICAL_COLUMNS
from src.models import SentenceEmbedder
from src.schema import FeedbackRecord


class _SpyDomainEmbedder:
    """Deterministic keyword embedder tracking unload() calls for offline pipeline tests."""

    def __init__(self) -> None:
        self.unloaded = False

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        del batch_size
        pos_kws = ("AI", "SW", "소프트웨어", "인턴", "성남시", "지원금", "장학")
        neg_kws = ("회계", "영업", "서울캠퍼스", "경력직", "생산관리")
        rows = []
        for text in texts:
            s = str(text)
            rows.append(
                [
                    float(sum(1 for k in pos_kws if k in s)) + 0.1,
                    float(sum(1 for k in neg_kws if k in s)) + 0.1,
                ]
            )
        arr = np.asarray(rows, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)
        return arr

    def unload(self) -> None:
        self.unloaded = True


def _sample_notices_df(rows: list[dict[str, str]]) -> pd.DataFrame:
    out = []
    for r in rows:
        base = {col: "" for col in CANONICAL_COLUMNS}
        base.update(r)
        out.append(base)
    return pd.DataFrame(out, columns=CANONICAL_COLUMNS)


def test_sentence_embedder_unload_releases_model(monkeypatch) -> None:
    embedder = SentenceEmbedder(device="cpu")
    embedder._model = MagicMock()
    embedder.unload()
    assert embedder._model is None


def test_seed_sent_ids_and_feedback_dataset_override(tmp_path: Path) -> None:
    raw_df = _sample_notices_df(
        [
            {"id": "n_pos", "source": "swcon_career", "category": "채용", "title": "AI SW 인턴 채용", "posted_at": "2026-11-26", "url": "https://example.com/1"},
            {"id": "n_neg", "source": "goodjob_general", "category": "채용", "title": "회계 영업 경력직 채용", "posted_at": "2026-11-26", "url": "https://example.com/2"},
            {"id": "n_new", "source": "seongnam_youth", "category": "복지", "title": "성남시 청년 지원금", "posted_at": "2026-11-26", "url": "https://example.com/3"},
        ]
    )
    raw_csv = tmp_path / "raw.csv"
    raw_df.to_csv(raw_csv, index=False, encoding="utf-8")

    train_df = raw_df.iloc[:2].copy()
    train_df["judge_label"] = [1, 0]
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8")

    state_file = tmp_path / "sent.json"
    added = seed_sent_ids_from_csv(state_file, raw_csv)
    assert added == 3
    assert set(load_sent_ids(state_file)) == {"n_pos", "n_neg", "n_new"}

    feedback_csv = tmp_path / "feedback.csv"
    append_feedback(feedback_csv, FeedbackRecord("n_new", 1, "discord", "2026-11-26T00:00:00Z", "관심 👍"))
    texts, labels = build_training_dataset(
        base_train_csv=train_csv,
        raw_notices_csv=raw_csv,
        latest_store_csv=tmp_path / "store.csv",
        feedback_csv=feedback_csv,
        text_mode="title",
        feedback_weight=3,
    )
    assert labels.count(1) == 4  # 1 base pos + 3 weighted feedback copies
    assert labels.count(0) == 1
    assert any("성남시 청년 지원금" in t for t in texts)


def test_run_alert_cycle_deduplicates_summarizes_and_unloads(tmp_path: Path) -> None:
    train_df = _sample_notices_df(
        [
            {"id": "tr_1", "source": "swcon_career", "category": "채용", "title": "AI SW 인턴 개발자", "posted_at": "2026-11-26", "url": "https://example.com/tr1"},
            {"id": "tr_2", "source": "seongnam_youth", "category": "복지", "title": "성남시 청년 장학 지원금", "posted_at": "2026-11-26", "url": "https://example.com/tr2"},
            {"id": "tr_3", "source": "goodjob_general", "category": "채용", "title": "회계 영업 경력직", "posted_at": "2026-11-26", "url": "https://example.com/tr3"},
            {"id": "tr_4", "source": "goodjob_general", "category": "채용", "title": "서울캠퍼스 생산관리 경력직", "posted_at": "2026-11-26", "url": "https://example.com/tr4"},
        ]
    )
    train_df["judge_label"] = [1, 1, 0, 0]
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8")

    fresh_df = _sample_notices_df(
        [
            {
                "id": "already_sent",
                "source": "swcon_career",
                "category": "채용",
                "title": "AI 백엔드 인턴 채용",
                "posted_at": "2026-11-26",
                "url": "https://example.com/old",
                "content": "AI 백엔드 인턴 모집 안내입니다.",
            },
            {
                "id": "fresh_pos",
                "source": "swcon_academic",
                "category": "장학",
                "title": "성남시 소프트웨어 장학 지원금 안내",
                "posted_at": "2026-11-26",
                "url": "https://example.com/pos",
                "content": (
                    "2026-2학기 소프트웨어융합학과 성남시 장학금 선발 안내입니다. "
                    "지원대상은 재학생 중 성적 우수 학생입니다. "
                    "혜택으로 1인당 200만원을 지원합니다. "
                    "신청기간은 12월 5일까지 온라인 접수입니다. "
                    "[제출서류] 1) 성적증명서 1부 2) 재학증명서 1부 "
                    "문의: 031-123-4567"
                ),
            },
            {
                "id": "fresh_neg",
                "source": "goodjob_general",
                "category": "채용",
                "title": "서울캠퍼스 회계 영업 경력직 채용",
                "posted_at": "2026-11-26",
                "url": "https://example.com/neg",
                "content": "회계 영업 경력직 5년 이상 모집입니다.",
            },
        ]
    )

    state_file = tmp_path / "sent.json"
    mark_sent_ids(state_file, ["already_sent"])
    spy_embedder = _SpyDomainEmbedder()

    batch = run_alert_cycle(
        threshold=0.50,
        max_alerts_per_run=5,
        state_file=state_file,
        feedback_csv=tmp_path / "feedback.csv",
        store_csv=tmp_path / "store.csv",
        base_train_csv=train_csv,
        raw_notices_csv=train_csv,
        config_path=tmp_path / "missing_cfg.json",
        embedder=spy_embedder,
        collect_fn=lambda **_: fresh_df,
    )

    assert spy_embedder.unloaded is True
    assert [notice.id for notice, _, _ in batch] == ["fresh_pos"]
    _, score, summary = batch[0]
    assert score >= 0.50
    assert "성적증명서 1부" not in summary
    assert "031-123-4567" not in summary
    assert "200만원을 지원합니다" in summary or "장학금 선발 안내" in summary
