"""Offline unit tests for src/eval (metrics, stratified split, mocked LLM judge)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.config import load_profile
from src.eval import (
    AnthropicNoticeJudge,
    evaluate_classification,
    evaluate_judge_agreement,
    evaluate_korean_rouge,
    stratified_three_way_split,
)


def test_evaluate_classification_metrics_and_confusion_matrix() -> None:
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0]
    res = evaluate_classification(y_true, y_pred)

    assert pytest.approx(res["accuracy"], rel=1e-6) == 4 / 6
    assert pytest.approx(res["precision"], rel=1e-6) == 2 / 3
    assert pytest.approx(res["recall"], rel=1e-6) == 2 / 3
    assert pytest.approx(res["f1"], rel=1e-6) == 2 / 3
    assert res["confusion_matrix"] == [[2, 1], [1, 2]]

    # zero_division=0 when model predicts all negatives
    zero_pred = evaluate_classification([1, 0, 1, 0], [0, 0, 0, 0])
    assert zero_pred["precision"] == 0.0
    assert zero_pred["recall"] == 0.0
    assert zero_pred["f1"] == 0.0
    assert zero_pred["confusion_matrix"] == [[2, 0], [2, 0]]

    with pytest.raises(ValueError):
        evaluate_classification([1, 0], [1])
    with pytest.raises(ValueError):
        evaluate_classification([], [])


def test_evaluate_korean_rouge_space_and_okt() -> None:
    references = [
        "소프트웨어융합학과 4학년 졸업심사 일정을 안내합니다.",
        "성남시 분당구 거주 대학생에게 자격증 응시료를 지원합니다.",
    ]
    predictions = [
        "소프트웨어융합학과 4학년 졸업심사 일정 안내",
        "성남시 분당구 대학생 자격증 응시료 지원",
    ]

    space_scores = evaluate_korean_rouge(references, predictions, tokenizer="space")
    assert 0.0 < space_scores["rouge1"] < 1.0
    assert 0.0 < space_scores["rougeL"] < 1.0

    okt_scores = evaluate_korean_rouge(references, predictions, tokenizer="okt")
    assert 0.0 < okt_scores["rouge1"] <= 1.0
    assert 0.0 < okt_scores["rougeL"] <= 1.0
    # Okt strips particles ("을", "에게", "를") into separate morphemes, improving overlap on Korean text.
    assert okt_scores["rouge1"] > space_scores["rouge1"]

    identical = evaluate_korean_rouge(references, references, tokenizer="space")
    assert pytest.approx(identical["rouge1"], rel=1e-6) == 1.0
    assert pytest.approx(identical["rougeL"], rel=1e-6) == 1.0

    assert evaluate_korean_rouge([], [], tokenizer="space") == {"rouge1": 0.0, "rougeL": 0.0}

    with pytest.raises(ValueError, match="Unsupported tokenizer"):
        evaluate_korean_rouge(references, predictions, tokenizer="mecab")
    with pytest.raises(ValueError):
        evaluate_korean_rouge(references[:1], predictions, tokenizer="space")


def test_evaluate_judge_agreement() -> None:
    res = evaluate_judge_agreement([0, 1, 0, 1], [0, 1, 1, 1])
    assert pytest.approx(res["accuracy"], rel=1e-6) == 0.75
    assert pytest.approx(res["cohen_kappa"], rel=1e-6) == 0.5

    perfect_const = evaluate_judge_agreement([1, 1, 1], [1, 1, 1])
    assert perfect_const["accuracy"] == 1.0
    assert perfect_const["cohen_kappa"] == 1.0

    with pytest.raises(ValueError):
        evaluate_judge_agreement([0, 1], [0])


def test_stratified_three_way_split_disjoint_and_fallback() -> None:
    rows = []
    sources = ["swcon_academic", "swcon_career", "goodjob_general", "seongnam_youth"]
    for idx in range(60):
        rows.append(
            {
                "id": f"n_{idx:03d}",
                "source": sources[idx % len(sources)],
                "label": idx % 2,
                "title": f"공지 {idx}",
            }
        )
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = stratified_three_way_split(
        df, label_col="label", source_col="source", test_size=12, val_size=10, random_state=42
    )
    assert len(train_df) == 38
    assert len(val_df) == 10
    assert len(test_df) == 12

    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(df["id"])

    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert set(val_df.index).isdisjoint(set(test_df.index))

    for split in (train_df, val_df, test_df):
        assert set(split["label"].unique()) == {0, 1}

    # Tiny singleton stratum fallback
    tiny_df = pd.concat(
        [
            df,
            pd.DataFrame([{"id": "singleton_0", "source": "rare_source", "label": 1, "title": "희귀"}]),
        ],
        ignore_index=True,
    )
    tr, va, te = stratified_three_way_split(tiny_df, test_size=12, val_size=10, random_state=42)
    assert len(tr) + len(va) + len(te) == len(tiny_df)
    assert set(tr["id"]).isdisjoint(set(va["id"]))
    assert set(tr["id"]).isdisjoint(set(te["id"]))
    assert set(va["id"]).isdisjoint(set(te["id"]))

    with pytest.raises(ValueError):
        stratified_three_way_split(df, test_size=40, val_size=25)


class _FakeMessagesAPI:
    def __init__(self, fail_on_id: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on_id = fail_on_id

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        user_msg = kwargs["messages"][0]["content"]
        if self.fail_on_id and f"ID: {self.fail_on_id}\n" in user_msg:
            raise RuntimeError("Simulated transient API error")
        is_positive = "소프트웨어융합" in user_msg or "성남시" in user_msg
        payload = (
            '{"label": 1, "reason": "전공 및 거주지 관심 기준 충족"}'
            if is_positive
            else '{"label": 0, "reason": "관심 직무 및 지역과 무관"}'
        )
        return SimpleNamespace(content=[SimpleNamespace(text=payload)])


class _FakeAnthropicClient:
    def __init__(self, fail_on_id: str | None = None) -> None:
        self.messages = _FakeMessagesAPI(fail_on_id=fail_on_id)


def test_anthropic_judge_prompt_and_strict_json_parsing() -> None:
    profile = load_profile()
    judge = AnthropicNoticeJudge(profile=profile, client=_FakeAnthropicClient())

    assert "소프트웨어융합학과" in judge.system_prompt
    assert "성남시" in judge.system_prompt
    assert judge.model == "claude-haiku-4-5-20251001"

    assert judge.parse_response('{"label": 1, "reason": "적합"}') == {
        "label": 1,
        "reason": "적합",
    }
    assert judge.parse_response('```json\n{"label": 0, "reason": "제외 대상"}\n```') == {
        "label": 0,
        "reason": "제외 대상",
    }

    for bad_output in (
        "",
        "not json",
        '["label", 1]',
        '{"label": 1}',
        '{"label": true, "reason": "bool rejected"}',
        '{"label": 2, "reason": "out of range"}',
        '{"label": 1, "reason": "   "}',
    ):
        with pytest.raises(ValueError):
            judge.parse_response(bad_output)


def test_anthropic_judge_mocked_checkpoint_resume(tmp_path: Path) -> None:
    df_initial = pd.DataFrame(
        [
            {"id": "j1", "source": "swcon_academic", "category": "학사", "title": "소프트웨어융합 졸업", "meta_tags": "", "content": ""},
            {"id": "j2", "source": "goodjob_general", "category": "채용", "title": "영업관리 채용", "meta_tags": "", "content": ""},
            {"id": "j3", "source": "seongnam_youth", "category": "복지", "title": "성남시 청년 지원", "meta_tags": "", "content": ""},
        ]
    )
    out_csv = tmp_path / "silver_labels.csv"

    # Simulate failure on j3 after checkpointing j1 and j2
    failing_client = _FakeAnthropicClient(fail_on_id="j3")
    judge_fail = AnthropicNoticeJudge(client=failing_client)
    with pytest.raises(RuntimeError, match="Simulated transient API error"):
        judge_fail.judge_dataframe(df_initial, out_csv, save_every=1)

    saved_partial = pd.read_csv(out_csv, encoding="utf-8")
    assert saved_partial["id"].tolist() == ["j1", "j2"]
    assert len(failing_client.messages.calls) == 3

    # Resume on expanded DataFrame (j1, j2 cached; j3, j4 newly judged)
    df_expanded = pd.concat(
        [
            df_initial,
            pd.DataFrame(
                [{"id": "j4", "source": "goodjob_general", "category": "채용", "title": "회계 채용", "meta_tags": "", "content": ""}]
            ),
        ],
        ignore_index=True,
    )
    resume_client = _FakeAnthropicClient()
    judge_resume = AnthropicNoticeJudge(client=resume_client)
    resumed = judge_resume.judge_dataframe(df_expanded, out_csv, save_every=1)

    assert len(resume_client.messages.calls) == 2  # only j3 and j4
    assert resumed["id"].tolist() == ["j1", "j2", "j3", "j4"]
    assert resumed["judge_label"].tolist() == [1, 0, 1, 0]

    persisted = pd.read_csv(out_csv, encoding="utf-8")
    assert persisted["id"].tolist() == ["j1", "j2", "j3", "j4"]
    assert persisted["judge_label"].tolist() == [1, 0, 1, 0]
