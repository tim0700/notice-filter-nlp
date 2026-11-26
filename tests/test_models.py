"""Offline unit tests for src/models (embedder, scorers, TextRank summarizer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import find_repo_root, load_profile
from src.models import (
    CosineProfileScorer,
    EmbeddingLogRegScorer,
    PreprocessedTextRankSummarizer,
    RocchioScorer,
    SentenceEmbedder,
    TextRankSummarizer,
    TfidfLogRegScorer,
)
from src.models.scorer import extract_texts_from_df


class FakeSentenceTransformer:
    """Deterministic offline stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.encode_calls: int = 0

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.encode_calls += 1
        rows = []
        for idx, text in enumerate(texts):
            seed = sum(ord(ch) for ch in str(text)) + idx
            vec = np.array(
                [
                    float((seed % 17) + 1),
                    float((seed % 11) + 1),
                    float(len(str(text)) % 13 + 1),
                    1.0,
                ],
                dtype=np.float32,
            )
            rows.append(vec)
        arr = np.vstack(rows)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / norms
        return arr


class FakeDomainEmbedder:
    """Deterministic keyword-projecting embedder for offline scorer tests."""

    _GROUPS: tuple[tuple[str, ...], ...] = (
        ("AI", "인공지능", "게임", "보안", "백엔드", "클라우드", "소프트웨어", "SW", "개발", "인턴", "신입"),
        ("학사", "졸업", "장학", "수강", "캡스톤", "국제캠퍼스", "4학년"),
        ("성남시", "분당구", "자격증", "응시료", "주거", "지원금", "연수"),
        ("영업", "회계", "생산관리", "기계", "순수예술", "디자인", "부산", "타지역"),
        ("서울캠퍼스", "경력직", "5년", "신입생", "편입생", "일반대학원", "석사"),
    )

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        rows = []
        for text in texts:
            s = str(text)
            vec = [float(sum(1 for kw in group if kw in s)) for group in self._GROUPS]
            vec.append(0.05)  # ensures non-zero norm even if no keyword hits
            rows.append(vec)
        arr = np.asarray(rows, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)
        return arr


def _make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "pos_1",
                "source": "swcon_academic",
                "category": "학사/졸업",
                "title": "소프트웨어융합학과 4학년 졸업논문 안내",
                "meta_tags": "국제캠퍼스|졸업",
                "content": "졸업 심사 일정 안내입니다.",
            },
            {
                "id": "pos_2",
                "source": "goodjob_general",
                "category": "일반채용",
                "title": "AI 백엔드 클라우드 신입 개발자 인턴 채용",
                "meta_tags": "IT|SW|신입",
                "content": "백엔드 및 AI 개발 신입 채용입니다.",
            },
            {
                "id": "pos_3",
                "source": "seongnam_youth",
                "category": "청년복지",
                "title": "성남시 분당구 청년 자격증 응시료 지원금",
                "meta_tags": "성남시|분당구|청년",
                "content": "대학생 응시료 지원 사업입니다.",
            },
            {
                "id": "neg_1",
                "source": "goodjob_general",
                "category": "일반채용",
                "title": "해외영업 및 회계 생산관리 채용",
                "meta_tags": "영업|회계",
                "content": "비IT 직무 채용입니다.",
            },
            {
                "id": "neg_2",
                "source": "swcon_career",
                "category": "특강/예술",
                "title": "서울캠퍼스 순수예술 디자인 특강",
                "meta_tags": "서울캠퍼스|디자인",
                "content": "서울캠퍼스 오프라인 행사입니다.",
            },
            {
                "id": "neg_3",
                "source": "goodjob_general",
                "category": "일반채용",
                "title": "AI 경력직 5년 이상 시니어 채용",
                "meta_tags": "경력직|5년이상",
                "content": "경력직 전용 채용 공고입니다.",
            },
        ]
    )


def test_sentence_embedder_lazy_loading_and_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeSentenceTransformer] = []

    def fake_loader(self: SentenceEmbedder) -> FakeSentenceTransformer:
        if self._model is None:
            self._model = FakeSentenceTransformer(self.model_name, device=self.device)
            created.append(self._model)
        return self._model

    monkeypatch.setattr(SentenceEmbedder, "_load_model", fake_loader)

    embedder = SentenceEmbedder(model_name="nlpai-lab/KURE-v1", device="cpu")
    assert embedder._model is None
    assert created == []

    assert embedder.encode([]).shape == (0, 0)
    assert created == []

    vecs = embedder.encode(["경희대 SW 인턴", "성남시 청년 지원"], normalize=True)
    assert len(created) == 1
    assert vecs.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_sentence_embedder_encode_cached_hit_and_stale_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models_by_name: dict[str, FakeSentenceTransformer] = {}

    def fake_loader(self: SentenceEmbedder) -> FakeSentenceTransformer:
        if self._model is None:
            model = FakeSentenceTransformer(self.model_name, device=self.device)
            models_by_name[self.model_name] = model
            self._model = model
        return self._model

    monkeypatch.setattr(SentenceEmbedder, "_load_model", fake_loader)

    texts = ["소프트웨어융합학과 졸업 안내", "성남시 청년 자격증 지원"]
    embedder = SentenceEmbedder(model_name="jhgan/ko-sroberta-multitask", cache_dir=tmp_path)

    first = embedder.encode_cached("notices_v1", texts)
    assert (tmp_path / "notices_v1.npy").is_file()
    assert (tmp_path / "notices_v1.meta.json").is_file()
    meta = json.loads((tmp_path / "notices_v1.meta.json").read_text(encoding="utf-8"))
    assert meta["model_name"] == "jhgan/ko-sroberta-multitask"
    assert len(meta["texts_sha256"]) == 64
    assert models_by_name["jhgan/ko-sroberta-multitask"].encode_calls == 1

    # Cache hit: must not invoke model.encode again
    second = embedder.encode_cached("notices_v1", texts)
    assert np.allclose(first, second)
    assert models_by_name["jhgan/ko-sroberta-multitask"].encode_calls == 1

    # Stale texts: must regenerate
    third = embedder.encode_cached("notices_v1", ["변경된 공지사항", "두 번째 공지"])
    assert models_by_name["jhgan/ko-sroberta-multitask"].encode_calls == 2
    assert not np.allclose(first, third)

    # Different model name with same cache_key and texts: must regenerate
    other_embedder = SentenceEmbedder(model_name="BAAI/bge-m3", cache_dir=tmp_path)
    other_embedder.encode_cached("notices_v1", ["변경된 공지사항", "두 번째 공지"])
    assert models_by_name["BAAI/bge-m3"].encode_calls == 1


def test_extract_texts_from_df_modes_and_nan() -> None:
    df = pd.DataFrame(
        [
            {
                "category": "학사",
                "meta_tags": "국제캠퍼스|4학년",
                "title": "졸업 안내",
                "content": np.nan,
            }
        ]
    )
    assert extract_texts_from_df(df, "title") == ["졸업 안내"]
    assert extract_texts_from_df(df, "meta_title") == ["학사 국제캠퍼스 4학년 졸업 안내"]
    assert extract_texts_from_df(df, "full") == ["학사 국제캠퍼스 4학년 졸업 안내"]
    assert extract_texts_from_df(df.iloc[0:0], "meta_title") == []

    with pytest.raises(ValueError, match="Unsupported text_mode"):
        extract_texts_from_df(df, "invalid_mode")


def test_cosine_profile_scorer_cold_start_and_negative_penalty() -> None:
    profile = load_profile()
    embedder = FakeDomainEmbedder()
    df = _make_sample_df()

    pos_only = CosineProfileScorer(
        embedder=embedder,
        profile_queries=profile["interests"]["positive_queries"],
    )
    scores_pos = pos_only.score(df)
    assert scores_pos.shape == (6,)
    assert scores_pos[0] > scores_pos[3]
    assert scores_pos[1] > scores_pos[4]
    assert scores_pos[2] > scores_pos[3]

    with_neg = CosineProfileScorer(
        embedder=embedder,
        profile_queries=profile["interests"]["positive_queries"],
        negative_queries=profile["interests"]["negative_queries"],
    )
    scores_neg = with_neg.score(df)
    # Row 5 ('AI 경력직 5년 이상') matches both AI and negative career keywords, so negative penalty lowers it.
    assert scores_neg[5] < scores_pos[5]
    preds = with_neg.predict(df, threshold=0.35)
    assert set(preds.tolist()).issubset({0, 1})
    assert preds[:3].tolist() == [1, 1, 1]
    assert preds[3:].tolist() == [0, 0, 0]


def test_rocchio_scorer_feedback_and_missing_polarity() -> None:
    profile = load_profile()
    embedder = FakeDomainEmbedder()
    df = _make_sample_df()

    scorer = RocchioScorer(
        embedder=embedder,
        profile_queries=profile["interests"]["positive_queries"],
        alpha=0.5,
        beta=0.25,
    )
    scorer.fit(
        feedback_texts=["AI 보안 백엔드 인턴 채용", "서울캠퍼스 회계 경력직"],
        feedback_labels=[1, 0],
    )
    assert scorer._profile_vector is not None
    assert np.isclose(np.linalg.norm(scorer._profile_vector), 1.0, atol=1e-5)

    scores = scorer.score(df)
    assert scores[1] > scores[3]

    # Missing negative samples
    pos_only_rocchio = RocchioScorer(embedder, profile["interests"]["positive_queries"]).fit(
        ["성남시 청년 자격증 지원"], [1]
    )
    assert np.isclose(np.linalg.norm(pos_only_rocchio._profile_vector), 1.0, atol=1e-5)

    # Missing positive samples
    neg_only_rocchio = RocchioScorer(embedder, profile["interests"]["positive_queries"]).fit(
        ["서울캠퍼스 일반대학원 모집"], [0]
    )
    assert np.isclose(np.linalg.norm(neg_only_rocchio._profile_vector), 1.0, atol=1e-5)


def test_embedding_and_tfidf_logreg_scorers() -> None:
    embedder = FakeDomainEmbedder()
    df = _make_sample_df()
    train_texts = [
        "학사 졸업 소프트웨어융합학과 4학년 안내",
        "AI 백엔드 클라우드 신입 개발자 인턴",
        "성남시 분당구 청년 자격증 응시료 지원금",
        "해외영업 회계 생산관리 채용",
        "서울캠퍼스 순수예술 디자인 특강",
        "경력직 5년 이상 일반대학원 모집",
    ]
    train_labels = [1, 1, 1, 0, 0, 0]

    emb_clf = EmbeddingLogRegScorer(embedder=embedder, c_value=1.0)
    with pytest.raises(RuntimeError):
        emb_clf.score(df)
    emb_clf.fit(train_texts, train_labels)
    emb_scores = emb_clf.score(df)
    assert emb_scores.shape == (6,)
    assert np.all((emb_scores >= 0.0) & (emb_scores <= 1.0))
    assert emb_scores[:3].min() > emb_scores[3:].max()

    tfidf_clf = TfidfLogRegScorer(c_value=1.0)
    with pytest.raises(RuntimeError):
        tfidf_clf.score(df)
    tfidf_clf.fit(train_texts, train_labels)
    tfidf_scores = tfidf_clf.score(df)
    assert tfidf_scores.shape == (6,)
    assert np.all((tfidf_scores >= 0.0) & (tfidf_scores <= 1.0))
    assert tfidf_scores[:3].min() > tfidf_scores[3:].max()

    with pytest.raises(ValueError):
        TfidfLogRegScorer().fit(["only positive"], [1])


def test_textrank_summarizer_on_sample_notices_and_edge_cases() -> None:
    summarizer = TextRankSummarizer(top_k=3, min_sentences=5)

    assert summarizer.split_sentences("") == []
    assert summarizer.summarize("") == ""

    short_text = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
    assert summarizer.summarize(short_text) == short_text

    # Multi-sentence Korean notice (> min_sentences) preserves order and returns top_k=3
    long_text = (
        "소프트웨어융합학과 4학년 대상 2026학년도 2학기 졸업심사 일정을 안내합니다. "
        "캡스톤디자인 결과보고서와 발표 자료를 12월 11일까지 업로드하세요. "
        "국제캠퍼스 전자정보대학관에서 대면 질의응답 심사가 진행됩니다. "
        "성적 우수 학생에게는 산학협력 장학금 150만 원이 별도 지급됩니다. "
        "졸업논문 표절 검사 확인서를 반드시 함께 제출해야 합니다. "
        "2026. 12. 11. 기한 내 미제출 시 졸업 사정에서 제외됩니다. "
        "문의 사항은 소프트웨어융합학과 학과 사무실로 연락 바랍니다."
    )
    all_sents = summarizer.split_sentences(long_text)
    assert len(all_sents) == 7

    summary = summarizer.summarize(long_text)
    summary_sents = summarizer.split_sentences(summary)
    assert len(summary_sents) == 3
    indices = [all_sents.index(s) for s in summary_sents]
    assert indices == sorted(indices)

    # Real data/sample_notices.csv multi-sentence notices
    sample_df = pd.read_csv(find_repo_root() / "data" / "sample_notices.csv", encoding="utf-8")
    non_empty_contents = [c for c in sample_df["content"].fillna("") if str(c).strip()]
    assert len(non_empty_contents) >= 5
    for content in non_empty_contents:
        out = summarizer.summarize(str(content))
        assert out
        assert len(summarizer.split_sentences(out)) <= len(summarizer.split_sentences(str(content)))

    # Disjoint vocabulary fallback (no edges in similarity graph)
    disjoint = "가나다라. 마바사아. 자차카타. 파하거너. 더러머버. 서어저처."
    fallback_summary = summarizer.summarize(disjoint)
    assert len(summarizer.split_sentences(fallback_summary)) == 3


def test_preprocessed_textrank_filters_checklist_and_splits_structure() -> None:
    prep = PreprocessedTextRankSummarizer(top_k=3, min_sentences=4)

    scholarship_notice = (
        "2026-2학기 문숙장학재단 신규장학생 선발 안내입니다. "
        "지원대상은 소프트웨어융합학과 4학년 재학생 중 직전학기 평점 3.5 이상 학생입니다. "
        "선발 인원에게는 학기당 등록금 200만원을 지원합니다. "
        "신청기간은 2026. 11. 21.부터 12. 05.까지이며 온라인으로 접수합니다. "
        "[제출서류] "
        "4) 직전학기 성적증명서 1부 - 각 학교 소정 양식 "
        "5) 최근 월이 포함된 건강·장기요양보험료 납부확인서 1부 "
        "• 부모 따로 가입된 경우 각 1부씩 -국민건강관리공단에서 확인 가능) "
        "6) 건강보험증 사본 또는 건강보험 자격확인(통보)서 1부 "
        "• 자격득실확인서가 아님 - 부모 따로 가입된 경우 각 1부씩) "
        "문의: 학생지원센터 장학팀 02-1234-5678"
    )
    summary = prep.summarize(scholarship_notice)
    assert "문숙장학재단 신규장학생 선발 안내입니다" in summary
    assert "등록금 200만원을 지원합니다" in summary or "소프트웨어융합학과 4학년" in summary
    assert "2026. 11. 21." in summary  # Date must not be broken across inline dot rule
    assert "성적증명서 1부" not in summary
    assert "납부확인서" not in summary
    assert "02-1234-5678" not in summary

    municipal_notice = (
        "[사업개요] "
        "❍ 일 시: 2026. 11. 21.(토) 14:00-17:00 "
        "❍ 대 상: 성남시 거주·재학 중인 청년 개인 혹은 팀 "
        "❍ 총시상금: 금 4,700,000원 및 우수 제안 시상 진행 "
        "❍ 접수방법: 홈페이지 접수 https://www.snyouth.or.kr/fmcs/215 "
        "[제출서류] "
        "❍ 신청서, 개인정보제공동의서 각 1부 제출 "
        "문의처: 청년지원팀 admin@example.com"
    )
    muni_summary = prep.summarize(municipal_notice)
    assert "https://www.snyouth.or.kr" not in muni_summary
    assert "admin@example.com" not in muni_summary
    assert "개인정보제공동의서" not in muni_summary
    assert "총시상금" in muni_summary or "성남시 거주" in muni_summary

    # Fallback when every line is boilerplate/checklist: never return empty string
    only_boilerplate = (
        "1) 재학증명서 1부 제출 "
        "2) 성적증명서 1부 제출 "
        "3) 가족관계증명서 1부 제출 "
        "4) 주민등록등본 1부 제출 "
        "문의: 031-729-4419"
    )
    bp_summary = prep.summarize(only_boilerplate)
    assert bp_summary.strip() != ""
    assert "재학증명서" in bp_summary

    # Real data/sample_notices.csv multi-sentence notices must never return empty
    sample_df = pd.read_csv(find_repo_root() / "data" / "sample_notices.csv", encoding="utf-8")
    for content in sample_df["content"].fillna(""):
        if str(content).strip():
            assert prep.summarize(str(content)).strip() != ""

    # H3: preserve core schedule/target/location when trailing '문의:' is appended on the same line
    inline_contact = (
        "일시: 2026년 5월 1일 장소: 야탑유스센터 대상: 성남시 청년 문의: 031-729-4419 "
        "주요내용: AI 취업 멘토링을 무료로 제공합니다. "
        "신청기간: 2026. 4. 10.까지 온라인 접수입니다."
    )
    inline_out = prep.summarize(inline_contact)
    assert "야탑유스센터" in inline_out
    assert "031-729-4419" not in inline_out

    # H2: when non-boilerplate sentences < top_k (but >= 1), do not resurrect document checklist lines
    sparse_notice = (
        "2026-2학기 문숙장학재단 신규장학생을 선발합니다. "
        "[제출서류] "
        "1) 지원 신청서 및 추천서 1부 "
        "2) 직전학기 성적증명서 1부 "
        "3) 건강보험료 납부확인서 1부 "
        "4) 주민등록등본 1부"
    )
    sparse_out = prep.summarize(sparse_notice)
    assert "문숙장학재단 신규장학생" in sparse_out
    assert "성적증명서 1부" not in sparse_out
    assert "납부확인서 1부" not in sparse_out
