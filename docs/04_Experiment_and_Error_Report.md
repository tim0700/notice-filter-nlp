# 04. 실험 결과 및 오류 분석 리포트

> 본 문서는 Phase 1 완료 후 Baseline 평가부터 Phase 2 단계적 개선 루프의 정량 지표와 오류 사례를 기록하는 살아있는 리포트입니다.
> **실험 원칙**: 한 번에 하나의 변수만 변경하여 동일한 Hold-out Test Set(150건, Human Gold)에서 성능을 비교한다.

---

## 1. 데이터셋 및 라벨 신뢰도 검증 (Phase 1 후 작성)

- 수집 총량: `(TBD) 건` (6개 필수 소스 + 선택 온통청년 API)
- 3-way 층화 분할:
  - Train / Feedback Pool: `(TBD) 건` (LLM-as-a-Judge Silver)
  - Validation Set: `100 건`
  - Hold-out Test Set: `150 건` (Human Gold)
  - Summary Eval Set: `20 건` (Human 참조 요약)
- **LLM-as-a-Judge vs Human Gold 일치율** (Test Set 일부 또는 교차 검증 구간):
  - Accuracy: `(TBD)` / Cohen's Kappa: `(TBD)`

---

## 2. 단계적 선별 모델 비교 실험 (Hold-out Test Set, n=150)

| 실험 단계 | Jira 티켓 | 변경 사항 (단일 변수) | Positive F1 | Precision | Recall | Accuracy | 비고 |
|---|---|---|---:|---:|---:|---:|---|
| Exp 0 (Baseline) | `NTF-14` | `ko-sroberta` 프로필 코사인 유사도 (`title` 기반, 피드백 0건) | `(TBD)` | `(TBD)` | `(TBD)` | `(TBD)` | Cold Start 기준선 |
| Exp 1 (Input/Thresh) | `(TBD)` | 입력 구성(`meta_title`) 및 Val 튜닝 임계값 적용 | `(TBD)` | `(TBD)` | `(TBD)` | `(TBD)` | 메타 태그 반영 |
| Exp 2 (Rocchio) | `(TBD)` | 소량 피드백 Rocchio 벡터 보정 ($\alpha, \beta$ Val 튜닝) | `(TBD)` | `(TBD)` | `(TBD)` | `(TBD)` | 소량 피드백 적응 |
| Exp 3a (대조군) | `(TBD)` | 누적 피드백 `TF-IDF + LogisticRegression` | `(TBD)` | `(TBD)` | `(TBD)` | `(TBD)` | 강의 교안 대조군 |
| Exp 3b (누적 ML) | `(TBD)` | 누적 피드백 `ko-sroberta + LogisticRegression` | `(TBD)` | `(TBD)` | `(TBD)` | `(TBD)` | 밀집 임베딩 분류 |
| Exp 4 (최신 NLP) | `(TBD)` | 최신 한국어 임베딩 (`nlpai-lab/KURE-v1` 또는 `BAAI/bge-m3`) 적용 | `(TBD)` | `(TBD)` | `(TBD)` | `(TBD)` | 128토큰 한계·영문 직무 개선 검증 |

---

## 3. 오류 분석 (Error Analysis — Jira `버그` 티켓 연동)

Baseline(Exp 0) 실행 후 오분류(FP/FN) 사례를 최소 3건 이상 실제 문장 단위로 분석해 아래 표와 Jira `버그` 티켓에 기록한다.

| Jira Bug Key | 오류 유형 (FP/FN) | 공지 원문 (ID / 제목) | 모델 예측 vs 정답 | 원인 분석 | 연결된 개선 실험 |
|---|---|---|---|---|---|
| `(TBD)` | `FP` 또는 `FN` | `(실제 오분류 문장 기록)` | `(TBD)` | `(TBD)` | `(TBD)` |

---

## 4. 추출 요약(`TextRank`) 한국어 `ROUGE` 평가 (Summary Eval Set, n=20)

> **주의**: `rouge_score` 기본 토크나이저는 ASCII 영숫자만 남겨 한글 토큰이 소실되므로, 공백 기준 및 `KoNLPy Okt` 형태소 기준 커스텀 토크나이저로 측정한다.

| 요약 방식 | 토크나이저 | ROUGE-1 (F1) | ROUGE-L (F1) | 비고 |
|---|---|---:|---:|---|
| `TextRank` (top_k=3) | 공백 기준 (`space`) | `(TBD)` | `(TBD)` | 어절 일치율 |
| `TextRank` (top_k=3) | 형태소 기준 (`okt`) | `(TBD)` | `(TBD)` | 조사 분리 실질 일치율 |
