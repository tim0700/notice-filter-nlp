# 01. 요구사항 명세서(SRS) 및 추적 매트릭스(RTM)

## 1. 문제 정의 (한 문장)
> **경희대학교 국제캠퍼스 소프트웨어융합학과 4학년(성남시 분당구 거주)** 학생이 학과·취업·지자체 청년 포털에 흩어진 공지 중 비IT·타지역·경력직 노이즈를 걸러내고, 본인의 관심 직무(AI·게임·보안·SW개발), 학사 일정, 청년 복지 공지만 선별·요약해 받아볼 수 있는 **문장 임베딩 + 피드백 적응형 공지 선별 시스템**을 구축한다.

---

## 2. 기능 요구사항 (Functional Requirements)

| ID | 요구사항 | 상세 기준 |
|---|---|---|
| `FR-01` | 다중 소스 공지·복지 수집 | `swcon` 2종, `goodjob` 3종(비로그인 공개 목록), `성남청년다해` 필수 수집 + `온통청년 Open API`(`YOUTH_API_KEY` 설정 시 선택 수집), `robots.txt` 준수 및 9컬럼 Canonical Schema 정규화 |
| `FR-02` | 한국어 텍스트 정제·키워드 분석 | 정규식 정제, `KoNLPy Okt` 형태소/명사 분석, 도메인 불용어 제거, `TF-IDF` 키워드 추출 및 `WordCloud` 시각화 |
| `FR-03` | Cold-Start 임베딩 선별 | `config/profile.yaml` 관심/제외 쿼리와 공지 간 코사인 유사도 계산 (`jhgan/ko-sroberta-multitask` 기준선) |
| `FR-04` | 사용자 피드백 기반 단계적 성능 개선 | 입력 구성·임계값 튜닝 → 소량 피드백 `Rocchio` 벡터 보정 → 누적 피드백 분류(`SBERT + LogReg` vs `TF-IDF + LogReg`) + 최신 한국어 임베딩(`KURE-v1` / `bge-m3`) 비교 |
| `FR-05` | 장문 공지 추출 요약 및 한국어 ROUGE 평가 | 본문 5문장 이상 공지 대상 `TextRank` (Raw) 및 `PreprocessedTextRankSummarizer`(개조식·서류 제거·핵심키워드 가중치) 2~3줄 요약과 커스텀 한국어 토크나이저(공백·Okt) 기반 `ROUGE-1/ROUGE-L` 측정 |
| `FR-06` | 정량 검증·오류 분석·데이터 누수 방지 | 3-way 층화 분할(Hold-out Test 150 / Val 100 / Train ~65% + Summary Eval 20), LLM-as-a-Judge(Silver) vs Human Gold 일치율(Cohen's Kappa), 오류 사례 3건 이상 분석 및 Jira `버그` 연동 |
| `FR-07` | 24/7 Discord 알림 및 영속 피드백 루프 | 선별 공지·요약 Embed 발송, `on_interaction` 기반 상시 `관심(1)/제외(0)` 피드백 `data/feedback.csv` 저장, `data/sent_notices.json` 중복 방지 및 매일 `09:00 KST` 자동 수집·재학습·알림 데몬 (`scripts/run_alert_pipeline.py`) |

---

## 3. 비기능 요구사항 (Non-Functional Requirements)

| ID | 요구사항 | 검증 기준 |
|---|---|---|
| `NFR-01` | CPU 재현성 및 노트북 완주 | 최종 노트북 `Kernel → Restart & Run All`이 GPU 없는 CPU 환경에서도 오류 없이 실행(`src.config.get_device()` 자동 분기 + `.npy` 임베딩 캐시) |
| `NFR-02` | 보안 및 저작권·크롤링 윤리 | INFO21 로그인 크롤링 배제(ADR-02), `.env` 비밀키 및 `.local/` 작업 문서 `.gitignore` 격리 |
| `NFR-03` | 모듈 독립성 및 Worktree 병렬성 | Core(`src/models/`, `src/eval/`)가 Adapter(`src/collectors/`, `src/adapters/discord/`)를 직접 호출하지 않고 정적 CSV 계약으로만 연결 |
| `NFR-04` | 추적성 (Traceability) | Jira `NTF` 티켓 ↔ Git 브랜치/커밋(`[NTF-N]`) ↔ PR(`Closes NTF-N`) ↔ 테스트 1:1 연결 |

---

## 4. 평가 항목(100점) 및 요구사항 매핑

| 평가 항목 (배점) | 매핑 요구사항 | 주요 산출물 |
|---|---|---|
| 문제 정의 (10) | 문제 정의 1문장, `config/profile.yaml` | `docs/01_SRS_and_RTM.md`, `README.md` |
| 데이터와 전처리 (15) | `FR-01`, `FR-02`, `NFR-02` | `src/collectors/`, `src/preprocess/`, `docs/03_ADR.md` (ADR-02) |
| NLP 기능 적용 (20) | `FR-03`, `FR-04`, `FR-05` | `src/models/` (임베딩 검색·분류·TextRank 3기능 결합) |
| 결과 검증 (20) | `FR-04`, `FR-05`, `FR-06` | `src/eval/metrics.py`, `docs/04_Experiment_and_Error_Report.md` |
| 오류 분석 (15) | `FR-06`, Jira `버그` 티켓 | `docs/04_Experiment_and_Error_Report.md`, Jira `NTF` `버그` 이슈 |
| 시각화·전달력 (10) | `FR-02`, `FR-06`, 발표 PPT | WordCloud, 혼동행렬, 실험별 F1 비교 차트, PPT |
| 재현성·정리 (10) | `NFR-01`, `NFR-04` | `README.md`, `requirements.txt`, 최종 `.ipynb` (`Restart & Run All`) |

---

## 5. 요구사항 추적 매트릭스 (RTM)

| Req ID | Jira Epic / Task | Worktree / 브랜치 | 구현 모듈 | 검증 테스트 |
|---|---|---|---|---|
| `NFR-03`, `NFR-04` | `NTF-9` / `NTF-11` | `chore/NTF-11-setup` | `docs/`, `src/config.py`, `src/schema.py`, `config/profile.yaml` | `tests/test_contracts.py` |
| `FR-01`, `NFR-02` | `NTF-10` / `NTF-12` | `wt-collector` (`feat/NTF-12-collector`) | `src/collectors/` | `tests/test_collectors.py` |
| `FR-02` | `NTF-10` / `NTF-13` | `wt-preprocess` (`feat/NTF-13-preprocess`) | `src/preprocess/` | `tests/test_preprocess.py` |
| `FR-03`, `FR-04`, `FR-05`, `FR-06`, `NFR-01` | `NTF-10` / `NTF-14` | `wt-nlp-core` (`feat/NTF-14-nlp-core`) | `src/models/`, `src/eval/` | `tests/test_models.py`, `tests/test_eval.py` |
| `FR-07`, `NFR-03` | `NTF-10` / `NTF-15` | `wt-discord` (`feat/NTF-15-discord`) | `src/adapters/discord/` | `tests/test_discord.py` |
| `FR-04`, `FR-05`, `FR-06`, `NFR-01` | `NTF-16` / `NTF-17`~`NTF-24` | `main` (Phase 2 실험·버그·노트북) | `scripts/run_experiments.py`, `src/models/summarizer.py`, `notebooks/notice_filter_pipeline.ipynb` | `results/final_test_results.json`, CPU 노트북 완주 (`8/8` 0에러) |
| `FR-07`, `NFR-02`, `NFR-03` | `NTF-25` | `main` (Phase 3 운영·배포) | `scripts/run_alert_pipeline.py`, `src/adapters/discord/bot.py`, `src/models/embedder.py`, `deploy/` | `tests/test_alert_pipeline.py` (48 passed), `risuai-server` `notice-filter.service` (`active (running)`) |
