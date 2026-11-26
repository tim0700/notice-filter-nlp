# 03. 아키텍처 의사결정 기록 (ADR)

## ADR-01: Python 3.11 공유 `.venv` 및 CPU/CUDA 자동 분기 + 플랫폼 중립 `requirements.txt`

- **상태**: 채택됨 (Phase 0)
- **배경**: 로컬 개발 PC에는 Python 3.10~3.14와 RTX 4070 Ti(CUDA)가 함께 존재하지만, PyTorch CUDA 빌드·`JPype1(KoNLPy)`·`sentence-transformers` 간 안정 호환성과 제출 노트북의 CPU 전용 평가 환경(`Restart & Run All`)을 동시에 만족해야 함.
- **결정**:
  1. 모든 Worktree가 `C:\Users\LEE\notice-filter-nlp\.venv`(Python 3.11.9) 단일 인터프리터를 공유한다.
  2. `requirements.txt`에는 `+cu124` 태그가 없는 일반 `torch>=2.0.0`만 기록해 CPU 환경 `pip install -r requirements.txt` 실패를 방지하고, GPU 설치 명령은 `README.md`로 분리한다.
  3. 런타임에서는 `src.config.get_device()`가 `torch.cuda.is_available()`을 점검해 `"cuda"` 또는 `"cpu"`를 자동 선택한다.
- **결과**: Worktree별 2.4GB PyTorch 중복 설치를 제거하고, 평가자 CPU 환경에서도 동일 코드가 재현된다.

---

## ADR-02: `goodjob` INFO21 로그인 크롤링 배제 (공개 목록 수집 한정)

- **상태**: 채택됨 (Data Spike 실증 기반)
- **배경**: 경희대학교 미래인재센터(`goodjob.khu.ac.kr`)의 채용·비교과 상세 본문을 로그인 크롤링할지 여부를 6건 실측 샘플로 점검함.
- **근거 (Spike 실측)**:
  1. **이미지 포스터 비중**: 점검한 6건 중 2건(33%, 핵심 IT 공고인 NHN Cloud 포함)의 모집요강 전체가 텍스트 없는 통짜 이미지 포스터로 구성되어 HTML 본문 추출의 실익이 낮음.
  2. **보일러플레이트 및 개인정보 노출**: 모든 상세 페이지 상단에 미래인재센터 공통 안내 박스가 반복되어 임베딩 노이즈를 유발하며, 채용 담당자 개인 이메일이 원문에 포함됨.
  3. **계정 보안 및 약관 리스크**: 학교 통합계정(INFO21) 자격증명을 `.env`에 보관해 로그인 세션을 자동화하는 것은 권한 범위가 과도함.
- **결정**: `goodjob` 3개 게시판(`s4_1`, `s3_3`, `s3_2`)은 `robots.txt`가 허용하는 **비로그인 공개 목록(제목 + 카테고리/캠퍼스/기업규모/직무 메타 + 마감일)**만 수집한다. `.env`에는 INFO21 계정을 절대 저장하지 않는다.
- **후속 확장**: 이미지 포스터 기반 채용 공고 파싱은 향후 OCR/VLM 파이프라인 확장 과제로 분리한다.

---

## ADR-03: LLM-as-a-Judge(Silver) + Human Gold 3-Way 층화 분할로 데이터 누수 방지

- **상태**: 채택됨 (Phase 0)
- **배경**: 600~1,000건 공지에 대해 일관된 정답지를 확보하되, 모델 튜닝·학습 과정에 평가 데이터가 섞이는 Data Leakage와 LLM 편향 의존을 차단해야 함.
- **결정**:
  1. 데이터셋을 `(소스 그룹, 라벨)` 기준 3-way 층화 분할한다.
     - **Hold-out Test Set (150건, ~20%)**: 100% 사용자 직접 채점(**Human Gold**). 최종 평가와 오류 분석에만 사용하며 학습·임계값 튜닝에 절대 사용하지 않는다.
     - **Validation Set (100건, ~15%)**: 임계값, Rocchio $\alpha/\beta$, LogReg $C$ 튜닝 전용.
     - **Train / Feedback Pool (~65%)**: `config/profile.yaml`의 `judge_rubric`을 적용한 **LLM-as-a-Judge Silver 라벨링**으로 피드백 누적 시뮬레이션 및 분류기 학습에 사용.
     - **Summary Eval Set (20건, 별도)**: 본문 5문장 이상 공지 대상 사용자 정답 요약 작성 후 ROUGE 평가.
  2. Hold-out Test Set을 통해 LLM Judge vs Human Gold 간 **Accuracy 및 Cohen's Kappa**를 반드시 공개 검증한다.
- **결과**: 추론 파이프라인은 LLM API 없이 완전 오프라인으로 동작하며, 정량 지표의 신뢰성을 방어할 수 있다.

---

## ADR-04: Pluggable 문장 임베딩 구조 (강의 베이스라인 + 최신 한국어 임베딩 비교)

- **상태**: 채택됨 (Phase 0)
- **배경**: 강의 교안의 표준 모델은 `jhgan/ko-sroberta-multitask`(2021, 128토큰 제한)이나, 최신 한국어 임베딩 모델(`nlpai-lab/KURE-v1` 또는 `BAAI/bge-m3`)은 긴 컨텍스트(8192토큰)와 한·영 IT 직무명(`Backend`, `Unity`, `Cloud`) 이해력이 크게 향상됨.
- **결정**:
  1. `src/models/embedder.py::SentenceEmbedder`가 `model_name`을 인자로 받아 강의 베이스라인(`jhgan/ko-sroberta-multitask`)과 최신 모델(`nlpai-lab/KURE-v1` / `BAAI/bge-m3`)을 동일 인터페이스로 교체 실행하도록 설계한다.
  2. 동일 Hold-out Test Set(150건)에서 베이스라인 대비 최신 모델의 F1 변화량을 Phase 2 통제 실험(한 번에 하나씩 변경)으로 측정한다.
  3. 대용량 모델의 CPU 재현 시간 단축을 위해 `encode_cached()`로 `data/embeddings/*.npy` 디스크 캐시를 지원한다.
- **결과**: 강의 기준 준수(베이스라인 구현·비교)와 최신 NLP 기술 적용 두 요구사항을 정량 근거와 함께 동시에 충족한다.

---

## ADR-05: 공지 구조 인식 TextRank 전처리 (`PreprocessedTextRankSummarizer`)

- **상태**: 채택됨 (Phase 2 오류 분석 `NTF-24` 기반)
- **배경**: 실제 장학금·복지 공지를 Discord로 발송 검증한 결과, 원문 `TextRank`가 (1) 마침표(`.`) 없이 `❍/•/[사업개요]` 개조식으로 작성된 공지를 분리하지 못해 1,000자 이상을 통째로 반환하고, (2) 제출서류 체크리스트(`1부`, `건강보험료 납부확인서`, `성적증명서`)의 단어 반복으로 TF-IDF PageRank 중심도가 편향되는 문제가 확인됨.
- **결정**:
  1. 기존 `TextRankSummarizer` 시그니처와 Baseline 동작은 그대로 유지(`tests/test_contracts.py` 보존)하고, 서브클래스 `PreprocessedTextRankSummarizer`를 추가한다.
  2. 개조식(`■/❍/•/1)`·`[섹션]`) 경계 정규화(날짜 `2026. 11. 21.` 보존), URL/이메일/제출서류·문의처 보일러플레이트 필터링(+ 보수적 fallback), 도입부·핵심키워드(`대상/지원/일정/장학/시상`) 가중치를 적용한다.
- **결과**: 20건 Human Summary Gold 기준 **Okt ROUGE-L `0.3802 → 0.4049` (`+0.0247`)**, **Space ROUGE-L `0.3482 → 0.3648` (`+0.0166`)** 개선을 정량 입증함. (문의처 꼬리만 제거해 핵심 조건을 보존하고, 비서류 문장이 1건 이상 있으면 서류 체크리스트를 복구하지 않도록 보완함.)

---

## ADR-06: 24/7 상시 Discord 피드백 데몬 · 순수 Python 서빙 · 추론 후 모델 메모리 해제

- **상태**: 채택됨 (Phase 3 Oracle Cloud ARM64 배포)
- **배경**: 일회성 `send_batch()` 호출 후 봇 연결을 종료하면 사용자가 추후 Discord에서 `관심 👍 / 제외 👎` 버튼을 눌렀을 때 상호작용이 실패해 실시간 피드백 루프가 끊기며, 상시 클라우드 서버에서는 JVM(`KoNLPy`) 의존성과 유휴 RAM 점유를 최소화해야 함.
- **결정**:
  1. `scripts/run_alert_pipeline.py --daemon --daily-at 09:00` 통합 러너를 구축해 Discord 어댑터와 NLP 코어 간 디커플링 계약을 유지하면서 매일 지정 시각 자동 수집·중복 제외(`data/sent_notices.json`)·피드백 반영 재학습(`feedback_weight=3`)·요약·알림을 수행한다.
  2. `src/adapters/discord/bot.py`에 `on_interaction` 기반 영속 버튼 핸들러(`notice_fb:<id>:<0|1>`)를 추가해 봇 재시작 후에도 클릭 피드백이 `data/feedback.csv`에 즉시 기록되도록 한다.
  3. 실운영 추론 경로는 `KoNLPy/Java` 없는 경량 의존성(`deploy/requirements-serve.txt`)으로 분리하고, 매 주기 추론 종료 직후 `SentenceEmbedder.unload()`로 모델 가중치를 메모리에서 해제한다.
  4. Oracle Cloud(Ubuntu ARM64, `Linger=yes`)에서는 root 권한 없이 격리된 사용자 단위 `systemd --user` 서비스(`notice-filter.service`)로 운영해 기존 서버 서비스와의 간섭을 원천 차단한다.
- **결과**: 기존 서버 환경에 무영향으로 24시간 실시간 피드백 수집 + 매일 09:00 KST 자동 공지 알림이 상시 동작한다.

---

## ADR-07: 임베딩 기반 SemanticMMRTextRank 요약 및 구조화 핵심 슬롯 추출 채택

- **상태**: 채택됨 (Phase 3 요약 고도화)
- **배경**: `PreprocessedTextRankSummarizer`는 어휘 일치(`TF-IDF`) 기반 그래프를 사용하므로 조사·표현 차이가 큰 한국어 원자적 불릿 간 유사도 연결이 약하고, 동일 의미 문장 중복 선택 및 공지 제목 주제 반영 부재, 이메일+조사(`.kr로`) 잔여물 등의 한계가 존재함.
- **대안 검토**:
  - `KoBART` 등 생성 요약(Abstractive) 모델은 ~500MB 추가 메모리 적재가 필요하고 마감일·지원금액 등 숫자 환각(Hallucination) 위험이 있으며 Oracle ARM64 CPU 지연이 큼.
- **결정**:
  1. 분류에 이미 메모리에 로드된 `ko-sroberta-multitask` 임베딩을 재사용해 코사인 유사도 PageRank + 공지 `title` 유사도(`title_weight=0.25`) + **MMR**(`mmr_lambda=0.70`)로 중복 없이 핵심 문장을 선택하는 `SemanticMMRTextRankSummarizer`를 도입한다.
  2. 이메일+조사/빈 괄호/플랫폼 안내 배너/`* `·`- ` 인라인 불릿 정제 및 중복 문장 제거를 보강한다.
  3. 실사용 가독성을 위해 규칙 기반 `extract_notice_slots()`로 `일정/기한 · 대상 · 지원/혜택 · 장소` 슬롯을 추출해 Discord 임베드 `📋 핵심 정보` 필드로 선택 표기한다(미전달 시 기존 3필드 계약 유지).
- **결과**: 추가 모델 용량/RAM 0 증가, 숫자 환각 0%를 유지한 채 20건 Human Summary Gold 기준 **Okt ROUGE-L `0.3802 → 0.5040` (`+0.1238`)**, **Space ROUGE-L `0.3482 → 0.4638` (`+0.1156`)** 및 `20/20` 슬롯 커버리지를 달성함.
