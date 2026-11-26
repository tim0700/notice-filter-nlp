# notice-filter-nlp

문장 임베딩과 사용자 피드백 기반 학생 맞춤형 공지 및 복지 선별 알림 모델  
*(타겟: 경희대학교 국제캠퍼스 소프트웨어융합학과 4학년 · 성남시 분당구 거주)*

---

## 1. 프로젝트 개요

- **해결하려는 문제**: 대학 학과·취업 게시판과 지자체 청년 복지 포털에 올라오는 공지 중 비IT 직무, 타캠퍼스/타지역, 경력직 공고 등 노이즈를 걸러내고, 사용자(AI/게임/보안/SW개발·학사·성남시 청년 복지)에게 맞는 공지만 선별·요약해 전달합니다.
- **핵심 NLP 파이프라인**:
  1. **전처리 및 키워드 분석**: 정규식 정제 + `KoNLPy Okt` 형태소 분석 + 도메인 불용어 제거 + `TF-IDF` / `WordCloud`
  2. **Cold-Start 및 피드백 적응형 선별**:
     - Baseline: `jhgan/ko-sroberta-multitask` 프로필-공지 코사인 유사도
     - 단계적 개선: 입력 텍스트 구성·임계값 튜닝 → 소량 피드백 `Rocchio` 벡터 보정 → 누적 피드백 분류(`SBERT + LogReg` vs `TF-IDF + LogReg`)
     - 최신 NLP 비교: 최신 한국어 임베딩(`nlpai-lab/KURE-v1` / `BAAI/bge-m3`) vs `ko-sroberta` 정량 비교
  3. **추출 요약 및 정량 평가**: `TextRank` (`networkx` + `TF-IDF`) 2~3줄 요약 + 커스텀 한국어 `ROUGE-1 / ROUGE-L` (공백·Okt 형태소 기준)

---

## 2. 아키텍처 및 디렉터리 구조

```text
notice-filter-nlp/
├── config/
│   └── profile.yaml          # 사용자 프로필, 관심/제외 쿼리, Judge Rubric, 소스 설정
├── data/
│   └── sample_notices.csv    # 9컬럼 Canonical Schema 샘플 20건 (계약 테스트용)
├── docs/
│   ├── 01_SRS_and_RTM.md
│   ├── 02_Architecture_and_Contract.md
│   ├── 03_ADR.md
│   └── 04_Experiment_and_Error_Report.md
├── src/
│   ├── config.py             # .env 자동 탐색(Worktree 공유), profile 로더, CUDA/CPU 분기
│   ├── schema.py             # 9컬럼 NoticeRecord & 5컬럼 FeedbackRecord 계약
│   ├── collectors/           # 데이터 수집 어댑터 (swcon, goodjob, seongnam, youth_api)
│   ├── preprocess/           # 한국어 정제, Okt 토크나이저, 불용어, TF-IDF
│   ├── models/               # SentenceEmbedder, Scorer 계층, TextRankSummarizer
│   ├── eval/                 # 분류 지표(F1), 한국어 ROUGE, Judge 일치율, 층화 분할
│   └── adapters/discord/     # Discord Embed 알림 및 피드백(feedback.csv) 수집 어댑터
├── tests/
│   └── test_contracts.py     # Phase 0 스키마·인터페이스 계약 테스트
├── .env.example
├── pyproject.toml
└── requirements.txt
```

---

## 3. 환경 구축 (Python 3.11 + Java 17)

### 기본 설치 (CPU 재현 환경)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### GPU 가속 (선택 — 로컬 RTX 등 CUDA 환경)
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
- 코드 실행 시 `src.config.get_device()`가 `torch.cuda.is_available()`를 확인해 `"cuda"` 또는 `"cpu"`를 자동 선택합니다.
- `KoNLPy Okt` 구동을 위해 JDK 17(Temurin 등)이 필요합니다.

### 환경변수 및 테스트 실행
```powershell
Copy-Item .env.example .env
# .env에 YOUTH_API_KEY, DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID, GEMINI_API_KEY 등을 입력
.\.venv\Scripts\pytest.exe -v
```
