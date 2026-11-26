# notice-filter-nlp

문장 임베딩과 사용자 피드백 기반 학생 맞춤형 공지 및 복지 선별 알림 모델  
*(타겟: 경희대학교 국제캠퍼스 소프트웨어융합학과 4학년 · 성남시 분당구 거주)*

---

## 1. 프로젝트 개요

- **해결하려는 문제**: 대학 학과·취업 게시판(`swcon`, `goodjob`)과 지자체 청년 복지 포털(`성남청년다해`, `온통청년 API`)에 올라오는 공지 중 비IT 직무, 타캠퍼스/타지역, 경력직 공고 등 노이즈를 걸러내고, 사용자(AI/게임/보안/SW개발·학사·성남시 청년 복지)에게 맞는 공지만 선별·요약해 전달합니다.
- **핵심 NLP 파이프라인**:
  1. **전처리 및 키워드 분석**: 정규식 정제 + `KoNLPy Okt` 형태소 분석 + 도메인 불용어 제거 + `TF-IDF` / `WordCloud`
  2. **Cold-Start 및 피드백 적응형 선별**:
     - Baseline (`Exp 0`): `jhgan/ko-sroberta-multitask` 프로필-공지 코사인 유사도 (`title@0.45`)
     - 단계적 개선: 입력 텍스트 구성·부정 감점·임계값 튜닝(`Exp 1`) → 소량 피드백 `Rocchio` 벡터 보정(`Exp 2`) → 누적 피드백 분류(`TF-IDF + LogReg` `Exp 3a` vs `ko-sroberta + LogReg` `Exp 3b`)
     - 최신 NLP 비교 (`Exp 4a~4c`): `nlpai-lab/KURE-v1` (1024d) vs `ko-sroberta` (768d) 1:1 통제 비교
  3. **구조 인식 추출 요약 및 한국어 ROUGE 평가**: `TextRank` (Raw Baseline) vs `PreprocessedTextRankSummarizer` (개조식 분리·제출서류 보일러플레이트 제거·핵심키워드 가중치) + 커스텀 한국어 `ROUGE-1 / ROUGE-L` (공백·Okt 형태소 기준)

---

## 2. 주요 정량 결과 요약 (`Hold-out Human Gold n=150`, `Summary Gold n=20`)

- **LLM-as-a-Judge (`claude-haiku-4-5-20251001`) vs Human Gold (`n=150`)**: Accuracy `0.8400` (`84.0%`) / Cohen's Kappa `0.6785`

| 태스크 | 모델 / 실험 단계 | 핵심 지표 | 비고 |
|---|---|---:|---|
| 공지 선별 (`n=150`) | `Exp 0` Baseline (`ko-sroberta title@0.45`) | F1 `0.6761` / Acc `0.6933` | Cold-Start 기준선 |
| 공지 선별 (`n=150`) | `Exp 1` (`full` + 부정 감점 + `thr=0.37`) | F1 `0.7853` / Acc `0.7667` | Baseline 대비 `+0.1092 F1` |
| 공지 선별 (`n=150`) | `Exp 2` (`Rocchio n=50`, `thr=0.51`) | F1 `0.7904` / Acc `0.7667` | 소량 피드백 보정 |
| 공지 선별 (`n=150`) | `Exp 3a` (`TF-IDF + LogReg`, 대조군) | F1 `0.7746` / Acc `0.7867` | 어휘 기반 선형 분류기 |
| 공지 선별 (`n=150`) | **`Exp 3b` (`ko-sroberta + LogReg`)** | **F1 `0.8250` / Acc `0.8133`** | **전체 최고 성능 (`+0.1489 F1`)** |
| 공지 선별 (`n=150`) | `Exp 4c` (`KURE-v1 + LogReg`) | F1 `0.8125` / Acc `0.8000` | 1024d 최신 모델 비교군 |
| 추출 요약 (`n=20`) | `TextRank` (Raw Baseline) | Okt ROUGE-1 `0.4232` / ROUGE-L `0.3802` | 공백 ROUGE-L `0.3482` |
| 추출 요약 (`n=20`) | **`PreprocessedTextRankSummarizer`** | **Okt ROUGE-1 `0.4387` / ROUGE-L `0.4049`** | **공백 ROUGE-L `0.3648` (`+0.0247` Okt)** |

상세 실험 표·학습 곡선·Jira `버그` 연동 오류 분석은 [04_Experiment_and_Error_Report.md](docs/04_Experiment_and_Error_Report.md)를 참고하세요.

---

## 3. 아키텍처 및 디렉터리 구조

```text
notice-filter-nlp/
├── config/
│   └── profile.yaml              # 사용자 프로필, 관심/제외 쿼리, Judge Rubric, 소스 설정
├── data/
│   ├── sample_notices.csv        # 9컬럼 Canonical Schema 샘플 20건 (계약 테스트용)
│   ├── raw_notices.csv           # 수집·정규화 통합 공지 979건
│   ├── silver_labels.csv         # Claude Haiku Silver 라벨 979건
│   └── splits/                   # 누수 방지 3-way 분할 (train/val/test_gold/summary_eval_gold)
├── deploy/                       # Oracle Cloud 24/7 상시 데몬 배포 가이드·경량 의존성·systemd 유닛
├── docs/
│   ├── 01_SRS_and_RTM.md
│   ├── 02_Architecture_and_Contract.md
│   ├── 03_ADR.md
│   └── 04_Experiment_and_Error_Report.md
├── notebooks/                    # 최종 제출용 파이프라인 노트북 (CPU Restart & Run All 검증 완료)
├── results/                      # Val/Test 정량 결과 JSON, 오류 사례 CSV, 시각화 PNG
├── scripts/                      # 수집, 분할, 라벨링, 임베딩 캐시, 실험, 24/7 알림 데몬 CLI
├── src/
│   ├── config.py                 # .env 자동 탐색(Worktree 공유), profile 로더, CUDA/CPU 분기
│   ├── schema.py                 # 9컬럼 NoticeRecord & 5컬럼 FeedbackRecord 계약
│   ├── collectors/               # 데이터 수집 어댑터 (swcon, goodjob, seongnam, youth_api)
│   ├── preprocess/               # 한국어 정제, Okt 토크나이저, 불용어, TF-IDF, WordCloud
│   ├── models/                   # SentenceEmbedder(+unload), Scorer 계층, Raw & Preprocessed TextRank
│   ├── eval/                     # 분류 지표(F1), 한국어 ROUGE, Judge 일치율, 층화 분할
│   └── adapters/discord/         # Discord Embed 알림 및 영속 버튼 피드백(feedback.csv) 데몬 어댑터
├── tests/                        # 오프라인 계약 및 단위 테스트 (50 tests)
├── .env.example
├── pyproject.toml
└── requirements.txt
```

---

## 4. 환경 구축 (Python 3.11 + Java 17)

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
# .env에 YOUTH_POLICY_API_KEY, DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID, ANTHROPIC_API_KEY 등 입력
.\.venv\Scripts\python.exe -m pytest -v
```

### 24/7 상시 Discord 피드백 & 일일 자동 알림 실행
```powershell
# 1회 드라이런 (신규 공지 수집 -> 분류 -> 전처리 요약 -> 콘솔 확인)
.\.venv\Scripts\python.exe scripts\run_alert_pipeline.py --once

# 24/7 상시 봇 데몬 (매일 09:00 KST 자동 수집·알림 + 상시 버튼 피드백 수집 및 재학습)
.\.venv\Scripts\python.exe scripts\run_alert_pipeline.py --daemon --daily-at 09:00
```
Oracle Cloud Ubuntu 서버 상시 배포 절차는 [deploy/ORACLE_DEPLOY.md](deploy/ORACLE_DEPLOY.md)를 참고하세요.
