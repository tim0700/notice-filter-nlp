# Oracle Cloud 24/7 상시 알림 봇 배포 가이드

`scripts/run_alert_pipeline.py --daemon`은 다음 두 가지 역할을 한 프로세스에서 동시에 수행합니다.
1. **상시 Discord 봇**: 사용자가 언제든 공지 카드의 `관심 👍 / 제외 👎` 버튼을 누르면 즉시 `data/feedback.csv`에 저장
2. **매일 지정 시각(`09:00 KST`) 자동 파이프라인**: 최신 공지 수집 → 기발송 중복 제외(`data/sent_notices.json`) → `train.csv + feedback.csv` 재학습 → `ko-sroberta + LogReg` 점수 산출 → `PreprocessedTextRankSummarizer` 요약 → Discord 발송 → 모델 메모리 해제(`unload()`)

> **참고**: 실운영 추론 경로는 `KoNLPy/Java`를 사용하지 않는 순수 Python이므로 JDK 설치가 필요 없습니다.

---

## 1. 서버 환경 준비 (Ubuntu 22.04/24.04 기준)

```bash
# 1GB Micro VM 사용 시 임베딩 추론 안정화를 위해 2GB Swap 권장 (ARM 24GB는 생략 가능)
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

sudo apt update && sudo apt install -y python3 python3-venv git
git clone https://github.com/tim0700/notice-filter-nlp.git
cd notice-filter-nlp

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r deploy/requirements-serve.txt
```

## 2. `.env` 설정 및 기존 공지 시딩(중복 발송 방지)

```bash
cp .env.example .env
nano .env  # DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID, YOUTH_*_API_KEY 입력

# 기존 수집된 979건을 기발송 처리해 첫 실행 시 과거 공지가 쏟아지지 않도록 방지
.venv/bin/python scripts/run_alert_pipeline.py --seed-seen-from data/raw_notices.csv

# 1회 드라이런 검증 (실제 Discord 발송 없음)
.venv/bin/python scripts/run_alert_pipeline.py --once --limit-per-source 5
```

## 3. `systemd` 24/7 데몬 등록 및 실행

```bash
# deploy/notice-filter.service 내 경로(/home/ubuntu/notice-filter-nlp)가 다르면 수정
sudo cp deploy/notice-filter.service /etc/systemd/system/notice-filter.service
sudo systemctl daemon-reload
sudo systemctl enable --now notice-filter

# 상태 및 실시간 로그 확인
sudo systemctl status notice-filter
journalctl -u notice-filter -f
```
