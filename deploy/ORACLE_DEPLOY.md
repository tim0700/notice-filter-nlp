# Oracle Cloud 24/7 상시 알림 봇 배포 가이드

`scripts/run_alert_pipeline.py --daemon`은 다음 두 가지 역할을 한 프로세스에서 동시에 수행합니다.
1. **상시 Discord 봇**: 사용자가 언제든 공지 카드의 `관심 👍 / 제외 👎` 버튼을 누르면 봇 재시작 후에도 `on_interaction`이 즉시 `data/feedback.csv`에 저장
2. **매일 지정 시각(`09:00 KST`) 자동 파이프라인**: 최신 공지 수집 → 기발송 중복 제외(`data/sent_notices.json`) → `train.csv + feedback.csv`(가중치 `3x`) 재학습 → `ko-sroberta + LogReg` 점수 산출 → `PreprocessedTextRankSummarizer` 요약 → Discord 발송 → 모델 메모리 해제(`unload()`)

> **참고**: 실운영 추론 경로는 `KoNLPy/Java`를 사용하지 않는 순수 Python이므로 JDK 설치가 필요 없습니다.

---

## 1. 서버 환경 준비 (Oracle Cloud Ubuntu 22.04/24.04 `aarch64` / `x86_64`)

```bash
# (선택) 1GB Micro VM 사용 시 2GB Swap 권장 (ARM 24GB는 생략 가능)
# sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

git clone https://github.com/tim0700/notice-filter-nlp.git ~/ai/notice-filter-nlp
cd ~/ai/notice-filter-nlp

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# ARM64(aarch64) Linux에서는 표준 PyPI torch가 공식 ARM64 CPU 휠을 자동 설치합니다.
# x86_64 CPU 전용 서버에서는 아래 CPU 전용 인덱스 옵션을 먼저 실행하세요:
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r deploy/requirements-serve.txt
```

## 2. `.env` 설정 및 기존 공지 시딩(중복 발송 방지)

```bash
cp .env.example .env
chmod 600 .env
# .env에 DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID, YOUTH_*_API_KEY 입력 (Windows 복사 시 CRLF 제거: sed -i 's/\r$//' .env)

# 기존 수집된 979건을 기발송 처리해 첫 실행 시 과거 공지가 쏟아지지 않도록 방지
.venv/bin/python scripts/run_alert_pipeline.py --seed-seen-from data/raw_notices.csv

# 1회 드라이런 검증 (실제 Discord 발송 없음)
.venv/bin/python scripts/run_alert_pipeline.py --once --input-csv data/sample_notices.csv --max-alerts-per-run 3
```

## 3. 격리된 사용자 단위 `systemd --user` 데몬 등록 (권장, `sudo` 불필요)

기존 서버 서비스·root 설정을 건드리지 않고 `ubuntu` 계정 하위에 완전히 격리해 상시 운영할 수 있습니다 (`loginctl show-user $USER -p Linger`가 `Linger=yes` 확인).

```bash
mkdir -p ~/.config/systemd/user
cat << 'EOF' > ~/.config/systemd/user/notice-filter.service
[Unit]
Description=Notice Filter NLP 24/7 Discord Bot & Daily Notice Alert Daemon
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/ubuntu/ai/notice-filter-nlp
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/ubuntu/ai/notice-filter-nlp/.venv/bin/python scripts/run_alert_pipeline.py --daemon --daily-at 09:00 --timezone Asia/Seoul
Restart=always
RestartSec=15

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now notice-filter
systemctl --user status notice-filter --no-pager
journalctl --user -u notice-filter -n 30 --no-pager
```
