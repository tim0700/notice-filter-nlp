"""Concurrent offline Silver labeling CLI using AnthropicNoticeJudge (Claude Haiku)."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.llm_judge import AnthropicNoticeJudge

logger = logging.getLogger(__name__)


def _judge_with_retry(
    judge: AnthropicNoticeJudge,
    row: dict[str, Any],
    max_attempts: int = 5,
    base_delay: float = 1.5,
) -> dict[str, Any]:
    """Judge one notice with exponential backoff for transient API/rate-limit errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return judge.judge_notice(row)
        except TypeError:
            raise
        except Exception as exc:  # pragma: no cover - network/API error path
            last_exc = exc
            if attempt >= max_attempts:
                break
            sleep_sec = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Judge retry %d/%d for %s after error: %s",
                attempt,
                max_attempts,
                row.get("id", ""),
                exc,
            )
            time.sleep(sleep_sec)
    assert last_exc is not None
    raise last_exc


def run_silver_labeling(
    input_csv: str | Path,
    output_csv: str | Path,
    limit: int | None = None,
    workers: int = 5,
    checkpoint_every: int = 20,
) -> pd.DataFrame:
    """Run resume-capable concurrent Claude Haiku Silver labeling over raw notices CSV."""
    raw_df = pd.read_csv(input_csv, encoding="utf-8")
    if limit is not None and limit > 0:
        raw_df = raw_df.head(limit).copy()

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "judge_label", "judge_reason", "judge_model"]

    judge = AnthropicNoticeJudge()
    ordered_ids = [str(val).strip() for val in raw_df["id"].tolist()]
    completed: dict[str, dict[str, Any]] = {}

    if out_path.is_file() and out_path.stat().st_size > 0:
        existing = pd.read_csv(out_path, encoding="utf-8")
        if "id" in existing.columns and "judge_label" in existing.columns:
            for row in existing.to_dict(orient="records"):
                nid = str(row.get("id", "")).strip()
                raw_label = row.get("judge_label")
                if nid and raw_label in (0, 1, "0", "1"):
                    completed[nid] = {
                        "id": nid,
                        "judge_label": int(raw_label),
                        "judge_reason": str(row.get("judge_reason", "") or "").strip(),
                        "judge_model": str(row.get("judge_model", "") or judge.model).strip(),
                    }

    pending_rows = [
        row
        for row in raw_df.to_dict(orient="records")
        if str(row.get("id", "")).strip() not in completed
    ]
    logger.info(
        "Silver labeling: total=%d, cached=%d, pending=%d (workers=%d, model=%s)",
        len(raw_df),
        len(completed),
        len(pending_rows),
        workers,
        judge.model,
    )

    lock = threading.Lock()

    def _flush() -> None:
        ordered = [completed[nid] for nid in ordered_ids if nid in completed]
        # Preserve any cached IDs outside current slice (e.g. when running with --limit)
        ordered_set = set(ordered_ids)
        extras = [v for k, v in completed.items() if k not in ordered_set]
        pd.DataFrame(ordered + extras, columns=columns).to_csv(
            out_path, index=False, encoding="utf-8"
        )

    if pending_rows:
        thread_local = threading.local()

        def _worker(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            if not hasattr(thread_local, "judge"):
                thread_local.judge = AnthropicNoticeJudge(model=judge.model, profile=judge.profile)
            nid = str(row.get("id", "")).strip()
            verdict = _judge_with_retry(thread_local.judge, row)
            return nid, verdict

        processed = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(_worker, row) for row in pending_rows]
            for fut in as_completed(futures):
                nid, verdict = fut.result()
                with lock:
                    completed[nid] = {
                        "id": nid,
                        "judge_label": int(verdict["label"]),
                        "judge_reason": verdict["reason"],
                        "judge_model": judge.model,
                    }
                    processed += 1
                    if processed % checkpoint_every == 0:
                        _flush()
                        logger.info("Checkpointed %d/%d pending rows.", processed, len(pending_rows))

        with lock:
            _flush()
    elif not out_path.is_file():
        _flush()

    result_rows = [completed[nid] for nid in ordered_ids if nid in completed]
    return pd.DataFrame(result_rows, columns=columns)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run Claude Haiku Silver labeling on raw notices.")
    parser.add_argument("--input", default="data/raw_notices.csv")
    parser.add_argument("--output", default="data/silver_labels.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    df = run_silver_labeling(
        input_csv=args.input,
        output_csv=args.output,
        limit=args.limit,
        workers=args.workers,
    )
    pos = int((df["judge_label"] == 1).sum()) if not df.empty else 0
    neg = int((df["judge_label"] == 0).sum()) if not df.empty else 0
    logger.info("Finished Silver labeling: %d rows (positive=%d, negative=%d)", len(df), pos, neg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
