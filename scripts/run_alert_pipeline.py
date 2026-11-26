"""End-to-end periodic notice crawler, classifier, summarizer, and Discord alert runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.discord import NoticeDiscordClient, load_feedback_df
from src.collectors import collect_all
from src.config import CANONICAL_COLUMNS, load_env, load_profile
from src.models import (
    EmbeddingLogRegScorer,
    SemanticMMRTextRankSummarizer,
    SentenceEmbedder,
    extract_notice_slots,
    extract_texts_from_df,
)
from src.schema import NoticeRecord, normalize_notices_df

logger = logging.getLogger(__name__)


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def load_sent_ids(state_path: str | Path) -> dict[str, str]:
    """Load persisted `{notice_id: iso_timestamp}` deduplication map."""
    path = Path(state_path)
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read state file %s (%s); starting empty.", path, exc)
    return {}


def save_sent_ids(state_path: str | Path, sent_map: dict[str, str]) -> None:
    """Persist `{notice_id: iso_timestamp}` deduplication state as UTF-8 JSON."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sent_map, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_sent_ids(state_path: str | Path, notice_ids: Sequence[str]) -> dict[str, str]:
    """Append newly sent notice IDs into state_path and return the updated mapping."""
    sent_map = load_sent_ids(state_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    for nid in notice_ids:
        key = str(nid).strip()
        if key and key not in sent_map:
            sent_map[key] = now_iso
    save_sent_ids(state_path, sent_map)
    return sent_map


def seed_sent_ids_from_csv(state_path: str | Path, csv_path: str | Path) -> int:
    """Mark existing notices from csv_path as already seen so initial daemon only alerts new posts."""
    src = Path(csv_path)
    if not src.is_file():
        raise FileNotFoundError(f"Seed CSV not found: {src}")
    df = pd.read_csv(src, encoding="utf-8")
    before = len(load_sent_ids(state_path))
    ids = [_safe_str(v) for v in df.get("id", pd.Series(dtype=str)).tolist() if _safe_str(v)]
    updated = mark_sent_ids(state_path, ids)
    return len(updated) - before


def upsert_notices_store(store_path: str | Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge newly collected notices into a local store CSV for later feedback re-training lookup."""
    path = Path(store_path)
    incoming = normalize_notices_df(new_df)
    if path.is_file() and path.stat().st_size > 0:
        existing = normalize_notices_df(pd.read_csv(path, encoding="utf-8"))
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    if not combined.empty:
        combined = combined.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=CANONICAL_COLUMNS)

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    return combined


def _load_locked_model_config(config_path: Path) -> dict[str, Any]:
    """Load Val-tuned Exp 3b settings with safe defaults."""
    profile = load_profile()
    defaults: dict[str, Any] = {
        "model_name": str(profile["models"]["baseline_embedding"]),
        "text_mode": "full",
        "c_value": 2.0,
        "threshold": 0.35,
    }
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            defaults["model_name"] = str(cfg.get("baseline_model", defaults["model_name"]))
            defaults["text_mode"] = str(cfg.get("exp1", {}).get("text_mode", defaults["text_mode"]))
            defaults["c_value"] = float(cfg.get("exp3b", {}).get("c_value", defaults["c_value"]))
            defaults["threshold"] = float(cfg.get("exp3b", {}).get("threshold", defaults["threshold"]))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return defaults


def build_training_dataset(
    base_train_csv: str | Path,
    raw_notices_csv: str | Path,
    latest_store_csv: str | Path,
    feedback_csv: str | Path,
    *,
    text_mode: str = "full",
    feedback_weight: int = 3,
) -> tuple[list[str], list[int]]:
    """Combine Train Silver labels with live Discord user feedback for adaptive classifier training."""
    text_by_id: dict[str, str] = {}
    label_by_id: dict[str, int] = {}

    base_path = Path(base_train_csv)
    if base_path.is_file() and base_path.stat().st_size > 0:
        tdf = pd.read_csv(base_path, encoding="utf-8")
        texts = extract_texts_from_df(tdf, text_mode=text_mode)
        for nid, txt, lbl in zip(tdf["id"], texts, tdf["judge_label"]):
            key = _safe_str(nid)
            if key and pd.notna(lbl):
                text_by_id[key] = txt
                label_by_id[key] = int(lbl)

    for lookup_path in (Path(raw_notices_csv), Path(latest_store_csv)):
        if lookup_path.is_file() and lookup_path.stat().st_size > 0:
            ldf = pd.read_csv(lookup_path, encoding="utf-8")
            texts = extract_texts_from_df(ldf, text_mode=text_mode)
            for nid, txt in zip(ldf["id"], texts):
                key = _safe_str(nid)
                if key:
                    text_by_id[key] = txt

    fb_df = load_feedback_df(feedback_csv)
    fb_samples: list[tuple[str, int]] = []
    if not fb_df.empty:
        latest_fb = fb_df.drop_duplicates(subset=["notice_id"], keep="last")
        for _, row in latest_fb.iterrows():
            nid = _safe_str(row.get("notice_id"))
            if nid in text_by_id:
                lbl = int(row["label"])
                label_by_id[nid] = lbl
                fb_samples.append((text_by_id[nid], lbl))

    train_texts = [text_by_id[nid] for nid in label_by_id]
    train_labels = [label_by_id[nid] for nid in label_by_id]

    extra_repeats = max(0, int(feedback_weight) - 1)
    for txt, lbl in fb_samples:
        for _ in range(extra_repeats):
            train_texts.append(txt)
            train_labels.append(lbl)

    return train_texts, train_labels


def _row_to_notice(row: pd.Series) -> NoticeRecord:
    return NoticeRecord(
        id=_safe_str(row.get("id")),
        source=_safe_str(row.get("source")),
        category=_safe_str(row.get("category")) or "일반",
        title=_safe_str(row.get("title")),
        meta_tags=_safe_str(row.get("meta_tags")),
        posted_at=_safe_str(row.get("posted_at")),
        url=_safe_str(row.get("url")),
        content=_safe_str(row.get("content")),
        deadline=_safe_str(row.get("deadline")),
    )


def run_alert_cycle(
    *,
    sources: Sequence[str] | None = None,
    limit_per_source: int = 20,
    max_alerts_per_run: int = 5,
    threshold: float | None = None,
    state_file: str | Path = _REPO_ROOT / "data" / "sent_notices.json",
    feedback_csv: str | Path = _REPO_ROOT / "data" / "feedback.csv",
    store_csv: str | Path = _REPO_ROOT / "data" / "latest_notices_store.csv",
    base_train_csv: str | Path = _REPO_ROOT / "data" / "splits" / "train.csv",
    raw_notices_csv: str | Path = _REPO_ROOT / "data" / "raw_notices.csv",
    config_path: str | Path = _REPO_ROOT / "results" / "val_tuned_config.json",
    input_csv: str | Path | None = None,
    embedder: Any | None = None,
    collect_fn: Callable[..., pd.DataFrame] | None = None,
) -> list[tuple[NoticeRecord, float, str, dict[str, str] | None]]:
    """Collect fresh notices, score unseen items, summarize positives, and free model RAM."""
    if input_csv is not None:
        fresh_df = normalize_notices_df(pd.read_csv(Path(input_csv), encoding="utf-8"))
    else:
        fetcher = collect_fn if collect_fn is not None else collect_all
        fresh_df = fetcher(sources=sources, limit_per_source=limit_per_source, output_path=None)

    if fresh_df.empty:
        logger.info("No notices returned from collectors.")
        return []

    upsert_notices_store(store_csv, fresh_df)
    sent_ids = load_sent_ids(state_file)
    unseen_df = fresh_df[~fresh_df["id"].astype(str).isin(sent_ids)].reset_index(drop=True)
    if unseen_df.empty:
        logger.info("All %d collected notices were already seen.", len(fresh_df))
        return []

    cfg = _load_locked_model_config(Path(config_path))
    active_threshold = float(cfg["threshold"] if threshold is None else threshold)
    text_mode = str(cfg["text_mode"])

    active_embedder = (
        embedder
        if embedder is not None
        else SentenceEmbedder(model_name=str(cfg["model_name"]))
    )

    try:
        train_texts, train_labels = build_training_dataset(
            base_train_csv=base_train_csv,
            raw_notices_csv=raw_notices_csv,
            latest_store_csv=store_csv,
            feedback_csv=feedback_csv,
            text_mode=text_mode,
        )
        if len(set(train_labels)) < 2:
            raise RuntimeError("Need both positive and negative labeled samples to fit alert classifier.")

        scorer = EmbeddingLogRegScorer(
            embedder=active_embedder,
            c_value=float(cfg["c_value"]),
            text_mode=text_mode,
        ).fit(train_texts, train_labels)

        scores = scorer.score(unseen_df)
        scored_df = unseen_df.copy()
        scored_df["_score"] = scores
        positives = scored_df[scored_df["_score"] >= active_threshold].sort_values(
            "_score", ascending=False
        )
        if max_alerts_per_run > 0:
            positives = positives.head(max_alerts_per_run)

        summarizer = SemanticMMRTextRankSummarizer(
            embedder=active_embedder,
            top_k=3,
            min_sentences=5,
        )
        batch: list[tuple[NoticeRecord, float, str, dict[str, str] | None]] = []
        for _, row in positives.iterrows():
            notice = _row_to_notice(row)
            summary = summarizer.summarize(notice.content, title=notice.title) if notice.content else ""
            key_info = (
                extract_notice_slots(notice.content, deadline=notice.deadline)
                if notice.content
                else {}
            )
            batch.append((notice, float(row["_score"]), summary, key_info or None))
        return batch
    finally:
        if hasattr(active_embedder, "unload"):
            active_embedder.unload()


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for single-shot and 24/7 daemon alert modes."""
    parser = argparse.ArgumentParser(
        description="Collect latest notices, filter with adaptive NLP model, summarize, and send to Discord.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--once", action="store_true", help="Run a single crawl-score-alert cycle (default).")
    mode_group.add_argument("--daemon", action="store_true", help="Run 24/7 Discord bot with scheduled daily crawls and persistent feedback collection.")

    parser.add_argument("--live", action="store_true", help="Send alerts to Discord live channel in --once mode (default is dry-run).")
    parser.add_argument("--daily-at", default="09:00", help="Daily HH:MM schedule in --daemon mode (default: 09:00).")
    parser.add_argument("--interval-minutes", type=int, default=None, help="Override --daily-at with fixed interval in minutes.")
    parser.add_argument("--timezone", default="Asia/Seoul", help="Timezone for --daily-at (default: Asia/Seoul).")
    parser.add_argument("--run-on-startup", action="store_true", help="Run one crawl/alert cycle immediately when --daemon boots.")
    parser.add_argument("--limit-per-source", type=int, default=20, help="Max recent notices to fetch per source (default: 20).")
    parser.add_argument("--max-alerts-per-run", type=int, default=5, help="Max Discord alerts per cycle (default: 5).")
    parser.add_argument("--threshold", type=float, default=None, help="Override classification threshold (default: locked Val threshold 0.35).")
    parser.add_argument("--sources", nargs="*", default=None, help="Optional subset of collector sources.")
    parser.add_argument("--input-csv", default=None, help="Optional local CSV input instead of live web collection.")
    parser.add_argument("--seed-seen-from", default=None, help="Mark IDs from this CSV as already sent and exit without dispatching.")
    parser.add_argument("--state-file", default="data/sent_notices.json", help="Sent notice deduplication JSON path.")
    parser.add_argument("--feedback-csv", default="data/feedback.csv", help="Interactive feedback CSV path.")
    parser.add_argument("--store-csv", default="data/latest_notices_store.csv", help="Collected notice store CSV path.")
    parser.add_argument("--channel-id", default=None, help="Override DISCORD_CHANNEL_ID from environment.")
    return parser


def _resolve_channel_id(cli_channel_id: str | None, dry_run: bool) -> int:
    raw = (cli_channel_id if cli_channel_id is not None else os.getenv("DISCORD_CHANNEL_ID", "")).strip()
    if not raw:
        if dry_run:
            return 0
        raise ValueError("DISCORD_CHANNEL_ID is required for live Discord dispatch/daemon.")
    return int(raw)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for single-run and 24/7 notice alerting pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    load_env()

    if args.seed_seen_from:
        added = seed_sent_ids_from_csv(args.state_file, args.seed_seen_from)
        print(f"Seeded {added} notice ID(s) into {args.state_file} from {args.seed_seen_from}.")
        return 0

    def _cycle() -> list[tuple[NoticeRecord, float, str, dict[str, str] | None]]:
        return run_alert_cycle(
            sources=args.sources,
            limit_per_source=args.limit_per_source,
            max_alerts_per_run=args.max_alerts_per_run,
            threshold=args.threshold,
            state_file=args.state_file,
            feedback_csv=args.feedback_csv,
            store_csv=args.store_csv,
            input_csv=args.input_csv,
        )

    if args.daemon:
        channel_id = _resolve_channel_id(args.channel_id, dry_run=False)
        client = NoticeDiscordClient(
            channel_id=channel_id,
            feedback_csv=args.feedback_csv,
            dry_run=False,
        )
        asyncio.run(
            client.run_daemon(
                fetch_batch_fn=_cycle,
                daily_at=args.daily_at,
                interval_minutes=args.interval_minutes,
                timezone_name=args.timezone,
                run_on_startup=args.run_on_startup,
                on_dispatched=lambda ids: mark_sent_ids(args.state_file, ids),
            )
        )
        return 0

    # Single-shot (--once)
    dry_run = not args.live
    channel_id = _resolve_channel_id(args.channel_id, dry_run=dry_run)
    batch = _cycle()
    if not batch:
        print("[ONCE] No new relevant notices to alert.")
        return 0

    client = NoticeDiscordClient(
        channel_id=channel_id,
        feedback_csv=args.feedback_csv,
        dry_run=dry_run,
    )
    asyncio.run(
        client.send_batch(
            batch,
            on_sent=(None if dry_run else lambda nid: mark_sent_ids(args.state_file, [nid])),
        )
    )

    mode_lbl = "LIVE" if args.live else "DRY-RUN"
    print(f"[{mode_lbl}] Dispatched {len(batch)} notice(s):")
    for notice, score, summary, _key_info in batch:
        preview = (summary or notice.title)[:90].replace("\n", " ")
        print(f"  - [{notice.source}] ({score:.2%}) {notice.title} | {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
