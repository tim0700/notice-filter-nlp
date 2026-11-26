"""CLI entrypoint for sending formatted notices to Discord (live or dry-run)."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.adapters.discord.bot import NoticeBatchItem, NoticeDiscordClient
from src.config import load_env
from src.schema import NoticeRecord, normalize_notices_df


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the Discord notice sender CLI."""
    parser = argparse.ArgumentParser(
        description="Send canonical notices from CSV to a Discord channel.",
    )
    parser.add_argument(
        "--input",
        default="data/sample_notices.csv",
        help="Path to input CSV containing canonical notice columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of notices to send (default: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Format and record payloads locally without calling Discord API.",
    )
    parser.add_argument(
        "--channel-id",
        default=None,
        help="Override DISCORD_CHANNEL_ID from environment.",
    )
    parser.add_argument(
        "--feedback-csv",
        default="data/feedback.csv",
        help="Path to feedback CSV file for interactive button persistence.",
    )
    return parser


def load_notice_batch(input_path: str | Path, limit: int = 5) -> list[NoticeBatchItem]:
    """Load up to limit notices from CSV into (NoticeRecord, score, summary) items."""
    if limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")
    if limit == 0:
        return []

    raw_df = pd.read_csv(Path(input_path), encoding="utf-8").head(limit).reset_index(drop=True)
    normalized = normalize_notices_df(raw_df)

    has_score = "score" in raw_df.columns
    has_summary = "summary" in raw_df.columns

    batch: list[NoticeBatchItem] = []
    for idx, row in normalized.iterrows():
        notice = NoticeRecord(
            id=row["id"],
            source=row["source"],
            category=row["category"],
            title=row["title"],
            meta_tags=row["meta_tags"],
            posted_at=row["posted_at"],
            url=row["url"],
            content=row["content"],
            deadline=row["deadline"],
        )
        score = 1.0
        if has_score and pd.notna(raw_df.at[idx, "score"]):
            score = float(raw_df.at[idx, "score"])

        summary = ""
        if has_summary and pd.notna(raw_df.at[idx, "summary"]):
            summary = str(raw_df.at[idx, "summary"])

        batch.append((notice, score, summary))

    return batch


def _resolve_channel_id(cli_channel_id: str | None, dry_run: bool) -> int:
    """Resolve channel ID from CLI argument or DISCORD_CHANNEL_ID env variable."""
    raw = (
        cli_channel_id
        if cli_channel_id is not None
        else os.getenv("DISCORD_CHANNEL_ID", "")
    ).strip()
    if not raw:
        if dry_run:
            return 0
        raise ValueError("DISCORD_CHANNEL_ID is required when --dry-run is omitted.")
    return int(raw)


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Execute the Discord CLI workflow and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    load_env()
    channel_id = _resolve_channel_id(args.channel_id, dry_run=args.dry_run)
    token = None if args.dry_run else os.getenv("DISCORD_BOT_TOKEN", "").strip()

    batch = load_notice_batch(args.input, limit=args.limit)
    client = NoticeDiscordClient(
        channel_id=channel_id,
        feedback_csv=args.feedback_csv,
        token=token,
        dry_run=args.dry_run,
    )
    sent = asyncio.run(client.send_batch(batch))

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] Dispatched {len(sent)} notice(s) to channel {channel_id}.")
    return 0


def main() -> int:
    """CLI entrypoint for `python -m src.adapters.discord.cli`."""
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
