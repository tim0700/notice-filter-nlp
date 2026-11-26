"""Decoupled Discord preview utility comparing Raw vs Preprocessed TextRank summaries."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.discord import NoticeDiscordClient
from src.config import load_env, load_profile
from src.models.embedder import SentenceEmbedder
from src.models.summarizer import (
    PreprocessedTextRankSummarizer,
    SemanticMMRTextRankSummarizer,
    TextRankSummarizer,
    extract_notice_slots,
)
from src.schema import NoticeRecord, normalize_notices_df


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for notice summary preview and Discord dispatch."""
    parser = argparse.ArgumentParser(
        description="Preview Raw vs Preprocessed TextRank summaries and optionally send to Discord.",
    )
    parser.add_argument(
        "--input",
        default="data/splits/train.csv",
        help="Input notice CSV path (default: data/splits/train.csv).",
    )
    parser.add_argument(
        "--notice-id",
        default=None,
        help="Optional specific notice ID to preview.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of multi-sentence notices to preview (default: 3).",
    )
    parser.add_argument(
        "--min-content-chars",
        type=int,
        default=250,
        help="Minimum content length filter when --notice-id is omitted (default: 250).",
    )
    parser.add_argument(
        "--summarizer",
        choices=["raw", "preprocessed", "semantic_mmr"],
        default="semantic_mmr",
        help="Which summarizer output to attach to Discord embeds (default: semantic_mmr).",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=0.90,
        help="Preview relevance score shown on Discord embed (default: 0.90).",
    )
    parser.add_argument(
        "--channel-id",
        default=None,
        help="Override DISCORD_CHANNEL_ID from environment.",
    )
    parser.add_argument(
        "--feedback-csv",
        default="data/feedback.csv",
        help="Path to feedback CSV file.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send to Discord live channel instead of dry-run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Raw vs Preprocessed TextRank comparison and optional Discord dispatch."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    load_env()

    df = pd.read_csv(Path(args.input), encoding="utf-8")
    normalized = normalize_notices_df(df)

    if args.notice_id:
        subset = normalized[normalized["id"] == args.notice_id]
        if subset.empty:
            raise ValueError(f"Notice ID not found in {args.input}: {args.notice_id}")
    else:
        subset = normalized[normalized["content"].str.len() >= args.min_content_chars]
        if subset.empty:
            subset = normalized[normalized["content"].str.len() > 0]
        subset = subset.head(max(0, args.limit))

    profile = load_profile()
    embedder = SentenceEmbedder(model_name=str(profile["models"]["baseline_embedding"]))
    raw_summarizer = TextRankSummarizer(top_k=3, min_sentences=5)
    prep_summarizer = PreprocessedTextRankSummarizer(top_k=3, min_sentences=5)
    sem_summarizer = SemanticMMRTextRankSummarizer(embedder=embedder, top_k=3, min_sentences=5)

    batch = []
    try:
        for _, row in subset.iterrows():
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
            raw_sum = raw_summarizer.summarize(notice.content)
            prep_sum = prep_summarizer.summarize(notice.content)
            sem_sum = sem_summarizer.summarize(notice.content, title=notice.title)
            slots = extract_notice_slots(notice.content, deadline=notice.deadline)
            if args.summarizer == "semantic_mmr":
                chosen_sum = sem_sum
            elif args.summarizer == "preprocessed":
                chosen_sum = prep_sum
            else:
                chosen_sum = raw_sum

            print(f"=== [{notice.source}] {notice.title} ({notice.id}) ===")
            print(f"[RAW TextRank]          {raw_sum}")
            print(f"[PREPROCESSED TextRank] {prep_sum}")
            print(f"[SEMANTIC+MMR TextRank] {sem_sum}")
            print(f"[KEY SLOTS]             {slots}")
            print()
            batch.append((notice, float(args.score), chosen_sum, slots or None))
    finally:
        embedder.unload()

    dry_run = not args.live
    raw_channel = (
        args.channel_id
        if args.channel_id is not None
        else os.getenv("DISCORD_CHANNEL_ID", "")
    ).strip()
    channel_id = int(raw_channel) if raw_channel else 0
    if not dry_run and channel_id == 0:
        raise ValueError("DISCORD_CHANNEL_ID is required when --live is set.")
    token = None if dry_run else os.getenv("DISCORD_BOT_TOKEN", "").strip()

    client = NoticeDiscordClient(
        channel_id=channel_id,
        feedback_csv=args.feedback_csv,
        token=token,
        dry_run=dry_run,
    )
    sent = asyncio.run(client.send_batch(batch))
    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"[{mode}] Dispatched {len(sent)} preview notice(s) using '{args.summarizer}' summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
