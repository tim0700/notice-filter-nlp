"""CLI and orchestration runner for collecting, deduplicating, and validating notices."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.collectors.base import COLLECTOR_REGISTRY, BaseCollector
from src.config import CANONICAL_COLUMNS
from src.schema import NoticeRecord, normalize_notices_df, validate_notices_df

logger = logging.getLogger(__name__)


def collect_all(
    sources: Sequence[str] | None = None,
    limit_per_source: int = 150,
    output_path: str | Path | None = None,
    delay_seconds: float = 0.5,
    registry: Mapping[str, type[BaseCollector]] | None = None,
) -> pd.DataFrame:
    """Run registered collectors with per-source resilience, deduplicate, validate, and optionally save CSV."""
    active_registry = registry if registry is not None else COLLECTOR_REGISTRY
    target_sources = list(sources) if sources is not None else list(active_registry.keys())

    all_records: list[NoticeRecord | dict[str, str]] = []
    for index, source_name in enumerate(target_sources):
        collector_cls = active_registry.get(source_name)
        if collector_cls is None:
            logger.warning("Unknown collector source '%s'; skipping.", source_name)
            continue

        if index > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)

        try:
            collector = collector_cls()
            fetched = collector.fetch(limit=limit_per_source)
            logger.info("Collected %d notices from '%s'.", len(fetched), source_name)
            all_records.extend(fetched)
        except Exception as exc:
            logger.exception("Collector '%s' failed; continuing remaining sources: %s", source_name, exc)

    seen_ids: set[str] = set()
    deduped_rows: list[dict[str, str]] = []
    for item in all_records:
        row = item.to_dict() if isinstance(item, NoticeRecord) else dict(item)
        notice_id = str(row.get("id", "")).strip()
        if not notice_id or notice_id in seen_ids:
            continue
        seen_ids.add(notice_id)
        deduped_rows.append(row)

    raw_df = pd.DataFrame(deduped_rows, columns=CANONICAL_COLUMNS)
    normalized_df = normalize_notices_df(raw_df)
    validate_notices_df(normalized_df)

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        normalized_df.to_csv(out_file, index=False, encoding="utf-8")
        logger.info("Wrote %d notices to %s", len(normalized_df), out_file)

    return normalized_df


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for the collector runner."""
    parser = argparse.ArgumentParser(description="Collect public notices into canonical CSV.")
    parser.add_argument(
        "--output",
        default="data/raw_notices.csv",
        help="Output CSV path (UTF-8).",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=150,
        help="Maximum number of notices to collect per source.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional subset of source names to collect.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Polite delay in seconds between collectors.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m src.collectors.runner`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    df = collect_all(
        sources=args.sources,
        limit_per_source=args.limit_per_source,
        output_path=args.output,
        delay_seconds=args.delay,
    )
    logger.info("Collection complete: %d unique validated notices.", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
