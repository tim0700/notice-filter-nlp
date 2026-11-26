"""Discord adapter contract decoupled from NLP core (writes only to data/feedback.csv)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.config import FEEDBACK_COLUMNS
from src.schema import FeedbackRecord, NoticeRecord


def append_feedback(feedback_csv: str | Path, record: FeedbackRecord) -> None:
    """Append a single validated FeedbackRecord row to feedback_csv, creating header if needed."""
    path = Path(feedback_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = record.to_dict()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEEDBACK_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def format_notice_message(
    notice: NoticeRecord,
    score: float,
    summary: str = "",
) -> dict[str, Any]:
    """Build a Discord embed payload dictionary for a filtered notice."""
    description = summary.strip() if summary.strip() else notice.title
    return {
        "title": notice.title,
        "url": notice.url,
        "description": description,
        "fields": [
            {"name": "출처", "value": notice.source, "inline": True},
            {"name": "적합도", "value": f"{score:.2f}", "inline": True},
            {"name": "마감일", "value": notice.deadline or "상시/미정", "inline": True},
        ],
    }


class NoticeDiscordClient:
    """Discord bot adapter stub for sending alerts and collecting Good/Skip feedback."""

    def __init__(
        self,
        channel_id: int | str,
        feedback_csv: str | Path = "data/feedback.csv",
    ) -> None:
        self.channel_id = int(channel_id)
        self.feedback_csv = Path(feedback_csv)

    async def send_notice(
        self,
        notice: NoticeRecord,
        score: float,
        summary: str = "",
    ) -> None:
        """Send formatted notice embed with feedback controls to Discord."""
        raise NotImplementedError("Implemented in Phase 1 (wt-discord)")
