"""Discord adapter client, interactive feedback view, and feedback CSV persistence."""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord
import pandas as pd

from src.adapters.discord.formatter import build_discord_embed, format_notice_message
from src.config import FEEDBACK_COLUMNS
from src.schema import FeedbackRecord, NoticeRecord

NoticeBatchItem = tuple[NoticeRecord, float] | tuple[NoticeRecord, float, str]


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


def load_feedback_df(feedback_csv: str | Path) -> pd.DataFrame:
    """Load feedback CSV as a DataFrame ordered by FEEDBACK_COLUMNS, or empty if missing."""
    path = Path(feedback_csv)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    df = pd.read_csv(path, encoding="utf-8")
    for col in FEEDBACK_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    ordered = df[FEEDBACK_COLUMNS].copy()
    for col in ("notice_id", "source", "timestamp", "comment"):
        ordered[col] = ordered[col].fillna("").astype(str)
    if not ordered.empty:
        ordered["label"] = ordered["label"].astype(int)
    return ordered


def _unpack_batch_item(item: NoticeBatchItem) -> tuple[NoticeRecord, float, str]:
    """Normalize a 2-tuple or 3-tuple batch item into (notice, score, summary)."""
    if len(item) == 2:
        notice, score = item
        return notice, float(score), ""
    if len(item) == 3:
        notice, score, summary = item
        return notice, float(score), summary
    raise ValueError(f"Expected 2-tuple or 3-tuple batch item, got length {len(item)}")


class NoticeFeedbackView(discord.ui.View):
    """Interactive Discord view providing 관심/제외 feedback buttons for a notice."""

    def __init__(
        self,
        notice: NoticeRecord,
        feedback_csv: str | Path = "data/feedback.csv",
        *,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self.notice = notice
        self.feedback_csv = Path(feedback_csv)
        self.interested_button.custom_id = f"notice_fb:{notice.id}:1"[:100]
        self.exclude_button.custom_id = f"notice_fb:{notice.id}:0"[:100]

    async def _record_feedback(
        self,
        interaction: discord.Interaction,
        label: int,
        action_text: str,
    ) -> None:
        """Persist FeedbackRecord to CSV, disable buttons, and acknowledge the interaction."""
        record = FeedbackRecord(
            notice_id=self.notice.id,
            label=label,
            source="discord",
            timestamp=datetime.now(timezone.utc).isoformat(),
            comment=action_text,
        )
        append_feedback(self.feedback_csv, record)

        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

        await interaction.response.edit_message(
            content=f"피드백 기록 완료: {action_text}",
            view=self,
        )

    @discord.ui.button(label="관심 👍", style=discord.ButtonStyle.success)
    async def interested_button(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[NoticeFeedbackView],
    ) -> None:
        """Record positive user interest (label=1) for this notice."""
        del button
        await self._record_feedback(interaction, label=1, action_text="관심 👍")

    @discord.ui.button(label="제외 👎", style=discord.ButtonStyle.secondary)
    async def exclude_button(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[NoticeFeedbackView],
    ) -> None:
        """Record negative user feedback (label=0) for this notice."""
        del button
        await self._record_feedback(interaction, label=0, action_text="제외 👎")


class NoticeDiscordClient:
    """Discord bot adapter for sending notice alerts and collecting Good/Skip feedback."""

    def __init__(
        self,
        channel_id: int | str,
        feedback_csv: str | Path = "data/feedback.csv",
        token: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self.channel_id = int(channel_id)
        self.feedback_csv = Path(feedback_csv)
        self.token = token
        self.dry_run = dry_run
        self.sent_messages: list[dict[str, Any]] = []

    def _resolve_token(self) -> str:
        """Return configured bot token or read DISCORD_BOT_TOKEN from environment."""
        token = (self.token if self.token is not None else os.getenv("DISCORD_BOT_TOKEN", "")).strip()
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN is required when dry_run=False.")
        return token

    def _create_discord_client(self) -> discord.Client:
        """Create a discord.Client instance with default intents."""
        return discord.Client(intents=discord.Intents.default())

    async def send_notice(
        self,
        notice: NoticeRecord,
        score: float,
        summary: str = "",
        *,
        channel: Any | None = None,
    ) -> dict[str, Any]:
        """Send a formatted notice embed with feedback buttons or record it in dry-run mode."""
        payload = format_notice_message(notice, score=score, summary=summary)
        entry: dict[str, Any] = {
            **payload,
            "channel_id": self.channel_id,
            "notice_id": notice.id,
            "score": float(score),
        }

        if self.dry_run:
            self.sent_messages.append(entry)
            return entry

        if channel is not None:
            embed = build_discord_embed(payload)
            view = NoticeFeedbackView(notice, feedback_csv=self.feedback_csv)
            await channel.send(embed=embed, view=view)
            self.sent_messages.append(entry)
            return entry

        results = await self.send_batch([(notice, score, summary)])
        return results[0]

    async def send_batch(
        self,
        items: Sequence[NoticeBatchItem],
    ) -> list[dict[str, Any]]:
        """Send a batch of (notice, score[, summary]) items over one Discord client session."""
        if not items:
            return []

        unpacked = [_unpack_batch_item(item) for item in items]
        if self.dry_run:
            return [
                await self.send_notice(notice, score, summary)
                for notice, score, summary in unpacked
            ]

        token = self._resolve_token()
        client = self._create_discord_client()
        results: list[dict[str, Any]] = []
        try:
            await client.login(token)
            channel = client.get_channel(self.channel_id)
            if channel is None:
                channel = await client.fetch_channel(self.channel_id)
            for notice, score, summary in unpacked:
                results.append(
                    await self.send_notice(notice, score, summary, channel=channel)
                )
        finally:
            await client.close()

        return results
