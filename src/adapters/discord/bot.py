"""Discord adapter client, interactive feedback view, and feedback CSV persistence."""

from __future__ import annotations

import asyncio
import csv
import logging
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import discord
import pandas as pd

from src.adapters.discord.formatter import build_discord_embed, format_notice_message
from src.config import FEEDBACK_COLUMNS
from src.schema import FeedbackRecord, NoticeRecord

logger = logging.getLogger(__name__)

NoticeBatchItem = (
    tuple[NoticeRecord, float]
    | tuple[NoticeRecord, float, str]
    | tuple[NoticeRecord, float, str, dict[str, str] | None]
)


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


def _unpack_batch_item(item: NoticeBatchItem) -> tuple[NoticeRecord, float, str, dict[str, str] | None]:
    """Normalize a 2-, 3-, or 4-tuple batch item into (notice, score, summary, key_info)."""
    if len(item) == 2:
        notice, score = item
        return notice, float(score), "", None
    if len(item) == 3:
        notice, score, summary = item
        return notice, float(score), summary, None
    if len(item) == 4:
        notice, score, summary, key_info = item
        return notice, float(score), summary, key_info
    raise ValueError(f"Expected 2-tuple, 3-tuple, or 4-tuple batch item, got length {len(item)}")


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
        url = str(notice.url or "").strip()
        if url.startswith(("http://", "https://")):
            self.add_item(
                discord.ui.Button(
                    label="원문 보기",
                    emoji="🔗",
                    style=discord.ButtonStyle.link,
                    url=url,
                )
            )

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
            if isinstance(child, discord.ui.Button) and child.style != discord.ButtonStyle.link:
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


def _extract_link_url(components: Sequence[Any] | None) -> str:
    """Extract the first URL link button target from existing message ActionRow components."""
    for row in components or []:
        for child in getattr(row, "children", []):
            url = getattr(child, "url", None)
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url
    return ""


def parse_feedback_custom_id(custom_id: str) -> tuple[str, int] | None:
    """Parse persistent button custom_id (`notice_fb:<notice_id>:<0|1>`) into (notice_id, label)."""
    raw = str(custom_id or "").strip()
    if not raw.startswith("notice_fb:"):
        return None
    body = raw[len("notice_fb:") :]
    if ":" not in body:
        return None
    notice_id, label_str = body.rsplit(":", 1)
    if not notice_id or label_str not in {"0", "1"}:
        return None
    return notice_id, int(label_str)


def compute_next_delay_seconds(
    now: datetime,
    daily_at: str | None = "09:00",
    interval_minutes: int | None = None,
) -> float:
    """Compute sleep seconds until the next periodic run (`interval_minutes` or daily `HH:MM`)."""
    if interval_minutes is not None:
        if interval_minutes <= 0:
            raise ValueError(f"interval_minutes must be > 0, got {interval_minutes}")
        return float(interval_minutes * 60)

    raw_time = str(daily_at or "09:00").strip()
    parts = raw_time.split(":")
    if len(parts) != 2:
        raise ValueError(f"daily_at must be in HH:MM format, got {daily_at!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid HH:MM time in daily_at: {daily_at!r}")

    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return max(1.0, (target - now).total_seconds())


def _resolve_timezone(timezone_name: str) -> timezone | ZoneInfo:
    """Resolve an IANA timezone with KST (+09:00) fallback if system tzdata is unavailable."""
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        if timezone_name == "Asia/Seoul":
            return timezone(timedelta(hours=9))
        return timezone.utc


def build_persistent_notice_view(
    notice_id: str,
    url: str = "",
    *,
    disabled: bool = False,
) -> discord.ui.View:
    """Build a stateless View (`timeout=None`) handled via `on_interaction` custom_id routing."""
    view = discord.ui.View(timeout=None)
    view.add_item(
        discord.ui.Button(
            label="관심 👍",
            style=discord.ButtonStyle.success,
            custom_id=f"notice_fb:{notice_id}:1"[:100],
            disabled=disabled,
        )
    )
    view.add_item(
        discord.ui.Button(
            label="제외 👎",
            style=discord.ButtonStyle.secondary,
            custom_id=f"notice_fb:{notice_id}:0"[:100],
            disabled=disabled,
        )
    )
    clean_url = str(url or "").strip()
    if clean_url.startswith(("http://", "https://")):
        view.add_item(
            discord.ui.Button(
                label="원문 보기",
                emoji="🔗",
                style=discord.ButtonStyle.link,
                url=clean_url,
            )
        )
    return view


async def handle_persistent_feedback_interaction(
    interaction: Any,
    feedback_csv: str | Path = "data/feedback.csv",
) -> bool:
    """Handle persistent button clicks from surviving/restarted bot processes and log CSV feedback."""
    if getattr(interaction, "type", None) != discord.InteractionType.component:
        return False
    data = getattr(interaction, "data", None) or {}
    parsed = parse_feedback_custom_id(str(data.get("custom_id", "")))
    if parsed is None:
        return False

    notice_id, label = parsed
    action_text = "관심 👍" if label == 1 else "제외 👎"
    record = FeedbackRecord(
        notice_id=notice_id,
        label=label,
        source="discord",
        timestamp=datetime.now(timezone.utc).isoformat(),
        comment=action_text,
    )
    append_feedback(feedback_csv, record)

    message = getattr(interaction, "message", None)
    link_url = _extract_link_url(getattr(message, "components", None))
    disabled_view = build_persistent_notice_view(notice_id, url=link_url, disabled=True)

    response = getattr(interaction, "response", None)
    if response is not None:
        try:
            await response.edit_message(content=f"피드백 기록 완료: {action_text}", view=disabled_view)
        except Exception:
            await response.send_message(f"피드백 기록 완료: {action_text}", ephemeral=True)
    return True


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
        key_info: dict[str, str] | None = None,
        channel: Any | None = None,
        persistent: bool = False,
    ) -> dict[str, Any]:
        """Send a formatted notice embed with feedback buttons or record it in dry-run mode."""
        payload = format_notice_message(notice, score=score, summary=summary, key_info=key_info)
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
            view = (
                build_persistent_notice_view(notice.id, notice.url)
                if persistent
                else NoticeFeedbackView(notice, feedback_csv=self.feedback_csv)
            )
            await channel.send(embed=embed, view=view)
            self.sent_messages.append(entry)
            return entry

        results = await self.send_batch([(notice, score, summary, key_info)])
        return results[0]

    async def send_batch(
        self,
        items: Sequence[NoticeBatchItem],
        *,
        on_sent: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Send a batch of (notice, score[, summary[, key_info]]) items over one Discord client session."""
        if not items:
            return []

        unpacked = [_unpack_batch_item(item) for item in items]
        if self.dry_run:
            results: list[dict[str, Any]] = []
            for notice, score, summary, key_info in unpacked:
                entry = await self.send_notice(notice, score, summary, key_info=key_info)
                results.append(entry)
                if on_sent is not None:
                    on_sent(notice.id)
            return results

        token = self._resolve_token()
        client = self._create_discord_client()
        results = []
        try:
            await client.login(token)
            channel = client.get_channel(self.channel_id)
            if channel is None:
                channel = await client.fetch_channel(self.channel_id)
            for notice, score, summary, key_info in unpacked:
                entry = await self.send_notice(
                    notice,
                    score,
                    summary,
                    key_info=key_info,
                    channel=channel,
                )
                results.append(entry)
                if on_sent is not None:
                    await asyncio.to_thread(on_sent, notice.id)
        finally:
            await client.close()

        return results

    async def run_daemon(
        self,
        fetch_batch_fn: Callable[[], Sequence[NoticeBatchItem]],
        *,
        daily_at: str | None = "09:00",
        interval_minutes: int | None = None,
        timezone_name: str = "Asia/Seoul",
        run_on_startup: bool = False,
        on_dispatched: Callable[[list[str]], None] | None = None,
    ) -> None:
        """Run a 24/7 Discord bot handling persistent feedback buttons and scheduled notice dispatches."""
        if self.dry_run:
            raise ValueError("Daemon mode requires dry_run=False.")

        token = self._resolve_token()
        tz = _resolve_timezone(timezone_name)
        # Validate schedule syntax upfront before connecting to Discord Gateway.
        compute_next_delay_seconds(datetime.now(tz), daily_at=daily_at, interval_minutes=interval_minutes)
        client = self._create_discord_client()
        scheduler_task: asyncio.Task[None] | None = None

        async def _dispatch_once(channel: Any) -> None:
            try:
                raw_batch = await asyncio.to_thread(fetch_batch_fn)
                unpacked = [_unpack_batch_item(item) for item in raw_batch]
                if not unpacked:
                    logger.info("Scheduled cycle: no new notices above threshold.")
                    return

                dispatched_count = 0
                for notice, score, summary, key_info in unpacked:
                    await self.send_notice(
                        notice,
                        score,
                        summary,
                        key_info=key_info,
                        channel=channel,
                        persistent=True,
                    )
                    dispatched_count += 1
                    if on_dispatched is not None:
                        await asyncio.to_thread(on_dispatched, [notice.id])

                logger.info("Scheduled cycle: dispatched %d notice(s).", dispatched_count)
            except Exception:
                logger.exception("Scheduled notice cycle failed; will retry at next interval.")

        async def _scheduler_loop() -> None:
            await client.wait_until_ready()
            channel = client.get_channel(self.channel_id)
            if channel is None:
                channel = await client.fetch_channel(self.channel_id)

            if run_on_startup:
                await _dispatch_once(channel)

            while not client.is_closed():
                delay = compute_next_delay_seconds(
                    datetime.now(tz),
                    daily_at=daily_at,
                    interval_minutes=interval_minutes,
                )
                logger.info(
                    "Next notice crawl & alert cycle scheduled in %.0f seconds (~%.1f hours).",
                    delay,
                    delay / 3600.0,
                )
                await asyncio.sleep(delay)
                if client.is_closed():
                    break
                await _dispatch_once(channel)

        @client.event
        async def on_ready() -> None:
            nonlocal scheduler_task
            logger.info("Discord daemon logged in as %s (channel=%s)", client.user, self.channel_id)
            if scheduler_task is None or scheduler_task.done():
                scheduler_task = asyncio.create_task(_scheduler_loop())

        @client.event
        async def on_interaction(interaction: discord.Interaction) -> None:
            try:
                await handle_persistent_feedback_interaction(
                    interaction,
                    feedback_csv=self.feedback_csv,
                )
            except Exception:
                logger.exception("Failed to handle persistent Discord feedback interaction.")

        try:
            async with client:
                await client.start(token)
        finally:
            if scheduler_task is not None and not scheduler_task.done():
                scheduler_task.cancel()
