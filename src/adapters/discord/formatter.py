"""Discord embed payload formatting and discord.Embed builder utilities."""

from __future__ import annotations

from typing import Any

import discord

from src.schema import NoticeRecord

EMBED_TITLE_MAX_LEN = 256
EMBED_DESCRIPTION_MAX_LEN = 4096
EMBED_FIELD_VALUE_MAX_LEN = 1024
EMBED_FOOTER_MAX_LEN = 2048

HIGH_SCORE_COLOR = 0x2ECC71
MEDIUM_SCORE_COLOR = 0xF39C12
LOW_SCORE_COLOR = 0x95A5A6


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length using an ellipsis suffix when needed."""
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_length:
        return cleaned
    if max_length <= 3:
        return cleaned[:max_length]
    return f"{cleaned[: max_length - 3].rstrip()}..."


def score_to_color(score: float) -> int:
    """Map a relevance score in [0, 1] to a Discord embed hex color integer."""
    if score >= 0.75:
        return HIGH_SCORE_COLOR
    if score >= 0.50:
        return MEDIUM_SCORE_COLOR
    return LOW_SCORE_COLOR


def _parse_color(raw_color: int | str | None) -> int | None:
    """Normalize an integer or hex color string into an integer color value."""
    if raw_color is None:
        return None
    if isinstance(raw_color, int):
        return raw_color
    text = str(raw_color).strip().lstrip("#")
    if text.lower().startswith("0x"):
        text = text[2:]
    return int(text, 16)


def format_notice_message(
    notice: NoticeRecord,
    score: float,
    summary: str = "",
) -> dict[str, Any]:
    """Build a Discord embed payload dictionary for a filtered notice."""
    raw_description = summary.strip() or notice.content.strip() or notice.title
    title = _truncate_text(notice.title, EMBED_TITLE_MAX_LEN)
    description = _truncate_text(raw_description, EMBED_DESCRIPTION_MAX_LEN)
    deadline_value = _truncate_text(notice.deadline.strip() or "상시/미정", EMBED_FIELD_VALUE_MAX_LEN)
    source_value = _truncate_text(notice.source.strip() or "-", EMBED_FIELD_VALUE_MAX_LEN)
    footer_text = _truncate_text(
        f"ID: {notice.id} | 게시일: {notice.posted_at or '-'}",
        EMBED_FOOTER_MAX_LEN,
    )

    return {
        "title": title,
        "url": notice.url,
        "description": description,
        "color": score_to_color(float(score)),
        "fields": [
            {"name": "출처", "value": source_value, "inline": True},
            {"name": "적합도", "value": f"{score:.2f}", "inline": True},
            {"name": "마감일", "value": deadline_value, "inline": True},
        ],
        "footer": {"text": footer_text},
    }


def build_discord_embed(payload: dict[str, Any]) -> discord.Embed:
    """Convert an embed payload dictionary into a discord.Embed instance."""
    title = _truncate_text(str(payload.get("title", "")), EMBED_TITLE_MAX_LEN) or None
    description = _truncate_text(str(payload.get("description", "")), EMBED_DESCRIPTION_MAX_LEN) or None
    url = str(payload.get("url", "")).strip() or None
    color = _parse_color(payload.get("color"))

    embed = discord.Embed(
        title=title,
        description=description,
        url=url,
        color=color,
    )

    for field in payload.get("fields", []):
        name = _truncate_text(str(field.get("name", "")), EMBED_TITLE_MAX_LEN) or "-"
        value = _truncate_text(str(field.get("value", "")), EMBED_FIELD_VALUE_MAX_LEN) or "-"
        inline = bool(field.get("inline", False))
        embed.add_field(name=name, value=value, inline=inline)

    footer = payload.get("footer")
    if isinstance(footer, dict) and footer.get("text"):
        embed.set_footer(text=_truncate_text(str(footer["text"]), EMBED_FOOTER_MAX_LEN))

    return embed
