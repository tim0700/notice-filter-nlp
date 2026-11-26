"""Discord embed payload formatting and discord.Embed builder utilities."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import discord

from src.schema import NoticeRecord

EMBED_TITLE_MAX_LEN = 256
EMBED_DESCRIPTION_MAX_LEN = 4096
EMBED_FIELD_VALUE_MAX_LEN = 1024
EMBED_FOOTER_MAX_LEN = 2048
EMBED_AUTHOR_MAX_LEN = 256
SUMMARY_PREVIEW_MAX_LEN = 360

HIGH_SCORE_COLOR = 0x2ECC71
MEDIUM_SCORE_COLOR = 0xF39C12
LOW_SCORE_COLOR = 0x95A5A6

SOURCE_DISPLAY_MAP: dict[str, str] = {
    "swcon_academic": "🎓 SW융합 학사/장학",
    "swcon_career": "💡 SW융합 취업/대외활동",
    "goodjob_extracurricular": "📌 미래인재 비교과(국제)",
    "goodjob_general": "💼 미래인재 일반채용",
    "goodjob_recommend": "🚀 미래인재 추천채용",
    "seongnam_youth": "🏛️ 성남청년다해",
    "youth_api": "🌐 온통청년 API",
}

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_NOISY_BULLET_RE = re.compile(r"[■▪▶▷◆◇💭★☆※]+")
_DANGLING_INDEX_RE = re.compile(r"\s+\d+\.$")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?!\d+\.)|\n+")
_WHITESPACE_RE = re.compile(r"\s+")


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


def format_source_label(source: str) -> str:
    """Convert an internal source key into a human-readable Korean label."""
    key = str(source or "").strip()
    if not key:
        return "-"
    return SOURCE_DISPLAY_MAP.get(key, key)


def format_score_badge(score: float) -> str:
    """Format relevance score into an intuitive percentage + badge label."""
    clamped = max(0.0, min(1.0, float(score)))
    pct = int(round(clamped * 100))
    if clamped >= 0.75:
        badge = "🟢"
        tier = "높음"
    elif clamped >= 0.50:
        badge = "🟡"
        tier = "보통"
    else:
        badge = "⚪"
        tier = "낮음"
    return f"{badge} {pct}% ({tier})"


def format_deadline_dday(deadline: str, today: date | None = None) -> str:
    """Format YYYY-MM-DD deadlines with D-day indicator when parseable."""
    raw = str(deadline or "").strip()
    if not raw:
        return "상시/미정"

    ref_date = today if today is not None else datetime.now().date()
    try:
        target_date = date.fromisoformat(raw[:10])
    except ValueError:
        return raw

    delta = (target_date - ref_date).days
    if delta > 0:
        suffix = f"D-{delta}"
    elif delta == 0:
        suffix = "D-Day"
    else:
        suffix = "마감"
    return f"{raw[:10]} ({suffix})"


def _format_summary_description(
    raw_text: str,
    meta_tags: str = "",
    max_summary_chars: int = SUMMARY_PREVIEW_MAX_LEN,
) -> str:
    """Clean raw summary/content into up to 3 readable bullet points plus inline tag chips."""
    cleaned = _URL_RE.sub("", str(raw_text or ""))
    cleaned = _NOISY_BULLET_RE.sub("\n", cleaned)
    cleaned = _DANGLING_INDEX_RE.sub("", cleaned.strip())

    raw_segments = [
        _WHITESPACE_RE.sub(" ", seg).strip(" -•·")
        for seg in _SENTENCE_BREAK_RE.split(cleaned)
    ]
    segments = [seg for seg in raw_segments if len(seg) >= 2 and not re.fullmatch(r"\d+\.?", seg)]

    if not segments:
        body_text = _truncate_text(_WHITESPACE_RE.sub(" ", str(raw_text or "")).strip(), max_summary_chars)
    elif len(segments) == 1:
        body_text = _truncate_text(segments[0], max_summary_chars)
    else:
        bullets: list[str] = []
        used_chars = 0
        for seg in segments[:3]:
            line = f"• {seg}"
            if used_chars + len(line) > max_summary_chars and bullets:
                break
            bullets.append(_truncate_text(line, max_summary_chars))
            used_chars += len(line) + 1
        body_text = "\n".join(bullets)

    tags = [t.strip() for t in str(meta_tags or "").split("|") if t.strip()][:5]
    if tags:
        chips = " ".join(f"`{tag}`" for tag in tags)
        return f"{body_text}\n\n🏷️ {chips}" if body_text else f"🏷️ {chips}"
    return body_text


_SLOT_LABELS: tuple[tuple[str, str], ...] = (
    ("schedule", "📅 일정/기한"),
    ("target", "👥 대상"),
    ("benefit", "💰 지원/혜택"),
    ("location", "📍 장소"),
)


def format_key_info_slots(key_info: dict[str, str] | None) -> str:
    """Format extracted notice slots into concise labeled lines, omitting empty values."""
    if not key_info:
        return ""
    lines: list[str] = []
    for key, label in _SLOT_LABELS:
        val = _WHITESPACE_RE.sub(" ", str(key_info.get(key, "") or "")).strip()
        if val:
            lines.append(f"{label}: {val}")
    return "\n".join(lines)


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
    *,
    key_info: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a polished Discord embed payload dictionary for a filtered notice."""
    raw_description = summary.strip() or notice.content.strip() or notice.title
    formatted_desc = _format_summary_description(raw_description, meta_tags=notice.meta_tags)
    title = _truncate_text(notice.title, EMBED_TITLE_MAX_LEN)
    description = _truncate_text(formatted_desc, EMBED_DESCRIPTION_MAX_LEN)

    source_label = format_source_label(notice.source)
    category = str(notice.category or "").strip()
    author_name = _truncate_text(
        f"{source_label} · {category}" if category else source_label,
        EMBED_AUTHOR_MAX_LEN,
    )

    source_value = _truncate_text(source_label, EMBED_FIELD_VALUE_MAX_LEN)
    score_value = _truncate_text(format_score_badge(float(score)), EMBED_FIELD_VALUE_MAX_LEN)
    deadline_value = _truncate_text(format_deadline_dday(notice.deadline), EMBED_FIELD_VALUE_MAX_LEN)
    footer_text = _truncate_text(
        f"게시일: {notice.posted_at or '-'} · ID: {notice.id}",
        EMBED_FOOTER_MAX_LEN,
    )

    fields: list[dict[str, Any]] = [
        {"name": "📌 출처", "value": source_value, "inline": True},
        {"name": "🎯 적합도", "value": score_value, "inline": True},
        {"name": "⏰ 마감일", "value": deadline_value, "inline": True},
    ]
    slot_text = format_key_info_slots(key_info)
    if slot_text:
        fields.append(
            {
                "name": "📋 핵심 정보",
                "value": _truncate_text(slot_text, EMBED_FIELD_VALUE_MAX_LEN),
                "inline": False,
            }
        )

    return {
        "author": {"name": author_name},
        "title": title,
        "url": notice.url,
        "description": description,
        "color": score_to_color(float(score)),
        "fields": fields,
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

    author = payload.get("author")
    if isinstance(author, dict) and author.get("name"):
        embed.set_author(name=_truncate_text(str(author["name"]), EMBED_AUTHOR_MAX_LEN))

    for field in payload.get("fields", []):
        name = _truncate_text(str(field.get("name", "")), EMBED_TITLE_MAX_LEN) or "-"
        value = _truncate_text(str(field.get("value", "")), EMBED_FIELD_VALUE_MAX_LEN) or "-"
        inline = bool(field.get("inline", False))
        embed.add_field(name=name, value=value, inline=inline)

    footer = payload.get("footer")
    if isinstance(footer, dict) and footer.get("text"):
        embed.set_footer(text=_truncate_text(str(footer["text"]), EMBED_FOOTER_MAX_LEN))

    return embed
