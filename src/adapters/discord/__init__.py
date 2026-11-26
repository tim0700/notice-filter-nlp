"""Discord notification and feedback collection adapter package."""

from src.adapters.discord.bot import (
    NoticeDiscordClient,
    NoticeFeedbackView,
    append_feedback,
    build_persistent_notice_view,
    compute_next_delay_seconds,
    handle_persistent_feedback_interaction,
    load_feedback_df,
    parse_feedback_custom_id,
)
from src.adapters.discord.formatter import (
    build_discord_embed,
    format_notice_message,
)

__all__ = [
    "NoticeDiscordClient",
    "NoticeFeedbackView",
    "append_feedback",
    "build_discord_embed",
    "build_persistent_notice_view",
    "compute_next_delay_seconds",
    "format_notice_message",
    "handle_persistent_feedback_interaction",
    "load_feedback_df",
    "parse_feedback_custom_id",
]
