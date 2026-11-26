"""Offline unit tests for the Discord adapter (formatter, bot, feedback view, CLI)."""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pandas as pd
import pytest

from src.adapters.discord import (
    NoticeDiscordClient,
    NoticeFeedbackView,
    append_feedback,
    build_discord_embed,
    compute_next_delay_seconds,
    format_notice_message,
    handle_persistent_feedback_interaction,
    load_feedback_df,
    parse_feedback_custom_id,
)
from src.adapters.discord.cli import load_notice_batch, run_cli
from src.adapters.discord.formatter import (
    EMBED_DESCRIPTION_MAX_LEN,
    EMBED_FIELD_VALUE_MAX_LEN,
    EMBED_TITLE_MAX_LEN,
    HIGH_SCORE_COLOR,
    LOW_SCORE_COLOR,
    MEDIUM_SCORE_COLOR,
    format_deadline_dday,
    format_score_badge,
    format_source_label,
    score_to_color,
)
from src.config import CANONICAL_COLUMNS, FEEDBACK_COLUMNS, find_repo_root
from src.schema import FeedbackRecord, NoticeRecord


def _make_notice(**overrides: str) -> NoticeRecord:
    base = {
        "id": "swcon_001",
        "source": "swcon_academic",
        "category": "학사/장학",
        "title": "2026학년도 2학기 SW중심대학 장학생 선발",
        "meta_tags": "국제캠퍼스|장학금",
        "posted_at": "2026-11-26",
        "url": "https://swcon.khu.ac.kr/notice/1001",
        "content": "소프트웨어융합학과 재학생 대상 장학금 공지입니다.",
        "deadline": "2026-12-10",
    }
    base.update(overrides)
    return NoticeRecord(**base)


def test_discord_adapter_decoupled_from_core_nlp() -> None:
    discord_dir = find_repo_root() / "src" / "adapters" / "discord"
    forbidden_prefixes = ("src.models", "src.preprocess", "src.collectors")

    for py_file in discord_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), (
                        f"{py_file.name} imports forbidden core module {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_prefixes), (
                    f"{py_file.name} imports forbidden core module {node.module}"
                )


def test_format_notice_message_and_score_colors() -> None:
    assert score_to_color(0.90) == HIGH_SCORE_COLOR
    assert score_to_color(0.75) == HIGH_SCORE_COLOR
    assert score_to_color(0.60) == MEDIUM_SCORE_COLOR
    assert score_to_color(0.50) == MEDIUM_SCORE_COLOR
    assert score_to_color(0.20) == LOW_SCORE_COLOR
    assert format_source_label("seongnam_youth") == "🏛️ 성남청년다해"
    assert format_score_badge(0.92) == "🟢 92% (높음)"
    assert format_deadline_dday("2026-12-03", today=datetime(2026, 11, 26).date()) == "2026-12-03 (D-7)"
    assert format_deadline_dday("2026-11-26", today=datetime(2026, 11, 26).date()) == "2026-11-26 (D-Day)"

    notice = _make_notice(deadline="")
    payload = format_notice_message(
        notice,
        score=0.876,
        summary="첫 번째 핵심 요약입니다. https://instagram.com/example ■ 두 번째 요약입니다. 3.",
    )

    assert payload["author"]["name"] == "🎓 SW융합 학사/장학 · 학사/장학"
    assert payload["title"] == notice.title
    assert payload["url"] == notice.url
    assert "• 첫 번째 핵심 요약입니다." in payload["description"]
    assert "• 두 번째 요약입니다." in payload["description"]
    assert "https://instagram.com" not in payload["description"]
    assert "🏷️ `국제캠퍼스` `장학금`" in payload["description"]
    assert payload["color"] == HIGH_SCORE_COLOR
    assert len(payload["fields"]) == 3
    assert payload["fields"][0] == {"name": "📌 출처", "value": "🎓 SW융합 학사/장학", "inline": True}
    assert payload["fields"][1] == {"name": "🎯 적합도", "value": "🟢 88% (높음)", "inline": True}
    assert payload["fields"][2] == {"name": "⏰ 마감일", "value": "상시/미정", "inline": True}
    assert "swcon_001" in payload["footer"]["text"]


def test_format_notice_message_truncates_limits_and_fallbacks() -> None:
    long_title = "가" * 300
    long_summary = "나" * 4200
    long_source = "src_" + ("x" * 1100)
    notice = _make_notice(title=long_title, source=long_source, content="본문 내용")

    payload = format_notice_message(notice, score=0.40, summary=long_summary)
    assert len(payload["title"]) == EMBED_TITLE_MAX_LEN
    assert payload["title"].endswith("...")
    assert len(payload["description"]) <= EMBED_DESCRIPTION_MAX_LEN
    assert "..." in payload["description"]
    assert len(payload["fields"][0]["value"]) == EMBED_FIELD_VALUE_MAX_LEN
    assert payload["color"] == LOW_SCORE_COLOR

    # Empty summary falls back to content, then title
    fallback_content = format_notice_message(_make_notice(content="본문 우선"), score=0.55, summary="  ")
    assert "본문 우선" in fallback_content["description"]
    fallback_title = format_notice_message(
        _make_notice(title="제목 폴백", content="", meta_tags=""),
        score=0.55,
        summary="",
    )
    assert fallback_title["description"] == "제목 폴백"


def test_build_discord_embed() -> None:
    notice = _make_notice()
    payload = format_notice_message(notice, score=0.65, summary="요약 테스트")
    embed = build_discord_embed(payload)

    assert isinstance(embed, discord.Embed)
    assert embed.author.name == "🎓 SW융합 학사/장학 · 학사/장학"
    assert embed.title == notice.title
    assert "요약 테스트" in (embed.description or "")
    assert embed.url == notice.url
    assert embed.color is not None
    assert embed.color.value == MEDIUM_SCORE_COLOR
    assert len(embed.fields) == 3
    assert embed.fields[0].name == "📌 출처"
    assert embed.fields[0].value == "🎓 SW융합 학사/장학"
    assert embed.footer.text is not None
    assert "swcon_001" in embed.footer.text

    hex_str_embed = build_discord_embed({"title": "Hex", "color": "#2ECC71"})
    assert hex_str_embed.color is not None
    assert hex_str_embed.color.value == 0x2ECC71


def test_append_and_load_feedback_df(tmp_path: Path) -> None:
    feedback_csv = tmp_path / "nested" / "feedback.csv"
    empty_df = load_feedback_df(feedback_csv)
    assert empty_df.empty
    assert list(empty_df.columns) == FEEDBACK_COLUMNS

    append_feedback(
        feedback_csv,
        FeedbackRecord(
            notice_id="n_1",
            label=1,
            source="discord",
            timestamp="2026-11-26T09:00:00+00:00",
            comment="관심 👍",
        ),
    )
    append_feedback(
        feedback_csv,
        FeedbackRecord(
            notice_id="n_2",
            label=0,
            source="discord",
            timestamp="2026-11-26T09:01:00+00:00",
        ),
    )

    loaded = load_feedback_df(feedback_csv)
    assert list(loaded.columns) == FEEDBACK_COLUMNS
    assert len(loaded) == 2
    assert loaded["notice_id"].tolist() == ["n_1", "n_2"]
    assert loaded["label"].tolist() == [1, 0]
    assert loaded["comment"].tolist() == ["관심 👍", ""]


def test_notice_feedback_view_button_callbacks_persist_feedback(tmp_path: Path) -> None:
    feedback_csv = tmp_path / "feedback.csv"
    notice_pos = _make_notice(id="notice_pos")
    notice_neg = _make_notice(id="notice_neg")

    async def _run() -> tuple[NoticeFeedbackView, MagicMock, MagicMock]:
        view_pos = NoticeFeedbackView(notice_pos, feedback_csv=feedback_csv)
        assert [btn.label for btn in view_pos.children if isinstance(btn, discord.ui.Button)] == [
            "관심 👍",
            "제외 👎",
            "원문 보기",
        ]

        interaction_pos = MagicMock()
        interaction_pos.response = AsyncMock()
        await view_pos.interested_button.callback(interaction_pos)

        view_neg = NoticeFeedbackView(notice_neg, feedback_csv=feedback_csv)
        interaction_neg = MagicMock()
        interaction_neg.response = AsyncMock()
        await view_neg.exclude_button.callback(interaction_neg)

        return view_pos, interaction_pos, interaction_neg

    view_pos, interaction_pos, interaction_neg = asyncio.run(_run())

    interaction_pos.response.edit_message.assert_awaited_once()
    interaction_neg.response.edit_message.assert_awaited_once()
    vote_buttons = [
        child
        for child in view_pos.children
        if isinstance(child, discord.ui.Button) and child.style != discord.ButtonStyle.link
    ]
    link_buttons = [
        child
        for child in view_pos.children
        if isinstance(child, discord.ui.Button) and child.style == discord.ButtonStyle.link
    ]
    assert len(vote_buttons) == 2 and all(child.disabled for child in vote_buttons)
    assert len(link_buttons) == 1 and not link_buttons[0].disabled

    saved = load_feedback_df(feedback_csv)
    assert len(saved) == 2
    assert saved["notice_id"].tolist() == ["notice_pos", "notice_neg"]
    assert saved["label"].tolist() == [1, 0]
    assert saved["source"].tolist() == ["discord", "discord"]
    assert saved["comment"].tolist() == ["관심 👍", "제외 👎"]
    for ts in saved["timestamp"]:
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None


def test_notice_discord_client_dry_run(tmp_path: Path) -> None:
    client = NoticeDiscordClient(
        channel_id="987654321",
        feedback_csv=tmp_path / "feedback.csv",
        dry_run=True,
    )
    n1 = _make_notice(id="dry_1", title="공지 1")
    n2 = _make_notice(id="dry_2", title="공지 2")

    async def _run() -> list[dict]:
        first = await client.send_notice(n1, score=0.91, summary="요약 1")
        batch = await client.send_batch([(n2, 0.42)])
        return [first, *batch]

    results = asyncio.run(_run())
    assert len(results) == 2
    assert len(client.sent_messages) == 2
    assert client.sent_messages[0]["notice_id"] == "dry_1"
    assert client.sent_messages[0]["channel_id"] == 987654321
    assert client.sent_messages[1]["notice_id"] == "dry_2"
    assert n2.content in client.sent_messages[1]["description"]


def test_notice_discord_client_live_send_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    missing_token_client = NoticeDiscordClient(channel_id=111222, dry_run=False)
    with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
        asyncio.run(missing_token_client.send_notice(_make_notice(), score=0.8))

    mock_channel = MagicMock()
    mock_channel.send = AsyncMock()

    mock_discord = MagicMock()
    mock_discord.login = AsyncMock()
    mock_discord.get_channel = MagicMock(return_value=None)
    mock_discord.fetch_channel = AsyncMock(return_value=mock_channel)
    mock_discord.close = AsyncMock()

    client = NoticeDiscordClient(
        channel_id=111222,
        feedback_csv=tmp_path / "feedback.csv",
        token="fake-bot-token-for-test",
        dry_run=False,
    )
    notices = [
        (_make_notice(id="live_1"), 0.85, "요약 A"),
        (_make_notice(id="live_2"), 0.60, "요약 B"),
    ]

    with patch("src.adapters.discord.bot.discord.Client", return_value=mock_discord):
        sent = asyncio.run(client.send_batch(notices))

    assert len(sent) == 2
    mock_discord.login.assert_awaited_once_with("fake-bot-token-for-test")
    mock_discord.fetch_channel.assert_awaited_once_with(111222)
    assert mock_channel.send.await_count == 2
    for call in mock_channel.send.await_args_list:
        assert isinstance(call.kwargs["embed"], discord.Embed)
        assert isinstance(call.kwargs["view"], NoticeFeedbackView)
    mock_discord.close.assert_awaited_once()


def test_cli_dry_run_on_sample_notices(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sample_csv = find_repo_root() / "data" / "sample_notices.csv"
    feedback_csv = tmp_path / "feedback.csv"

    exit_code = run_cli(
        [
            "--input",
            str(sample_csv),
            "--limit",
            "5",
            "--dry-run",
            "--feedback-csv",
            str(feedback_csv),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "[DRY-RUN] Dispatched 5 notice(s)" in captured


def test_cli_load_batch_with_score_summary_and_live_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "scored.csv"
    row = {col: "" for col in CANONICAL_COLUMNS}
    row.update(
        {
            "id": "cli_001",
            "source": "swcon_career",
            "category": "특강",
            "title": "AI 특강",
            "posted_at": "2026-11-26",
            "url": "https://example.com/cli_001",
            "score": 0.93,
            "summary": "맞춤형 AI 특강 요약",
        }
    )
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8")

    batch = load_notice_batch(csv_path, limit=5)
    assert len(batch) == 1
    notice, score, summary = batch[0]
    assert notice.id == "cli_001"
    assert score == pytest.approx(0.93)
    assert summary == "맞춤형 AI 특강 요약"

    monkeypatch.delenv("DISCORD_CHANNEL_ID", raising=False)
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    with patch("src.adapters.discord.cli.load_env", return_value=None):
        with pytest.raises(ValueError, match="DISCORD_CHANNEL_ID"):
            run_cli(["--input", str(csv_path), "--limit", "1"])

    monkeypatch.setenv("DISCORD_CHANNEL_ID", "555666777")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-cli-token")
    with (
        patch("src.adapters.discord.cli.load_env", return_value=None),
        patch(
            "src.adapters.discord.cli.NoticeDiscordClient.send_batch",
            new_callable=AsyncMock,
            return_value=[{"notice_id": "cli_001"}],
        ) as mock_send_batch,
    ):
        code = run_cli(["--input", str(csv_path), "--limit", "1"])

    assert code == 0
    mock_send_batch.assert_awaited_once()


def test_persistent_feedback_helpers_and_scheduler_timing(tmp_path: Path) -> None:
    assert parse_feedback_custom_id("notice_fb:swcon_academic_100:1") == ("swcon_academic_100", 1)
    assert parse_feedback_custom_id("notice_fb:prefix:with:colon:0") == ("prefix:with:colon", 0)
    assert parse_feedback_custom_id("other_button:1") is None
    assert parse_feedback_custom_id("notice_fb:empty_id:") is None

    now = datetime(2026, 11, 26, 8, 30, 0)
    assert compute_next_delay_seconds(now, daily_at="09:00") == pytest.approx(1800.0)
    after = datetime(2026, 11, 26, 9, 15, 0)
    assert compute_next_delay_seconds(after, daily_at="09:00") == pytest.approx(85500.0)
    assert compute_next_delay_seconds(now, interval_minutes=15) == pytest.approx(900.0)

    feedback_csv = tmp_path / "persistent_feedback.csv"
    link_child = MagicMock()
    link_child.url = "https://swcon.khu.ac.kr/notice/100"
    action_row = MagicMock()
    action_row.children = [link_child]

    interaction = MagicMock()
    interaction.type = discord.InteractionType.component
    interaction.data = {"custom_id": "notice_fb:swcon_academic_100:1"}
    interaction.message = MagicMock()
    interaction.message.components = [action_row]
    interaction.response = AsyncMock()

    async def _run() -> bool:
        handled = await handle_persistent_feedback_interaction(interaction, feedback_csv=feedback_csv)
        other = MagicMock()
        other.type = discord.InteractionType.application_command
        not_handled = await handle_persistent_feedback_interaction(other, feedback_csv=feedback_csv)
        assert not not_handled
        return handled

    assert asyncio.run(_run()) is True
    interaction.response.edit_message.assert_awaited_once()
    edited_view = interaction.response.edit_message.call_args.kwargs["view"]
    buttons = [c for c in edited_view.children if isinstance(c, discord.ui.Button)]
    assert len(buttons) == 3
    assert buttons[0].disabled is True and buttons[1].disabled is True
    assert buttons[2].style == discord.ButtonStyle.link and buttons[2].url == "https://swcon.khu.ac.kr/notice/100"

    saved = load_feedback_df(feedback_csv)
    assert saved["notice_id"].tolist() == ["swcon_academic_100"]
    assert saved["label"].tolist() == [1]


def test_send_batch_partial_failure_preserves_already_sent_ids(tmp_path: Path) -> None:
    mock_channel = MagicMock()
    mock_channel.send = AsyncMock(side_effect=[None, RuntimeError("transient send error")])

    mock_discord = MagicMock()
    mock_discord.login = AsyncMock()
    mock_discord.get_channel = MagicMock(return_value=mock_channel)
    mock_discord.close = AsyncMock()

    client = NoticeDiscordClient(
        channel_id=111222,
        feedback_csv=tmp_path / "feedback.csv",
        token="fake-token",
        dry_run=False,
    )
    recorded: list[str] = []
    batch = [(_make_notice(id="ok_first"), 0.9, "요약1"), (_make_notice(id="fail_second"), 0.8, "요약2")]

    with patch("src.adapters.discord.bot.discord.Client", return_value=mock_discord):
        with pytest.raises(RuntimeError, match="transient send error"):
            asyncio.run(client.send_batch(batch, on_sent=recorded.append))

    assert recorded == ["ok_first"]
    mock_discord.close.assert_awaited_once()
