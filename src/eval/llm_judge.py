"""Anthropic LLM-as-a-Judge silver labeler with strict JSON parsing and CSV checkpointing."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_env, load_profile

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _clean_cell(value: Any) -> str:
    """Convert a DataFrame/mapping value to stripped text, treating NaN/None as empty."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


class AnthropicNoticeJudge:
    """LLM-as-a-Judge using Claude Haiku and the profile.yaml personalization rubric."""

    def __init__(
        self,
        model: str | None = None,
        profile: dict[str, Any] | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 256,
    ) -> None:
        self.profile = profile if profile is not None else load_profile()
        models_cfg = self.profile.get("models", {})
        self.model = model or str(models_cfg.get("llm_judge_model") or DEFAULT_JUDGE_MODEL)
        self.max_tokens = int(max_tokens)
        self._api_key = api_key
        self._client = client
        self.system_prompt = self.build_system_prompt(self.profile)

    def _get_client(self) -> Any:
        """Lazily initialize the Anthropic SDK client using explicit key or shared .env."""
        if self._client is None:
            import anthropic

            if self._api_key is None:
                load_env()
                self._api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not self._api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set in the environment or .env.")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def build_system_prompt(profile: dict[str, Any]) -> str:
        """Build rubric system prompt from user profile and positive/negative rules."""
        user = profile.get("user", {})
        rubric = profile.get("judge_rubric", {})
        pos_rules = rubric.get("positive_rules", [])
        neg_rules = rubric.get("negative_rules", [])

        pos_block = "\n".join(f"- {rule}" for rule in pos_rules)
        neg_block = "\n".join(f"- {rule}" for rule in neg_rules)

        return (
            "당신은 대학생 맞춤형 공지사항 및 청년복지 필터링 평가자(LLM Judge)입니다.\n"
            "아래 학생 프로필과 판정 루브릭을 엄격하게 적용하여 공지가 관심 대상(1)인지 제외 대상(0)인지 판정하세요.\n\n"
            "[학생 프로필]\n"
            f"- 소속: {user.get('university', '')} {user.get('college', '')} {user.get('major', '')}\n"
            f"- 학년/졸업예정: {user.get('grade', '')} (졸업예정: {user.get('expected_graduation', '')})\n"
            f"- 거주지: {user.get('region', '')}\n\n"
            "[관심(1) 판정 기준]\n"
            f"{pos_block}\n\n"
            "[제외(0) 판정 기준]\n"
            f"{neg_block}\n\n"
            "[출력 형식]\n"
            '반드시 마크다운 코드블록 없이 순수 JSON 객체 하나만 출력하세요: {"label": 0 또는 1, "reason": "한 문장 한국어 사유"}'
        )

    @staticmethod
    def format_notice_prompt(notice: Mapping[str, Any], max_content_chars: int = 2000) -> str:
        """Format a canonical notice record into the user prompt for evaluation."""
        content = _clean_cell(notice.get("content"))
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "..."

        return (
            f"ID: {_clean_cell(notice.get('id'))}\n"
            f"출처: {_clean_cell(notice.get('source'))}\n"
            f"카테고리: {_clean_cell(notice.get('category'))}\n"
            f"제목: {_clean_cell(notice.get('title'))}\n"
            f"메타태그: {_clean_cell(notice.get('meta_tags'))}\n"
            f"게시일: {_clean_cell(notice.get('posted_at'))}\n"
            f"마감일: {_clean_cell(notice.get('deadline'))}\n"
            f"본문: {content or '(본문 없음)'}"
        )

    @staticmethod
    def parse_response(raw_text: str) -> dict[str, Any]:
        """Strictly parse {'label': 0|1, 'reason': '...'} JSON from model output."""
        if not raw_text or not str(raw_text).strip():
            raise ValueError("Empty LLM judge response.")

        cleaned = str(raw_text).strip()
        fence_match = _CODE_FENCE_RE.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM judge: {cleaned!r}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from LLM judge, got {type(parsed).__name__}")

        if "label" not in parsed or "reason" not in parsed:
            raise ValueError(f"Missing required keys 'label'/'reason' in {parsed!r}")

        label = parsed["label"]
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"Judge 'label' must be integer 0 or 1, got {label!r}")

        reason = parsed["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Judge 'reason' must be a non-empty string, got {reason!r}")

        return {"label": int(label), "reason": reason.strip()}

    @staticmethod
    def _extract_text_from_message(response: Any) -> str:
        """Extract concatenated text content from an Anthropic Messages API response."""
        blocks = getattr(response, "content", None)
        if isinstance(blocks, str):
            return blocks
        if not blocks:
            return ""
        parts: list[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, Mapping):
                text = block.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(parts)

    def judge_notice(self, notice: Mapping[str, Any]) -> dict[str, Any]:
        """Call Anthropic Messages API to judge a single notice record."""
        client = self._get_client()
        user_prompt = self.format_notice_prompt(notice)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = self._extract_text_from_message(response)
        return self.parse_response(raw_text)

    def judge_dataframe(
        self,
        df: pd.DataFrame,
        output_csv: str | Path,
        id_col: str = "id",
        save_every: int = 1,
        continue_on_error: bool = False,
    ) -> pd.DataFrame:
        """Judge all rows in df with resume-capable CSV checkpointing."""
        if id_col not in df.columns:
            raise ValueError(f"Missing id column: {id_col!r}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")

        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [id_col, "judge_label", "judge_reason", "judge_model"]

        completed: dict[str, dict[str, Any]] = {}
        if out_path.is_file() and out_path.stat().st_size > 0:
            existing_df = pd.read_csv(out_path, encoding="utf-8")
            if id_col in existing_df.columns and "judge_label" in existing_df.columns:
                for row in existing_df.to_dict(orient="records"):
                    row_id = _clean_cell(row.get(id_col))
                    raw_label = row.get("judge_label")
                    if row_id and raw_label in (0, 1, "0", "1"):
                        completed[row_id] = {
                            id_col: row_id,
                            "judge_label": int(raw_label),
                            "judge_reason": _clean_cell(row.get("judge_reason")),
                            "judge_model": _clean_cell(row.get("judge_model")) or self.model,
                        }

        def _flush_checkpoint() -> None:
            ordered_rows = [completed[nid] for nid in ordered_ids if nid in completed]
            pd.DataFrame(ordered_rows, columns=columns).to_csv(
                out_path, index=False, encoding="utf-8"
            )

        ordered_ids = [_clean_cell(val) for val in df[id_col].tolist()]
        new_since_flush = 0

        for row in df.to_dict(orient="records"):
            notice_id = _clean_cell(row.get(id_col))
            if not notice_id:
                raise ValueError(f"Encountered empty {id_col!r} value in DataFrame.")
            if notice_id in completed:
                continue

            try:
                verdict = self.judge_notice(row)
            except Exception:
                if not continue_on_error:
                    if new_since_flush > 0:
                        _flush_checkpoint()
                    raise
                continue

            completed[notice_id] = {
                id_col: notice_id,
                "judge_label": verdict["label"],
                "judge_reason": verdict["reason"],
                "judge_model": self.model,
            }
            new_since_flush += 1
            if new_since_flush >= save_every:
                _flush_checkpoint()
                new_since_flush = 0

        if new_since_flush > 0 or not out_path.is_file():
            _flush_checkpoint()

        result_rows = [
            completed.get(
                nid,
                {
                    id_col: nid,
                    "judge_label": pd.NA,
                    "judge_reason": "",
                    "judge_model": self.model,
                },
            )
            for nid in ordered_ids
        ]
        return pd.DataFrame(result_rows, columns=columns)
