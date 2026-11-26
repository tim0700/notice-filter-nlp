"""Canonical notice and user-feedback data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.config import CANONICAL_COLUMNS, FEEDBACK_COLUMNS

REQUIRED_NON_EMPTY_FIELDS = ("id", "source", "title", "url")


@dataclass(slots=True)
class NoticeRecord:
    """9-column canonical representation for a collected notice or welfare post."""

    id: str
    source: str
    category: str
    title: str
    meta_tags: str
    posted_at: str
    url: str
    content: str = ""
    deadline: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return fields ordered strictly by CANONICAL_COLUMNS."""
        raw = asdict(self)
        return {col: str(raw.get(col, "") or "") for col in CANONICAL_COLUMNS}


@dataclass(slots=True)
class FeedbackRecord:
    """5-column user feedback event persisted to data/feedback.csv."""

    notice_id: str
    label: int
    source: str
    timestamp: str
    comment: str = ""

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"Feedback label must be 0 or 1, got {self.label!r}")

    def to_dict(self) -> dict[str, str | int]:
        """Return fields ordered strictly by FEEDBACK_COLUMNS."""
        return {
            "notice_id": self.notice_id,
            "label": int(self.label),
            "source": self.source,
            "timestamp": self.timestamp,
            "comment": self.comment,
        }


def normalize_notices_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical column order and replace NaN strings with empty strings."""
    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")
    normalized = df[CANONICAL_COLUMNS].copy()
    for col in CANONICAL_COLUMNS:
        normalized[col] = normalized[col].fillna("").astype(str)
    return normalized


def validate_notices_df(df: pd.DataFrame) -> None:
    """Validate schema, required non-empty fields, and unique notice IDs."""
    normalized = normalize_notices_df(df)
    for col in REQUIRED_NON_EMPTY_FIELDS:
        empty_mask = normalized[col].str.strip() == ""
        if empty_mask.any():
            bad_indices = normalized.index[empty_mask].tolist()
            raise ValueError(f"Column '{col}' has empty values at rows {bad_indices}")

    if normalized["id"].duplicated().any():
        dup_ids = normalized.loc[normalized["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicate notice IDs detected: {dup_ids}")
