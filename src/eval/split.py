"""Stratified three-way dataset splitting with tiny-stratum fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _resolve_split_count(size: int | float, total: int, name: str) -> int:
    """Convert an integer count or fractional proportion into a validated row count."""
    if isinstance(size, float):
        if not 0.0 < size < 1.0:
            raise ValueError(f"{name} as float must be in (0, 1), got {size}")
        return max(1, int(round(total * size)))
    count = int(size)
    if count < 1:
        raise ValueError(f"{name} must be >= 1, got {count}")
    return count


def _choose_stratify_array(
    df_subset: pd.DataFrame,
    label_col: str,
    source_col: str,
    split_count: int,
) -> np.ndarray | None:
    """Build a valid stratification key array, gracefully falling back for tiny strata."""
    n_rows = len(df_subset)
    remaining = n_rows - split_count
    if split_count <= 0 or remaining <= 0:
        return None

    labels = df_subset[label_col].astype(str)
    if source_col in df_subset.columns:
        sources = df_subset[source_col].fillna("unknown").astype(str)
        combined = sources + "__" + labels
        counts = combined.value_counts()
        # Merge strata with fewer than 2 samples into a per-label fallback bucket.
        merged = combined.where(combined.map(counts) >= 2, "__other__" + "__" + labels)
        merged_counts = merged.value_counts()
        if (
            int(merged_counts.min()) >= 2
            and merged.nunique() <= split_count
            and merged.nunique() <= remaining
        ):
            return merged.to_numpy()

    label_counts = labels.value_counts()
    if (
        int(label_counts.min()) >= 2
        and labels.nunique() <= split_count
        and labels.nunique() <= remaining
    ):
        return labels.to_numpy()

    return None


def stratified_three_way_split(
    df: pd.DataFrame,
    label_col: str = "label",
    source_col: str = "source",
    test_size: int | float = 150,
    val_size: int | float = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into strictly disjoint (train_df, val_df, test_df) preserving strata."""
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col!r}")

    total = len(df)
    if total < 3:
        raise ValueError(f"Need at least 3 rows for a 3-way split, got {total}")

    test_count = _resolve_split_count(test_size, total, "test_size")
    val_count = _resolve_split_count(val_size, total, "val_size")
    if test_count + val_count >= total:
        raise ValueError(
            f"test_size ({test_count}) + val_size ({val_count}) must be < len(df) ({total})"
        )

    positions = np.arange(total)
    strata_test = _choose_stratify_array(df, label_col, source_col, test_count)
    train_val_pos, test_pos = train_test_split(
        positions,
        test_size=test_count,
        stratify=strata_test,
        random_state=random_state,
    )

    train_val_df = df.iloc[train_val_pos]
    strata_val = _choose_stratify_array(train_val_df, label_col, source_col, val_count)
    train_rel_pos, val_rel_pos = train_test_split(
        np.arange(len(train_val_pos)),
        test_size=val_count,
        stratify=strata_val,
        random_state=random_state,
    )

    train_pos = train_val_pos[train_rel_pos]
    val_pos = train_val_pos[val_rel_pos]

    return (
        df.iloc[train_pos].copy(),
        df.iloc[val_pos].copy(),
        df.iloc[test_pos].copy(),
    )
