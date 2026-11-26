"""Create leakage-free 3-way stratified splits and a 20-item Summary Eval Set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import CANONICAL_COLUMNS
from src.eval.split import stratified_three_way_split
from src.models.summarizer import TextRankSummarizer


def prepare_splits(
    raw_csv: str | Path = "data/raw_notices.csv",
    silver_csv: str | Path = "data/silver_labels.csv",
    output_dir: str | Path = "data/splits",
    test_size: int = 150,
    val_size: int = 100,
    summary_size: int = 20,
    random_state: int = 42,
) -> dict[str, int]:
    """Join raw notices with Silver labels, split 3-way, and build blind/summary sets."""
    raw_df = pd.read_csv(raw_csv, encoding="utf-8")
    silver_df = pd.read_csv(silver_csv, encoding="utf-8")

    merged = raw_df.merge(
        silver_df[["id", "judge_label", "judge_reason", "judge_model"]],
        on="id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("Merged dataset between raw_notices and silver_labels is empty.")

    total = len(merged)
    eff_test = test_size
    eff_val = val_size
    if total < 350:
        eff_test = max(1, int(round(total * 0.20)))
        eff_val = max(1, int(round(total * 0.15)))

    train_df, val_df, test_df = stratified_three_way_split(
        merged,
        label_col="judge_label",
        source_col="source",
        test_size=eff_test,
        val_size=eff_val,
        random_state=random_state,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir / "test_with_silver.csv", index=False, encoding="utf-8")
    # Blind hold-out test CSV contains ONLY the 9 canonical columns to prevent Silver leakage.
    test_df[CANONICAL_COLUMNS].to_csv(out_dir / "test_blind.csv", index=False, encoding="utf-8")

    # Select balanced multi-sentence (>= 5 sentences) notices for Summary Eval Set.
    summarizer = TextRankSummarizer()
    sentence_counts = merged["content"].fillna("").astype(str).map(
        lambda txt: len(summarizer.split_sentences(txt))
    )
    candidates = merged.loc[sentence_counts >= 5].copy()
    candidates["_sent_cnt"] = sentence_counts.loc[candidates.index]

    selected_parts: list[pd.DataFrame] = []
    sources_with_body = candidates["source"].value_counts().index.tolist()
    if sources_with_body:
        per_source = max(1, summary_size // len(sources_with_body))
        for src in sources_with_body:
            sub = candidates[candidates["source"] == src].sample(
                n=min(per_source, len(candidates[candidates["source"] == src])),
                random_state=random_state,
            )
            selected_parts.append(sub)

    selected_df = (
        pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["id"])
        if selected_parts
        else candidates.head(0)
    )
    if len(selected_df) < summary_size:
        remaining = candidates[~candidates["id"].isin(set(selected_df["id"]))]
        need = min(summary_size - len(selected_df), len(remaining))
        if need > 0:
            selected_df = pd.concat(
                [selected_df, remaining.sample(n=need, random_state=random_state)],
                ignore_index=True,
            )

    summary_df = selected_df.head(summary_size)[CANONICAL_COLUMNS].copy()
    summary_df.to_csv(out_dir / "summary_eval_20.csv", index=False, encoding="utf-8")

    return {
        "total": total,
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "summary_candidates": len(candidates),
        "summary_eval": len(summary_df),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare 3-way splits and summary eval set.")
    parser.add_argument("--raw", default="data/raw_notices.csv")
    parser.add_argument("--silver", default="data/silver_labels.csv")
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()

    stats = prepare_splits(raw_csv=args.raw, silver_csv=args.silver, output_dir=args.output_dir)
    print("Split summary:", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
