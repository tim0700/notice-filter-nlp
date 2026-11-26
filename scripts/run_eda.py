"""Exploratory Data Analysis & Korean WordCloud generation using strictly Train Pool (zero test leakage)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import koreanize_matplotlib  # noqa: F401 - registers Korean font for matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.summarizer import TextRankSummarizer
from src.preprocess.stopwords import load_stopwords
from src.preprocess.text import compose_notice_text, extract_tfidf_keywords
from src.preprocess.wordcloud_viz import generate_wordcloud

SOURCE_LABELS: dict[str, str] = {
    "swcon_academic": "SW융합 학사/장학",
    "swcon_career": "SW융합 취업/특강",
    "goodjob_extracurricular": "미래인재 비교과(국제)",
    "goodjob_general": "미래인재 일반채용",
    "goodjob_recommend": "미래인재 추천채용",
    "seongnam_youth": "성남청년다해",
    "youth_api": "온통청년 API",
}


def run_eda() -> dict[str, Any]:
    raw_df = pd.read_csv(_REPO_ROOT / "data" / "raw_notices.csv", encoding="utf-8")
    train_df = pd.read_csv(_REPO_ROOT / "data" / "splits" / "train.csv", encoding="utf-8")

    fig_dir = _REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    # 1. Source distribution & multi-sentence counts (raw_notices.csv - unlabeled)
    summarizer = TextRankSummarizer()
    raw_df["sent_count"] = raw_df["content"].fillna("").astype(str).map(
        lambda t: len(summarizer.split_sentences(t))
    )
    src_counts = raw_df["source"].value_counts()
    multi_by_src = raw_df.groupby("source")["sent_count"].apply(lambda s: int((s >= 5).sum()))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = [SOURCE_LABELS.get(s, s) for s in src_counts.index]
    bars = ax.barh(labels[::-1], src_counts.values[::-1], color="#3B82F6", alpha=0.88)
    for bar, src_key in zip(bars, src_counts.index[::-1]):
        cnt = int(bar.get_width())
        multi = int(multi_by_src.get(src_key, 0))
        note = f" {cnt}건 (본문>=5문장: {multi}건)" if multi > 0 else f" {cnt}건"
        ax.text(cnt + 2, bar.get_y() + bar.get_height() / 2, note, va="center", fontsize=9.5)
    ax.set_xlim(0, max(src_counts.values) * 1.32)
    ax.set_xlabel("수집 공지 건수")
    ax.set_title(f"수집 소스별 공지 현황 (총 {len(raw_df)}건, 본문>=5문장 {int((raw_df['sent_count'] >= 5).sum())}건)")
    plt.tight_layout()
    fig.savefig(fig_dir / "01_source_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Train Pool (n=729) Positive vs Negative by Source (strictly train.csv)
    ct = pd.crosstab(train_df["source"], train_df["judge_label"]).reindex(
        columns=[1, 0], fill_value=0
    )
    ct["total"] = ct[1] + ct[0]
    ct = ct.sort_values("total", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y_labels = [SOURCE_LABELS.get(s, s) for s in ct.index]
    ax.barh(y_labels, ct[1].values, label="관심 (Positive=1)", color="#10B981")
    ax.barh(y_labels, ct[0].values, left=ct[1].values, label="제외 (Negative=0)", color="#94A3B8")
    for idx, (pos, neg) in enumerate(zip(ct[1].values, ct[0].values)):
        ax.text(pos + neg + 1.5, idx, f"Pos {pos} / Neg {neg}", va="center", fontsize=9)
    ax.set_xlim(0, ct["total"].max() * 1.28)
    ax.set_xlabel("Train Pool 건수")
    ax.set_title(f"Train Pool(n={len(train_df)}) 소스별 관심/제외 분포 (LLM Silver)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(fig_dir / "02_train_label_by_source.png", dpi=150)
    plt.close(fig)

    # 3. Top-15 TF-IDF Keywords (Okt + Stopwords) on Train Pool Positive vs Negative
    pos_texts = [
        compose_notice_text(r, mode="meta_title")
        for r in train_df[train_df["judge_label"] == 1].to_dict(orient="records")
    ]
    neg_texts = [
        compose_notice_text(r, mode="meta_title")
        for r in train_df[train_df["judge_label"] == 0].to_dict(orient="records")
    ]
    pos_kw = extract_tfidf_keywords(pos_texts, top_k=15)
    neg_kw = extract_tfidf_keywords(neg_texts, top_k=15)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for ax, kws, title, color in (
        (axes[0], pos_kw, "관심(Positive) 공지 Top-15 TF-IDF 키워드", "#10B981"),
        (axes[1], neg_kw, "제외(Negative) 공지 Top-15 TF-IDF 키워드", "#EF4444"),
    ):
        words = [w for w, _ in kws][::-1]
        scores = [float(s) for _, s in kws][::-1]
        ax.barh(words, scores, color=color, alpha=0.88)
        ax.set_title(title)
        ax.set_xlabel("TF-IDF 합산 가중치")
    fig.suptitle("Train Pool 기반 Okt 형태소 TF-IDF 핵심 키워드 비교", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(fig_dir / "03_tfidf_keywords_pos_vs_neg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4 & 5. Korean WordClouds (Positive vs Negative)
    stopwords = load_stopwords()
    generate_wordcloud(pos_texts, fig_dir / "04_wordcloud_positive.png", stopwords=stopwords)
    generate_wordcloud(neg_texts, fig_dir / "05_wordcloud_negative.png", stopwords=stopwords)

    summary = {
        "raw_total": int(len(raw_df)),
        "raw_by_source": {str(k): int(v) for k, v in src_counts.items()},
        "multi_sentence_ge5": int((raw_df["sent_count"] >= 5).sum()),
        "train_total": int(len(train_df)),
        "train_pos": int((train_df["judge_label"] == 1).sum()),
        "train_neg": int((train_df["judge_label"] == 0).sum()),
        "top15_positive_keywords": [[w, round(float(s), 4)] for w, s in pos_kw],
        "top15_negative_keywords": [[w, round(float(s), 4)] for w, s in neg_kw],
        "figures": [
            "results/figures/01_source_distribution.png",
            "results/figures/02_train_label_by_source.png",
            "results/figures/03_tfidf_keywords_pos_vs_neg.png",
            "results/figures/04_wordcloud_positive.png",
            "results/figures/05_wordcloud_negative.png",
        ],
    }
    out_json = _REPO_ROOT / "results" / "eda_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()
