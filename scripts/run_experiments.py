"""Controlled experiment runner supporting leakage-free Val tuning and Final Gold evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import koreanize_matplotlib  # noqa: F401 - Korean font support
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_profile
from src.eval.metrics import (
    evaluate_classification,
    evaluate_judge_agreement,
    evaluate_korean_rouge,
)
from src.models.embedder import SentenceEmbedder
from src.models.scorer import (
    CosineProfileScorer,
    EmbeddingLogRegScorer,
    RocchioScorer,
    TfidfLogRegScorer,
    extract_texts_from_df,
)
from src.models.summarizer import TextRankSummarizer


class LookupEmbedder:
    """Fast lookup wrapper over pre-cached .npy vectors with live SentenceEmbedder fallback."""

    def __init__(
        self,
        model_name: str,
        lookup: dict[str, np.ndarray] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self._lookup: dict[str, np.ndarray] = lookup or {}
        self._cache_dir = cache_dir
        self._live: SentenceEmbedder | None = None

    def register(self, texts: list[str], vectors: np.ndarray) -> None:
        for text, vec in zip(texts, vectors):
            self._lookup[str(text)] = np.asarray(vec, dtype=np.float32)

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        missing_indices = [i for i, t in enumerate(texts) if str(t) not in self._lookup]
        if missing_indices:
            if self._live is None:
                self._live = SentenceEmbedder(model_name=self.model_name, cache_dir=self._cache_dir)
            missing_texts = [str(texts[i]) for i in missing_indices]
            encoded = self._live.encode(missing_texts, normalize=normalize, batch_size=batch_size)
            self.register(missing_texts, encoded)

        stacked = np.vstack([self._lookup[str(t)] for t in texts]).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(stacked, axis=1, keepdims=True)
            stacked = np.divide(stacked, norms, out=np.zeros_like(stacked), where=norms > 0)
        return stacked


def _build_lookup_embedder(
    model_name: str,
    tag: str,
    modes: tuple[str, ...],
    raw_df: pd.DataFrame,
    profile: dict[str, Any],
    cache_dir: Path,
) -> LookupEmbedder:
    """Load pre-computed raw notice and profile query caches into a LookupEmbedder."""
    emb = LookupEmbedder(model_name=model_name, cache_dir=cache_dir)
    live = SentenceEmbedder(model_name=model_name, cache_dir=cache_dir)
    batch_size = 16 if "kure" in model_name.lower() or "bge" in model_name.lower() else 64

    for mode in modes:
        texts = extract_texts_from_df(raw_df, text_mode=mode)
        vecs = live.encode_cached(f"raw_{mode}_{tag}", texts, batch_size=batch_size)
        emb.register(texts, vecs)

    pos_q = list(profile["interests"]["positive_queries"])
    neg_q = list(profile["interests"]["negative_queries"])
    emb.register(pos_q, live.encode_cached(f"profile_pos_{tag}", pos_q, batch_size=batch_size))
    emb.register(neg_q, live.encode_cached(f"profile_neg_{tag}", neg_q, batch_size=batch_size))
    return emb


def _resolve_modern_model_name(profile: dict[str, Any]) -> str:
    info_path = _REPO_ROOT / "results" / "embedding_cache_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        modern = info.get("modern", {}).get("model_name")
        if modern:
            return str(modern)
    return str(profile["models"]["modern_embedding"])


def _sample_few_shot(
    train_df: pd.DataFrame,
    n_samples: int,
    label_col: str = "judge_label",
    random_state: int = 42,
) -> pd.DataFrame:
    if n_samples >= len(train_df):
        return train_df.copy()
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    idx, _ = next(splitter.split(train_df, train_df[label_col]))
    return train_df.iloc[idx].copy()


def _sweep_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    best_thresh = float(thresholds[0])
    best_metrics: dict[str, Any] = {"f1": -1.0}
    for thr in thresholds:
        preds = (scores >= float(thr)).astype(int)
        metrics = evaluate_classification(y_true, preds)
        if (metrics["f1"], metrics["accuracy"]) > (best_metrics["f1"], best_metrics.get("accuracy", -1.0)):
            best_metrics = metrics
            best_thresh = float(round(float(thr), 4))
    return best_thresh, best_metrics


def run_val_tuning() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune all experiment stages strictly on train.csv and val.csv (zero test leakage)."""
    profile = load_profile()
    raw_df = pd.read_csv(_REPO_ROOT / "data" / "raw_notices.csv", encoding="utf-8")
    train_df = pd.read_csv(_REPO_ROOT / "data" / "splits" / "train.csv", encoding="utf-8")
    val_df = pd.read_csv(_REPO_ROOT / "data" / "splits" / "val.csv", encoding="utf-8")
    y_val = val_df["judge_label"].astype(int).to_numpy()

    cache_dir = _REPO_ROOT / "data" / "embeddings"
    base_name = str(profile["models"]["baseline_embedding"])
    modern_name = _resolve_modern_model_name(profile)

    base_emb = _build_lookup_embedder(
        base_name, "kosroberta", ("title", "meta_title", "full"), raw_df, profile, cache_dir
    )
    modern_emb = _build_lookup_embedder(
        modern_name, "modern", ("meta_title", "full"), raw_df, profile, cache_dir
    )

    pos_q = list(profile["interests"]["positive_queries"])
    neg_q = list(profile["interests"]["negative_queries"])
    val_rows: list[dict[str, Any]] = []
    cos_thresholds = np.arange(0.15, 0.76, 0.02)
    prob_thresholds = np.arange(0.30, 0.71, 0.05)

    # Exp 0: Baseline Cold-Start (title, threshold=0.45, no negative penalty)
    exp0_scorer = CosineProfileScorer(base_emb, pos_q, negative_queries=None, text_mode="title")
    exp0_scores = exp0_scorer.score(val_df)
    exp0_metrics = evaluate_classification(y_val, (exp0_scores >= 0.45).astype(int))
    val_rows.append({"exp": "Exp 0 (Baseline: ko-sroberta title@0.45)", **exp0_metrics, "threshold": 0.45})

    # Exp 1: Input mode (meta_title vs full) + negative penalty + Val threshold sweep
    best_exp1: dict[str, Any] = {"f1": -1.0}
    for mode in ("meta_title", "full"):
        for use_neg in (False, True):
            scorer = CosineProfileScorer(
                base_emb,
                pos_q,
                negative_queries=neg_q if use_neg else None,
                text_mode=mode,
                neg_weight=0.25,
            )
            thr, m = _sweep_threshold(y_val, scorer.score(val_df), cos_thresholds)
            if (m["f1"], m["accuracy"]) > (best_exp1["f1"], best_exp1.get("accuracy", -1.0)):
                best_exp1 = {
                    "text_mode": mode,
                    "use_neg": use_neg,
                    "threshold": thr,
                    **m,
                }
    best_mode = str(best_exp1["text_mode"])
    val_rows.append({"exp": f"Exp 1 (ko-sroberta {best_mode}, tuned thr)", **{k: v for k, v in best_exp1.items() if k in {"accuracy", "precision", "recall", "f1", "confusion_matrix", "threshold"}}})

    # Exp 2: Few-shot Rocchio (n=50) hyperparameter sweep on Val
    fb50_df = _sample_few_shot(train_df, 50, random_state=42)
    fb50_texts = extract_texts_from_df(fb50_df, text_mode=best_mode)
    fb50_labels = fb50_df["judge_label"].astype(int).tolist()
    best_exp2: dict[str, Any] = {"f1": -1.0}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        for beta in (0.1, 0.25, 0.5):
            scorer = RocchioScorer(base_emb, pos_q, alpha=alpha, beta=beta, text_mode=best_mode)
            scorer.fit(fb50_texts, fb50_labels)
            thr, m = _sweep_threshold(y_val, scorer.score(val_df), cos_thresholds)
            if (m["f1"], m["accuracy"]) > (best_exp2["f1"], best_exp2.get("accuracy", -1.0)):
                best_exp2 = {"alpha": alpha, "beta": beta, "threshold": thr, "n_feedback": 50, **m}
    val_rows.append({"exp": "Exp 2 (Rocchio n=50, ko-sroberta)", **{k: v for k, v in best_exp2.items() if k in {"accuracy", "precision", "recall", "f1", "confusion_matrix", "threshold"}}})

    # Exp 3a: Lecture Baseline Control (TF-IDF + LogReg on full train)
    train_texts = extract_texts_from_df(train_df, text_mode=best_mode)
    train_labels = train_df["judge_label"].astype(int).tolist()
    best_exp3a: dict[str, Any] = {"f1": -1.0}
    for c_val in (0.1, 0.5, 1.0, 2.0, 5.0):
        scorer = TfidfLogRegScorer(c_value=c_val, text_mode=best_mode).fit(train_texts, train_labels)
        thr, m = _sweep_threshold(y_val, scorer.score(val_df), prob_thresholds)
        if (m["f1"], m["accuracy"]) > (best_exp3a["f1"], best_exp3a.get("accuracy", -1.0)):
            best_exp3a = {"c_value": c_val, "threshold": thr, **m}
    val_rows.append({"exp": "Exp 3a (TF-IDF + LogReg, control)", **{k: v for k, v in best_exp3a.items() if k in {"accuracy", "precision", "recall", "f1", "confusion_matrix", "threshold"}}})

    # Exp 3b: Cumulative Embedding + LogReg (ko-sroberta on full train)
    best_exp3b: dict[str, Any] = {"f1": -1.0}
    for c_val in (0.1, 0.5, 1.0, 2.0, 5.0):
        scorer = EmbeddingLogRegScorer(base_emb, c_value=c_val, text_mode=best_mode).fit(train_texts, train_labels)
        thr, m = _sweep_threshold(y_val, scorer.score(val_df), prob_thresholds)
        if (m["f1"], m["accuracy"]) > (best_exp3b["f1"], best_exp3b.get("accuracy", -1.0)):
            best_exp3b = {"c_value": c_val, "threshold": thr, **m}
    val_rows.append({"exp": "Exp 3b (ko-sroberta + LogReg, full)", **{k: v for k, v in best_exp3b.items() if k in {"accuracy", "precision", "recall", "f1", "confusion_matrix", "threshold"}}})

    # Exp 4a: Modern Embedding Cold-Start Cosine
    best_exp4a: dict[str, Any] = {"f1": -1.0}
    for use_neg in (False, True):
        scorer = CosineProfileScorer(
            modern_emb,
            pos_q,
            negative_queries=neg_q if use_neg else None,
            text_mode=best_mode,
            neg_weight=0.25,
        )
        thr, m = _sweep_threshold(y_val, scorer.score(val_df), cos_thresholds)
        if (m["f1"], m["accuracy"]) > (best_exp4a["f1"], best_exp4a.get("accuracy", -1.0)):
            best_exp4a = {"use_neg": use_neg, "threshold": thr, **m}
    val_rows.append({"exp": f"Exp 4a ({modern_name.split('/')[-1]} Cosine)", **{k: v for k, v in best_exp4a.items() if k in {"accuracy", "precision", "recall", "f1", "confusion_matrix", "threshold"}}})

    # Exp 4b: Modern Embedding Few-Shot Rocchio (n=50)
    best_exp4b: dict[str, Any] = {"f1": -1.0}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        for beta in (0.1, 0.25, 0.5):
            scorer = RocchioScorer(modern_emb, pos_q, alpha=alpha, beta=beta, text_mode=best_mode)
            scorer.fit(fb50_texts, fb50_labels)
            thr, m = _sweep_threshold(y_val, scorer.score(val_df), cos_thresholds)
            if (m["f1"], m["accuracy"]) > (best_exp4b["f1"], best_exp4b.get("accuracy", -1.0)):
                best_exp4b = {"alpha": alpha, "beta": beta, "threshold": thr, **m}
    val_rows.append({"exp": f"Exp 4b ({modern_name.split('/')[-1]} Rocchio n=50)", **{k: v for k, v in best_exp4b.items() if k in {"accuracy", "precision", "recall", "f1", "confusion_matrix", "threshold"}}})

    # Exp 4c: Modern Embedding + LogReg (full train)
    best_exp4c: dict[str, Any] = {"f1": -1.0}
    for c_val in (0.1, 0.5, 1.0, 2.0, 5.0):
        scorer = EmbeddingLogRegScorer(modern_emb, c_value=c_val, text_mode=best_mode).fit(train_texts, train_labels)
        thr, m = _sweep_threshold(y_val, scorer.score(val_df), prob_thresholds)
        if (m["f1"], m["accuracy"]) > (best_exp4c["f1"], best_exp4c.get("accuracy", -1.0)):
            best_exp4c = {"c_value": c_val, "threshold": thr, **m}
    val_rows.append({"exp": f"Exp 4c ({modern_name.split('/')[-1]} + LogReg, full)", **{k: v for k, v in best_exp4c.items() if k in {"accuracy", "precision", "recall", "f1", "confusion_matrix", "threshold"}}})

    tuned_cfg = {
        "baseline_model": base_name,
        "modern_model": modern_name,
        "exp1": {"text_mode": best_mode, "use_neg": best_exp1["use_neg"], "threshold": best_exp1["threshold"]},
        "exp2": {"alpha": best_exp2["alpha"], "beta": best_exp2["beta"], "threshold": best_exp2["threshold"], "n_feedback": 50},
        "exp3a": {"c_value": best_exp3a["c_value"], "threshold": best_exp3a["threshold"]},
        "exp3b": {"c_value": best_exp3b["c_value"], "threshold": best_exp3b["threshold"]},
        "exp4a": {"use_neg": best_exp4a["use_neg"], "threshold": best_exp4a["threshold"]},
        "exp4b": {"alpha": best_exp4b["alpha"], "beta": best_exp4b["beta"], "threshold": best_exp4b["threshold"], "n_feedback": 50},
        "exp4c": {"c_value": best_exp4c["c_value"], "threshold": best_exp4c["threshold"]},
    }

    res_dir = _REPO_ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "val_tuned_config.json").write_text(json.dumps(tuned_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (res_dir / "val_results.json").write_text(json.dumps(val_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return tuned_cfg, val_rows


def run_final_eval() -> dict[str, Any]:
    """Evaluate all Val-locked configurations ONCE on Hold-out Human Gold Test Set & Summary Eval."""
    profile = load_profile()
    raw_df = pd.read_csv(_REPO_ROOT / "data" / "raw_notices.csv", encoding="utf-8")
    train_df = pd.read_csv(_REPO_ROOT / "data" / "splits" / "train.csv", encoding="utf-8")
    test_silver_df = pd.read_csv(_REPO_ROOT / "data" / "splits" / "test_with_silver.csv", encoding="utf-8")
    gold_path = _REPO_ROOT / "data" / "splits" / "test_gold.csv"
    sum_eval_path = _REPO_ROOT / "data" / "splits" / "summary_eval_20.csv"
    sum_gold_path = _REPO_ROOT / "data" / "splits" / "summary_eval_gold.csv"

    if not gold_path.is_file():
        raise FileNotFoundError(f"Human Gold file not found: {gold_path}")

    gold_df = pd.read_csv(gold_path, encoding="utf-8")
    test_df = test_silver_df.merge(gold_df[["id", "human_label"]], on="id", how="inner")
    if len(test_df) != len(test_silver_df):
        raise ValueError(f"Expected {len(test_silver_df)} Human Gold labels, got {len(test_df)}")

    y_gold = test_df["human_label"].astype(int).to_numpy()
    y_silver = test_df["judge_label"].astype(int).to_numpy()
    judge_agreement = evaluate_judge_agreement(y_gold, y_silver)

    cfg_path = _REPO_ROOT / "results" / "val_tuned_config.json"
    if cfg_path.is_file():
        tuned_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        tuned_cfg, _ = run_val_tuning()

    cache_dir = _REPO_ROOT / "data" / "embeddings"
    base_emb = _build_lookup_embedder(
        tuned_cfg["baseline_model"], "kosroberta", ("title", "meta_title", "full"), raw_df, profile, cache_dir
    )
    modern_emb = _build_lookup_embedder(
        tuned_cfg["modern_model"], "modern", ("meta_title", "full"), raw_df, profile, cache_dir
    )

    pos_q = list(profile["interests"]["positive_queries"])
    neg_q = list(profile["interests"]["negative_queries"])
    best_mode = str(tuned_cfg["exp1"]["text_mode"])
    train_texts = extract_texts_from_df(train_df, text_mode=best_mode)
    train_labels = train_df["judge_label"].astype(int).tolist()
    fb50_df = _sample_few_shot(train_df, 50, random_state=42)
    fb50_texts = extract_texts_from_df(fb50_df, text_mode=best_mode)
    fb50_labels = fb50_df["judge_label"].astype(int).tolist()

    specs: list[tuple[str, Any, float]] = [
        ("Exp 0: Baseline (ko-sroberta title@0.45)", CosineProfileScorer(base_emb, pos_q, text_mode="title"), 0.45),
        (
            f"Exp 1: Input/Thresh ({best_mode})",
            CosineProfileScorer(
                base_emb, pos_q, negative_queries=neg_q if tuned_cfg["exp1"]["use_neg"] else None, text_mode=best_mode, neg_weight=0.25
            ),
            float(tuned_cfg["exp1"]["threshold"]),
        ),
        (
            "Exp 2: Rocchio (ko-sroberta, n=50)",
            RocchioScorer(base_emb, pos_q, alpha=tuned_cfg["exp2"]["alpha"], beta=tuned_cfg["exp2"]["beta"], text_mode=best_mode).fit(fb50_texts, fb50_labels),
            float(tuned_cfg["exp2"]["threshold"]),
        ),
        (
            "Exp 3a: TF-IDF + LogReg (Control)",
            TfidfLogRegScorer(c_value=tuned_cfg["exp3a"]["c_value"], text_mode=best_mode).fit(train_texts, train_labels),
            float(tuned_cfg["exp3a"]["threshold"]),
        ),
        (
            "Exp 3b: ko-sroberta + LogReg (n=729)",
            EmbeddingLogRegScorer(base_emb, c_value=tuned_cfg["exp3b"]["c_value"], text_mode=best_mode).fit(train_texts, train_labels),
            float(tuned_cfg["exp3b"]["threshold"]),
        ),
        (
            f"Exp 4a: {tuned_cfg['modern_model'].split('/')[-1]} Cosine",
            CosineProfileScorer(
                modern_emb, pos_q, negative_queries=neg_q if tuned_cfg["exp4a"]["use_neg"] else None, text_mode=best_mode, neg_weight=0.25
            ),
            float(tuned_cfg["exp4a"]["threshold"]),
        ),
        (
            f"Exp 4b: {tuned_cfg['modern_model'].split('/')[-1]} Rocchio (n=50)",
            RocchioScorer(modern_emb, pos_q, alpha=tuned_cfg["exp4b"]["alpha"], beta=tuned_cfg["exp4b"]["beta"], text_mode=best_mode).fit(fb50_texts, fb50_labels),
            float(tuned_cfg["exp4b"]["threshold"]),
        ),
        (
            f"Exp 4c: {tuned_cfg['modern_model'].split('/')[-1]} + LogReg (n=729)",
            EmbeddingLogRegScorer(modern_emb, c_value=tuned_cfg["exp4c"]["c_value"], text_mode=best_mode).fit(train_texts, train_labels),
            float(tuned_cfg["exp4c"]["threshold"]),
        ),
    ]

    test_results: list[dict[str, Any]] = []
    cached_preds: dict[str, tuple[np.ndarray, np.ndarray, list[list[int]]]] = {}
    for name, scorer, thr in specs:
        scores = scorer.score(test_df)
        preds = (scores >= thr).astype(int)
        m = evaluate_classification(y_gold, preds)
        test_results.append({"exp": name, "threshold": thr, **m})
        cached_preds[name] = (scores, preds, m["confusion_matrix"])

    # Export Baseline (Exp 0) and Final Best errors for Jira Bug creation
    exp0_scores, exp0_preds, _ = cached_preds[specs[0][0]]
    err_df = test_df[["id", "source", "category", "title", "meta_tags", "human_label", "judge_label", "judge_reason", "url"]].copy()
    err_df["exp0_score"] = np.round(exp0_scores, 4)
    err_df["exp0_pred"] = exp0_preds
    err_df["exp0_error_type"] = np.where(
        (err_df["exp0_pred"] == 1) & (err_df["human_label"] == 0),
        "FP",
        np.where((err_df["exp0_pred"] == 0) & (err_df["human_label"] == 1), "FN", "OK"),
    )
    err_only = err_df[err_df["exp0_error_type"] != "OK"].copy()
    err_only.to_csv(_REPO_ROOT / "results" / "error_cases.csv", index=False, encoding="utf-8")

    # Learning curve over feedback sizes
    learning_curve: list[dict[str, Any]] = []
    for n_fb in (20, 50, 100, 200, 400, len(train_df)):
        sub_df = _sample_few_shot(train_df, n_fb, random_state=42)
        sub_texts = extract_texts_from_df(sub_df, text_mode=best_mode)
        sub_labels = sub_df["judge_label"].astype(int).tolist()
        for label_name, clf, thr in (
            ("TF-IDF + LogReg", TfidfLogRegScorer(c_value=tuned_cfg["exp3a"]["c_value"], text_mode=best_mode).fit(sub_texts, sub_labels), float(tuned_cfg["exp3a"]["threshold"])),
            ("ko-sroberta + LogReg", EmbeddingLogRegScorer(base_emb, c_value=tuned_cfg["exp3b"]["c_value"], text_mode=best_mode).fit(sub_texts, sub_labels), float(tuned_cfg["exp3b"]["threshold"])),
            (f"{tuned_cfg['modern_model'].split('/')[-1]} + LogReg", EmbeddingLogRegScorer(modern_emb, c_value=tuned_cfg["exp4c"]["c_value"], text_mode=best_mode).fit(sub_texts, sub_labels), float(tuned_cfg["exp4c"]["threshold"])),
        ):
            preds = clf.predict(test_df, threshold=thr)
            m = evaluate_classification(y_gold, preds)
            learning_curve.append({"n_feedback": n_fb, "model": label_name, "f1": m["f1"], "accuracy": m["accuracy"]})

    # Korean ROUGE evaluation on 20-item Summary Eval Set if Human Gold exists
    rouge_results: dict[str, Any] = {}
    if sum_eval_path.is_file() and sum_gold_path.is_file():
        s_eval = pd.read_csv(sum_eval_path, encoding="utf-8")
        s_gold = pd.read_csv(sum_gold_path, encoding="utf-8")
        s_merged = s_eval.merge(s_gold[["id", "human_summary"]], on="id", how="inner")
        summarizer = TextRankSummarizer(top_k=3, min_sentences=5)
        preds = [summarizer.summarize(str(t)) for t in s_merged["content"].fillna("")]
        refs = [str(r) for r in s_merged["human_summary"].fillna("")]
        rouge_results = {
            "count": len(s_merged),
            "space": evaluate_korean_rouge(refs, preds, tokenizer="space"),
            "okt": evaluate_korean_rouge(refs, preds, tokenizer="okt"),
        }

    # Figures 06-08
    fig_dir = _REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    # 06: Experiment F1 comparison
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    exp_names = [r["exp"] for r in test_results][::-1]
    exp_f1s = [r["f1"] for r in test_results][::-1]
    bars = ax.barh(exp_names, exp_f1s, color="#2563EB", alpha=0.88)
    for bar in bars:
        val = bar.get_width()
        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Hold-out Test Positive F1 (Human Gold n=150)")
    ax.set_title("단계적 개선 실험별 Positive F1 비교")
    plt.tight_layout()
    fig.savefig(fig_dir / "06_experiment_f1_comparison.png", dpi=150)
    plt.close(fig)

    # 07: Baseline vs Best Confusion Matrix
    best_row = max(test_results, key=lambda r: (r["f1"], r["accuracy"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, title_key in ((axes[0], specs[0][0]), (axes[1], best_row["exp"])):
        cm = np.array(cached_preds[title_key][2])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["제외(0)", "관심(1)"], yticklabels=["제외(0)", "관심(1)"])
        ax.set_xlabel("예측 라벨")
        ax.set_ylabel("실제 정답 (Human Gold)")
        ax.set_title(title_key, fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_dir / "07_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # 08: Learning Curve
    lc_df = pd.DataFrame(learning_curve)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for model_label, grp in lc_df.groupby("model"):
        ax.plot(grp["n_feedback"], grp["f1"], marker="o", linewidth=2, label=model_label)
    ax.set_xlabel("누적 사용자 피드백 건수 (Train Pool)")
    ax.set_ylabel("Hold-out Test Positive F1")
    ax.set_title("피드백 누적 건수에 따른 분류기 F1 학습 곡선")
    ax.set_ylim(0, 1.02)
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "08_feedback_learning_curve.png", dpi=150)
    plt.close(fig)

    summary = {
        "judge_agreement_on_test150": judge_agreement,
        "test_results": test_results,
        "best_experiment": best_row,
        "exp0_error_count": int(len(err_only)),
        "learning_curve": learning_curve,
        "rouge_eval": rouge_results,
    }
    (_REPO_ROOT / "results" / "final_test_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Val tuning or Final Gold evaluation.")
    parser.add_argument("--mode", choices=["val-only", "final-eval"], default="val-only")
    args = parser.parse_args()
    if args.mode == "val-only":
        _, val_rows = run_val_tuning()
        print("=== Validation Set Tuning Summary (n=100) ===")
        for row in val_rows:
            print(
                f"- {row['exp']}: F1={row['f1']:.4f} | P={row['precision']:.4f} | R={row['recall']:.4f} | Acc={row['accuracy']:.4f} | thr={row['threshold']}"
            )
    else:
        run_final_eval()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
