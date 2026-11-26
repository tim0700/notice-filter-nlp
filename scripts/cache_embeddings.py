"""Pre-compute and cache sentence embeddings on GPU/CPU for reproducible fast experiments."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_profile
from src.models.embedder import SentenceEmbedder
from src.models.scorer import extract_texts_from_df


def _encode_model_suite(
    embedder: SentenceEmbedder,
    tag: str,
    raw_df: pd.DataFrame,
    modes: tuple[str, ...],
    pos_queries: list[str],
    neg_queries: list[str],
    batch_size: int,
) -> dict[str, Any]:
    """Encode all requested text modes and profile queries for one embedding model."""
    info: dict[str, Any] = {
        "model_name": embedder.model_name,
        "device": embedder.device,
        "modes": {},
    }
    for mode in modes:
        texts = extract_texts_from_df(raw_df, text_mode=mode)
        t0 = time.perf_counter()
        vecs = embedder.encode_cached(
            f"raw_{mode}_{tag}",
            texts,
            batch_size=batch_size,
        )
        info["modes"][mode] = {
            "cache_key": f"raw_{mode}_{tag}",
            "shape": list(vecs.shape),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    t_q = time.perf_counter()
    pos_vecs = embedder.encode_cached(f"profile_pos_{tag}", pos_queries, batch_size=batch_size)
    neg_vecs = embedder.encode_cached(f"profile_neg_{tag}", neg_queries, batch_size=batch_size)
    info["profile_queries"] = {
        "pos_shape": list(pos_vecs.shape),
        "neg_shape": list(neg_vecs.shape),
        "elapsed_sec": round(time.perf_counter() - t_q, 2),
    }
    return info


def main() -> int:
    profile = load_profile()
    raw_df = pd.read_csv(_REPO_ROOT / "data" / "raw_notices.csv", encoding="utf-8")
    notices = raw_df.to_dict(orient="records")
    pos_queries = list(profile["interests"]["positive_queries"])
    neg_queries = list(profile["interests"]["negative_queries"])

    cache_dir = _REPO_ROOT / "data" / "embeddings"
    results: dict[str, Any] = {"total_notices": len(notices)}

    # 1. Lecture baseline model: jhgan/ko-sroberta-multitask
    baseline_name = str(profile["models"]["baseline_embedding"])
    baseline_emb = SentenceEmbedder(model_name=baseline_name, cache_dir=cache_dir)
    results["baseline"] = _encode_model_suite(
        baseline_emb,
        tag="kosroberta",
        raw_df=raw_df,
        modes=("title", "meta_title", "full"),
        pos_queries=pos_queries,
        neg_queries=neg_queries,
        batch_size=64,
    )

    # 2. Modern Korean embedding: try KURE-v1 first, fall back to BGE-M3
    modern_candidates = [
        str(profile["models"]["modern_embedding"]),
        str(profile["models"]["fallback_modern_embedding"]),
    ]
    last_err: Exception | None = None
    for candidate in modern_candidates:
        try:
            modern_emb = SentenceEmbedder(model_name=candidate, cache_dir=cache_dir)
            results["modern"] = _encode_model_suite(
                modern_emb,
                tag="modern",
                raw_df=raw_df,
                modes=("meta_title", "full"),
                pos_queries=pos_queries,
                neg_queries=neg_queries,
                batch_size=16,
            )
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            print(f"[WARN] Failed loading modern embedding '{candidate}': {exc}")

    if last_err is not None:
        raise RuntimeError(f"All modern embedding candidates failed: {modern_candidates}") from last_err

    out_json = _REPO_ROOT / "results" / "embedding_cache_info.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
