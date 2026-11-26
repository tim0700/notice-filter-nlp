"""Pluggable sentence embedding contract supporting ko-sroberta and modern Korean models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import get_device


class SentenceEmbedder:
    """Sentence embedding wrapper with CUDA/CPU auto-fallback and disk .npy caching."""

    def __init__(
        self,
        model_name: str = "jhgan/ko-sroberta-multitask",
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = get_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts into L2-normalized dense embeddings."""
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")

    def encode_cached(
        self,
        cache_key: str,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Load cached embeddings from cache_dir if present, otherwise encode and persist."""
        raise NotImplementedError("Implemented in Phase 1 (wt-nlp-core)")
