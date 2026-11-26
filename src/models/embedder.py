"""Pluggable sentence embedding contract supporting ko-sroberta and modern Korean models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import find_repo_root, get_device

SUPPORTED_MODELS: tuple[str, ...] = (
    "jhgan/ko-sroberta-multitask",
    "nlpai-lab/KURE-v1",
    "BAAI/bge-m3",
)


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
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazily load the underlying SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows safely without producing NaNs for zero vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts into dense (optionally L2-normalized) embeddings."""
        if not texts:
            dim = 0
            if self._model is not None and hasattr(self._model, "get_sentence_embedding_dimension"):
                dim = int(self._model.get_sentence_embedding_dimension() or 0)
            return np.empty((0, dim), dtype=np.float32)

        clean_texts = ["" if t is None else str(t) for t in texts]
        model = self._load_model()
        raw = model.encode(
            clean_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(raw, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if normalize:
            embeddings = self._l2_normalize(embeddings)
        return embeddings

    @staticmethod
    def _compute_texts_digest(texts: list[str]) -> str:
        """Compute deterministic SHA-256 hex digest over ordered input texts."""
        payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def encode_cached(
        self,
        cache_key: str,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Load cached embeddings from cache_dir if fresh, otherwise encode and persist."""
        clean_texts = ["" if t is None else str(t) for t in texts]
        base_dir = self.cache_dir if self.cache_dir is not None else find_repo_root() / "data" / "cache"
        stem = cache_key[:-4] if cache_key.endswith(".npy") else cache_key
        npy_path = base_dir / f"{stem}.npy"
        meta_path = base_dir / f"{stem}.meta.json"
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        texts_sha256 = self._compute_texts_digest(clean_texts)

        if npy_path.is_file() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("model_name") == self.model_name
                    and meta.get("texts_sha256") == texts_sha256
                    and meta.get("normalize") == normalize
                ):
                    cached = np.load(npy_path)
                    if cached.ndim == 2 and cached.shape[0] == len(clean_texts):
                        return np.asarray(cached, dtype=np.float32)
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        embeddings = self.encode(clean_texts, normalize=normalize, batch_size=batch_size)
        np.save(npy_path, embeddings)
        meta_payload = {
            "model_name": self.model_name,
            "texts_sha256": texts_sha256,
            "normalize": normalize,
            "num_texts": len(clean_texts),
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return embeddings
