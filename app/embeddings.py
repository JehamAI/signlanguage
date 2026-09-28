from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np

from .config import Settings
from .arabic import normalize_arabic


class _LocalHashEmbedder:
    """Offline-safe lexical fallback; keeps the app running when no transformer cache exists."""

    dimensions = 384

    def encode(self, items, normalize_embeddings=True, show_progress_bar=False):
        matrix = np.zeros((len(items), self.dimensions), dtype=np.float32)
        for row, text in enumerate(items):
            value = f"  {normalize_arabic(str(text))}  "
            for width in (2, 3, 4):
                for start in range(max(0, len(value) - width + 1)):
                    token = value[start : start + width].encode("utf-8")
                    digest = hashlib.blake2b(token, digest_size=8).digest()
                    number = int.from_bytes(digest, "little")
                    matrix[row, number % self.dimensions] += 1.0 if number & 1 else -1.0
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix /= np.maximum(norms, 1e-9)
        return matrix


@lru_cache(maxsize=1)
def model():
    # This project deliberately uses the model already cached on the machine.
    # Prevent a normal startup from contacting Hugging Face or changing versions.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from sentence_transformers import SentenceTransformer

    settings = Settings.load()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if Path(settings.embedding_model).exists():
            return SentenceTransformer(settings.embedding_model, device=device)
        return SentenceTransformer(settings.embedding_model, device=device, local_files_only=True)
    except (OSError, TypeError, ValueError):
        return _LocalHashEmbedder()


def backend_name() -> str:
    loaded = model()
    return "local_hash_fallback" if isinstance(loaded, _LocalHashEmbedder) else Settings.load().embedding_model


def embed(texts: list[str] | str) -> np.ndarray:
    items = [texts] if isinstance(texts, str) else list(texts)
    return np.asarray(
        model().encode(items, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    candidates = np.asarray(candidates, dtype=np.float32)
    return candidates @ query
