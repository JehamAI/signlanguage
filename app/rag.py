from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arabic import normalize_arabic, tokenize
from .embeddings import embed


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


class LocalRAG:
    KARSL_KNOWLEDGE_MARK = "karsl100_closed_qa"

    def __init__(self, root: Path):
        self.chunks = self._load(root)
        self.vectors = embed([c.text for c in self.chunks]) if self.chunks else np.empty((0, 384))
        self._token_sets = [set(tokenize(c.text)) for c in self.chunks]
        self._karsl_indexes = [
            index
            for index, chunk in enumerate(self.chunks)
            if self.KARSL_KNOWLEDGE_MARK in chunk.source.lower()
        ]

    @staticmethod
    def _load(root: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        if not root.exists():
            return chunks
        for path in sorted(root.rglob("*.txt")) + sorted(root.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunks.extend(Chunk(path.name, paragraph) for paragraph in paragraphs)
        return chunks

    def _score_indexes(self, query: str, indexes: list[int], recognized_glosses: list[str] | None = None) -> np.ndarray:
        if not indexes:
            return np.asarray([], dtype=np.float32)
        semantic = self.vectors[indexes] @ embed(query)[0]
        query_tokens = set(tokenize(query))
        query_norm = normalize_arabic(query)
        recognized_norm = [normalize_arabic(g) for g in (recognized_glosses or []) if normalize_arabic(g)]
        lexical_values: list[float] = []
        for index in indexes:
            tokens = self._token_sets[index]
            chunk = self.chunks[index]
            chunk_norm = normalize_arabic(chunk.text)
            overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
            phrase_hit = 1.0 if query_norm and query_norm in chunk_norm else 0.0
            gloss_hit = sum(2.0 for gloss in recognized_norm if gloss and gloss in chunk_norm)
            lexical_values.append(overlap + phrase_hit + gloss_hit)
        lexical = np.asarray(lexical_values, dtype=np.float32)
        return semantic + (2.0 * lexical)

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        if not self.chunks:
            return []
        scores = self._score_indexes(query, list(range(len(self.chunks))))
        indexes = np.argsort(scores)[::-1][: max(1, k)]
        return [self.chunks[int(i)] for i in indexes]

    def retrieve_karsl100(
        self, query: str, recognized_glosses: list[str] | None = None, k: int = 4
    ) -> list[Chunk]:
        """Retrieve only from KArSL closed-QA knowledge (demo RAG stage)."""
        if not self._karsl_indexes:
            return self.retrieve(query, k=k)
        scores = self._score_indexes(query, self._karsl_indexes, recognized_glosses)
        order = np.argsort(scores)[::-1][: max(1, k)]
        return [self.chunks[self._karsl_indexes[int(i)]] for i in order]

    def context(self, query: str, k: int = 4) -> str:
        return "\n\n".join(f"[{c.source}] {c.text}" for c in self.retrieve(query, k))
