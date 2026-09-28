from __future__ import annotations

import json
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .arabic import normalize_arabic
from .config import Settings
from .embeddings import cosine_scores, embed


MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class SignEntry:
    gloss: str
    normalized_gloss: str
    media_path: str
    media_type: str
    category: str = ""


@dataclass(frozen=True)
class Match:
    query: str
    gloss: str | None
    score: float
    media_path: str | None
    fallback: str | None = None


class SignDictionary:
    def __init__(self, entries: list[SignEntry], vectors: np.ndarray | None = None):
        self.entries = entries
        self._vectors = vectors
        self._exact = {entry.normalized_gloss: entry for entry in entries}

    @classmethod
    def scan(cls, *roots: Path) -> "SignDictionary":
        by_gloss: dict[str, SignEntry] = {}
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                    continue
                # Word datasets normally encode the gloss as the parent directory.
                label_file = path.parent / "label.txt"
                gloss = (
                    label_file.read_text(encoding="utf-8").strip()
                    if label_file.exists()
                    else path.parent.name.strip()
                )
                normalized = normalize_arabic(gloss)
                if not normalized or normalized in by_gloss:
                    continue
                category = path.parent.parent.name if path.parent.parent != root else ""
                media_type = "video" if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else "image"
                by_gloss[normalized] = SignEntry(gloss, normalized, str(path.resolve()), media_type, category)
        return cls(list(by_gloss.values()))

    def build_embeddings(self) -> None:
        if self.entries:
            self._vectors = embed([entry.gloss for entry in self.entries])

    def match(self, query: str, threshold: float | None = None) -> Match:
        normalized = normalize_arabic(query)
        exact = self._exact.get(normalized)
        if exact:
            return Match(query, exact.gloss, 1.0, exact.media_path)
        if not self.entries:
            return Match(query, None, 0.0, None, fallback="fingerspell")
        if self._vectors is None:
            self.build_embeddings()
        scores = cosine_scores(embed(normalized)[0], self._vectors)
        index = int(np.argmax(scores))
        score = float(scores[index])
        required = threshold if threshold is not None else Settings.load().semantic_threshold
        if score < required:
            return Match(query, None, score, None, fallback="fingerspell")
        entry = self.entries[index]
        lexical_score = SequenceMatcher(None, normalized, entry.normalized_gloss).ratio()
        if lexical_score < Settings.load().lexical_similarity_threshold:
            return Match(query, None, score, None, fallback="fingerspell")
        return Match(query, entry.gloss, score, entry.media_path)

    def save_index(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is None:
            self.build_embeddings()
        (directory / "dictionary.json").write_text(
            json.dumps([asdict(item) for item in self.entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(directory / "dictionary_vectors.npy", self._vectors)

    @classmethod
    def load_index(cls, directory: Path) -> "SignDictionary":
        raw = json.loads((directory / "dictionary.json").read_text(encoding="utf-8"))
        return cls([SignEntry(**item) for item in raw], np.load(directory / "dictionary_vectors.npy"))


class AlphabetDictionary:
    """Exact character lookup kept separate from semantic word matching."""

    def __init__(self, entries: dict[str, SignEntry]):
        self.entries = entries

    @classmethod
    def scan(cls, root: Path) -> "AlphabetDictionary":
        entries: dict[str, SignEntry] = {}
        if not root.exists():
            return cls(entries)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            label_file = path.parent / "label.txt"
            if not label_file.exists():
                continue
            character = label_file.read_text(encoding="utf-8").strip()
            if character and character not in entries:
                entries[character] = SignEntry(character, character, str(path.resolve()), "video", "alphabet")
        return cls(entries)

    def spell(self, text: str) -> list[Match]:
        ignored = set(" \t\r\nـًٌٍَُِّْ،؛؟,.!?:;()[]{}\"'")
        result: list[Match] = []
        index = 0
        while index < len(text):
            character = text[index]
            if character in ignored:
                index += 1
                continue
            # Prefer KArSL's dedicated common combinations (لا and ال).
            token = text[index:index + 2]
            entry = self.entries.get(token)
            if entry:
                character = token
                index += 2
            else:
                entry = self.entries.get(character)
                index += 1
            # KArSL has the presentation form ئـ; ordinary ئ uses the same sign.
            if entry is None and character == "ئ":
                entry = self.entries.get("ئـ")
            if entry:
                result.append(Match(character, entry.gloss, 1.0, entry.media_path, fallback="fingerspell"))
            else:
                result.append(Match(character, None, 0.0, None, fallback="unmapped_character"))
        return result
