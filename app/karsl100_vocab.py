from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .arabic import normalize_arabic
from .config import ROOT


@dataclass(frozen=True)
class KArSL100Vocabulary:
    labels: tuple[str, ...]
    _by_norm: dict[str, str]

    @classmethod
    def load(cls, manifest_path: Path | None = None) -> "KArSL100Vocabulary":
        path = manifest_path or (ROOT / "data" / "karsl100_manifest.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = tuple(str(item["arabic"]).strip() for item in payload["classes"])
        by_norm = {normalize_arabic(label): label for label in labels}
        return cls(labels=labels, _by_norm=by_norm)

    def resolve(self, value: str) -> str | None:
        return self._by_norm.get(normalize_arabic(value))

    def sanitize_glosses(self, glosses: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in glosses:
            canonical = self.resolve(item)
            if canonical and canonical not in cleaned:
                cleaned.append(canonical)
        return cleaned

    def extract_glosses_from_text(self, text: str) -> list[str]:
        """Pull KArSL-100 labels from Arabic text (longest phrases first)."""
        norm_text = normalize_arabic(text)
        if not norm_text:
            return []
        ordered = sorted(self.labels, key=len, reverse=True)
        found: list[tuple[int, str]] = []
        for label in ordered:
            norm_label = normalize_arabic(label)
            if not norm_label:
                continue
            start = norm_text.find(norm_label)
            if start >= 0:
                found.append((start, label))
        found.sort(key=lambda pair: pair[0])
        glosses: list[str] = []
        for _, label in found:
            if label not in glosses:
                glosses.append(label)
        return glosses

    def sanitize_text_to_glosses(self, text: str, llm_glosses: list[str] | None = None) -> list[str]:
        from_llm = self.sanitize_glosses(llm_glosses or [])
        if from_llm:
            return from_llm
        return self.extract_glosses_from_text(text)


@lru_cache(maxsize=1)
def karsl100_vocabulary() -> KArSL100Vocabulary:
    return KArSL100Vocabulary.load()


@lru_cache(maxsize=1)
def karsl502_vocabulary() -> KArSL100Vocabulary:
    """Semantic output vocabulary; alphabet IDs 32-70 are reserved for fallback spelling."""
    from openpyxl import load_workbook

    workbook = ROOT / "references" / "karsl_word_recognition" / "KARSL-502_Labels.xlsx"
    sheet = load_workbook(workbook, read_only=True, data_only=True).active
    labels = tuple(
        str(row[1]).strip()
        for row in sheet.iter_rows(min_row=2, values_only=True)
        if row[0] is not None and row[1] is not None and not 32 <= int(row[0]) <= 70
    )
    return KArSL100Vocabulary(labels=labels, _by_norm={normalize_arabic(x): x for x in labels})
