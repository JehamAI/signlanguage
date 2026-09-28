from __future__ import annotations

import re
import unicodedata


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
NON_WORD = re.compile(r"[^\w-]+", re.UNICODE)


def normalize_arabic(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").strip().lower()
    value = ARABIC_DIACRITICS.sub("", value)
    value = value.replace("ـ", "")
    value = re.sub("[إأآٱ]", "ا", value)
    value = value.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    value = NON_WORD.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str) -> list[str]:
    return [token for token in normalize_arabic(value).split() if token]
