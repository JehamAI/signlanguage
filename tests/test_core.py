from pathlib import Path

import numpy as np

from app.arabic import normalize_arabic, tokenize
from app.dictionary import SignDictionary, SignEntry


def test_arabic_normalization():
    assert normalize_arabic("إِسْتَيْقَظَ") == "استيقظ"
    assert tokenize("أب، وأم") == ["اب", "وام"]


def test_exact_dictionary_match_does_not_load_model():
    entry = SignEntry("أب", "اب", str(Path("father.jpg")), "image", "Family")
    match = SignDictionary([entry]).match("أَب")
    assert match.gloss == "أب"
    assert match.score == 1.0


def test_empty_dictionary_falls_back():
    match = SignDictionary([]).match("مطار")
    assert match.fallback == "fingerspell"
    assert match.media_path is None

