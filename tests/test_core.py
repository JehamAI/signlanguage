from pathlib import Path

import numpy as np

from app.arabic import normalize_arabic, tokenize
from app.dictionary import AlphabetDictionary, SignDictionary, SignEntry
from app.karsl100_vocab import karsl100_vocabulary, karsl502_vocabulary


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


def test_semantic_neighbor_also_requires_lexical_safety():
    entry = SignEntry("شهيد", "شهيد", str(Path("martyr.mp4")), "video", "KArSL")
    dictionary = SignDictionary([entry], np.asarray([[1.0, 0.0]], dtype=np.float32))
    import app.dictionary as dictionary_module
    original = dictionary_module.embed
    dictionary_module.embed = lambda _: np.asarray([[1.0, 0.0]], dtype=np.float32)
    try:
        match = dictionary.match("شديد", threshold=0.90)
    finally:
        dictionary_module.embed = original
    assert match.fallback == "fingerspell"
    assert match.gloss is None


def test_alphabet_dictionary_spells_each_character():
    entries = {
        "م": SignEntry("م", "م", "m.mp4", "video", "alphabet"),
        "ن": SignEntry("ن", "ن", "n.mp4", "video", "alphabet"),
    }
    matches = AlphabetDictionary(entries).spell("مَن")
    assert [match.gloss for match in matches] == ["م", "ن"]
    assert all(match.fallback == "fingerspell" for match in matches)


def test_input_and_output_vocabularies_are_separate():
    assert len(karsl100_vocabulary().labels) == 100
    assert len(karsl502_vocabulary().labels) == 463
    assert karsl502_vocabulary().resolve("طبيب") == "طبيب"
    assert karsl502_vocabulary().resolve("ا") is None
