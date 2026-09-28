from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_ENV = Path(
    r"C:\Users\Jeham\evalia\ai-interviewer_AM_june10\dev2-ai-interviewer-gpu\src\.env"
)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    shared_env: Path = Path(os.getenv("SIGN_SHARED_ENV", str(DEFAULT_SHARED_ENV)))
    embedding_model: str = os.getenv(
        "SIGN_EMBEDDING_MODEL", str(ROOT / "artifacts" / "embedding" / "all-MiniLM-L6-v2")
    )
    llm_model: str = "gpt-5.6-luna"
    llm_api_key: str | None = None
    dictionary_root: Path = ROOT / "data" / "ARSLW" / "Dataset"
    video_root: Path = ROOT / "data" / "arabic_words"
    karsl_dictionary_root: Path = ROOT / "data" / "KArSL100_dictionary"
    karsl502_dictionary_root: Path = ROOT / "data" / "KArSL502_dictionary"
    karsl_alphabet_root: Path = ROOT / "data" / "KArSL_alphabet"
    knowledge_root: Path = ROOT / "knowledge"
    artifact_root: Path = ROOT / "artifacts"
    output_root: Path = ROOT / "outputs"
    semantic_threshold: float = 0.94
    lexical_similarity_threshold: float = 0.86
    visual_threshold: float = 0.60
    pretrained_word_threshold: float = 0.45

    @classmethod
    def load(cls) -> "Settings":
        # Values are read at runtime from the existing file; secrets are never copied here.
        shared_path = Path(os.getenv("SIGN_SHARED_ENV", str(DEFAULT_SHARED_ENV)))
        values = dotenv_values(shared_path) if shared_path.exists() else {}
        return cls(
            shared_env=shared_path,
            embedding_model=os.getenv(
                "SIGN_EMBEDDING_MODEL", str(ROOT / "artifacts" / "embedding" / "all-MiniLM-L6-v2")
            ),
            llm_model=os.getenv("SIGN_LLM_MODEL", values.get("LLM_MODEL") or "gpt-5.6-luna"),
            llm_api_key=os.getenv("OPENAI_API_KEY") or values.get("OPENAI_API_KEY"),
            dictionary_root=Path(os.getenv("SIGN_DICTIONARY_ROOT", str(ROOT / "data" / "ARSLW" / "Dataset"))),
            video_root=Path(os.getenv("SIGN_VIDEO_ROOT", str(ROOT / "data" / "arabic_words"))),
            karsl_dictionary_root=Path(
                os.getenv("SIGN_KARSL_DICTIONARY_ROOT", str(ROOT / "data" / "KArSL100_dictionary"))
            ),
            karsl502_dictionary_root=Path(
                os.getenv("SIGN_KARSL502_DICTIONARY_ROOT", str(ROOT / "data" / "KArSL502_dictionary"))
            ),
            karsl_alphabet_root=Path(
                os.getenv("SIGN_KARSL_ALPHABET_ROOT", str(ROOT / "data" / "KArSL_alphabet"))
            ),
            knowledge_root=Path(os.getenv("SIGN_KNOWLEDGE_ROOT", str(ROOT / "knowledge"))),
            semantic_threshold=float(os.getenv("SIGN_SEMANTIC_THRESHOLD", "0.94")),
            lexical_similarity_threshold=float(os.getenv("SIGN_LEXICAL_THRESHOLD", "0.86")),
            visual_threshold=float(os.getenv("SIGN_VISUAL_THRESHOLD", "0.60")),
            pretrained_word_threshold=float(os.getenv("SIGN_WORD_THRESHOLD", "0.45")),
        )
