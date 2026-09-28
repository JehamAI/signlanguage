"""Evaluate SignBart-502 recognition + full pipeline on five confident dictionary clips."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = [
    {"id": "q1", "name": "skeleton", "expected": ["هيكل عظمي"], "clips": ["0071"]},
    {"id": "q2", "name": "headache", "expected": ["صداع"], "clips": ["0115"]},
    {"id": "q3", "name": "fever", "expected": ["حمى"], "clips": ["0117"]},
    {"id": "q4", "name": "hospital", "expected": ["مستشفى"], "clips": ["0092"]},
    {"id": "q5", "name": "patient_headache_pain", "expected": ["مريض / مرض", "صداع", "ألم"], "clips": ["0134", "0115", "0116"]},
]


def clip_path(sign_id: str) -> Path:
    return ROOT / "data" / "KArSL502_dictionary" / sign_id / "sign.mp4"


def main() -> None:
    from app.karsl502_recognizer import karsl502_recognizer
    from app.karsl100_vocab import karsl100_vocabulary, karsl502_vocabulary

    try:
        from app.karsl100_recognizer import karsl100_recognizer
    except (ImportError, ModuleNotFoundError):
        karsl100_recognizer = None  # type: ignore[misc, assignment]
    from app.pipeline import ConversationPipeline
    from app.config import Settings
    from app.dictionary import AlphabetDictionary, SignDictionary
    from app.llm import LLMService
    from app.rag import LocalRAG

    rec502 = karsl502_recognizer()
    rec100 = None
    if karsl100_recognizer is not None:
        try:
            rec100 = karsl100_recognizer()
        except FileNotFoundError:
            rec100 = None
    v100 = karsl100_vocabulary()
    v502 = karsl502_vocabulary()
    settings = Settings.load()
    pipeline = ConversationPipeline(
        SignDictionary.scan(
            settings.dictionary_root,
            settings.video_root,
            settings.karsl502_dictionary_root,
            settings.karsl_dictionary_root,
        ),
        LocalRAG(settings.knowledge_root),
        LLMService(settings),
        settings.output_root,
        AlphabetDictionary.scan(settings.karsl_alphabet_root),
    )

    report: list[dict] = []
    for query in QUERIES:
        paths = [clip_path(s) for s in query["clips"]]
        missing = [str(p) for p in paths if not p.exists()]
        preds502 = []
        preds100 = []
        for path in paths:
            p502 = rec502.predict_video(path)
            preds502.append({"gloss": p502.gloss, "confidence": p502.confidence, "accepted": p502.accepted})
            if rec100:
                p100 = rec100.predict_video(path)
                preds100.append({"gloss": p100.gloss, "confidence": p100.confidence, "accepted": p100.accepted})
        glosses502 = [p["gloss"] for p in preds502]
        rec_ok502 = glosses502 == query["expected"]
        sanitized100 = v100.sanitize_glosses(glosses502)
        sanitized502 = v502.sanitize_glosses(glosses502)
        entry = {
            "id": query["id"],
            "expected": query["expected"],
            "clips": query["clips"],
            "missing_clips": missing,
            "signbart502": preds502,
            "recognition_exact_match_502": rec_ok502,
            "sanitized_for_rag_karsl100_vocab": sanitized100,
            "sanitized_karsl502_vocab": sanitized502,
            "rag_query_would_drop_glosses": glosses502 != sanitized100,
        }
        if preds100:
            entry["karsl100_bilstm"] = preds100
            entry["recognition_exact_match_100"] = [p["gloss"] for p in preds100] == query["expected"]
        if not missing and all(p["accepted"] for p in preds502):
            try:
                result = pipeline.answer_glosses(glosses502, compose_video=False)
                entry["pipeline"] = {
                    "reconstructed_question": result.get("reconstructed_question"),
                    "answer": result.get("answer"),
                    "answer_glosses": result.get("answer_glosses"),
                    "sign_language_sentence": result.get("sign_language_sentence"),
                }
            except Exception as exc:
                entry["pipeline_error"] = str(exc)
        report.append(entry)

    out = ROOT / "artifacts" / "eval_five_queries.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
