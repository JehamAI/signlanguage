"""Evaluate continuous segmentation + SignBart on Isharah sentence clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.arabic import normalize_arabic
from app.continuous_sign import recognize_continuous_glosses, split_continuous_signs
from app.karsl100_vocab import karsl502_vocabulary
from app.karsl502_recognizer import karsl502_recognizer
from app.segmenter import split_isolated_signs


def _normalize_tokens(tokens: list[str]) -> list[str]:
    vocab = karsl502_vocabulary()
    normalized: list[str] = []
    for token in tokens:
        cleaned = token.replace("_", " ").strip()
        canonical = vocab.resolve(cleaned) or vocab.resolve(token)
        normalized.append(canonical or cleaned)
    return normalized


def _token_accuracy(expected: list[str], predicted: list[str]) -> tuple[int, int]:
    overlap = min(len(expected), len(predicted))
    correct = sum(
        normalize_arabic(expected[index]) == normalize_arabic(predicted[index]) for index in range(overlap)
    )
    return correct, max(len(expected), len(predicted))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "isharah" / "samples" / "manifest.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.manifest.exists():
        print("Manifest missing. Run: python scripts/fetch_isharah_samples.py", flush=True)
        raise SystemExit(1)

    samples = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit:
        samples = samples[: args.limit]

    recognizer = karsl502_recognizer()
    work_dir = ROOT / "artifacts" / "isharah_continuous_eval"
    work_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for sample in samples:
        video = ROOT / sample["video_path"]
        expected_raw = sample["gloss_tokens"]
        expected = _normalize_tokens(expected_raw)

        legacy_dir = work_dir / f"legacy-{sample['id']}"
        stable_glosses, stable_confidences, stable_method = recognize_continuous_glosses(video, recognizer)
        stable_predicted = _normalize_tokens(stable_glosses)

        split = split_continuous_signs(video, work_dir / f"split-{sample['id']}", recognizer)
        split_glosses: list[str] = []
        split_confidences: list[float] = []
        for segment in split.segments:
            prediction = recognizer.predict_video(segment.path)
            if prediction.gloss:
                split_glosses.append(prediction.gloss)
                split_confidences.append(prediction.confidence)
        split_predicted = _normalize_tokens(split_glosses)

        legacy_segments = split_isolated_signs(video, legacy_dir)
        legacy_glosses: list[str] = []
        for segment in legacy_segments:
            prediction = recognizer.predict_video(segment.path)
            if prediction.gloss:
                legacy_glosses.append(prediction.gloss)
        legacy_predicted = _normalize_tokens(legacy_glosses)

        stable_correct, stable_denom = _token_accuracy(expected, stable_predicted)
        split_correct, split_denom = _token_accuracy(expected, split_predicted)
        legacy_correct, legacy_denom = _token_accuracy(expected, legacy_predicted)

        record = {
            "id": sample["id"],
            "expected_tokens": expected,
            "stable_recognition": {
                "method": stable_method,
                "predicted": stable_predicted,
                "confidences": stable_confidences,
                "token_correct": stable_correct,
                "token_denominator": stable_denom,
                "count_expected": len(expected),
                "count_predicted": len(stable_predicted),
            },
            "split_pipeline": {
                "segmentation_method": split.method,
                "segment_count": len(split.segments),
                "predicted": split_predicted,
                "token_correct": split_correct,
                "token_denominator": split_denom,
            },
            "legacy_pause_split": {
                "segment_count": len(legacy_segments),
                "predicted": legacy_predicted,
                "token_correct": legacy_correct,
                "token_denominator": legacy_denom,
            },
        }
        records.append(record)
        print(
            f"{sample['id']}: expected={len(expected)} "
            f"stable={stable_correct}/{stable_denom} ({split.method} {split_correct}/{split_denom}) "
            f"legacy={legacy_correct}/{legacy_denom}",
            flush=True,
        )

    def _summarize(key: str) -> dict:
        correct = sum(record[key]["token_correct"] for record in records)
        denom = sum(record[key]["token_denominator"] for record in records)
        return {"token_correct": correct, "token_denominator": denom, "token_accuracy": correct / denom if denom else 0.0}

    summary = {
        "samples": len(records),
        "stable_recognition": _summarize("stable_recognition"),
        "split_pipeline": _summarize("split_pipeline"),
        "legacy_pause_split": _summarize("legacy_pause_split"),
        "records": records,
    }
    output = work_dir / "results.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"SUMMARY stable={summary['stable_recognition']['token_accuracy']:.1%} "
        f"split={summary['split_pipeline']['token_accuracy']:.1%} "
        f"legacy={summary['legacy_pause_split']['token_accuracy']:.1%}",
        flush=True,
    )
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
