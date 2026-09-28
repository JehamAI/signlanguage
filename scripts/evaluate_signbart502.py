"""Evaluate SignBart on one isolated local dictionary clip per KArSL-502 class."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.karsl502_recognizer import karsl502_recognizer


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = ROOT / "artifacts" / "karsl502_isolated_evaluation.json"
    records: list[dict] = []
    if args.resume and output.exists():
        records = json.loads(output.read_text(encoding="utf-8"))
    completed = {int(record["sign_id"]) for record in records}

    root = ROOT / "data" / "KArSL502_dictionary"
    candidates = sorted(root.iterdir(), key=lambda path: int(path.name))
    if args.limit:
        candidates = candidates[: args.limit]
    recognizer = karsl502_recognizer()

    for directory in candidates:
        sign_id = int(directory.name)
        if sign_id in completed:
            continue
        label_path = directory / "label.txt"
        video_path = directory / "sign.mp4"
        expected = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
        prediction = recognizer.predict_video(video_path)
        record = {
            "sign_id": sign_id,
            "expected": expected,
            "predicted": prediction.gloss,
            "correct": prediction.gloss == expected,
            "readable": prediction.sampled_frames > 0,
            **asdict(prediction),
        }
        records.append(record)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "PASS" if record["correct"] else "FAIL"
        print(
            f"{sign_id:04d} {status}: {expected} -> {prediction.gloss} "
            f"({prediction.confidence:.3f}, frames={prediction.sampled_frames})",
            flush=True,
        )

    evaluated = [record for record in records if record["readable"]]
    correct = sum(bool(record["correct"]) for record in evaluated)
    unreadable = sum(not bool(record["readable"]) for record in records)
    accuracy = correct / len(evaluated) if evaluated else 0.0
    print(
        f"SUMMARY: correct={correct}/{len(evaluated)} "
        f"top1_accuracy={accuracy:.2%} unreadable={unreadable}",
        flush=True,
    )


if __name__ == "__main__":
    main()
