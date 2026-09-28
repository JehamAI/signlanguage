"""Build and evaluate controlled KArSL-502 conversation examples.

The resulting artifacts are intended for a reproducible proof-of-concept evaluation,
not for claiming unconstrained continuous-sign recognition.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from app.browser_mp4 import write_browser_mp4
from app.config import ROOT
from app.karsl502_recognizer import karsl502_recognizer
from app.main import pipeline
from app.segmenter import split_isolated_signs


@dataclass(frozen=True)
class Example:
    example_id: str
    title_ar: str
    glosses: tuple[str, ...]


EXAMPLES = (
    Example("airport-medical", "مسافر مريض يطلب المساعدة", ("مريض / مرض", "ألم", "يساعد")),
    Example("airport-luggage", "طلب مساعدة في حمل حقيبة سفر ثقيلة", ("حقيبة سفر", "ثقيل", "يساعد")),
    Example("library-access", "طلب مترجم لغة إشارة في المكتبة", ("إعاقة سمعية", "مترجم لغة الإشارة", "يساعد")),
    Example("restroom-directions", "السؤال عن دورة مياه قريبة", ("دورة مياه (حمام)", "قريب", "هنا")),
    Example("headache-fever", "طلب إرشاد للصداع والحمى", ("صداع", "حمى", "دواء")),
)


def _readable_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def dictionary_index() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory in sorted((ROOT / "data" / "KArSL502_dictionary").iterdir()):
        label_path = directory / "label.txt"
        video_path = directory / "sign.mp4"
        if not label_path.exists() or not video_path.exists():
            continue
        if _readable_frame_count(video_path) <= 0:
            continue
        result[label_path.read_text(encoding="utf-8").strip()] = video_path
    return result


def combine_with_neutral_pauses(paths: list[Path], output: Path, fps: int = 24) -> None:
    size = (640, 480)
    neutral = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pause = [neutral] * int(round(0.70 * fps))
    combined: list[np.ndarray] = list(pause)
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps)
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        capture.release()
        if not frames:
            raise RuntimeError(f"No frames in {path}")
        sample_count = max(1, int(round(len(frames) * fps / source_fps)))
        indexes = np.linspace(0, len(frames) - 1, sample_count).astype(int)
        combined.extend(frames[int(index)] for index in indexes)
        combined.extend(pause)
    write_browser_mp4(combined, float(fps), output)


def video_seconds(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    return round(frames / fps, 3)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Only build and test input recognition")
    parser.add_argument("--only", choices=[item.example_id for item in EXAMPLES])
    parser.add_argument("--answers-only", action="store_true", help="Reuse saved recognition and rerun RAG/LLM/output")
    args = parser.parse_args()
    root = ROOT / "paper_examples"
    input_dir = root / "input"
    output_dir = root / "output"
    segment_root = root / "segments"
    for directory in (input_dir, output_dir, segment_root):
        directory.mkdir(parents=True, exist_ok=True)

    index = dictionary_index()
    recognizer = karsl502_recognizer()
    manifest = root / "results.json"
    if args.answers_only:
        records = json.loads(manifest.read_text(encoding="utf-8"))
        for record in records:
            recognized = [str(x) for x in record.get("recognized_glosses", [])]
            if not recognized:
                continue
            result = pipeline().answer_glosses(recognized, compose_video=True)
            generated = Path(result["video_path"]) if result.get("video_path") else None
            if generated and generated.exists():
                final_video = output_dir / f"{record['example_id']}-answer.mp4"
                shutil.copy2(generated, final_video)
                result["video_path"] = str(final_video.relative_to(ROOT))
                result["output_seconds"] = video_seconds(final_video)
            record["pipeline"] = result
            print(f"{record['example_id']}: answer={result['answer']}")
        manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {manifest}")
        return

    records: list[dict] = []
    selected = [item for item in EXAMPLES if not args.only or item.example_id == args.only]
    for example in selected:
        sources = [index[gloss] for gloss in example.glosses]
        input_path = input_dir / f"{example.example_id}.mp4"
        combine_with_neutral_pauses(sources, input_path)
        segment_dir = segment_root / example.example_id
        shutil.rmtree(segment_dir, ignore_errors=True)
        segments = split_isolated_signs(input_path, segment_dir)
        predictions = [recognizer.predict_video(item.path) for item in segments]
        recognized = [item.gloss for item in predictions if item.gloss]
        record = {
            "example_id": example.example_id,
            "title_ar": example.title_ar,
            "expected_glosses": list(example.glosses),
            "input_video": str(input_path.relative_to(ROOT)),
            "input_seconds": video_seconds(input_path),
            "segments": [asdict(item) | {"path": str(item.path.relative_to(ROOT))} for item in segments],
            "predictions": [asdict(item) for item in predictions],
            "recognized_glosses": recognized,
            "exact_recognition": recognized == list(example.glosses),
        }
        if not args.skip_llm and recognized:
            result = pipeline().answer_glosses(recognized, compose_video=True)
            generated = Path(result["video_path"]) if result.get("video_path") else None
            if generated and generated.exists():
                final_video = output_dir / f"{example.example_id}-answer.mp4"
                shutil.copy2(generated, final_video)
                result["video_path"] = str(final_video.relative_to(ROOT))
                result["output_seconds"] = video_seconds(final_video)
            record["pipeline"] = result
        records.append(record)
        print(f"{example.example_id}: expected={list(example.glosses)} recognized={recognized}")

    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
