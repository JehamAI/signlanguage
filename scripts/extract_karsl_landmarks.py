"""Resumable MediaPipe landmark extraction for KArSL frame sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def points(landmarks, count: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((count, 3), dtype=np.float32)
    return np.asarray([(p.x, p.y, p.z) for p in landmarks.landmark], dtype=np.float32)


def normalize_pose_hands(pose: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Translation/scale normalization improves generalization between signers and camera positions.
    origin = (pose[11] + pose[12]) / 2 if np.any(pose[[11, 12]]) else np.zeros(3, dtype=np.float32)
    shoulder_width = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
    scale = max(shoulder_width, 1e-3)
    return np.concatenate(((pose - origin) / scale, (left - origin) / scale, (right - origin) / scale)).reshape(-1)


def sample_paths(sequence: Path, length: int) -> list[Path]:
    frames = sorted(sequence.glob("*.jpg"))
    if not frames:
        return []
    indices = np.linspace(0, len(frames) - 1, length).round().astype(int)
    return [frames[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "karsl100_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "KArSL100_landmarks")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Pilot limit; zero processes all sequences")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = payload["sequences"][: args.limit or None]
    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.4,
    ) as holistic:
        for number, row in enumerate(rows, 1):
            source = ROOT / row["sequence"]
            target = args.output / f"{source.name}.npz"
            if target.exists():
                completed += 1
                continue
            features, detected = [], []
            for image_path in sample_paths(source, args.frames):
                image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
                result = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pose = points(result.pose_landmarks, 33)
                left = points(result.left_hand_landmarks, 21)
                right = points(result.right_hand_landmarks, 21)
                features.append(normalize_pose_hands(pose, left, right))
                detected.append(bool(np.any(left) or np.any(right)))
            if features:
                np.savez_compressed(
                    target,
                    x=np.asarray(features, dtype=np.float32),
                    y=np.int64(row["class_index"]),
                    signer=np.int64(row["signer"]),
                    detected=np.asarray(detected, dtype=np.bool_),
                )
                completed += 1
            if number % 25 == 0:
                print(f"{number}/{len(rows)} scanned; {completed} available")
    print(f"Landmarks ready: {completed}/{len(rows)} in {args.output}")


if __name__ == "__main__":
    main()
