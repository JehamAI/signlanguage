"""Extract MediaPipe pose/hand/lip landmarks for every Isharah video (parallel, resumable)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "data" / "isharah" / "selfi"
POSE_ROOT = ROOT / "data" / "isharah" / "pose"
MAX_SIDE = 640


def _lip_indices() -> list[int]:
    import mediapipe as mp

    return sorted({index for edge in mp.solutions.face_mesh.FACEMESH_LIPS for index in edge})


def _extract(job: tuple[str, str]) -> tuple[str, int, str]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    video, target = Path(job[0]), Path(job[1])
    if target.exists():
        return job[0], -1, "cached"
    import cv2
    import mediapipe as mp

    lips = _lip_indices()
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    rows: list[np.ndarray] = []
    flags: list[tuple[bool, bool, bool]] = []
    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        ) as holistic:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                scale = MAX_SIDE / max(height, width)
                if scale < 1:
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                points = np.zeros((115, 2), dtype=np.float32)
                if result.pose_landmarks:
                    points[:33] = [(p.x, p.y) for p in result.pose_landmarks.landmark]
                if result.left_hand_landmarks:
                    points[33:54] = [(p.x, p.y) for p in result.left_hand_landmarks.landmark]
                if result.right_hand_landmarks:
                    points[54:75] = [(p.x, p.y) for p in result.right_hand_landmarks.landmark]
                if result.face_landmarks:
                    face = result.face_landmarks.landmark
                    points[75:115] = [(face[i].x, face[i].y) for i in lips]
                rows.append(points)
                flags.append(
                    (
                        result.pose_landmarks is not None,
                        result.left_hand_landmarks is not None,
                        result.right_hand_landmarks is not None,
                    )
                )
    except Exception as exc:  # noqa: BLE001 - one corrupt clip must not stop the batch
        return job[0], 0, f"error: {exc}"
    finally:
        capture.release()
    if not rows:
        return job[0], 0, "empty"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        keypoints=np.asarray(rows, dtype=np.float16),
        flags=np.asarray(flags, dtype=bool),
        fps=np.float32(fps),
    )
    return job[0], len(rows), "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    videos = sorted(VIDEO_ROOT.glob("Group*/*/*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    jobs = [
        (str(video), str(POSE_ROOT / video.relative_to(VIDEO_ROOT).with_suffix(".npz")))
        for video in videos
    ]
    started = time.time()
    done = failures = 0
    with Pool(args.workers) as pool:
        for video, frames, status in pool.imap_unordered(_extract, jobs, chunksize=2):
            done += 1
            if status not in ("ok", "cached"):
                failures += 1
                print(f"FAIL {video}: {status}", flush=True)
            if done % 100 == 0 or done == len(jobs):
                rate = done / max(time.time() - started, 1e-6)
                remaining = (len(jobs) - done) / max(rate, 1e-6)
                print(f"{done}/{len(jobs)} failures={failures} eta={remaining / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
