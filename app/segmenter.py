"""Boundary detection for deliberately paused isolated signs in one video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class Segment:
    path: Path
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float


def _close_short_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = mask.copy()
    inactive = np.flatnonzero(~mask)
    if not len(inactive):
        return result
    start = 0
    while start < len(inactive):
        end = start
        while end + 1 < len(inactive) and inactive[end + 1] == inactive[end] + 1:
            end += 1
        left, right = inactive[start] - 1, inactive[end] + 1
        if end - start + 1 <= maximum_gap and left >= 0 and right < len(mask) and mask[left] and mask[right]:
            result[inactive[start : end + 1]] = True
        start = end + 1
    return result


def _runs(mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= minimum_frames]


def split_paused_signs(video: Path, output_dir: Path, minimum_pause_seconds: float = 0.32) -> list[Segment]:
    """Split on sustained intervals where neither hand is detected.

    This is intentionally conservative: it supports a controlled signer who returns to a
    neutral pose between words. It is not a general continuous-sign segmentation claim.
    """
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    active: list[bool] = []
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
            frames.append(frame)
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            active.append(result.left_hand_landmarks is not None or result.right_hand_landmarks is not None)
    capture.release()
    if not frames:
        return []

    # Bridge brief detector misses inside a sign, but retain deliberate neutral pauses.
    pause_frames = max(4, int(round(minimum_pause_seconds * fps)))
    mask = _close_short_gaps(np.asarray(active, dtype=bool), pause_frames - 1)
    regions = _runs(mask, minimum_frames=max(8, int(round(0.30 * fps))))
    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []
    height, width = frames[0].shape[:2]
    for index, (start, end) in enumerate(regions, 1):
        start, end = max(0, start - 2), min(len(frames) - 1, end + 2)
        target = output_dir / f"segment-{index:02d}.mp4"
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create segmented video")
        for frame in frames[start : end + 1]:
            writer.write(frame)
        writer.release()
        segments.append(Segment(target, start, end, start / fps, (end + 1) / fps))
    return segments
