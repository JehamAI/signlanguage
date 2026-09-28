"""Boundary detection for deliberately paused isolated signs in one video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .browser_mp4 import write_browser_mp4


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


def _remove_short_runs(mask: np.ndarray, minimum_length: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in _runs(result, minimum_frames=1):
        if end - start + 1 < minimum_length:
            result[start : end + 1] = False
    return result


def _read_video(video: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def _frame_is_black(frame: np.ndarray, mean_threshold: float = 14.0) -> bool:
    return float(frame.mean()) <= mean_threshold


def _hand_active_flags(frames: list[np.ndarray]) -> np.ndarray:
    flags: list[bool] = []
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as holistic:
        for frame in frames:
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            flags.append(result.left_hand_landmarks is not None or result.right_hand_landmarks is not None)
    return np.asarray(flags, dtype=bool)


def _trim_to_hand_activity(start: int, end: int, hand_flags: np.ndarray, margin: int = 2) -> tuple[int, int] | None:
    region = hand_flags[start : end + 1]
    active = np.flatnonzero(region)
    if not len(active):
        return None
    left = max(start, start + int(active[0]) - margin)
    right = min(end, start + int(active[-1]) + margin)
    return left, right


def _write_segments(
    frames: list[np.ndarray],
    fps: float,
    regions: list[tuple[int, int]],
    output_dir: Path,
    hand_flags: np.ndarray | None,
) -> list[Segment]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []
    index = 0
    for start, end in regions:
        if hand_flags is not None:
            trimmed = _trim_to_hand_activity(start, end, hand_flags)
            if trimmed is None:
                continue
            start, end = trimmed
        index += 1
        target = output_dir / f"segment-{index:02d}.mp4"
        write_browser_mp4(frames[start : end + 1], fps, target)
        segments.append(Segment(target, start, end, start / fps, (end + 1) / fps))
    return segments


def split_on_black_pauses(
    video: Path,
    output_dir: Path,
    *,
    black_mean_threshold: float = 14.0,
    minimum_pause_seconds: float = 0.40,
    minimum_sign_seconds: float = 0.25,
    trim_to_hands: bool = True,
) -> list[Segment]:
    """Split on sustained black frames (demo composites with neutral pauses)."""
    frames, fps = _read_video(video)
    if not frames:
        return []

    black = np.array([_frame_is_black(frame, black_mean_threshold) for frame in frames])
    pause_frames = max(5, int(round(minimum_pause_seconds * fps)))
    black = _remove_short_runs(black, pause_frames)
    content = ~black
    min_sign_frames = max(6, int(round(minimum_sign_seconds * fps)))
    regions = _runs(content, minimum_frames=min_sign_frames)
    if len(regions) < 2:
        return []

    hand_flags = _hand_active_flags(frames) if trim_to_hands else None
    return _write_segments(frames, fps, regions, output_dir, hand_flags)


def split_paused_signs(video: Path, output_dir: Path, minimum_pause_seconds: float = 0.32) -> list[Segment]:
    """Split on sustained intervals where neither hand is detected."""
    frames, fps = _read_video(video)
    if not frames:
        return []

    hand_flags = _hand_active_flags(frames)
    pause_frames = max(4, int(round(minimum_pause_seconds * fps)))
    mask = _close_short_gaps(hand_flags, pause_frames - 1)
    regions = _runs(mask, minimum_frames=max(8, int(round(0.30 * fps))))
    return _write_segments(frames, fps, regions, output_dir, hand_flags)


def split_isolated_signs(video: Path, output_dir: Path) -> list[Segment]:
    """Split a controlled multi-sign upload for per-word recognition.

    Prefer explicit black pauses (merged dictionary demos). Fall back to hand pauses
    for live recordings without black separators.
    """
    segments = split_on_black_pauses(video, output_dir)
    if len(segments) >= 2:
        return segments
    return split_paused_signs(video, output_dir)
