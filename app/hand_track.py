"""Single-pass MediaPipe hand/pose tracks for continuous signing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .karsl502_recognizer import PARTS, _normalize_part, _points


@dataclass(frozen=True)
class HandFrameTrack:
    """Hand-active frames extracted from one video."""

    fps: float
    video_frame_count: int
    hand_video_indices: tuple[int, ...]
    keypoints: np.ndarray
    motion: np.ndarray


def extract_hand_track(video: Path) -> HandFrameTrack:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    keypoint_rows: list[np.ndarray] = []
    hand_indices: list[int] = []
    video_index = -1
    previous: np.ndarray | None = None
    motion_values: list[float] = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as holistic:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            video_index += 1
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.left_hand_landmarks is None and result.right_hand_landmarks is None:
                continue
            pose = _points(result.pose_landmarks, 33)
            left = _points(result.left_hand_landmarks, 21)
            right = _points(result.right_hand_landmarks, 21)
            left, right = right, left
            points = np.concatenate((pose, left, right), axis=0)
            for part in PARTS:
                points[part] = _normalize_part(points[part])
            hand_indices.append(video_index)
            keypoint_rows.append(points)
            if previous is None:
                motion_values.append(0.0)
            else:
                motion_values.append(float(np.linalg.norm(points - previous)))
            previous = points

    capture.release()
    video_frames = video_index + 1
    if not keypoint_rows:
        return HandFrameTrack(fps, max(video_frames, 0), (), np.empty((0, 75, 2), np.float32), np.empty(0, np.float32))
    return HandFrameTrack(
        fps,
        video_frames,
        tuple(hand_indices),
        np.asarray(keypoint_rows, dtype=np.float32),
        np.asarray(motion_values, dtype=np.float32),
    )


def motion_valley_hand_indices(
    motion: np.ndarray,
    *,
    smooth_radius: int = 3,
    valley_ratio: float = 0.35,
    minimum_gap: int = 6,
) -> list[int]:
    """Return hand-frame indexes where motion energy dips (candidate sign boundaries)."""
    if len(motion) < minimum_gap * 3:
        return []
    kernel = 2 * smooth_radius + 1
    padded = np.pad(motion, (smooth_radius, smooth_radius), mode="edge")
    smoothed = np.convolve(padded, np.ones(kernel) / kernel, mode="valid")
    threshold = float(np.quantile(smoothed, valley_ratio))
    valleys: list[int] = []
    for index in range(1, len(smoothed) - 1):
        if smoothed[index] > threshold:
            continue
        if smoothed[index] <= smoothed[index - 1] and smoothed[index] <= smoothed[index + 1]:
            if valleys and index - valleys[-1] < minimum_gap:
                if smoothed[index] < smoothed[valleys[-1]]:
                    valleys[-1] = index
                continue
            valleys.append(index)
    return valleys
