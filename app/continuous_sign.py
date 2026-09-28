"""Continuous signing: motion valleys + recognition-stable word boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .browser_mp4 import write_browser_mp4
from .hand_track import HandFrameTrack, extract_hand_track, motion_valley_hand_indices
from .segmentation_analysis import MotionValleyParams, motion_valley_video_regions
from .segmenter import Segment, _read_video, _write_segments, split_on_black_pauses, split_paused_signs

if TYPE_CHECKING:
    from .karsl502_recognizer import SignBartKArSL502Recognizer


@dataclass(frozen=True)
class ContinuousSplitResult:
    segments: list[Segment]
    method: str


def _hand_ranges_from_cuts(hand_count: int, cut_indices: list[int], minimum_hand_frames: int) -> list[tuple[int, int]]:
    boundaries = [0] + sorted({int(item) for item in cut_indices if 0 < int(item) < hand_count}) + [hand_count - 1]
    regions: list[tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        left = start if not regions else max(start, regions[-1][1] + 1)
        right = end
        if right - left + 1 >= minimum_hand_frames:
            regions.append((left, right))
    if not regions and hand_count >= minimum_hand_frames:
        regions.append((0, hand_count - 1))
    return regions


def _video_regions_from_hand_ranges(track: HandFrameTrack, hand_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not len(track.hand_video_indices):
        return []
    mapped: list[tuple[int, int]] = []
    for start, end in hand_ranges:
        video_start = int(track.hand_video_indices[start])
        video_end = int(track.hand_video_indices[end])
        mapped.append((video_start, video_end))
    return mapped


def split_on_motion_valleys(
    video: Path,
    output_dir: Path,
    *,
    minimum_sign_seconds: float = 0.28,
    valley_ratio: float = 0.35,
    smooth_radius: int = 3,
    minimum_gap: int | None = None,
) -> list[Segment]:
    frames, fps = _read_video(video)
    if not frames:
        return []
    track = extract_hand_track(video)
    gap = minimum_gap if minimum_gap is not None else max(4, max(8, int(round(minimum_sign_seconds * fps))) // 2)
    params = MotionValleyParams(
        smooth_radius=smooth_radius,
        valley_ratio=valley_ratio,
        minimum_gap=gap,
        minimum_sign_seconds=minimum_sign_seconds,
    )
    _valleys, regions = motion_valley_video_regions(track, params)
    if len(regions) < 2:
        return []
    return _write_segments(frames, fps, regions, output_dir, hand_flags=None)


def _sample_window(values: np.ndarray, maximum_frames: int = 64) -> np.ndarray:
    if len(values) <= maximum_frames:
        return values
    indexes = np.linspace(0, len(values) - 1, maximum_frames).astype(int)
    return values[indexes]


def _stable_label_segments(
    recognizer: SignBartKArSL502Recognizer,
    track: HandFrameTrack,
    *,
    window_hand_frames: int = 18,
    hop_hand_frames: int = 5,
    min_stable_hops: int = 2,
    min_confidence: float = 0.42,
    motion_valley_required: bool = False,
) -> list[tuple[int, int, str, float]]:
    """Emit gloss spans when the classifier stays stable, optionally at motion valleys."""
    total = len(track.keypoints)
    if total < window_hand_frames:
        prediction = recognizer.predict_keypoints(_sample_window(track.keypoints))
        if not prediction.gloss:
            return []
        return [
            (
                int(track.hand_video_indices[0]),
                int(track.hand_video_indices[-1]),
                prediction.gloss,
                prediction.confidence,
            )
        ]

    hops: list[tuple[int, str, float]] = []
    for start in range(0, total - window_hand_frames + 1, hop_hand_frames):
        window = track.keypoints[start : start + window_hand_frames]
        prediction = recognizer.predict_keypoints(_sample_window(window))
        center = start + window_hand_frames // 2
        hops.append((center, prediction.gloss, prediction.confidence))

    valleys = set(motion_valley_hand_indices(track.motion))
    emitted: list[tuple[int, int, str, float]] = []
    index = 0
    while index < len(hops):
        center, gloss, confidence = hops[index]
        if not gloss or confidence < min_confidence:
            index += 1
            continue
        end_index = index
        best_confidence = confidence
        while end_index + 1 < len(hops):
            next_center, next_gloss, next_confidence = hops[end_index + 1]
            if next_gloss != gloss or next_confidence < min_confidence:
                break
            best_confidence = max(best_confidence, next_confidence)
            end_index += 1
        stable_windows = end_index - index + 1
        if stable_windows < min_stable_hops:
            index += 1
            continue
        hand_start = hops[index][0] - window_hand_frames // 2
        hand_end = hops[end_index][0] + window_hand_frames // 2
        hand_start = max(0, hand_start)
        hand_end = min(total - 1, hand_end)
        if motion_valley_required and emitted:
            previous_end = emitted[-1][1]
            between = [item for item in valleys if previous_end < track.hand_video_indices[item] < track.hand_video_indices[hand_start]]
            if not between and track.hand_video_indices[hand_start] - previous_end < int(0.15 * track.fps):
                index = end_index + 1
                continue
        emitted.append(
            (
                int(track.hand_video_indices[hand_start]),
                int(track.hand_video_indices[hand_end]),
                gloss,
                best_confidence,
            )
        )
        index = end_index + 1

    merged: list[tuple[int, int, str, float]] = []
    for item in emitted:
        if merged and merged[-1][2] == item[2]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], item[1]), item[2], max(merged[-1][3], item[3]))
        else:
            merged.append(item)
    return merged


def split_by_stable_recognition(
    video: Path,
    output_dir: Path,
    recognizer: SignBartKArSL502Recognizer,
) -> list[Segment]:
    """Split when SignBart agrees on a gloss, then the pose/label changes (next word)."""
    frames, fps = _read_video(video)
    if not frames:
        return []
    track = extract_hand_track(video)
    labels = _stable_label_segments(recognizer, track)
    if len(labels) < 1:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []
    for position, (start, end, _gloss, _confidence) in enumerate(labels, start=1):
        target = output_dir / f"segment-{position:02d}.mp4"
        write_browser_mp4(frames[start : end + 1], fps, target)
        segments.append(Segment(target, start, end, start / fps, (end + 1) / fps))
    return segments


def recognize_continuous_glosses(
    video: Path,
    recognizer: SignBartKArSL502Recognizer,
) -> tuple[list[str], list[float], str]:
    """Return gloss sequence without writing segment files (fast path for evaluation)."""
    track = extract_hand_track(video)
    labels = _stable_label_segments(recognizer, track)
    if len(labels) >= 2:
        return [item[2] for item in labels], [item[3] for item in labels], "stable_recognition"
    if len(labels) == 1:
        return [labels[0][2]], [labels[0][3]], "stable_recognition_single"
    return [], [], "none"


def split_continuous_signs(
    video: Path,
    output_dir: Path,
    recognizer: SignBartKArSL502Recognizer | None = None,
) -> ContinuousSplitResult:
    """Prefer demo pauses, then motion, then recognition-stable boundaries."""
    segments = split_on_black_pauses(video, output_dir)
    if len(segments) >= 2:
        return ContinuousSplitResult(segments, "black_pause")

    segments = split_paused_signs(video, output_dir)
    if len(segments) >= 2:
        return ContinuousSplitResult(segments, "hand_pause")

    segments = split_on_motion_valleys(video, output_dir)
    if len(segments) >= 2:
        return ContinuousSplitResult(segments, "motion_valley")

    if recognizer is not None:
        segments = split_by_stable_recognition(video, output_dir, recognizer)
        if segments:
            return ContinuousSplitResult(segments, "stable_recognition")

    segments = split_paused_signs(video, output_dir)
    if segments:
        return ContinuousSplitResult(segments, "hand_pause_fallback")
    return ContinuousSplitResult([], "none")
