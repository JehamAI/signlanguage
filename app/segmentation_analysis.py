"""Extract motion/hand signals and deterministic cut proposals (no recognizer)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hand_track import HandFrameTrack, extract_hand_track, motion_valley_hand_indices
from .segmenter import _close_short_gaps, _hand_active_flags, _read_video, _runs, _trim_to_hand_activity


@dataclass(frozen=True)
class MotionValleyParams:
    smooth_radius: int = 3
    valley_ratio: float = 0.35
    minimum_gap: int = 6
    minimum_sign_seconds: float = 0.28


def smooth_motion(motion: np.ndarray, smooth_radius: int = 3) -> np.ndarray:
    if len(motion) == 0:
        return motion
    kernel = 2 * smooth_radius + 1
    padded = np.pad(motion, (smooth_radius, smooth_radius), mode="edge")
    return np.convolve(padded, np.ones(kernel) / kernel, mode="valid")


def _hand_ranges_from_cuts(hand_count: int, cut_indices: list[int], minimum_hand_frames: int) -> list[tuple[int, int]]:
    if hand_count < minimum_hand_frames:
        return []
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


def video_regions_from_hand_ranges(track: HandFrameTrack, hand_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not len(track.hand_video_indices):
        return []
    return [
        (int(track.hand_video_indices[start]), int(track.hand_video_indices[end])) for start, end in hand_ranges
    ]


def motion_valley_video_regions(track: HandFrameTrack, params: MotionValleyParams) -> tuple[list[int], list[tuple[int, int]]]:
    """Return valley indexes (hand timeline) and segment spans (video frame indexes)."""
    if len(track.keypoints) < 12:
        return [], []
    min_hand = max(8, int(round(params.minimum_sign_seconds * track.fps)))
    gap = params.minimum_gap if params.minimum_gap > 0 else max(4, min_hand // 2)
    valleys = motion_valley_hand_indices(
        track.motion,
        smooth_radius=params.smooth_radius,
        valley_ratio=params.valley_ratio,
        minimum_gap=gap,
    )
    hand_ranges = _hand_ranges_from_cuts(len(track.keypoints), valleys, min_hand)
    return valleys, video_regions_from_hand_ranges(track, hand_ranges)


def hand_pause_video_regions(video: Path, *, minimum_pause_seconds: float = 0.32) -> list[tuple[int, int]]:
    frames, fps = _read_video(video)
    if not frames:
        return []
    hand_flags = _hand_active_flags(frames)
    pause_frames = max(4, int(round(minimum_pause_seconds * fps)))
    mask = _close_short_gaps(hand_flags, pause_frames - 1)
    regions = _runs(mask, minimum_frames=max(8, int(round(0.30 * fps))))
    trimmed: list[tuple[int, int]] = []
    for start, end in regions:
        bounds = _trim_to_hand_activity(start, end, hand_flags)
        if bounds is None:
            continue
        trimmed.append(bounds)
    return trimmed


def hand_active_on_video_timeline(track: HandFrameTrack) -> np.ndarray:
    flags = np.zeros(track.video_frame_count, dtype=bool)
    for index in track.hand_video_indices:
        if 0 <= index < len(flags):
            flags[index] = True
    return flags


def timeline_payload(track: HandFrameTrack, params: MotionValleyParams) -> dict:
    """Serializable signals for plotting and manual calibration."""
    smoothed = smooth_motion(track.motion, params.smooth_radius)
    valleys, regions = motion_valley_video_regions(track, params)
    hand_seconds = [round(index / track.fps, 4) for index in track.hand_video_indices]
    valley_seconds = [round(track.hand_video_indices[index] / track.fps, 4) for index in valleys if index < len(track.hand_video_indices)]
    cut_seconds = [round((start + end) / 2 / track.fps, 4) for start, end in regions]
    segment_durations = [round((end - start + 1) / track.fps, 4) for start, end in regions]
    return {
        "fps": track.fps,
        "video_frame_count": track.video_frame_count,
        "hand_frame_count": len(track.keypoints),
        "hand_times_seconds": hand_seconds,
        "motion": [round(float(value), 6) for value in track.motion],
        "motion_smoothed": [round(float(value), 6) for value in smoothed],
        "valley_hand_indices": valleys,
        "valley_times_seconds": valley_seconds,
        "segment_video_regions": [{"start_frame": s, "end_frame": e} for s, e in regions],
        "cut_times_seconds": cut_seconds,
        "segment_durations_seconds": segment_durations,
        "segment_count": len(regions),
        "params": {
            "smooth_radius": params.smooth_radius,
            "valley_ratio": params.valley_ratio,
            "minimum_gap": params.minimum_gap,
            "minimum_sign_seconds": params.minimum_sign_seconds,
        },
    }


def analyze_video(video: Path, params: MotionValleyParams | None = None) -> dict:
    params = params or MotionValleyParams()
    track = extract_hand_track(video)
    payload = timeline_payload(track, params)
    pause_regions = hand_pause_video_regions(video)
    payload["hand_pause_segment_count"] = len(pause_regions)
    payload["hand_pause_regions"] = [{"start_frame": s, "end_frame": e} for s, e in pause_regions]
    return payload
