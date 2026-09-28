"""Write MP4 files that play in Chrome/Edge (H.264 + yuv420p)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def write_browser_mp4(frames: list[np.ndarray], fps: float, output: Path) -> None:
    if not frames:
        raise ValueError("No frames to encode")
    if shutil.which("ffmpeg") is None:
        _write_opencv_fallback(frames, fps, output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wusal-mp4-") as temp_dir:
        root = Path(temp_dir)
        for index, frame in enumerate(frames, 1):
            cv2.imwrite(str(root / f"{index:04d}.jpg"), frame)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(root / "%04d.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )


def _write_opencv_fallback(frames: list[np.ndarray], fps: float, output: Path) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output}")
    for frame in frames:
        writer.write(frame)
    writer.release()
