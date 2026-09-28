from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from .dictionary import Match
from .recognizer import read_image


def _text_card(text: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    # OpenCV cannot shape Arabic reliably; the card is an explicit review/fingerspelling placeholder.
    safe = text if text.isascii() else "UNMAPPED SIGN"
    cv2.putText(frame, safe, (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    return frame


def compose(
    matches: list[Match], output: Path, fps: int = 24,
    seconds_per_item: float = 1.2, transition_seconds: float = 0.16,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    intermediate = output.with_name(f"{output.stem}.intermediate.mp4")
    size = (640, 480)
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not initialize the video encoder")
    repeat = max(int(fps * seconds_per_item), 1)
    transition_frames = max(int(fps * transition_seconds), 1)
    for match in matches:
        path = Path(match.media_path) if match.media_path else None
        last_frame = None
        if path and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            image = read_image(path)
            frame = cv2.resize(image, size) if image is not None else _text_card(match.query, size)
            for _ in range(repeat):
                writer.write(frame)
            last_frame = frame
        elif path and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            capture = cv2.VideoCapture(str(path))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                last_frame = cv2.resize(frame, size)
                writer.write(last_frame)
            capture.release()
        else:
            frame = _text_card(match.query, size)
            for _ in range(repeat):
                writer.write(frame)
            last_frame = frame
        # A short held frame makes boundaries between signs and spelled letters visible.
        if last_frame is not None:
            for _ in range(transition_frames):
                writer.write(last_frame)
    writer.release()
    if not intermediate.exists() or intermediate.stat().st_size == 0:
        raise RuntimeError("Video construction failed")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        intermediate.replace(output)
        raise RuntimeError("FFmpeg is required to create a browser-compatible H.264 video")
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(intermediate),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {completed.stderr.strip()}")
    finally:
        intermediate.unlink(missing_ok=True)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Browser-compatible video construction failed")
    return output
