from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from app.pretrained_recognizer import WORD_LABELS, _normalized_landmarks, pretrained_recognizer
from app.recognizer import read_image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "ARSLW" / "Dataset" / "Verbs" / "اصمت"
OUTPUT = ROOT / "test_inputs" / "يسكت_demo.mp4"


def main() -> None:
    candidates = sorted(
        path for path in SOURCE.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not candidates:
        raise SystemExit(f"No source images found in {SOURCE}")
    selected_indexes = np.unique(np.linspace(0, len(candidates) - 1, min(80, len(candidates))).astype(int))
    recognizer = pretrained_recognizer()
    scored: list[tuple[float, Path, np.ndarray]] = []
    target_index = WORD_LABELS.index("يسكت")
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.30
    ) as hands:
        for index in selected_indexes:
            path = candidates[int(index)]
            image = read_image(path)
            if image is None:
                continue
            result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not result.multi_hand_landmarks:
                continue
            features = _normalized_landmarks(result.multi_hand_landmarks[0])
            if features is None:
                continue
            score = float(recognizer._probabilities(features)[target_index])
            scored.append((score, path, image))
    if not scored:
        raise SystemExit("MediaPipe did not find a hand in the candidate images")
    best = sorted(scored, key=lambda row: row[0], reverse=True)[:8]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    intermediate = OUTPUT.with_name(f"{OUTPUT.stem}.intermediate.mp4")
    fps, size = 24, (640, 480)
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    # Hold a few high-scoring signer variations long enough for uniform video sampling.
    for _, _, image in best:
        frame = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        for _ in range(12):
            writer.write(frame)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg is required")
    completed = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(intermediate), "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUTPUT),
        ],
        capture_output=True, text=True, check=False,
    )
    intermediate.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr)
    prediction = recognizer.predict_video(OUTPUT, max_samples=16)
    print(f"Created: {OUTPUT}")
    print(f"Best source-frame target score: {best[0][0]:.4f}")
    print(f"Video prediction: {prediction}")


if __name__ == "__main__":
    main()
