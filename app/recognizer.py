from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np


def read_image(path: Path) -> np.ndarray | None:
    """Read Unicode paths reliably on Windows (cv2.imread does not)."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def visual_features(image: np.ndarray) -> np.ndarray:
    """Compact appearance baseline. Replace with temporal landmarks for production."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9).compute(gray).reshape(-1)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).reshape(-1)
    hist = hist / max(float(hist.sum()), 1.0)
    return np.concatenate([hog / 255.0, hist]).astype(np.float32)


class VisualRecognizer:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = joblib.load(model_path) if model_path.exists() else None

    def predict_image(self, path: Path) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("Recognizer is not trained. Run scripts/train_visual_baseline.py first.")
        image = read_image(path)
        x = visual_features(image).reshape(1, -1)
        probabilities = self.model.predict_proba(x)[0]
        index = int(np.argmax(probabilities))
        return str(self.model.classes_[index]), float(probabilities[index])

    def predict_video(self, path: Path, samples: int = 12) -> tuple[str, float]:
        capture = cv2.VideoCapture(str(path))
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        predictions: list[np.ndarray] = []
        for frame_index in np.linspace(0, frame_count - 1, samples).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                predictions.append(self.model.predict_proba(visual_features(frame).reshape(1, -1))[0])
        capture.release()
        if not predictions:
            raise ValueError("No readable video frames")
        mean = np.mean(predictions, axis=0)
        index = int(np.argmax(mean))
        return str(self.model.classes_[index]), float(mean[index])
