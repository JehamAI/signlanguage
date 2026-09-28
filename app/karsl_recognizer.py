"""Inference adapter for the published 100-word KArSL BiLSTM checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class KArSLPrediction:
    gloss: str
    confidence: float
    accepted: bool
    frames_with_hands: int
    sampled_frames: int


def _relative(landmarks, count: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros(count * 3, dtype=np.float32)
    values = np.asarray([(p.x, p.y, p.z) for p in landmarks.landmark], dtype=np.float32)
    return (values - values[0]).reshape(-1)


class KArSL100Recognizer:
    """Recreates the original network because modern Keras cannot deserialize its legacy H5 config."""

    frames = 48

    def __init__(self, checkpoint: Path, labels: list[str], threshold: float = 0.60):
        if len(labels) != 100:
            raise ValueError(f"KArSL-100 requires 100 labels, got {len(labels)}")
        import tensorflow as tf

        self.labels = labels
        self.threshold = threshold
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input((self.frames, 225)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(100, activation="softmax"),
            ]
        )
        self.model.load_weights(checkpoint)

    def predict_video(self, path: Path) -> KArSLPrediction:
        import mediapipe as mp

        capture = cv2.VideoCapture(str(path))
        sequence: list[np.ndarray] = []
        hand_frames = 0
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        ) as holistic:
            while len(sequence) < self.frames:
                ok, frame = capture.read()
                if not ok:
                    break
                result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pose = _relative(result.pose_landmarks, 33)
                left = _relative(result.left_hand_landmarks, 21)
                right = _relative(result.right_hand_landmarks, 21)
                hand_frames += int(result.left_hand_landmarks is not None or result.right_hand_landmarks is not None)
                sequence.append(np.concatenate((pose, left, right)))
        capture.release()
        if not sequence:
            return KArSLPrediction("", 0.0, False, 0, 0)
        observed = len(sequence)
        while len(sequence) < self.frames:
            sequence.append(sequence[-1].copy())
        probabilities = self.model(np.asarray(sequence, dtype=np.float32)[None, ...], training=False).numpy()[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        # A confident softmax without observable hands is not an acceptable sign prediction.
        accepted = confidence >= self.threshold and hand_frames >= max(2, observed // 8)
        return KArSLPrediction(self.labels[index], confidence, accepted, hand_frames, observed)


@lru_cache(maxsize=1)
def karsl100_recognizer() -> KArSL100Recognizer:
    import json

    from .config import ROOT

    manifest = json.loads((ROOT / "data" / "karsl100_manifest.json").read_text(encoding="utf-8"))
    labels = [str(item["arabic"]) for item in sorted(manifest["classes"], key=lambda item: item["class_index"])]
    checkpoints = sorted((ROOT / "references" / "karsl_word_recognition").glob("*3_signers*Accuracy_*.h5"))
    if not checkpoints:
        raise FileNotFoundError("KArSL-100 three-signer checkpoint is missing")
    return KArSL100Recognizer(checkpoints[0], labels)
