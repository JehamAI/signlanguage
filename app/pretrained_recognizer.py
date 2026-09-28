from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration


WORD_LABELS = [
    "ينام",
    "يسكت",
    "حب",
    "يدخن",
    "دعم",
    "مرتبك",
    "قلق",
    "هنا",
    "السلام عليكم",
    "شكرا",
]


@dataclass(frozen=True)
class WordPrediction:
    gloss: str
    confidence: float
    accepted: bool
    frames_with_hands: int
    sampled_frames: int


class T5WordClassifier(nn.Module):
    """Architecture matching the MIT ArSL-Models word checkpoint."""

    def __init__(self):
        super().__init__()
        self.t5_model = T5ForConditionalGeneration(T5Config())
        self.custom_linear = nn.Sequential(
            nn.Linear(63, 512),
            nn.Dropout(0.1),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.classifier = nn.Linear(512, len(WORD_LABELS))

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        projected = self.custom_linear(landmarks)
        attention_mask = torch.ones(projected.shape[:2], device=projected.device, dtype=torch.long)
        encoded = self.t5_model.encoder(inputs_embeds=projected, attention_mask=attention_mask)
        return self.classifier(encoded.last_hidden_state[:, 0, :])


def _normalized_landmarks(hand_landmarks) -> np.ndarray | None:
    points = np.asarray([(p.x, p.y, p.z) for p in hand_landmarks.landmark], dtype=np.float32)
    points -= points[0]
    scale = float(np.linalg.norm(points[9]))
    if scale < 1e-6:
        return None
    return (points / scale).reshape(-1)


class PretrainedWordRecognizer:
    def __init__(self, checkpoint: Path, threshold: float = 0.45):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Pretrained word checkpoint not found: {checkpoint}")
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5WordClassifier()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def _probabilities(self, features: np.ndarray) -> np.ndarray:
        # The published model tiles a single normalized hand-landmark frame to length 100.
        sequence = np.repeat(features[None, :], 100, axis=0)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(tensor), dim=-1)[0].cpu().numpy()

    def predict_video(self, path: Path, max_samples: int = 24) -> WordPrediction:
        import mediapipe as mp

        capture = cv2.VideoCapture(str(path))
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        requested = np.unique(np.linspace(0, frame_count - 1, min(max_samples, frame_count)).astype(int))
        probabilities: list[np.ndarray] = []
        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.45,
        ) as hands:
            for frame_index in requested:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if not result.multi_hand_landmarks:
                    continue
                features = _normalized_landmarks(result.multi_hand_landmarks[0])
                if features is not None:
                    probabilities.append(self._probabilities(features))
        capture.release()
        if not probabilities:
            return WordPrediction("", 0.0, False, 0, len(requested))
        # Robust aggregation reduces the effect of transition and blurred frames.
        mean = np.mean(sorted(probabilities, key=lambda p: float(np.max(p)), reverse=True)[:8], axis=0)
        index = int(np.argmax(mean))
        confidence = float(mean[index])
        return WordPrediction(
            WORD_LABELS[index], confidence, confidence >= self.threshold, len(probabilities), len(requested)
        )


@lru_cache(maxsize=1)
def pretrained_recognizer() -> PretrainedWordRecognizer:
    from .config import Settings

    settings = Settings.load()
    return PretrainedWordRecognizer(
        settings.artifact_root / "pretrained" / "sign_word_t5_classifier_best_3d.pth",
        threshold=settings.pretrained_word_threshold,
    )

