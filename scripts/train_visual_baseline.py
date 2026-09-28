from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.recognizer import read_image, visual_features


def samples(root: Path, max_per_class: int):
    for word_dir in sorted(path for path in root.rglob("*") if path.is_dir()):
        images = [p for p in word_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            continue
        indexes = np.linspace(0, len(images) - 1, min(len(images), max_per_class)).astype(int)
        for index in indexes:
            image = read_image(images[int(index)])
            if image is not None:
                yield visual_features(image), word_dir.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ARSLW/Dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/visual_baseline.joblib"))
    parser.add_argument("--max-per-class", type=int, default=200)
    args = parser.parse_args()
    rows = list(samples(args.data, args.max_per_class))
    if len({label for _, label in rows}) < 2:
        raise SystemExit("At least two populated word folders are required")
    x = np.stack([features for features, _ in rows])
    y = np.asarray([label for _, label in rows])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=500, n_jobs=1, class_weight="balanced").fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test), zero_division=0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved {args.output}")
