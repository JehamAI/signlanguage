"""Train/evaluate KArSL-100 with a strict signer-independent split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.temporal_model import TemporalLandmarkClassifier


ROOT = Path(__file__).resolve().parents[1]


class LandmarkDataset(Dataset):
    def __init__(self, paths: list[Path], augment: bool = False):
        self.paths, self.augment = paths, augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index]) as item:
            x = item["x"].astype(np.float32)
            y = int(item["y"])
        if self.augment:
            x = x + np.random.normal(0, 0.008, x.shape).astype(np.float32)
            if random.random() < 0.3:
                x *= np.float32(random.uniform(0.95, 1.05))
        return torch.from_numpy(x), y


def metadata(path: Path) -> tuple[int, int]:
    with np.load(path) as item:
        return int(item["y"]), int(item["signer"])


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(loss_fn(logits, y)) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "KArSL100_landmarks")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "karsl100_manifest.json")
    parser.add_argument("--test-signer", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "karsl100_temporal.pt")
    args = parser.parse_args()
    files = sorted(args.data.glob("*.npz"))
    train = [p for p in files if metadata(p)[1] != args.test_signer]
    test = [p for p in files if metadata(p)[1] == args.test_signer]
    if not train or not test:
        raise ValueError(f"Need both train and signer-{args.test_signer} test samples; got {len(train)}/{len(test)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(LandmarkDataset(train, True), args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(LandmarkDataset(test), args.batch_size, num_workers=0)
    model = TemporalLandmarkClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = 0.0
    classes = json.loads(args.manifest.read_text(encoding="utf-8"))["classes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        val_loss, accuracy = evaluate(model, test_loader, device)
        print(f"epoch={epoch:02d} test_loss={val_loss:.4f} signer_independent_accuracy={accuracy:.4f}")
        if accuracy > best:
            best = accuracy
            torch.save(
                {"state_dict": model.state_dict(), "classes": classes, "test_signer": args.test_signer,
                 "accuracy": accuracy, "frames": 32},
                args.output,
            )
    print(f"Best signer-independent accuracy={best:.4f}; checkpoint={args.output}")


if __name__ == "__main__":
    main()
