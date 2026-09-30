"""Train and evaluate the CTC continuous-sign model on Isharah SI (signer-independent) splits."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.isharah_cslr import CSLRModel, edit_distance, greedy_decode

DATA = ROOT / "data" / "isharah" / "selfi"
OUTPUT = ROOT / "artifacts" / "isharah_cslr"


def load_split(protocol: str, split: str, feature_root: Path, gloss_to_id: dict[str, int]) -> list[dict]:
    info = np.load(DATA / protocol / f"{split}_info.npy", allow_pickle=True).item()
    samples: list[dict] = []
    for index in range(len(info) - 1):
        entry = info[index]
        path = feature_root / f"{entry['fileid']}.npy"
        if not path.exists():
            continue
        tokens = entry["label"].split()
        samples.append(
            {
                "fileid": entry["fileid"],
                "path": path,
                "tokens": tokens,
                "targets": [gloss_to_id[token] for token in tokens if token in gloss_to_id],
                "sentence": entry.get("sentence", ""),
            }
        )
    return samples


class FeatureDataset(Dataset):
    def __init__(self, samples: list[dict], train: bool):
        self.samples, self.train = samples, train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        features = np.load(sample["path"]).astype(np.float32)
        if self.train:
            factor = random.uniform(0.8, 1.2)
            length = max(8, int(round(len(features) * factor)))
            positions = np.linspace(0, len(features) - 1, length).round().astype(int)
            features = features[positions]
            features = features + np.random.normal(0, 0.02, features.shape).astype(np.float32)
        return torch.from_numpy(features), torch.tensor(sample["targets"], dtype=torch.long), index


def collate(batch):
    features, targets, indexes = zip(*batch)
    lengths = torch.tensor([len(item) for item in features])
    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    target_lengths = torch.tensor([len(item) for item in targets])
    return padded, lengths, torch.cat(targets), target_lengths, list(indexes)


def evaluate(model, loader, samples, id_to_gloss, device) -> tuple[float, list[dict]]:
    model.eval()
    errors = words = 0
    records: list[dict] = []
    with torch.inference_mode():
        for features, lengths, _targets, _target_lengths, indexes in loader:
            log_probs, _conv, out_lengths = model(features.to(device), lengths.to(device))
            log_probs = log_probs.transpose(0, 1).float().cpu().numpy()
            for row, sample_index in enumerate(indexes):
                sample = samples[sample_index]
                decoded = greedy_decode(log_probs[row, : int(out_lengths[row])], id_to_gloss)
                hypothesis = [item.gloss for item in decoded]
                distance = edit_distance(sample["tokens"], hypothesis)
                errors += distance
                words += len(sample["tokens"])
                records.append(
                    {
                        "fileid": sample["fileid"],
                        "reference": sample["tokens"],
                        "hypothesis": hypothesis,
                        "edit_distance": distance,
                        "spans_steps": [[item.start_step, item.end_step] for item in decoded],
                    }
                )
    return errors / max(words, 1), records


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("SI", "US"), default="SI")
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "isharah" / "features_dinov2s")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gloss_dict = np.load(DATA / args.protocol / "gloss_dict.npy", allow_pickle=True).item()
    gloss_to_id = {gloss: int(value[0]) for gloss, value in gloss_dict.items()}
    id_to_gloss = {index: gloss for gloss, index in gloss_to_id.items()}
    vocab_size = max(gloss_to_id.values()) + 1

    train = [s for s in load_split(args.protocol, "train", args.features, gloss_to_id) if s["targets"]]
    dev = load_split(args.protocol, "dev", args.features, gloss_to_id)
    print(f"train={len(train)} dev={len(dev)} vocab={vocab_size}", flush=True)
    if not train or not dev:
        raise SystemExit("Missing features for train or dev; run the download and feature scripts first")

    input_dim = int(np.load(train[0]["path"]).shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSLRModel(input_dim, vocab_size).to(device)
    train_loader = DataLoader(
        FeatureDataset(train, True), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=4,
        persistent_workers=True,
    )
    dev_loader = DataLoader(FeatureDataset(dev, False), batch_size=64, collate_fn=collate, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
    )
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    best_wer = float("inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for features, lengths, targets, target_lengths, _ in train_loader:
            features, lengths = features.to(device), lengths.to(device)
            log_probs, conv_log_probs, out_lengths = model(features, lengths)
            loss = ctc(log_probs, targets, out_lengths, target_lengths) + 0.5 * ctc(
                conv_log_probs, targets, out_lengths, target_lengths
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += float(loss)
        train_loss = running / len(train_loader)
        dev_wer, records = evaluate(model, dev_loader, dev, id_to_gloss, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_wer": dev_wer})
        marker = ""
        if dev_wer < best_wer:
            best_wer = dev_wer
            marker = " *"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "vocab_size": vocab_size,
                    "id_to_gloss": id_to_gloss,
                    "protocol": args.protocol,
                    "features": args.features.name,
                },
                OUTPUT / f"best_{args.protocol}.pt",
            )
            (OUTPUT / f"dev_predictions_{args.protocol}.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        print(
            f"epoch {epoch:02d} loss={train_loss:.3f} dev_WER={dev_wer:.1%} "
            f"({time.time() - started:.0f}s){marker}",
            flush=True,
        )
    (OUTPUT / f"history_{args.protocol}.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"BEST dev WER={best_wer:.1%}", flush=True)


if __name__ == "__main__":
    main()
