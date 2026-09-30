"""Wait until SI split features exist, then train the full signer-independent CSLR model."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "isharah" / "selfi"
FEATURES = ROOT / "data" / "isharah" / "features_dinov2s"
PYTHON = ROOT.parent / "gpu-env" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def si_feature_counts() -> tuple[int, int]:
    train = dev = 0
    for split, counter in (("train", lambda: train), ("dev", lambda: dev)):
        info = np.load(DATA / "SI" / f"{split}_info.npy", allow_pickle=True).item()
        total = len(info) - 1
        have = sum((FEATURES / f"{info[i]['fileid']}.npy").exists() for i in range(total))
        if split == "train":
            train = have
        else:
            dev = have
    return train, dev


def main() -> None:
    print("Waiting for SI train/dev features (feature extraction must be running)...", flush=True)
    while True:
        train, dev = si_feature_counts()
        print(f"SI features: train={train}/5996 dev={dev}/1499", flush=True)
        if train >= 5800 and dev >= 1400:
            break
        time.sleep(120)
    cmd = [str(PYTHON), str(ROOT / "scripts" / "train_isharah_cslr.py"), "--protocol", "SI", "--epochs", "60"]
    print("Starting:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
