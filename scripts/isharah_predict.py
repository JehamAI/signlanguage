"""Recognize the gloss sequence (with per-word time spans) in any sign video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.isharah_cslr import TEMPORAL_DOWNSAMPLE, CSLRModel, greedy_decode
from scripts.isharah_extract_features import MEAN, STD

DATA = ROOT / "data" / "isharah" / "selfi"


def encode_video(video: Path, stride: int = 2, size: int = 224) -> tuple[np.ndarray, float]:
    from transformers import Dinov2Model

    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames, position = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if position % stride == 0:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        position += 1
    capture.release()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Dinov2Model.from_pretrained("facebook/dinov2-small").to(device, dtype).eval()
    mean = torch.tensor(MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    batch = torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2).to(dtype).div(255).sub(mean).div(std)
    with torch.inference_mode():
        hidden = model(pixel_values=batch).last_hidden_state
        features = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
    return features.float().cpu().numpy(), fps


def reference_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for info_path in DATA.glob("*/*_info.npy"):
        info = np.load(info_path, allow_pickle=True).item()
        for index in range(len(info) - 1):
            labels[info[index]["fileid"]] = info[index]["label"]
    return labels


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "isharah_cslr" / "best_SI.pt")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    id_to_gloss = {int(key): value for key, value in checkpoint["id_to_gloss"].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSLRModel(checkpoint["input_dim"], checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    labels = reference_labels()

    for video in args.videos:
        features, fps = encode_video(video, args.stride)
        seconds_per_step = TEMPORAL_DOWNSAMPLE * args.stride / fps
        with torch.inference_mode():
            log_probs, _conv, lengths = model(
                torch.from_numpy(features).unsqueeze(0).to(device), torch.tensor([len(features)], device=device)
            )
        decoded = greedy_decode(log_probs[: int(lengths[0]), 0].cpu().numpy(), id_to_gloss)
        try:
            fileid = video.resolve().relative_to(DATA).with_suffix("").as_posix()
        except ValueError:
            fileid = ""
        print(f"\n{video}  ({len(features) / fps * args.stride:.1f}s)")
        if fileid in labels:
            print(f"  reference : {labels[fileid]}")
        print(f"  predicted : {' '.join(item.gloss for item in decoded) or '(nothing)'}")
        for item in decoded:
            start = item.start_step * seconds_per_step
            end = (item.end_step + 1) * seconds_per_step
            print(f"    {start:5.2f}s - {end:5.2f}s  {item.gloss}  (p={item.confidence:.2f})")


if __name__ == "__main__":
    main()
