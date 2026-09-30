"""Split a continuous sign video into per-gloss clips using a trained CTC checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.browser_mp4 import write_browser_mp4
from app.isharah_cslr import TEMPORAL_DOWNSAMPLE
from scripts.isharah_predict import encode_video

# Reuse predict logic
from scripts import isharah_predict as predict_mod


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "isharah_cslr" / "best_SI.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.15)
    args = parser.parse_args()

    import numpy as np
    import torch

    from app.isharah_cslr import CSLRModel, greedy_decode

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    id_to_gloss = {int(k): v for k, v in checkpoint["id_to_gloss"].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSLRModel(checkpoint["input_dim"], checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    features, fps = encode_video(args.video, args.stride)
    seconds_per_step = TEMPORAL_DOWNSAMPLE * args.stride / fps
    with torch.inference_mode():
        log_probs, _c, lengths = model(
            torch.from_numpy(features).unsqueeze(0).to(device), torch.tensor([len(features)], device=device)
        )
    decoded = greedy_decode(log_probs[: int(lengths[0]), 0].cpu().numpy(), id_to_gloss)
    decoded = [item for item in decoded if item.confidence >= args.min_confidence]

    capture = cv2.VideoCapture(str(args.video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(decoded, start=1):
        start_frame = max(0, int(item.start_step * seconds_per_step * fps))
        end_frame = min(len(frames) - 1, int((item.end_step + 1) * seconds_per_step * fps))
        if end_frame <= start_frame:
            continue
        target = args.out_dir / f"{index:02d}-{item.gloss}.mp4"
        write_browser_mp4(frames[start_frame : end_frame + 1], fps, target)
        print(f"{target.name}  frames {start_frame}-{end_frame}", flush=True)


if __name__ == "__main__":
    main()
