"""Encode every Isharah video frame with a frozen DINOv2-small backbone on the GPU (resumable).

Selfie recordings are 256x256 with hands often cropped, so hand-landmark trackers miss many
frames. Frozen whole-frame visual features avoid that dependency.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "data" / "isharah" / "selfi"
FEATURE_ROOT = ROOT / "data" / "isharah" / "features_dinov2s"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoFrames(Dataset):
    def __init__(self, videos: list[Path], stride: int, size: int):
        self.videos, self.stride, self.size = videos, stride, size

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        video = self.videos[index]
        capture = cv2.VideoCapture(str(video))
        frames: list[np.ndarray] = []
        position = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if position % self.stride == 0:
                frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            position += 1
        capture.release()
        array = np.stack(frames) if frames else np.zeros((0, self.size, self.size, 3), np.uint8)
        return str(video), torch.from_numpy(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=2, help="Keep every Nth frame (30 fps -> 15 fps)")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-frames", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from transformers import Dinov2Model

    videos = [
        video
        for video in sorted(VIDEO_ROOT.glob("Group*/*/*.mp4"))
        if not (FEATURE_ROOT / video.relative_to(VIDEO_ROOT).with_suffix(".npy")).exists()
    ]
    print(f"{len(videos)} videos to encode", flush=True)
    if not videos:
        return

    device = torch.device("cuda")
    model = Dinov2Model.from_pretrained("facebook/dinov2-small").to(device).half().eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1).half()
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1).half()
    loader = DataLoader(
        VideoFrames(videos, args.stride, args.size),
        batch_size=None,
        num_workers=args.workers,
        prefetch_factor=4,
    )

    started = time.time()
    for done, (video, frames) in enumerate(loader, start=1):
        target = FEATURE_ROOT / Path(video).relative_to(VIDEO_ROOT).with_suffix(".npy")
        if not len(frames):
            print(f"EMPTY {video}", flush=True)
            continue
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(frames), args.batch_frames):
                chunk = frames[start : start + args.batch_frames].to(device, non_blocking=True)
                chunk = chunk.permute(0, 3, 1, 2).half().div_(255).sub_(mean).div_(std)
                hidden = model(pixel_values=chunk).last_hidden_state
                feature = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
                outputs.append(feature.float().cpu().numpy().astype(np.float16))
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.concatenate(outputs))
        if done % 200 == 0 or done == len(videos):
            rate = done / (time.time() - started)
            print(f"{done}/{len(videos)} {rate:.1f} videos/s", flush=True)


if __name__ == "__main__":
    main()
