"""Selectively fetch KArSL IDs 0121-0170 from Kaggle's 25 GB ZIP via HTTP ranges."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import requests
from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DOWNLOAD = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "yousefdotpy/karsl-502?datasetVersionNumber=3"
)


def signed_archive_url() -> str:
    response = requests.get(
        KAGGLE_DOWNLOAD,
        headers={"Range": "bytes=0-0"},
        allow_redirects=True,
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    url = response.url
    response.close()
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=121)
    parser.add_argument("--end", type=int, default=170)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    manifest = json.loads((ROOT / "data" / "karsl100_manifest.json").read_text(encoding="utf-8"))
    labels = {int(row["sign_id"]): str(row["arabic"]) for row in manifest["classes"]}
    frame_root = ROOT / "data" / "KArSL100_dictionary_frames"
    video_root = ROOT / "data" / "KArSL100_dictionary"

    print("Reading the remote ZIP directory (one-time operation)...", flush=True)
    with RemoteZip(signed_archive_url()) as archive:
        members = archive.infolist()
        selected: dict[int, list] = {sign_id: [] for sign_id in range(args.start, args.end + 1)}
        first_sequences: dict[int, str] = {}
        for member in members:
            parts = member.filename.split("/")
            if (
                len(parts) < 6
                or parts[:3] != ["01", "01", "test"]
                or not parts[3].isdigit()
                or not member.filename.lower().endswith(".jpg")
            ):
                continue
            sign_id = int(parts[3])
            if sign_id not in selected:
                continue
            sequence = parts[4]
            first_sequences.setdefault(sign_id, sequence)
            if sequence == first_sequences[sign_id]:
                selected[sign_id].append(member)

        completed = 0
        for sign_id, frames in selected.items():
            output_dir = video_root / f"{sign_id:04d}"
            output = output_dir / "sign.mp4"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "label.txt").write_text(labels[sign_id], encoding="utf-8")
            if output.exists() and output.stat().st_size > 1000:
                completed += 1
                print(f"[{sign_id:04d}] ready ({completed})", flush=True)
                continue
            if not frames:
                print(f"[{sign_id:04d}] missing from archive", flush=True)
                continue
            local_dir = frame_root / f"{sign_id:04d}"
            local_dir.mkdir(parents=True, exist_ok=True)
            for index, member in enumerate(frames, 1):
                target = local_dir / f"{index:04d}.jpg"
                if not target.exists() or target.stat().st_size < 1000:
                    with archive.open(member) as source, target.open("wb") as destination:
                        destination.write(source.read())
            subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", str(args.fps),
                    "-i", str(local_dir / "%04d.jpg"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            completed += 1
            print(f"[{sign_id:04d}] {labels[sign_id]}: {len(frames)} frames ({completed})", flush=True)
    print(f"Missing-half dictionary videos ready: {completed}/{args.end - args.start + 1}", flush=True)


if __name__ == "__main__":
    main()
