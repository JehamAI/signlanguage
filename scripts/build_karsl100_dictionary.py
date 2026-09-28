"""Download one labeled KArSL sequence per class and create the 100-word video dictionary."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[1]
REPO = "FatimahEmadEldin/karsl-502-arabic-sign-language-v2"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def get_json(session: requests.Session, url: str, attempts: int = 12):
    for attempt in range(attempts):
        response = session.get(url, timeout=60)
        if response.ok:
            return response.json()
        if response.status_code == 404:
            return []
        if response.status_code != 429:
            response.raise_for_status()
        delay = min(60, 2 ** min(attempt, 5))
        print(f"Rate limited; retrying in {delay}s")
        time.sleep(delay)
    raise RuntimeError(f"Unable to query after {attempts} attempts: {url}")


def download(session: requests.Session, remote: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 1000:
        return
    url = f"{RAW}/{quote(remote, safe='/')}?download=true"
    for attempt in range(12):
        response = session.get(url, timeout=120)
        if response.ok:
            target.write_bytes(response.content)
            return
        if response.status_code != 429:
            response.raise_for_status()
        time.sleep(min(60, 2 ** min(attempt, 5)))
    raise RuntimeError(f"Unable to download {remote}")


def safe_folder(sign_id: int) -> str:
    return f"{sign_id:04d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=71)
    parser.add_argument("--end", type=int, default=170)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--rebuild-videos", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data" / "karsl100_manifest.json").read_text(encoding="utf-8"))
    labels = {int(x["sign_id"]): str(x["arabic"]) for x in manifest["classes"]}
    frame_root = ROOT / "data" / "KArSL100_dictionary_frames"
    video_root = ROOT / "data" / "KArSL100_dictionary"
    session = requests.Session()
    session.headers["User-Agent"] = "KArSL100-dictionary-builder/1.0"

    completed = 0
    for sign_id in range(args.start, args.end + 1):
        output_dir = video_root / safe_folder(sign_id)
        output = output_dir / "sign.mp4"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "label.txt").write_text(labels[sign_id], encoding="utf-8")
        if output.exists() and output.stat().st_size > 1000 and not args.rebuild_videos:
            completed += 1
            print(f"[{sign_id:04d}] ready ({completed})")
            continue

        class_items = get_json(session, f"{API}/{sign_id:04d}?limit=100")
        sequences = [x["path"] for x in class_items if x.get("type") == "directory"]
        if not sequences:
            print(f"[{sign_id:04d}] no sequence found")
            continue
        remote_sequence = sequences[0]
        files = get_json(session, f"{API}/{quote(remote_sequence, safe='/')}?limit=200")
        frames = [x["path"] for x in files if x.get("type") == "file" and x["path"].lower().endswith(".jpg")]
        if not frames:
            print(f"[{sign_id:04d}] empty sequence")
            continue
        local_frames = frame_root / safe_folder(sign_id)
        local_frames.mkdir(parents=True, exist_ok=True)
        jobs = [
            (remote, local_frames / f"{index:04d}.jpg")
            for index, remote in enumerate(frames, 1)
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda job: download(session, job[0], job[1]), jobs))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(local_frames / "%04d.jpg"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        completed += 1
        print(f"[{sign_id:04d}] {labels[sign_id]}: {len(frames)} frames ({completed})")
        time.sleep(0.15)
    print(f"Dictionary videos ready: {completed}/{args.end - args.start + 1}")


if __name__ == "__main__":
    main()
