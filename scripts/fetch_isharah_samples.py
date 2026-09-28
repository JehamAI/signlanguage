"""Download a small Isharah sentence set (mp4 + gloss labels) for local evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _annotation_map(meta_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in ("train.txt", "dev.txt"):
        path = meta_dir / "SI" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            if "|" not in line:
                continue
            sample_id, gloss = line.split("|", 1)
            mapping[sample_id.strip()] = gloss.strip()
    return mapping


def _candidate_paths(sample_id: str) -> list[str]:
    signer, number = sample_id.split("_", 1)
    number = number.zfill(4)
    return [
        f"Group1/{signer}/{number}.mp4",
        f"Group2/{signer}/{number}.mp4",
        f"Group3/{signer}/{number}.mp4",
    ]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--split", choices=("dev", "train", "both"), default="dev")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    repo = "GufranSabri/isharah-selfi"
    meta_dir = ROOT / "data" / "isharah" / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for relative in ("SI/dev.txt", "SI/train.txt"):
        hf_hub_download(repo, relative, repo_type="dataset", local_dir=str(meta_dir))

    available = set(list_repo_files(repo, repo_type="dataset"))
    labels = _annotation_map(meta_dir)
    order: list[str] = []
    if args.split in ("dev", "both"):
        order.extend(
            line.split("|", 1)[0].strip()
            for line in (meta_dir / "SI" / "dev.txt").read_text(encoding="utf-8").splitlines()[1:]
        )
    if args.split in ("train", "both"):
        order.extend(
            line.split("|", 1)[0].strip()
            for line in (meta_dir / "SI" / "train.txt").read_text(encoding="utf-8").splitlines()[1:]
        )

    output_dir = ROOT / "data" / "isharah" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_id in order:
        if len(manifest) >= args.limit:
            break
        gloss = labels.get(sample_id)
        if not gloss:
            continue
        remote = next((item for item in _candidate_paths(sample_id) if item in available), None)
        if remote is None:
            continue
        local_name = sample_id.replace("/", "_") + ".mp4"
        target = output_dir / local_name
        if not target.exists() or target.stat().st_size == 0:
            downloaded = Path(
                hf_hub_download(repo, remote, repo_type="dataset", local_dir=str(output_dir / "_hf"))
            )
            target.write_bytes(downloaded.read_bytes())
        manifest.append(
            {
                "id": sample_id,
                "video_path": str(target.relative_to(ROOT)).replace("\\", "/"),
                "gloss_sequence": gloss,
                "gloss_tokens": gloss.split(),
                "remote_path": remote,
            }
        )
        print(f"saved {sample_id} -> {target.name}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} samples to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
