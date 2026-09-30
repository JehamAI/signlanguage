"""Download the Isharah (Selfi release) videos and SI/US annotations, resumably.

Anonymous Hugging Face access is rate limited (HTTP 429); the loop waits and continues.
Set HF_TOKEN to lift the limit.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "GufranSabri/isharah-selfi"
TARGET = ROOT / "data" / "isharah" / "selfi"
WAIT_SECONDS = 330


def _wanted(name: str) -> bool:
    if name.startswith(("SI/", "US/")):
        return name.endswith((".txt", ".npy"))
    return name.startswith("Group") and name.endswith(".mp4")


def main() -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    TARGET.mkdir(parents=True, exist_ok=True)
    listing = TARGET / "file_list.txt"
    while not listing.exists():
        try:
            names = list_repo_files(REPO, repo_type="dataset")
            listing.write_text("\n".join(names), encoding="utf-8")
        except Exception:  # noqa: BLE001 - rate limited before the listing succeeds
            print(f"file listing rate limited; waiting {WAIT_SECONDS}s", flush=True)
            time.sleep(WAIT_SECONDS)
    pending = [name for name in listing.read_text(encoding="utf-8").splitlines() if _wanted(name)]

    def fetch(name: str) -> str | None:
        if (TARGET / name).exists():
            return None
        try:
            hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(TARGET))
            return None
        except Exception:  # noqa: BLE001 - rate limits surface as several error types; retry later
            return name

    while True:
        pending = [name for name in pending if not (TARGET / name).exists()]
        print(f"{len(pending)} files remaining", flush=True)
        if not pending:
            break
        with ThreadPoolExecutor(8) as pool:
            limited = [name for name in pool.map(fetch, pending) if name]
        if limited:
            print(f"{len(limited)} files failed (rate limit); waiting {WAIT_SECONDS}s", flush=True)
            time.sleep(WAIT_SECONDS)
    print(f"Done: {sum(1 for _ in TARGET.glob('Group*/*/*.mp4'))} videos in {TARGET}", flush=True)


if __name__ == "__main__":
    main()
