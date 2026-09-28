"""Create the KArSL-100 label map and a signer-aware sequence manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]


def load_labels(workbook: Path) -> dict[int, dict[str, str | int]]:
    sheet = load_workbook(workbook, read_only=True, data_only=True).active
    labels: dict[int, dict[str, str | int]] = {}
    for sign_id, arabic, english in list(sheet.values)[1:]:
        if sign_id is None or not 71 <= int(sign_id) <= 170:
            continue
        labels[int(sign_id)] = {
            "class_index": int(sign_id) - 71,
            "sign_id": int(sign_id),
            "arabic": str(arabic).strip(),
            "english": str(english).strip(),
        }
    if len(labels) != 100:
        raise ValueError(f"Expected 100 labels (71-170), found {len(labels)}")
    return labels


def build_manifest(dataset: Path, labels: dict[int, dict[str, str | int]]) -> list[dict]:
    manifest: list[dict] = []
    for sign_id, label in labels.items():
        class_dir = dataset / f"{sign_id:04d}"
        if not class_dir.exists():
            continue
        for sequence in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            parts = sequence.name.split("_")
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            images = sorted(sequence.glob("*.jpg"))
            if not images:
                continue
            manifest.append(
                {
                    **label,
                    "signer": int(parts[0]),
                    "take": int(parts[1]) if parts[1].isdigit() else -1,
                    "sequence": sequence.relative_to(ROOT).as_posix(),
                    "frames": len(images),
                }
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "KArSL100_frames")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "references" / "karsl_word_recognition" / "KARSL-502_Labels.xlsx",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "karsl100_manifest.json")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    manifest = build_manifest(args.dataset, labels)
    payload = {
        "dataset": "KArSL-100",
        "classes": list(labels.values()),
        "sequences": manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    signers = sorted({row["signer"] for row in manifest})
    print(f"Wrote {len(manifest)} sequences, {len(labels)} labels, signers={signers}: {args.output}")


if __name__ == "__main__":
    main()
