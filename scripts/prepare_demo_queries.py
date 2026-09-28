"""Copy combined demo inputs into demo_queries/ with numbered filenames."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_paper_examples import EXAMPLES

ORDER = [
    ("01-airport-medical.mp4", "airport-medical"),
    ("02-airport-luggage.mp4", "airport-luggage"),
    ("03-library-access.mp4", "library-access"),
    ("04-restroom-directions.mp4", "restroom-directions"),
    ("05-headache-fever.mp4", "headache-fever"),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    source_root = ROOT / "paper_examples" / "input"
    dest_root = ROOT / "demo_queries"
    dest_root.mkdir(parents=True, exist_ok=True)
    by_id = {item.example_id: item for item in EXAMPLES}
    manifest: list[dict] = []
    for filename, example_id in ORDER:
        source = source_root / f"{example_id}.mp4"
        if not source.exists():
            raise FileNotFoundError(f"Missing {source}; run scripts/build_paper_examples.py --skip-llm first")
        target = dest_root / filename
        shutil.copy2(source, target)
        example = by_id[example_id]
        manifest.append(
            {
                "file": str(target.relative_to(ROOT)),
                "example_id": example_id,
                "title_ar": example.title_ar,
                "expected_glosses": list(example.glosses),
            }
        )
        print(f"Wrote {target.name}")
    (dest_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(dest_root / "manifest.json")


if __name__ == "__main__":
    main()
