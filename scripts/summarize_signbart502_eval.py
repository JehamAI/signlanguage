"""Print top-1 accuracy from artifacts/karsl502_isolated_evaluation.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "artifacts" / "karsl502_isolated_evaluation.json"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not OUTPUT.exists():
        print("No evaluation file yet. Run: python scripts/evaluate_signbart502.py --resume")
        return
    records = json.loads(OUTPUT.read_text(encoding="utf-8"))
    evaluated = [r for r in records if r.get("readable")]
    correct = sum(1 for r in evaluated if r.get("correct"))
    fails = [r for r in evaluated if not r.get("correct")]
    acc = (correct / len(evaluated) * 100) if evaluated else 0.0
    print(f"Progress: {len(records)}/502 dictionary clips")
    print(f"Top-1 on evaluated: {correct}/{len(evaluated)} ({acc:.1f}%)")
    print(f"Failures ({len(fails)}):")
    for r in fails:
        sid = int(r["sign_id"])
        print(
            f"  {sid:04d}: {r['expected']} -> {r['predicted']} "
            f"(conf={float(r.get('confidence', 0)):.3f})"
        )


if __name__ == "__main__":
    main()
