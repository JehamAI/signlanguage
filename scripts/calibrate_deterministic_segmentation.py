"""Export motion/hand signals and grid-search deterministic cut parameters.

Isharah (and most CSLR sets) do not ship per-gloss frame boundaries. Calibration here uses:
  1) Rich signal + cut exports for visual inspection.
  2) Weak count target: |#segments - #gloss tokens| using sentence gloss lists only.

That does not prove cuts are temporally correct, but it helps tune thresholds when you
inspect overlays and merged KArSL demo clips with known pauses.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hand_track import extract_hand_track
from app.segmentation_analysis import MotionValleyParams, motion_valley_video_regions, timeline_payload, hand_pause_video_regions


def _count_error(segment_count: int, gloss_count: int) -> int:
    return abs(segment_count - gloss_count)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "isharah" / "samples" / "manifest.json",
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--fetch", type=int, default=0, help="Download this many Isharah clips first")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "segmentation_calibration")
    args = parser.parse_args()

    if args.fetch:
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "fetch_isharah_samples.py"),
                "--limit",
                str(args.fetch),
            ]
        )

    if not args.manifest.exists():
        print("Missing manifest. Run: python scripts/fetch_isharah_samples.py --limit 30", flush=True)
        raise SystemExit(1)

    samples = json.loads(args.manifest.read_text(encoding="utf-8"))[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    default_params = MotionValleyParams()
    timelines_dir = args.out / "timelines"
    timelines_dir.mkdir(parents=True, exist_ok=True)

    cached_tracks = {str(ROOT / sample["video_path"]): extract_hand_track(ROOT / sample["video_path"]) for sample in samples}

    records: list[dict] = []
    for sample in samples:
        video = ROOT / sample["video_path"]
        gloss_count = len(sample.get("gloss_tokens") or sample["gloss_sequence"].split())
        track = cached_tracks[str(video)]
        analysis = timeline_payload(track, default_params)
        pause_regions = hand_pause_video_regions(video)
        analysis["hand_pause_segment_count"] = len(pause_regions)
        analysis["hand_pause_regions"] = [{"start_frame": s, "end_frame": e} for s, e in pause_regions]
        record = {
            "id": sample["id"],
            "gloss_count": gloss_count,
            "gloss_sequence": sample.get("gloss_sequence"),
            "motion_valley_segment_count": analysis["segment_count"],
            "hand_pause_segment_count": analysis["hand_pause_segment_count"],
            "count_error_motion_default": _count_error(analysis["segment_count"], gloss_count),
            "count_error_hand_pause": _count_error(analysis["hand_pause_segment_count"], gloss_count),
        }
        records.append(record)
        timeline_path = timelines_dir / f"{sample['id']}.json"
        timeline_path.write_text(
            json.dumps({**analysis, "id": sample["id"], "gloss_count": gloss_count}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"{sample['id']}: glosses={gloss_count} "
            f"motion_segments={analysis['segment_count']} "
            f"hand_pause={analysis['hand_pause_segment_count']}",
            flush=True,
        )

    valley_ratios = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    smooth_radii = [2, 3, 4]
    minimum_gaps = [4, 6, 8, 10]
    minimum_sign_seconds_list = [0.22, 0.28, 0.34, 0.40]

    grid: list[dict] = []
    for valley_ratio, smooth_radius, minimum_gap, minimum_sign_seconds in product(
        valley_ratios, smooth_radii, minimum_gaps, minimum_sign_seconds_list
    ):
        params = MotionValleyParams(
            smooth_radius=smooth_radius,
            valley_ratio=valley_ratio,
            minimum_gap=minimum_gap,
            minimum_sign_seconds=minimum_sign_seconds,
        )
        errors: list[int] = []
        segment_counts: list[int] = []
        for sample in samples:
            video = ROOT / sample["video_path"]
            gloss_count = len(sample.get("gloss_tokens") or sample["gloss_sequence"].split())
            track = cached_tracks[str(video)]
            _valleys, regions = motion_valley_video_regions(track, params)
            segment_counts.append(len(regions))
            errors.append(_count_error(len(regions), gloss_count))
        grid.append(
            {
                "params": {
                    "valley_ratio": valley_ratio,
                    "smooth_radius": smooth_radius,
                    "minimum_gap": minimum_gap,
                    "minimum_sign_seconds": minimum_sign_seconds,
                },
                "mean_count_error": sum(errors) / len(errors) if errors else 0.0,
                "median_count_error": float(sorted(errors)[len(errors) // 2]) if errors else 0.0,
                "mean_segment_count": sum(segment_counts) / len(segment_counts) if segment_counts else 0.0,
            }
        )

    grid.sort(key=lambda item: (item["mean_count_error"], item["median_count_error"]))
    best = grid[0] if grid else None

    summary = {
        "samples": len(samples),
        "note": (
            "No frame-level gloss boundaries in Isharah. mean_count_error only compares "
            "segment count to gloss token count; inspect timelines/*.json signals to validate cuts."
        ),
        "default_motion_valley_params": {
            "smooth_radius": default_params.smooth_radius,
            "valley_ratio": default_params.valley_ratio,
            "minimum_gap": default_params.minimum_gap,
            "minimum_sign_seconds": default_params.minimum_sign_seconds,
        },
        "best_motion_valley_by_count_error": best,
        "top_grid_candidates": grid[:15],
        "per_sample": records,
    }
    summary_path = args.out / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if best:
        print(
            "BEST (weak count target): "
            f"valley_ratio={best['params']['valley_ratio']} "
            f"smooth_radius={best['params']['smooth_radius']} "
            f"minimum_gap={best['params']['minimum_gap']} "
            f"minimum_sign_seconds={best['params']['minimum_sign_seconds']} "
            f"mean_count_error={best['mean_count_error']:.2f}",
            flush=True,
        )
    print(f"Wrote {summary_path} and {len(records)} timelines under {timelines_dir}", flush=True)


if __name__ == "__main__":
    main()
