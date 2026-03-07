#!/usr/bin/env python3
"""
TALON — Extract coarse-scan positive frames for manual review.

Reads coarse_scan_results/*.json, copies positive frames to
data/coarse_positive/{code}/ for easy browsing in Finder.

Usage:
    python3 projects/talon/scripts/extract_coarse_positive.py
"""

import json
import shutil
from pathlib import Path

TALON_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = TALON_DIR / "data" / "coarse_scan_results"
RAW_FRAMES_DIR = TALON_DIR / "data" / "raw_frames"
OUT_DIR = TALON_DIR / "data" / "coarse_positive"


def main():
    results = sorted(RESULTS_DIR.glob("*.json"))
    if not results:
        print("No scan results found.")
        return

    total_copied = 0

    for rp in results:
        data = json.loads(rp.read_text())
        code = data["code"]
        positives = data.get("positive_frames", [])

        if not positives:
            print(f"  {code:<15s}   0 frames (skip)")
            continue

        src_dir = RAW_FRAMES_DIR / code
        dst_dir = OUT_DIR / code
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for name in positives:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)
                copied += 1

        total_copied += copied
        print(f"  {code:<15s} {copied:3d} frames")

    print(f"  {'-'*25}")
    print(f"  {'TOTAL':<15s} {total_copied:3d} frames")
    print(f"\n  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
