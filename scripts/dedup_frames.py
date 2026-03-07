#!/usr/bin/env python3
"""
Frame deduplication for labeled data.

Scans data/labels/handjob/ and data/labels/negative/*/ for near-duplicate
frames from the same video. Uses pixel MSE on downscaled grayscale images
for fast comparison.

Usage:
    python scripts/dedup_frames.py                      # dry-run (default)
    python scripts/dedup_frames.py --execute            # move dupes to data/dedup_removed/
    python scripts/dedup_frames.py --threshold 0.90     # stricter dedup
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS_DIR = DATA_DIR / "labels"
REMOVED_DIR = DATA_DIR / "dedup_removed"

COMPARE_SIZE = (128, 128)  # downscale for fast comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_video_code(filename: str) -> str:
    """Extract video code from filename.

    AVSA-325_frame_005337.jpg  → AVSA-325
    CAWD-937_scan1_0_12_25.jpg → CAWD-937
    frame_002660.jpg           → _unknown_
    """
    m = re.match(r'^([A-Za-z]+-\d+)', filename)
    return m.group(1) if m else "_unknown_"


def load_gray(path: Path) -> np.ndarray:
    """Load image as grayscale float32 array, downscaled."""
    img = Image.open(path).convert("L").resize(COMPARE_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def mse_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return similarity score 0..1 (1 = identical). Based on MSE."""
    mse = np.mean((a - b) ** 2)
    # MSE range for uint8 grayscale: 0 (identical) to 65025 (black vs white)
    return 1.0 - mse / 65025.0


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def collect_label_folders() -> list[tuple[Path, str]]:
    """Return [(folder_path, category_label), ...]."""
    folders = []

    hj = LABELS_DIR / "handjob"
    if hj.is_dir():
        folders.append((hj, "handjob"))

    neg = LABELS_DIR / "negative"
    if neg.is_dir():
        for sub in sorted(neg.iterdir()):
            if sub.is_dir():
                folders.append((sub, f"negative/{sub.name}"))

    return folders


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        (f for f in folder.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')),
        key=lambda p: p.name,
    )


# ---------------------------------------------------------------------------
# Dedup logic
# ---------------------------------------------------------------------------
def find_duplicates(images: list[Path], threshold: float) -> list[Path]:
    """Given sorted images from one video, return paths to remove (keep first of each cluster)."""
    if len(images) < 2:
        return []

    to_remove = []
    prev = load_gray(images[0])

    for img_path in images[1:]:
        curr = load_gray(img_path)
        sim = mse_similarity(prev, curr)
        if sim >= threshold:
            to_remove.append(img_path)
        else:
            prev = curr  # new cluster anchor

    return to_remove


def process_folder(
    folder: Path, category: str, threshold: float, execute: bool
) -> tuple[int, int]:
    """Process one label folder. Returns (total_images, removed_count)."""
    images = collect_images(folder)
    if not images:
        return 0, 0

    # Group by video code
    by_video: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        code = extract_video_code(img.name)
        by_video[code].append(img)

    total_removed = 0
    video_details = []

    for code in sorted(by_video):
        frames = by_video[code]
        dupes = find_duplicates(frames, threshold)
        if dupes:
            video_details.append((code, len(frames), len(dupes)))
            total_removed += len(dupes)

            if execute:
                # Mirror category subfolder structure in dedup_removed/
                dest_base = REMOVED_DIR / category
                dest_base.mkdir(parents=True, exist_ok=True)
                for dupe_path in dupes:
                    shutil.move(str(dupe_path), str(dest_base / dupe_path.name))

    # Print per-video details
    if video_details:
        for code, count, removed in video_details:
            print(f"    {code}: {count} frames → remove {removed}, keep {count - removed}")

    return len(images), total_removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Deduplicate labeled frames")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Similarity threshold 0..1 (default: 0.95)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually move duplicates (default: dry-run)")
    args = parser.parse_args()

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"{'=' * 60}")
    print(f"Frame deduplication — {mode} (threshold={args.threshold})")
    print(f"{'=' * 60}\n")

    folders = collect_label_folders()
    if not folders:
        print("No label folders found.")
        return

    grand_total = 0
    grand_removed = 0
    category_stats = []

    for folder, category in folders:
        print(f"[{category}]")
        total, removed = process_folder(folder, category, args.threshold, args.execute)
        kept = total - removed
        category_stats.append((category, total, removed, kept))
        grand_total += total
        grand_removed += removed

        if removed == 0:
            print(f"    no duplicates found ({total} frames)")
        print()

    # Summary
    print(f"{'=' * 60}")
    print(f"{'Category':<40} {'Before':>7} {'Remove':>7} {'After':>7}")
    print(f"{'-' * 60}")
    for cat, total, removed, kept in category_stats:
        print(f"{cat:<40} {total:>7} {removed:>7} {kept:>7}")
    print(f"{'-' * 60}")
    print(f"{'TOTAL':<40} {grand_total:>7} {grand_removed:>7} {grand_total - grand_removed:>7}")

    if not args.execute and grand_removed > 0:
        print(f"\nThis was a dry run. Use --execute to move {grand_removed} duplicates to {REMOVED_DIR}")


if __name__ == "__main__":
    main()
