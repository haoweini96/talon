"""
TALON v2 — Prepare dataset splits for CLIP fine-tuning.

Scans structured label directories:
  - data/labels/handjob/         → positives  (category [1,1,1])
  - data/labels/negative/[X,Y,Z] label/  → negatives (category from bracket prefix)

Builds train/val/test splits stratified by video (all frames from same video in same split).

Usage:
    python prepare_dataset.py [--train-ratio 0.6] [--val-ratio 0.2] [--seed 42]
"""

import re
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent
DATA_DIR = TALON_DIR / "data"
LABELS_DIR = DATA_DIR / "labels"
SPLITS_DIR = DATA_DIR / "splits"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CATEGORY_RE = re.compile(r"^\[[\d,]+\]")  # e.g. [0,1,1]


def extract_video_code(filename: str) -> str:
    """Extract video code from frame filename.

    Handles: MIDE-993_scan1_0_10_00.jpg, BBI-160_frame_005240.jpg,
             HMN-786_score0.50_frame_002270.jpg,
             DASS-885_s0.981_fine_DASS-885_frame_006517.jpg
    """
    # JAV code is always at the start: LETTERS-DIGITS
    m = re.match(r"^([A-Z]+-\d+)", filename)
    if m:
        return m.group(1)
    return "UNKNOWN"


def extract_category(folder_name: str) -> str:
    """Extract bracket category from folder name like '[0,1,1] 有胸动作没有脸' → '[0,1,1]'."""
    m = CATEGORY_RE.match(folder_name)
    return m.group(0) if m else folder_name


def collect_samples() -> tuple[list[dict], list[dict]]:
    """Scan label dirs to build positive/negative sample lists.

    Returns (positives, negatives).
    """
    positives = []
    negatives = []

    # ── Positives: labels/handjob/ (flat or subdirs) ──
    handjob_dir = LABELS_DIR / "handjob"
    if handjob_dir.exists():
        for f in sorted(handjob_dir.glob("**/*")):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            video = extract_video_code(f.name)
            positives.append({
                "path": str(f.relative_to(TALON_DIR)),
                "label": 1,
                "video": video,
                "category": "[1,1,1]",
            })

    # ── Negatives: labels/negative/[X,Y,Z] label/ ──
    neg_dir = LABELS_DIR / "negative"
    if neg_dir.exists():
        for subfolder in sorted(neg_dir.iterdir()):
            if not subfolder.is_dir() or subfolder.name.startswith("."):
                continue
            category = extract_category(subfolder.name)
            for f in sorted(subfolder.glob("*")):
                if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                    continue
                video = extract_video_code(f.name)
                negatives.append({
                    "path": str(f.relative_to(TALON_DIR)),
                    "label": 0,
                    "video": video,
                    "category": category,
                })

    return positives, negatives


def stratified_split_by_video(
    samples: list[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples so all frames from one video stay in the same split.

    Uses greedy min-max fill: videos sorted by total count (desc), each
    assigned to the split that minimizes max(pos_fill, neg_fill), keeping
    both pos and neg ratios balanced across all splits.
    """
    test_ratio = 1 - train_ratio - val_ratio

    by_video: dict[str, list[dict]] = defaultdict(list)
    pos_per_video: dict[str, int] = defaultdict(int)
    neg_per_video: dict[str, int] = defaultdict(int)
    for s in samples:
        by_video[s["video"]].append(s)
        if s["label"] == 1:
            pos_per_video[s["video"]] += 1
        else:
            neg_per_video[s["video"]] += 1

    videos = list(by_video.keys())

    if len(videos) <= 2:
        return [s for v in videos for s in by_video[v]], [], []

    total_pos = sum(pos_per_video.values())
    total_neg = sum(neg_per_video.values())

    target_pos = {
        "train": max(total_pos * train_ratio, 1),
        "val": max(total_pos * val_ratio, 1),
        "test": max(total_pos * test_ratio, 1),
    }
    target_neg = {
        "train": max(total_neg * train_ratio, 1),
        "val": max(total_neg * val_ratio, 1),
        "test": max(total_neg * test_ratio, 1),
    }

    # Largest videos first for better greedy packing
    videos.sort(key=lambda v: len(by_video[v]), reverse=True)

    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    current_pos = {"train": 0, "val": 0, "test": 0}
    current_neg = {"train": 0, "val": 0, "test": 0}

    for v in videos:
        vp = pos_per_video[v]
        vn = neg_per_video[v]

        best_split = min(
            assigned.keys(),
            key=lambda s: max(
                (current_pos[s] + vp) / target_pos[s],
                (current_neg[s] + vn) / target_neg[s],
            ),
        )
        assigned[best_split].append(v)
        current_pos[best_split] += vp
        current_neg[best_split] += vn

    train = [s for v in assigned["train"] for s in by_video[v]]
    val = [s for v in assigned["val"] for s in by_video[v]]
    test = [s for v in assigned["test"] for s in by_video[v]]

    return train, val, test


def print_split_stats(data: list[dict], name: str):
    """Print per-split statistics: pos/neg count, per-category, per-video."""
    n = len(data)
    pos = sum(1 for s in data if s["label"] == 1)
    neg = n - pos
    ratio = f"1:{neg/pos:.1f}" if pos > 0 else "n/a"

    # Per category
    by_cat: dict[str, int] = defaultdict(int)
    for s in data:
        by_cat[s["category"]] += 1

    # Per video
    by_vid: dict[str, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    for s in data:
        key = "pos" if s["label"] == 1 else "neg"
        by_vid[s["video"]][key] += 1

    pos_pct = f"{pos/n*100:.1f}%" if n > 0 else "n/a"
    print(f"\n  {name:6s}: {n:5d} samples ({pos:4d} pos / {neg:4d} neg) | ratio {ratio} | pos% {pos_pct} | {len(by_vid)} videos")

    # Category breakdown
    print(f"         categories:")
    for cat in sorted(by_cat):
        print(f"           {cat:10s}  {by_cat[cat]:5d}")

    # Video breakdown
    print(f"         videos:")
    for v in sorted(by_vid):
        d = by_vid[v]
        print(f"           {v:15s}  pos={d['pos']:4d}  neg={d['neg']:4d}")


def main():
    parser = argparse.ArgumentParser(description="TALON v2 — Prepare dataset splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning directories...")
    positives, negatives = collect_samples()

    n_pos = len(positives)
    n_neg = len(negatives)
    n_total = n_pos + n_neg

    # Per-category counts
    cat_counts: dict[str, int] = defaultdict(int)
    for s in positives + negatives:
        cat_counts[s["category"]] += 1

    print(f"\n  Dataset Summary")
    print(f"  {'─' * 40}")
    print(f"  Positive (handjob):     {n_pos}")
    print(f"  Negative (structured):  {n_neg}")
    print(f"  Total:                  {n_total}")
    print(f"\n  Per-category:")
    for cat in sorted(cat_counts):
        print(f"    {cat:10s}  {cat_counts[cat]:5d}")

    if n_pos == 0:
        print("\n  No positive samples found in data/labels/handjob/")
        print("  Please label some frames first.")

    all_samples = positives + negatives
    train, val, test = stratified_split_by_video(all_samples, args.train_ratio, args.val_ratio)

    # Save splits
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = SPLITS_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # Print stats
    print(f"\n  Splits (seed={args.seed}, ratio {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio:.1f}):")
    print_split_stats(train, "train")
    print_split_stats(val, "val")
    print_split_stats(test, "test")

    print(f"\n  Saved to: {SPLITS_DIR}/")
    print(f"    train.json: {len(train)} samples")
    print(f"    val.json:   {len(val)} samples")
    print(f"    test.json:  {len(test)} samples")


if __name__ == "__main__":
    main()
