#!/usr/bin/env python3
"""Extract high-score frames (score > 0.5) from v6 inference cache for manual review."""

import sys
import shutil
from pathlib import Path
from functools import partial

print = partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(TALON_DIR / "inference"))
from clip_classifier import TalonClassifier

MODEL_PATH = TALON_DIR / "models" / "talon_best_v6.pt"
CACHE_DIR = TALON_DIR / "data" / "inference_cache"
OUT_DIR = TALON_DIR / "data" / "false_positives_v6"

THRESHOLD = 0.5
CODES = ["EYAN-215", "DASS-885", "NMSL-034", "MADV-622"]


def process_video(classifier: TalonClassifier, code: str):
    out_dir = OUT_DIR / code
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect frames from both coarse and fine
    all_frames = []
    for stage in ("coarse", "fine"):
        stage_dir = CACHE_DIR / code / stage
        if not stage_dir.is_dir():
            continue
        frames = sorted(stage_dir.glob("*.jpg"))
        for f in frames:
            all_frames.append((f, stage))

    print(f"\n  [{code}] {len(all_frames)} total frames (coarse + fine)")
    if not all_frames:
        print(f"    No frames found")
        return

    # Batch inference on all frames
    print(f"    Running model...")
    paths = [str(f) for f, _ in all_frames]
    scores = classifier.predict_batch(paths, batch_size=64)

    # Filter and collect hits
    hits = []
    for (frame, stage), score in zip(all_frames, scores):
        if score < THRESHOLD:
            continue
        hits.append((frame, score, stage))

    # Sort by score descending
    hits.sort(key=lambda x: -x[1])

    # Copy with score-prefixed naming: {CODE}_s{score:.3f}_{stage}_{original_stem}.jpg
    # Sorting by filename = sorting by score (descending, since 0.999 > 0.500)
    for frame, score, stage in hits:
        new_name = f"{code}_s{score:.3f}_{stage}_{frame.stem}.jpg"
        shutil.copy2(frame, out_dir / new_name)

    # Stats
    buckets = {">0.9": 0, ">0.8": 0, ">0.7": 0, ">0.5": 0}
    for _, score, _ in hits:
        if score > 0.9:
            buckets[">0.9"] += 1
        if score > 0.8:
            buckets[">0.8"] += 1
        if score > 0.7:
            buckets[">0.7"] += 1
        buckets[">0.5"] += 1

    print(f"    Hits (>{THRESHOLD}): {len(hits)}")
    print(f"    Distribution:")
    print(f"      >0.9: {buckets['>0.9']:>4}")
    print(f"      >0.8: {buckets['>0.8']:>4}")
    print(f"      >0.7: {buckets['>0.7']:>4}")
    print(f"      >0.5: {buckets['>0.5']:>4}")

    if hits:
        print(f"    Top 10:")
        for frame, score, stage in hits[:10]:
            print(f"      {stage:6s}  {frame.name}  score={score:.3f}")

    print(f"    Saved to: {out_dir}")


def main():
    print(f"{'='*60}")
    print(f"  TALON v6 High-Score Frame Extractor")
    print(f"{'='*60}")

    classifier = TalonClassifier(MODEL_PATH)
    if not classifier.ready:
        print("FATAL: Model failed to load")
        sys.exit(1)

    total = 0
    for code in CODES:
        process_video(classifier, code)
        out_dir = OUT_DIR / code
        if out_dir.is_dir():
            total += sum(1 for f in out_dir.iterdir() if f.suffix == ".jpg")

    print(f"\n{'='*60}")
    print(f"  Done. {total} frames extracted to {OUT_DIR.relative_to(TALON_DIR)}/")
    print(f"  Browse and drag to labels/negative/[X,Y,Z] for hard negatives")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
