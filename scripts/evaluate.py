"""
TALON — Model evaluation script.

Evaluates a trained CLIP classifier on the test split.

Usage:
    python evaluate.py --model models/clip_handjob_v1.pt [--split test]
"""

import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent
SPLITS_DIR = TALON_DIR / "data" / "splits"
MODELS_DIR = TALON_DIR / "models"


def main():
    parser = argparse.ArgumentParser(description="TALON — Evaluate model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    split_path = SPLITS_DIR / f"{args.split}.json"
    if not split_path.exists():
        print(f"Split file not found: {split_path}")
        print("Run prepare_dataset.py first.")
        return

    samples = json.loads(split_path.read_text())
    print(f"Loaded {len(samples)} samples from {args.split} split")

    # TODO: Load model and run inference
    print(f"\nModel: {args.model}")
    print("Evaluation not yet implemented — waiting for trained model.")
    print("This script will compute: accuracy, precision, recall, F1, confusion matrix.")


if __name__ == "__main__":
    main()
