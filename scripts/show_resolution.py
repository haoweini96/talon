#!/usr/bin/env python3
"""
Generate a side-by-side comparison of Original / CLIP 224x224 / SigLIP2 378x378
preprocessing for randomly sampled handjob frames.

Usage:
    python scripts/show_resolution.py
    python scripts/show_resolution.py --samples 5
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HANDJOB_DIR = DATA_DIR / "labels" / "handjob"
OUTPUT_PATH = DATA_DIR / "resolution_compare.jpg"

DISPLAY_HEIGHT = 400
LABEL_HEIGHT = 30
PADDING = 10
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)


def resize_short_edge(img: Image.Image, target: int) -> Image.Image:
    """Resize so the shorter edge equals target, preserving aspect ratio."""
    w, h = img.size
    if w < h:
        new_w = target
        new_h = int(h * target / w)
    else:
        new_h = target
        new_w = int(w * target / h)
    return img.resize((new_w, new_h), Image.BICUBIC)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Center crop to size x size."""
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess_clip(img: Image.Image) -> Image.Image:
    """CLIP ViT-B/32 & ViT-L/14: resize short edge to 224, center crop 224x224."""
    return center_crop(resize_short_edge(img, 224), 224)


def preprocess_siglip2(img: Image.Image) -> Image.Image:
    """SigLIP2 ViT-SO400M-14: resize short edge to 378, center crop 378x378."""
    return center_crop(resize_short_edge(img, 378), 378)


def scale_to_height(img: Image.Image, h: int) -> Image.Image:
    """Scale image to given height, preserving aspect ratio."""
    w_new = int(img.width * h / img.height)
    return img.resize((w_new, h), Image.BICUBIC)


def make_row(img_path: Path) -> Image.Image:
    """Create one row: Original | CLIP 224 | SigLIP2 378, all scaled to DISPLAY_HEIGHT."""
    orig = Image.open(img_path).convert("RGB")

    clip_img = preprocess_clip(orig)
    siglip_img = preprocess_siglip2(orig)

    # Scale all to same display height
    orig_disp = scale_to_height(orig, DISPLAY_HEIGHT)
    clip_disp = scale_to_height(clip_img, DISPLAY_HEIGHT)
    siglip_disp = scale_to_height(siglip_img, DISPLAY_HEIGHT)

    panels = [
        (f"Original ({orig.width}x{orig.height})", orig_disp),
        ("CLIP 224x224", clip_disp),
        ("SigLIP2 378x378", siglip_disp),
    ]

    total_w = sum(p.width for _, p in panels) + PADDING * (len(panels) + 1)
    row_h = DISPLAY_HEIGHT + LABEL_HEIGHT + PADDING

    row = Image.new("RGB", (total_w, row_h), BG_COLOR)
    draw = ImageDraw.Draw(row)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    x = PADDING
    for label, panel in panels:
        # Draw label
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (panel.width - tw) // 2
        draw.text((tx, 4), label, fill=TEXT_COLOR, font=font)

        # Paste panel
        row.paste(panel, (x, LABEL_HEIGHT))
        x += panel.width + PADDING

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    images = [f for f in HANDJOB_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    chosen = random.sample(images, min(args.samples, len(images)))

    rows = [make_row(p) for p in chosen]

    # Stack rows vertically
    max_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + PADDING * (len(rows) + 1)

    canvas = Image.new("RGB", (max_w, total_h), BG_COLOR)
    y = PADDING
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + PADDING

    canvas.save(OUTPUT_PATH, quality=90)
    print(f"Saved to {OUTPUT_PATH} ({canvas.width}x{canvas.height})")

    for p in chosen:
        orig = Image.open(p)
        print(f"  {p.name}: {orig.width}x{orig.height}")


if __name__ == "__main__":
    main()
