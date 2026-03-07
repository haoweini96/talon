"""
TALON — GPT pre-screening for handjob frames.

Batches raw frames into sheets of 36, sends to GPT for classification,
copies predicted handjob frames to labels/gpt_predicted_handjob/.

Usage:
    python gpt_prescreening.py [--batch-size 36] [--workers 4]
"""

import sys
import os
import json
import time
import shutil
import base64
import argparse
import concurrent.futures
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent
PROJECTS_DIR = TALON_DIR.parent
MEGA_DIR = PROJECTS_DIR / "mega"

load_dotenv(MEGA_DIR / ".env")

DATA_DIR = TALON_DIR / "data"
RAW_FRAMES_DIR = DATA_DIR / "raw_frames"
GPT_PREDICTED_DIR = DATA_DIR / "labels" / "gpt_predicted_handjob"

GPT_MODEL = "gpt-5.2"

# Sheet layout (same style as scene_analyzer)
COLS = 6
SW, SH = 240, 135
LH, M, HH = 20, 3, 28

try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    FONT_H = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except Exception:
    FONT = ImageFont.load_default()
    FONT_H = FONT

PROMPT = """这些是 JAV 视频的帧截图。请列出哪些帧中女主正在用手服务男主（handjob）。
判断标准：女主面部可见、上半身可见、手臂从她身体向下伸出做握持动作。
男主的手触摸女主不算。
每张图片下方标注了该帧的文件名。请仔细看每一帧，只返回符合条件的帧的文件名。
只返回 JSON：{"handjob_frames": ["文件名1", "文件名2", ...]}
如果没有任何帧符合条件，返回 {"handjob_frames": []}"""


def build_screening_sheet(frame_paths: list[Path], out_path: Path) -> Path:
    """Build a sheet of frames with filename labels for GPT screening."""
    n = len(frame_paths)
    rows = (n + COLS - 1) // COLS
    sheet_w = COLS * (SW + M) + M
    sheet_h = HH + rows * (SH + LH + M) + M

    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((M + 2, 5), f"Screening batch ({n} frames)", fill=(255, 255, 100), font=FONT_H)

    for i, fp in enumerate(frame_paths):
        r, c = divmod(i, COLS)
        x = M + c * (SW + M)
        y = HH + M + r * (SH + LH + M)

        try:
            img = Image.open(fp).resize((SW, SH), Image.LANCZOS)
            sheet.paste(img, (x, y))
        except Exception:
            pass

        # Label with filename (truncated to fit)
        name = fp.name
        if len(name) > 28:
            name = name[:25] + "..."
        draw.rectangle([x, y + SH, x + SW, y + SH + LH], fill=(50, 50, 50))
        draw.text((x + 2, y + SH + 3), name, fill=(200, 200, 200), font=FONT)

    sheet.save(out_path, quality=92)
    return out_path


def classify_sheet(sheet_path: Path, frame_names: list[str]) -> list[str]:
    """Send a sheet to GPT and get predicted handjob frame names."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b64 = base64.b64encode(sheet_path.read_bytes()).decode()

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
        ]}],
        max_completion_tokens=2048,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(response.choices[0].message.content)
        predicted = data.get("handjob_frames", [])
        # Validate: only return names that are in the actual batch
        name_set = set(frame_names)
        return [n for n in predicted if n in name_set]
    except Exception as e:
        print(f"    Parse error: {e}")
        return []


def process_batch(batch_idx: int, total_batches: int, frame_paths: list[Path], tmp_dir: Path) -> list[str]:
    """Process one batch: build sheet → classify → return handjob filenames."""
    frame_names = [fp.name for fp in frame_paths]

    sheet_path = tmp_dir / f"screen_batch_{batch_idx:03d}.jpg"
    build_screening_sheet(frame_paths, sheet_path)

    t0 = time.time()
    predicted = classify_sheet(sheet_path, frame_names)
    elapsed = time.time() - t0

    n_hj = len(predicted)
    tag = f" → {n_hj} handjob" if n_hj > 0 else ""
    print(f"  [{batch_idx}/{total_batches}] {len(frame_paths)} frames, {elapsed:.1f}s{tag}")

    return predicted


def main():
    parser = argparse.ArgumentParser(description="TALON — GPT pre-screening")
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--workers", type=int, default=4, help="Parallel GPT calls")
    args = parser.parse_args()

    GPT_PREDICTED_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all frames (recursive: raw_frames/{code}/*.jpg)
    all_frames = sorted(RAW_FRAMES_DIR.glob("**/*.jpg"))
    if not all_frames:
        print("No frames in data/raw_frames/")
        return

    # Skip already-screened frames (already in gpt_predicted_handjob/)
    already_predicted = {f.name for f in GPT_PREDICTED_DIR.glob("**/*.jpg")}

    print(f"{'='*70}")
    print(f"  TALON GPT Pre-screening")
    print(f"{'='*70}")
    print(f"  Total frames:     {len(all_frames)}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Parallel workers: {args.workers}")
    print(f"{'='*70}")

    # Build batches
    batches = []
    for i in range(0, len(all_frames), args.batch_size):
        batches.append(all_frames[i:i + args.batch_size])

    total_batches = len(batches)
    print(f"  Batches: {total_batches}")

    # Temp dir for sheets
    tmp_dir = Path("/tmp/talon_screening")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Process batches in parallel
    all_predicted = []
    t_start = time.time()

    def _run(item):
        idx, batch = item
        return process_batch(idx, total_batches, batch, tmp_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run, enumerate(batches, 1)))

    for predicted in results:
        all_predicted.extend(predicted)

    total_elapsed = time.time() - t_start

    # Copy predicted handjob frames into per-video subdirs
    copied = 0
    for name in all_predicted:
        video = name.split("_scan")[0]
        src = RAW_FRAMES_DIR / video / name
        if not src.exists():
            # fallback: try flat
            src = RAW_FRAMES_DIR / name
        dst_dir = GPT_PREDICTED_DIR / video
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    # Statistics
    video_counts = {}
    for name in all_predicted:
        video = name.split("_scan")[0]
        video_counts[video] = video_counts.get(video, 0) + 1

    print(f"\n{'='*70}")
    print(f"  SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  Total frames scanned:    {len(all_frames)}")
    print(f"  GPT predicted handjob:   {len(all_predicted)}")
    print(f"  Newly copied:            {copied}")
    print(f"  Already in predicted/:   {len(already_predicted)}")
    print(f"  Time:                    {total_elapsed:.0f}s")
    print(f"\n  By video:")
    for video, count in sorted(video_counts.items()):
        print(f"    {video:15s}: {count} frames")

    print(f"\n  Next steps:")
    print(f"    1. Browse data/labels/gpt_predicted_handjob/ ({len(all_predicted)} frames)")
    print(f"    2. Drag confirmed handjob frames to data/labels/handjob/")
    print(f"    3. Leave false positives in gpt_predicted_handjob/ (auto-negative)")


if __name__ == "__main__":
    main()
