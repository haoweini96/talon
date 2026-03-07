#!/usr/bin/env python3
"""
TALON — GPT-5.2 coarse scan for handjob frame detection.

Builds contact sheets from raw frames and sends to GPT-5.2 for
binary classification (handjob = positive, everything else = negative).

Usage:
    cd projects/mega
    venv/bin/python3 ../talon/scripts/coarse_scan_frames.py

    # Scan specific videos only
    venv/bin/python3 ../talon/scripts/coarse_scan_frames.py --videos FPRE-161 BBI-214

    # Adjust batch size and parallelism
    venv/bin/python3 ../talon/scripts/coarse_scan_frames.py --batch-size 30 --workers 6
"""

import sys
import os
import json
import time
import base64
import argparse
import concurrent.futures
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent
MEGA_DIR = TALON_DIR.parent / "mega"

load_dotenv(MEGA_DIR / ".env")

RAW_FRAMES_DIR = TALON_DIR / "data" / "raw_frames"
RESULTS_DIR = TALON_DIR / "data" / "coarse_scan_results"

GPT_MODEL = "gpt-5.2"

# Default 10 videos (batch 1 + batch 2)
TARGET_VIDEOS = [
    "FPRE-161", "PFES-110", "MIDE-068", "BBI-214", "EBOD-530",
    "CJOD-493", "PRED-800", "CJOD-484", "DASS-714", "LUKE-034",
]

# ---------------------------------------------------------------------------
# Contact sheet layout (matches gpt_prescreening.py style)
# ---------------------------------------------------------------------------
COLS = 6
SW, SH = 240, 135   # thumbnail size
LH = 20              # label height below each thumbnail
M = 3                # margin
HH = 28              # header height

try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    FONT_H = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except Exception:
    FONT = ImageFont.load_default()
    FONT_H = FONT

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT = """这些是 JAV 视频的帧截图（每帧间隔 20 秒）。请判断哪些帧中女主正在用手服务男主（handjob）。

判断标准：
- 女主面部可见、上半身可见
- 手臂从身体向下伸出做握持动作
- 男主的手触摸女主不算

每张图片下方标有文件名，请仔细看每一帧。
只返回 JSON：{"handjob_frames": ["文件名1", "文件名2", ...]}
如果没有任何帧符合条件，返回 {"handjob_frames": []}"""


# ---------------------------------------------------------------------------
# Build contact sheet
# ---------------------------------------------------------------------------
def build_sheet(frame_paths: list[Path], out_path: Path) -> Path:
    """Build a contact sheet with frame thumbnails and filename labels."""
    n = len(frame_paths)
    rows = (n + COLS - 1) // COLS
    sheet_w = COLS * (SW + M) + M
    sheet_h = HH + rows * (SH + LH + M) + M

    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((M + 2, 5), f"Coarse scan ({n} frames)", fill=(255, 255, 100), font=FONT_H)

    for i, fp in enumerate(frame_paths):
        r, c = divmod(i, COLS)
        x = M + c * (SW + M)
        y = HH + M + r * (SH + LH + M)

        try:
            img = Image.open(fp).resize((SW, SH), Image.LANCZOS)
            sheet.paste(img, (x, y))
        except Exception:
            pass

        name = fp.name
        if len(name) > 28:
            name = name[:25] + "..."
        draw.rectangle([x, y + SH, x + SW, y + SH + LH], fill=(50, 50, 50))
        draw.text((x + 2, y + SH + 3), name, fill=(200, 200, 200), font=FONT)

    sheet.save(out_path, quality=92)
    return out_path


# ---------------------------------------------------------------------------
# GPT classification
# ---------------------------------------------------------------------------
def classify_sheet(client: OpenAI, sheet_path: Path, frame_names: list[str]) -> list[str]:
    """Send contact sheet to GPT-5.2, return list of positive frame names."""
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
        name_set = set(frame_names)
        return [n for n in predicted if n in name_set]
    except Exception as e:
        print(f"    Parse error: {e}")
        raw = getattr(response.choices[0].message, "content", "")
        print(f"    Raw: {str(raw)[:200]}")
        return []


# ---------------------------------------------------------------------------
# Process one batch
# ---------------------------------------------------------------------------
def process_batch(client: OpenAI, batch_idx: int, total_batches: int,
                  frame_paths: list[Path], tmp_dir: Path) -> list[str]:
    """Build sheet -> classify -> return positive filenames."""
    frame_names = [fp.name for fp in frame_paths]
    sheet_path = tmp_dir / f"sheet_{batch_idx:03d}.jpg"
    build_sheet(frame_paths, sheet_path)

    t0 = time.time()
    predicted = classify_sheet(client, sheet_path, frame_names)
    elapsed = time.time() - t0

    n = len(predicted)
    tag = f" -> {n} positive" if n > 0 else ""
    print(f"    [{batch_idx}/{total_batches}] {len(frame_paths)} frames, {elapsed:.1f}s{tag}")

    return predicted


# ---------------------------------------------------------------------------
# Scan one video
# ---------------------------------------------------------------------------
def scan_video(code: str, batch_size: int, workers: int) -> dict:
    """Scan all frames for one video. Returns results dict."""
    frame_dir = RAW_FRAMES_DIR / code
    if not frame_dir.exists():
        print(f"\n  SKIP {code}: directory not found")
        return {"code": code, "status": "missing", "total_frames": 0, "positive": 0, "ratio": 0}

    all_frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not all_frames:
        print(f"\n  SKIP {code}: no frames found")
        return {"code": code, "status": "no_frames", "total_frames": 0, "positive": 0, "ratio": 0}

    print(f"\n  {code}: {len(all_frames)} frames, {batch_size}/batch, {workers} workers")

    # Build batches
    batches = []
    for i in range(0, len(all_frames), batch_size):
        batches.append(all_frames[i:i + batch_size])

    tmp_dir = Path("/tmp/talon_coarse_scan") / code
    tmp_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    all_positive = []
    t0 = time.time()

    if workers <= 1:
        for idx, batch in enumerate(batches, 1):
            predicted = process_batch(client, idx, len(batches), batch, tmp_dir)
            all_positive.extend(predicted)
    else:
        def _run(item):
            idx, batch = item
            return process_batch(client, idx, len(batches), batch, tmp_dir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run, enumerate(batches, 1)))
        for predicted in results:
            all_positive.extend(predicted)

    elapsed = time.time() - t0
    ratio = len(all_positive) / len(all_frames) if all_frames else 0

    result = {
        "code": code,
        "status": "done",
        "total_frames": len(all_frames),
        "positive": len(all_positive),
        "ratio": round(ratio, 4),
        "positive_frames": sorted(all_positive),
        "elapsed_s": round(elapsed, 1),
    }

    print(f"    Done: {len(all_positive)}/{len(all_frames)} positive ({ratio:.1%}) in {elapsed:.0f}s")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TALON — GPT-5.2 coarse scan")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Video codes to scan (default: all 10)")
    parser.add_argument("--batch-size", type=int, default=36,
                        help="Frames per contact sheet (default: 36)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel GPT API calls per video (default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="Re-scan even if results already exist")
    args = parser.parse_args()

    videos = args.videos or TARGET_VIDEOS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  TALON Coarse Scan (GPT-5.2)")
    print(f"{'='*70}")
    print(f"  Videos:     {len(videos)} -- {', '.join(videos)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers:    {args.workers}")
    print(f"  Model:      {GPT_MODEL}")
    print(f"{'='*70}")

    results = []
    for code in videos:
        result_path = RESULTS_DIR / f"{code}.json"

        # Resume: skip already-scanned videos
        if result_path.exists() and not args.force:
            existing = json.loads(result_path.read_text())
            if existing.get("status") == "done":
                print(f"\n  SKIP {code}: already scanned "
                      f"({existing['positive']}/{existing['total_frames']} positive)")
                results.append(existing)
                continue

        result = scan_video(code, args.batch_size, args.workers)
        results.append(result)

        # Save per-video result
        if result.get("status") == "done":
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # --- Summary table ---
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Code':<15s} {'Frames':>8s} {'Positive':>10s} {'Ratio':>8s}  Status")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8}  {'-'*10}")

    total_frames = 0
    total_positive = 0
    for r in results:
        code = r.get("code", "?")
        total = r.get("total_frames", 0)
        pos = r.get("positive", 0)
        ratio = r.get("ratio", 0)
        status = r.get("status", "?")
        total_frames += total
        total_positive += pos
        print(f"  {code:<15s} {total:>8d} {pos:>10d} {ratio:>7.1%}  {status}")

    overall = total_positive / total_frames if total_frames else 0
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8}")
    print(f"  {'TOTAL':<15s} {total_frames:>8d} {total_positive:>10d} {overall:>7.1%}")
    print(f"\n  Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
