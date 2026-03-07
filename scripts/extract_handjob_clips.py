#!/usr/bin/env python3
"""
TALON — Extract per-second frames from specific video time ranges.

For each video code + time range, obtains the HLS m3u8 URL via
playwright CDN interception, then uses ffmpeg to extract 1 fps
frames within the specified window.

Usage:
    cd projects/talon
    venv/bin/python3 scripts/extract_handjob_clips.py
"""

import re
import time
import subprocess
from pathlib import Path
from functools import partial

print = partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent

RAW_FRAMES_DIR = TALON_DIR / "data" / "raw_frames"

# ---------------------------------------------------------------------------
# Clip definitions: code → list of "H:MM:SS-H:MM:SS" or "MM:SS-MM:SS"
# ---------------------------------------------------------------------------
CLIPS = {
    "ACHJ-039": ["3:00:00-3:01:00"],
    "ACHJ-058": ["1:05:00-1:06:00", "2:23:00-2:24:00"],
    "BLK-548": ["52:00-53:00"],
    "BLK-646": ["34:00-34:30", "1:10:40-1:11:20", "1:49:50-1:50:10"],
    "CESD-094": ["1:48:30-1:49:30"],
    "CJOD-316": ["49:00-50:00", "1:29:00-1:30:00"],
    "DASS-714": ["20:30-20:50"],
    "DVAJ-604": ["41:00-42:00", "1:59:30-2:00:30"],
    "FNS-050": ["2:40:40-2:41:20"],
    "FPRE-184": ["32:40-33:00", "1:27:50-1:28:10", "2:54:30-2:55:30"],
    "GVH-765": ["45:30-46:00"],
    "JUFD-369": ["1:16:00-1:17:00"],
    "JUQ-685": ["21:10-21:40"],
    "MIDE-068": ["54:00-55:00"],
    "MIDE-160": ["1:07:30-1:08:00"],
    "PFES-110": ["36:00-36:30", "49:00-50:00", "2:09:00-2:09:30"],
    "PRED-800": ["38:00-38:30"],
    "SOE-736": ["20:50-21:20", "1:12:50-1:13:20"],
    "SONE-154": ["2:00:50-2:01:10"],
    "SONE-479": ["27:30-28:00"],
    "SSIS-050": ["1:29:50-1:30:30"],
    "SSIS-357": ["22:30-22:50"],
    "SSNI-115": ["2:20:20-2:20:40"],
    "WAAA-103": ["1:42:20-1:43:30"],
    "WAAA-175": ["37:50-38:10"],
    "WO-020": ["1:30:50-1:31:20"],
}

BASE_URL = "https://24av.net/en/v/{slug}-uncensored-leaked"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def parse_time(t: str) -> int:
    """Parse H:MM:SS or MM:SS to total seconds."""
    parts = t.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0])


def fmt_time(s: int) -> str:
    """Format seconds as H:MM:SS."""
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def get_m3u8_url(code: str) -> str | None:
    """Get HLS m3u8 URL for a video code using playwright."""
    from playwright.sync_api import sync_playwright

    slug = code.lower()
    page_url = BASE_URL.format(slug=slug)
    print(f"    Getting CDN info from {page_url}")

    cdn_base = None
    hls_quality = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 720},
        )
        page = context.new_page()

        def on_response(response):
            nonlocal cdn_base, hls_quality
            url = response.url
            try:
                if "preview.vtt" in url and response.status == 200:
                    cdn_base = url.rsplit("/preview.vtt", 1)[0]
                if "/v.m3u8" in url and response.status == 200:
                    m = re.search(r"/(\w+)/v\.m3u8", url)
                    if m:
                        hls_quality = m.group(1)
                    if not cdn_base:
                        parts = url.split(f"/{hls_quality or 'qc'}/v.m3u8")
                        if len(parts) > 1:
                            cdn_base = parts[0]
            except Exception:
                pass

        page.on("response", on_response)
        page.goto(page_url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(8000)

        # Fallback: check performance entries in iframes
        if not cdn_base:
            for f in page.frames:
                if "surrit.store" not in f.url:
                    continue
                try:
                    entries = f.evaluate(
                        "performance.getEntriesByType('resource')"
                        ".filter(function(e){ return e.name.indexOf('wowstream') >= 0; })"
                        ".map(function(e){ return e.name; })"
                    )
                    for entry in entries:
                        if "preview.vtt" in entry:
                            cdn_base = entry.rsplit("/preview.vtt", 1)[0]
                            break
                        m2 = re.match(r"(https://[^/]+/[^/]+/[^/]+)", entry)
                        if m2:
                            cdn_base = m2.group(1)
                            break
                except Exception:
                    pass

        browser.close()

    if cdn_base:
        quality = hls_quality or "qc"
        m3u8 = f"{cdn_base}/{quality}/v.m3u8"
        print(f"    CDN: {cdn_base[:60]}...")
        print(f"    HLS quality: {quality}")
        return m3u8

    print(f"    FAILED: could not extract CDN URL")
    return None


def extract_clip(m3u8_url: str, code: str, start_s: int, end_s: int,
                 out_dir: Path) -> int:
    """Extract 1fps frames from start_s to end_s. Returns frame count."""
    duration = end_s - start_s
    out_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg outputs to tmp pattern, then rename
    tmp_pattern = str(out_dir / "tmp_%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-extension_picky", "0",
        "-ss", str(start_s),
        "-i", m3u8_url,
        "-t", str(duration),
        "-vf", "fps=1",
        "-q:v", "2",
        tmp_pattern,
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0

    if result.returncode != 0:
        err_lines = result.stderr.strip().split("\n")[-3:]
        print(f"    ffmpeg error (exit {result.returncode}):")
        for line in err_lines:
            print(f"      {line}")
        return 0

    # Rename tmp_NNNNNN.jpg → {code}_frame_{total_seconds:06d}.jpg
    tmp_files = sorted(out_dir.glob("tmp_*.jpg"))
    extracted = 0
    for f in tmp_files:
        idx = int(f.stem.split("_")[1]) - 1  # 1-based → 0-based
        total_s = start_s + idx
        target = out_dir / f"{code}_frame_{total_s:06d}.jpg"
        f.rename(target)
        extracted += 1

    print(f"    {fmt_time(start_s)}-{fmt_time(end_s)}: "
          f"{extracted} frames in {elapsed:.0f}s")
    return extracted


def main():
    # Calculate expected totals
    total_expected = 0
    for code, ranges in CLIPS.items():
        for r in ranges:
            dash_idx = r.index("-", r.index(":"))
            start_s = parse_time(r[:dash_idx])
            end_s = parse_time(r[dash_idx + 1:])
            total_expected += end_s - start_s

    print(f"{'='*60}")
    print(f"  TALON Clip Extractor")
    print(f"{'='*60}")
    print(f"  Videos:          {len(CLIPS)}")
    print(f"  Expected frames: ~{total_expected}")
    print(f"  Output:          {RAW_FRAMES_DIR}")
    print(f"{'='*60}")

    results = []

    for code, ranges in CLIPS.items():
        print(f"\n  [{code}] {len(ranges)} clip(s)")

        # Get m3u8 URL
        m3u8_url = get_m3u8_url(code)
        if not m3u8_url:
            print(f"    FAILED: could not get HLS URL, skipping")
            results.append({"code": code, "status": "cdn_fail", "frames": 0, "clips": []})
            continue

        out_dir = RAW_FRAMES_DIR / code
        video_total = 0
        clip_results = []

        for r in ranges:
            dash_idx = r.index("-", r.index(":"))
            start_str = r[:dash_idx]
            end_str = r[dash_idx + 1:]

            start_s = parse_time(start_str)
            end_s = parse_time(end_str)

            try:
                n = extract_clip(m3u8_url, code, start_s, end_s, out_dir)
            except Exception as e:
                print(f"    ERROR {r}: {e}")
                n = 0

            video_total += n
            clip_results.append({"range": r, "frames": n})

        results.append({"code": code, "status": "done", "frames": video_total,
                         "clips": clip_results})
        print(f"    Subtotal: {video_total} frames")

    # --- Summary ---
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    grand_total = 0
    for r in results:
        code = r["code"]
        status = r["status"]
        frames = r["frames"]
        grand_total += frames

        if status != "done":
            print(f"  {code:<12s}  FAILED ({status})")
            continue

        for c in r["clips"]:
            print(f"  {code:<12s}  {c['range']:<22s}  {c['frames']:>4d} frames")

    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<12s}  {'':<22s}  {grand_total:>4d} frames")
    print(f"\n  Output: {RAW_FRAMES_DIR}")
    print(f"  Note: frames saved to raw_frames/, NOT labels/")


if __name__ == "__main__":
    main()
