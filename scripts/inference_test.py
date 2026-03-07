#!/usr/bin/env python3
"""
TALON — End-to-end inference test with two-stage variable-rate sampling.

Stage 1 (coarse): Extract 1 frame per 10s across entire video, run model
Stage 2 (fine):   Extract 1 fps around hot zones, confirm detections

Usage:
    cd projects/talon
    venv/bin/python3 scripts/inference_test.py
"""

import sys
import os
import re
import json
import time
import subprocess
import tempfile
from pathlib import Path
from functools import partial

# Force unbuffered output
print = partial(print, flush=True)

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent
MEGA_DIR = TALON_DIR.parent / "mega"

sys.path.insert(0, str(TALON_DIR / "inference"))
from clip_classifier import TalonClassifier

# ── Config ──────────────────────────────────────────────────────────────

COARSE_INTERVAL = 10     # seconds between coarse frames
FINE_INTERVAL = 1        # seconds between fine frames
COARSE_THRESHOLD = 0.3   # score to trigger fine scan
FINE_THRESHOLD = 0.5     # score to confirm detection
HOT_ZONE_PADDING = 30    # seconds padding around coarse detection
MIN_CLUSTER_FRAMES = 3   # minimum sustained detections in fine scan

MODEL_PATH = TALON_DIR / "models" / "talon_best_v6.pt"
CACHE_DIR = TALON_DIR / "data" / "inference_cache"

BASE_URL = "https://24av.net/en/v/{slug}-uncensored-leaked"

TEST_VIDEOS = [
    {
        "code": "EYAN-215",
        "expected": True,
        "note": "确认有 handjob（预期：检出）",
    },
    {
        "code": "DASS-885",
        "expected": False,
        "note": "有脸有动作但没露胸（预期：不检出，三要素缺胸）",
    },
    {
        "code": "NMSL-034",
        "expected": False,
        "note": "有脸有动作但没露胸（预期：不检出，三要素缺胸）",
    },
    {
        "code": "MADV-622",
        "expected": False,
        "note": "确认没有 handjob（预期：不检出）",
    },
]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


# ── CDN extraction (standalone, no scene_analyzer dependency) ───────────

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
                        m = re.match(r"(https://[^/]+/[^/]+/[^/]+)", entry)
                        if m:
                            cdn_base = m.group(1)
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


def get_video_duration(m3u8_url: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        m3u8_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ── Frame extraction ────────────────────────────────────────────────────

def extract_frames(m3u8_url: str, code: str, start_s: int, end_s: int,
                   interval: int, out_dir: Path) -> list[Path]:
    """Extract frames at given interval. Returns list of frame paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = end_s - start_s

    # Use fps filter to control frame rate
    fps_val = f"1/{interval}" if interval > 1 else "1"
    tmp_pattern = str(out_dir / f"tmp_{start_s}_%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-extension_picky", "0",
        "-ss", str(start_s),
        "-i", m3u8_url,
        "-t", str(duration),
        "-vf", f"fps={fps_val}",
        "-q:v", "2",
        tmp_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        return []

    # Rename tmp files to timestamped names
    tmp_files = sorted(out_dir.glob(f"tmp_{start_s}_*.jpg"))
    frames = []
    for f in tmp_files:
        idx = int(f.stem.split("_")[-1]) - 1  # 1-based → 0-based
        total_s = start_s + idx * interval
        target = out_dir / f"{code}_frame_{total_s:06d}.jpg"
        f.rename(target)
        frames.append(target)

    return frames


def fmt_time(s: int) -> str:
    """Format seconds as H:MM:SS."""
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


# ── Two-stage inference ─────────────────────────────────────────────────

def find_hot_zones(frame_scores: list[tuple[int, float]],
                   threshold: float, padding: int) -> list[tuple[int, int]]:
    """Find time ranges where scores exceed threshold, with padding.

    Returns merged (start_s, end_s) intervals.
    """
    hits = [t for t, score in frame_scores if score >= threshold]
    if not hits:
        return []

    # Merge nearby hits into zones
    zones = []
    zone_start = hits[0] - padding
    zone_end = hits[0] + padding

    for t in hits[1:]:
        if t - padding <= zone_end:
            zone_end = t + padding
        else:
            zones.append((max(0, zone_start), zone_end))
            zone_start = t - padding
            zone_end = t + padding

    zones.append((max(0, zone_start), zone_end))
    return zones


def cluster_detections(frame_scores: list[tuple[int, float]],
                       threshold: float,
                       min_frames: int) -> list[dict]:
    """Find clusters of sustained high-score frames.

    Returns list of {start, end, peak_score, avg_score, n_frames}.
    """
    hits = [(t, s) for t, s in frame_scores if s >= threshold]
    if not hits:
        return []

    clusters = []
    cluster = [hits[0]]

    for i in range(1, len(hits)):
        t, s = hits[i]
        prev_t = cluster[-1][0]
        # Same cluster if within 5 seconds gap
        if t - prev_t <= 5:
            cluster.append(hits[i])
        else:
            if len(cluster) >= min_frames:
                times = [c[0] for c in cluster]
                scores = [c[1] for c in cluster]
                clusters.append({
                    "start": min(times),
                    "end": max(times),
                    "peak_score": max(scores),
                    "avg_score": sum(scores) / len(scores),
                    "n_frames": len(cluster),
                })
            cluster = [hits[i]]

    # Final cluster
    if len(cluster) >= min_frames:
        times = [c[0] for c in cluster]
        scores = [c[1] for c in cluster]
        clusters.append({
            "start": min(times),
            "end": max(times),
            "peak_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "n_frames": len(cluster),
        })

    return clusters


def run_inference_on_video(classifier: TalonClassifier, video: dict) -> dict:
    """Run two-stage inference on a single video."""
    code = video["code"]
    print(f"\n{'='*60}")
    print(f"  [{code}] {video['note']}")
    print(f"{'='*60}")

    cache_dir = CACHE_DIR / code
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Get m3u8 URL ──
    m3u8_url = get_m3u8_url(code)
    if not m3u8_url:
        return {"code": code, "status": "cdn_fail", "detections": []}

    # ── Get duration ──
    duration = get_video_duration(m3u8_url)
    if duration <= 0:
        # Fallback: assume 3 hours
        duration = 3 * 3600
        print(f"    Duration: unknown, assuming {fmt_time(int(duration))}")
    else:
        print(f"    Duration: {fmt_time(int(duration))}")

    duration_s = int(duration)

    # ── Stage 1: Coarse scan ──
    print(f"\n  Stage 1: Coarse scan ({COARSE_INTERVAL}s intervals)")
    print(f"    Extracting ~{duration_s // COARSE_INTERVAL} frames...")

    coarse_dir = cache_dir / "coarse"
    t0 = time.time()

    # Extract in chunks to avoid ffmpeg timeouts on long seeks
    CHUNK_SIZE = 600  # 10 minutes per chunk
    all_coarse_frames = []
    for chunk_start in range(0, duration_s, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, duration_s)
        frames = extract_frames(
            m3u8_url, code, chunk_start, chunk_end,
            COARSE_INTERVAL, coarse_dir,
        )
        all_coarse_frames.extend(frames)

    extract_time = time.time() - t0
    print(f"    Extracted {len(all_coarse_frames)} frames in {extract_time:.0f}s")

    if not all_coarse_frames:
        return {"code": code, "status": "extract_fail", "detections": []}

    # Run model on coarse frames
    print(f"    Running model...")
    t0 = time.time()
    coarse_paths = [str(f) for f in all_coarse_frames]
    coarse_scores_raw = classifier.predict_batch(coarse_paths)
    infer_time = time.time() - t0
    print(f"    Inference: {len(coarse_paths)} frames in {infer_time:.1f}s "
          f"({len(coarse_paths)/infer_time:.1f} fps)")

    # Parse timestamps from filenames and pair with scores
    coarse_scores = []
    for path, score in zip(all_coarse_frames, coarse_scores_raw):
        m = re.search(r"_frame_(\d+)\.jpg$", path.name)
        if m:
            t = int(m.group(1))
            coarse_scores.append((t, score))

    # Print coarse hits
    coarse_hits = [(t, s) for t, s in coarse_scores if s >= COARSE_THRESHOLD]
    if coarse_hits:
        print(f"    Coarse hits (>{COARSE_THRESHOLD}): {len(coarse_hits)}")
        for t, s in sorted(coarse_hits, key=lambda x: -x[1])[:10]:
            print(f"      {fmt_time(t)}  score={s:.3f}")
    else:
        print(f"    No coarse hits above {COARSE_THRESHOLD}")
        return {
            "code": code,
            "status": "no_detection",
            "detections": [],
            "coarse_max": max(s for _, s in coarse_scores) if coarse_scores else 0,
        }

    # ── Stage 2: Fine scan around hot zones ──
    hot_zones = find_hot_zones(coarse_scores, COARSE_THRESHOLD, HOT_ZONE_PADDING)
    # Cap zones to video duration
    hot_zones = [(max(0, s), min(e, duration_s)) for s, e in hot_zones]

    total_fine_seconds = sum(e - s for s, e in hot_zones)
    print(f"\n  Stage 2: Fine scan ({FINE_INTERVAL}s intervals)")
    print(f"    Hot zones: {len(hot_zones)} ({total_fine_seconds}s total)")
    for s, e in hot_zones:
        print(f"      {fmt_time(s)} - {fmt_time(e)}")

    fine_dir = cache_dir / "fine"
    all_fine_frames = []
    for zone_start, zone_end in hot_zones:
        frames = extract_frames(
            m3u8_url, code, zone_start, zone_end,
            FINE_INTERVAL, fine_dir,
        )
        all_fine_frames.extend(frames)

    print(f"    Extracted {len(all_fine_frames)} fine frames")

    if not all_fine_frames:
        return {"code": code, "status": "fine_extract_fail", "detections": []}

    # Run model on fine frames
    print(f"    Running model...")
    t0 = time.time()
    fine_paths = [str(f) for f in all_fine_frames]
    fine_scores_raw = classifier.predict_batch(fine_paths)
    infer_time = time.time() - t0
    print(f"    Inference: {len(fine_paths)} frames in {infer_time:.1f}s")

    fine_scores = []
    for path, score in zip(all_fine_frames, fine_scores_raw):
        m = re.search(r"_frame_(\d+)\.jpg$", path.name)
        if m:
            t = int(m.group(1))
            fine_scores.append((t, score))

    # Find sustained detection clusters
    detections = cluster_detections(fine_scores, FINE_THRESHOLD, MIN_CLUSTER_FRAMES)

    if detections:
        print(f"\n    DETECTIONS: {len(detections)}")
        for d in detections:
            print(f"      {fmt_time(d['start'])} - {fmt_time(d['end'])}  "
                  f"peak={d['peak_score']:.3f}  avg={d['avg_score']:.3f}  "
                  f"frames={d['n_frames']}")
    else:
        print(f"    No sustained detections above {FINE_THRESHOLD}")

    return {
        "code": code,
        "status": "detected" if detections else "no_detection",
        "detections": detections,
        "coarse_hits": len(coarse_hits),
        "fine_frames": len(all_fine_frames),
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*60}")
    print(f"  TALON End-to-End Inference Test")
    print(f"{'='*60}")
    print(f"  Model:    {MODEL_PATH.name}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"  Device:   {device}")
    print(f"  Videos:   {len(TEST_VIDEOS)}")
    print(f"  Coarse:   1 frame / {COARSE_INTERVAL}s, threshold {COARSE_THRESHOLD}")
    print(f"  Fine:     1 frame / {FINE_INTERVAL}s, threshold {FINE_THRESHOLD}")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading model...")
    classifier = TalonClassifier(MODEL_PATH)
    if not classifier.ready:
        print("FATAL: Model failed to load")
        sys.exit(1)

    results = []
    for video in TEST_VIDEOS:
        result = run_inference_on_video(classifier, video)
        results.append(result)

    # ── Summary ──
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    all_correct = True
    for video, result in zip(TEST_VIDEOS, results):
        code = video["code"]
        expected = video["expected"]
        detected = result["status"] == "detected"
        correct = detected == expected
        if not correct:
            all_correct = False

        status_str = "DETECTED" if detected else "CLEAR"
        expect_str = "should detect" if expected else "should not detect"
        mark = "✓" if correct else "✗"

        print(f"  {mark} {code:<10s}  {status_str:<10s}  ({expect_str})")

        if result.get("detections"):
            for d in result["detections"]:
                print(f"    └─ {fmt_time(d['start'])}-{fmt_time(d['end'])}  "
                      f"peak={d['peak_score']:.3f}  avg={d['avg_score']:.3f}")
        elif "coarse_max" in result:
            print(f"    └─ max coarse score: {result['coarse_max']:.3f}")

    print(f"\n  Result: {'ALL CORRECT' if all_correct else 'SOME MISMATCHES'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
