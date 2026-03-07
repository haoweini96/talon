#!/usr/bin/env python3
"""
TALON — Two-stage handjob scene detection pipeline.

Stage 1: CLIP coarse scan (fast, local)
  - Coarse pass: 1 frame / 30s → candidates above --coarse_threshold
  - Fine pass: 1 frame / 5s in ±60s windows → consecutive runs above --fine_threshold

Stage 2: GLM-4.6V-Flash confirmation (accurate, GPU)
  - Middle frame of each candidate segment → VLM YES/NO
  - Skippable with --skip_glm

Usage:
    # Stage 1 only (Mac, CPU/MPS)
    python scripts/two_stage_inference.py --input /path/to/video.mp4 --skip_glm

    # Page URL (auto-resolves to m3u8 via playwright CDN interception)
    python scripts/two_stage_inference.py --input "https://24av.net/en/v/avsa-325-uncensored-leaked" --skip_glm

    # Direct HLS stream
    python scripts/two_stage_inference.py --input "https://...m3u8" --skip_glm
"""

import argparse
import re
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent

# Allow importing TalonClassifier from inference/ and test_qwen3vl from scripts/
sys.path.insert(0, str(TALON_DIR))
sys.path.insert(0, str(SCRIPT_DIR))


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def fmt_time(s: int) -> str:
    """Format seconds as H:MM:SS."""
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def extract_video_code(source: str) -> str:
    """Extract JAV code from filename or path, e.g. 'MIDE-993'."""
    basename = Path(source).stem if not source.startswith("http") else source
    m = re.search(r'([A-Z]{2,5}-\d{3,5})', basename.upper())
    return m.group(1) if m else "unknown"


def get_duration(source: str) -> float | None:
    """Get video duration in seconds via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1"]
    if source.startswith("http"):
        cmd += ["-extension_picky", "0"]
    cmd.append(source)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"  ffprobe failed: {e}")
    return None


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def resolve_m3u8(page_url: str) -> str | None:
    """Resolve an HTML video page URL to an HLS m3u8 URL via playwright CDN interception."""
    from playwright.sync_api import sync_playwright

    print(f"  Resolving m3u8 from {page_url}")

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
        print(f"  CDN: {cdn_base[:60]}...")
        print(f"  HLS quality: {quality}")
        print(f"  m3u8: {m3u8[:80]}...")
        return m3u8

    print(f"  FAILED: could not extract m3u8 URL from page")
    return None


def extract_frame(source: str, second: int, out_dir: Path) -> Path | None:
    """Extract a single frame at the given timestamp. Returns path or None."""
    out_path = out_dir / f"tmp_frame_{second:06d}.jpg"
    if out_path.exists():
        return out_path

    cmd = ["ffmpeg", "-y", "-v", "error"]
    if source.startswith("http"):
        cmd += ["-extension_picky", "0"]
    cmd += ["-ss", str(second), "-i", source,
            "-frames:v", "1", "-q:v", "2", str(out_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────
# Stage 1: CLIP scan
# ─────────────────────────────────────────────────────────────────────

def stage1_coarse(source: str, clf, tmp_dir: Path,
                  start_time: int, duration: int, coarse_threshold: float,
                  ) -> list[tuple[int, float]]:
    """Coarse pass: 1 frame / 30s. Returns list of (second, score) above threshold."""
    end_time = int(duration)
    timestamps = list(range(start_time, end_time, 30))

    if not timestamps:
        print("  No timestamps to scan.")
        return []

    print(f"  Coarse pass: {len(timestamps)} frames "
          f"({fmt_time(start_time)} → {fmt_time(end_time)}, every 30s)")

    # Extract frames
    t0 = time.time()
    paths = []
    seconds = []
    for s in timestamps:
        p = extract_frame(source, s, tmp_dir)
        if p:
            paths.append(p)
            seconds.append(s)

    if not paths:
        print("  No frames extracted.")
        return []

    elapsed_extract = time.time() - t0
    print(f"  Extracted {len(paths)} frames in {elapsed_extract:.0f}s")

    # Predict
    t1 = time.time()
    scores = clf.predict_batch([str(p) for p in paths])
    elapsed_predict = time.time() - t1
    print(f"  CLIP inference: {elapsed_predict:.1f}s ({len(paths)/max(elapsed_predict,0.01):.0f} fps)")

    # Filter candidates
    candidates = [(s, sc) for s, sc in zip(seconds, scores) if sc >= coarse_threshold]
    print(f"  Candidates above {coarse_threshold}: {len(candidates)}")

    for s, sc in candidates:
        print(f"    {fmt_time(s)}  score={sc:.3f}")

    return candidates


def merge_windows(candidates: list[tuple[int, float]], margin: int = 60,
                  ) -> list[tuple[int, int]]:
    """Merge candidate timestamps into windows [t-margin, t+margin], merging overlaps."""
    if not candidates:
        return []

    windows = []
    for s, _ in sorted(candidates):
        lo, hi = s - margin, s + margin
        if windows and lo <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], hi))
        else:
            windows.append((lo, hi))
    return windows


def stage1_fine(source: str, clf, tmp_dir: Path,
                windows: list[tuple[int, int]], fine_threshold: float,
                confirm_frames: int, duration: float = float('inf'),
                ) -> list[dict]:
    """Fine pass: 1 frame / 5s within each window. Find consecutive runs."""
    if not windows:
        return []

    print(f"\n  Fine pass: {len(windows)} window(s), every 5s, threshold={fine_threshold}")

    segments = []

    for w_start, w_end in windows:
        w_start = max(w_start, 0)
        w_end = min(w_end, int(duration) - 1)
        if w_end < w_start:
            continue
        timestamps = list(range(w_start, w_end + 1, 5))

        # Extract
        paths = []
        seconds = []
        for s in timestamps:
            p = extract_frame(source, s, tmp_dir)
            if p:
                paths.append(p)
                seconds.append(s)

        if not paths:
            continue

        scores = clf.predict_batch([str(p) for p in paths])

        # Find consecutive runs of confirm_frames+ above fine_threshold
        above = [sc >= fine_threshold for sc in scores]
        run_start = None
        run_len = 0

        for i, is_above in enumerate(above):
            if is_above:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_len >= confirm_frames:
                    seg_scores = scores[run_start:run_start + run_len]
                    segments.append({
                        "start": seconds[run_start],
                        "end": seconds[run_start + run_len - 1],
                        "avg_score": sum(seg_scores) / len(seg_scores),
                        "max_score": max(seg_scores),
                        "n_frames": run_len,
                    })
                run_start = None
                run_len = 0

        # Flush trailing run
        if run_len >= confirm_frames:
            seg_scores = scores[run_start:run_start + run_len]
            segments.append({
                "start": seconds[run_start],
                "end": seconds[run_start + run_len - 1],
                "avg_score": sum(seg_scores) / len(seg_scores),
                "max_score": max(seg_scores),
                "n_frames": run_len,
            })

    # Merge adjacent segments (tolerance = one fine step)
    if len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg["start"] <= merged[-1]["end"] + 5:
                prev = merged[-1]
                total_frames = prev["n_frames"] + seg["n_frames"]
                prev["end"] = max(prev["end"], seg["end"])
                prev["avg_score"] = (prev["avg_score"] * prev["n_frames"] +
                                     seg["avg_score"] * seg["n_frames"]) / total_frames
                prev["max_score"] = max(prev["max_score"], seg["max_score"])
                prev["n_frames"] = total_frames
            else:
                merged.append(seg)
        segments = merged

    for seg in segments:
        print(f"    {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  "
              f"avg={seg['avg_score']:.3f}  max={seg['max_score']:.3f}  "
              f"({seg['n_frames']} frames)")

    return segments


# ─────────────────────────────────────────────────────────────────────
# Stage 2: GLM confirmation
# ─────────────────────────────────────────────────────────────────────

ROUND2_PROMPT = """这张图片是否属于 handjob 场景（女性用手给男性手淫）？
注意排除乳交（胸部夹住生殖器）和口交。
只回答 YES 或 NO"""


def parse_round2(response: str) -> str:
    """Parse GLM YES/NO response, stripping <think> tags."""
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    text = text.upper().strip()
    if "YES" in text:
        return "YES"
    if "NO" in text:
        return "NO"
    return text


def stage2_glm(segments: list[dict], source: str, tmp_dir: Path,
               ) -> list[dict]:
    """Run GLM-4.6V-Flash on middle frame of each segment. Returns confirmed list."""
    if not segments:
        return []

    # Lazy import — only when Stage 2 is actually needed (GPU)
    from test_qwen3vl import load_model, ask_glm

    print(f"\n  Loading GLM-4.6V-Flash...")
    model, processor = load_model("glm")
    print(f"  Running Stage 2 on {len(segments)} segment(s)...\n")

    confirmed = []

    for seg in segments:
        mid_sec = (seg["start"] + seg["end"]) // 2
        frame_path = extract_frame(source, mid_sec, tmp_dir)

        if not frame_path:
            print(f"    {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → SKIP (no frame)")
            seg["glm_result"] = "SKIP"
            continue

        raw = ask_glm(model, processor, str(frame_path), ROUND2_PROMPT)
        result = parse_round2(raw)
        seg["glm_result"] = result

        if result == "YES":
            confirmed.append(seg)
            print(f"    ✓ {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → YES")
        else:
            print(f"    ✗ {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → {result}")

    return confirmed


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TALON two-stage handjob scene detection")
    parser.add_argument("--input", "--video", required=True, dest="input",
                        help="Video path (.mp4), HLS m3u8 URL, or page URL (auto-resolved)")
    parser.add_argument("--model", "--model_clip", default="models/talon_best_v6.pt",
                        dest="model",
                        help="CLIP model path (default: models/talon_best_v6.pt)")
    parser.add_argument("--coarse_threshold", type=float, default=0.10,
                        help="Coarse scan threshold (default: 0.10)")
    parser.add_argument("--fine_threshold", type=float, default=0.50,
                        help="Fine scan threshold (default: 0.50)")
    parser.add_argument("--confirm_frames", type=int, default=3,
                        help="Consecutive frames needed (default: 3)")
    parser.add_argument("--start_time", type=int, default=600,
                        help="Start scanning from (seconds, default: 600)")
    parser.add_argument("--skip_glm", action="store_true",
                        help="Skip Stage 2 (GLM confirmation)")
    parser.add_argument("--keep_frames", action="store_true",
                        help="Don't delete extracted frames after processing")
    parser.add_argument("--device", default=None,
                        help="Force device (cpu/mps/cuda)")
    args = parser.parse_args()

    source = args.input
    video_code = extract_video_code(source)

    # Auto-resolve HTML page URLs to m3u8
    if source.startswith("http") and not source.endswith(".m3u8"):
        print(f"\n  Input is a page URL, resolving to m3u8...")
        m3u8 = resolve_m3u8(source)
        if not m3u8:
            print("  ERROR: Could not resolve m3u8 from page URL.")
            sys.exit(1)
        source = m3u8

    # Resolve model path relative to talon/
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = TALON_DIR / model_path

    # ── Header ──
    print(f"\n{'═' * 60}")
    print(f"  TALON Two-Stage Pipeline")
    print(f"{'═' * 60}")
    print(f"  视频: {video_code}")
    print(f"  输入: {source[:80]}{'...' if len(source) > 80 else ''}")
    print(f"  模型: {model_path.name}")
    print(f"  Stage 2: {'skip' if args.skip_glm else 'GLM-4.6V-Flash'}")
    print(f"{'═' * 60}")

    # ── Duration ──
    print(f"\n  Probing video duration...")
    duration = get_duration(source)
    if duration is None:
        print("  ERROR: Could not determine video duration.")
        sys.exit(1)
    print(f"  Duration: {fmt_time(int(duration))} ({int(duration)}s)")

    if args.start_time < 0 or args.start_time >= duration:
        print(f"  ERROR: --start_time must be in [0, {int(duration)})")
        sys.exit(1)

    # ── Load CLIP ──
    from inference.clip_classifier import TalonClassifier
    clf = TalonClassifier(str(model_path), device=args.device)
    if not clf.ready:
        print("  ERROR: CLIP model failed to load.")
        sys.exit(1)

    # ── Temp dir ──
    tmp_dir = Path(tempfile.mkdtemp(prefix="talon_"))
    print(f"  Temp dir: {tmp_dir}")

    try:
        # ══════════════════════════════════════════════════════════════
        # Stage 1: CLIP
        # ══════════════════════════════════════════════════════════════
        print(f"\n{'═' * 60}")
        print(f"  Stage 1: CLIP Coarse Scan")
        print(f"{'═' * 60}")

        t_stage1 = time.time()

        candidates = stage1_coarse(
            source, clf, tmp_dir,
            args.start_time, duration, args.coarse_threshold)

        windows = merge_windows(candidates, margin=60)

        segments = stage1_fine(
            source, clf, tmp_dir,
            windows, args.fine_threshold, args.confirm_frames,
            duration=duration)

        elapsed_s1 = time.time() - t_stage1

        # ── Stage 1 summary ──
        print(f"\n  Stage 1 完成: {len(segments)} 候选时间段 ({elapsed_s1:.0f}s)")
        for i, seg in enumerate(segments, 1):
            print(f"    {i}. {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  "
                  f"(avg score: {seg['avg_score']:.2f})")

        if not segments:
            print(f"\n  未检测到候选时间段。")
            return

        # ══════════════════════════════════════════════════════════════
        # Stage 2: GLM (optional)
        # ══════════════════════════════════════════════════════════════
        if args.skip_glm:
            confirmed = segments
            print(f"\n  (Stage 2 跳过)")
        else:
            print(f"\n{'═' * 60}")
            print(f"  Stage 2: GLM-4.6V-Flash Confirmation")
            print(f"{'═' * 60}")

            t_stage2 = time.time()
            confirmed = stage2_glm(segments, source, tmp_dir)
            elapsed_s2 = time.time() - t_stage2

            print(f"\n  Stage 2 完成: {len(confirmed)}/{len(segments)} 确认 ({elapsed_s2:.0f}s)")

        # ══════════════════════════════════════════════════════════════
        # Final results
        # ══════════════════════════════════════════════════════════════
        print(f"\n{'═' * 60}")
        print(f"  最终结果: {len(confirmed)} 个 handjob 时间段")
        print(f"{'═' * 60}")

        if confirmed:
            for i, seg in enumerate(confirmed, 1):
                print(f"    {i}. {fmt_time(seg['start'])} - {fmt_time(seg['end'])}")
        else:
            print(f"    (无)")

    finally:
        # Cleanup
        if not args.keep_frames:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\n  Cleaned up {tmp_dir}")
        else:
            n_frames = len(list(tmp_dir.glob("*.jpg")))
            print(f"\n  Kept {n_frames} frames in {tmp_dir}")


if __name__ == "__main__":
    main()
