#!/usr/bin/env python3
"""
TALON — Batch two-stage pipeline for multiple videos.

Runs CLIP Stage 1 locally (Mac/MPS), calls a remote GLM API for
Stage 2 confirmation. Supports resume via progress.json.

Parallelization (Approach A):
  - Phase 0: Pre-resolve all m3u8 URLs in parallel (ThreadPool, 4 workers)
  - Phase 1: Per-video Stage 1 with parallel frame extraction (8 workers)
  - Phase 2: Per-video GLM calls sent in parallel (4 workers)

Usage:
    # Stage 1 only (no RunPod needed)
    python scripts/batch_pipeline.py \\
        --video_list data/saved_urls.md --skip_glm

    # Full pipeline with RunPod Serverless
    python scripts/batch_pipeline.py \\
        --video_list data/saved_urls.md \\
        --runpod_endpoint YOUR_ENDPOINT_ID \\
        --runpod_key YOUR_API_KEY

    # Full pipeline with GPU Pod (legacy)
    python scripts/batch_pipeline.py \\
        --video_list data/saved_urls.md \\
        --gpu_endpoint https://xxx-8000.proxy.runpod.net

    # Resume interrupted run (just re-run the same command)
    python scripts/batch_pipeline.py \\
        --video_list data/saved_urls.md --skip_glm
"""

import argparse
import base64
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent

# Allow imports from scripts/ and talon/
sys.path.insert(0, str(TALON_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# Import Stage 1 logic from two_stage_inference.py
from two_stage_inference import (
    extract_frame,
    extract_video_code,
    fmt_time,
    get_duration,
    merge_windows,
    resolve_m3u8,
)

import requests


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def load_video_list(path: Path) -> list[str]:
    """Load URLs from a file (one per line, # for comments, skip blanks)."""
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def load_progress(path: Path) -> dict:
    """Load progress.json or return empty dict."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_progress(path: Path, progress: dict):
    """Save progress.json atomically."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2, ensure_ascii=False))
    tmp.replace(path)


def save_result(output_dir: Path, code: str, result: dict):
    """Save per-video result JSON."""
    out_path = output_dir / f"{code}.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    tmp.replace(out_path)


# ─────────────────────────────────────────────────────────────────────
# Parallel frame extraction
# ─────────────────────────────────────────────────────────────────────

def extract_frames_parallel(
    source: str, timestamps: list[int], tmp_dir: Path, max_workers: int = 8,
) -> tuple[list[Path], list[int]]:
    """Extract frames in parallel. Returns (paths, seconds) for successful extractions."""
    results = {}  # second -> Path or None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(extract_frame, source, s, tmp_dir): s
            for s in timestamps
        }
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                results[s] = fut.result()
            except Exception:
                results[s] = None

    # Maintain original order
    paths = []
    seconds = []
    for s in timestamps:
        p = results.get(s)
        if p:
            paths.append(p)
            seconds.append(s)
    return paths, seconds


def stage1_coarse_parallel(
    source: str, clf, tmp_dir: Path,
    start_time: int, duration: int, coarse_threshold: float,
    max_workers: int = 8,
) -> list[tuple[int, float]]:
    """Coarse pass with parallel frame extraction."""
    end_time = int(duration)
    timestamps = list(range(start_time, end_time, 30))

    if not timestamps:
        print("  No timestamps to scan.")
        return []

    print(f"  Coarse pass: {len(timestamps)} frames "
          f"({fmt_time(start_time)} → {fmt_time(end_time)}, every 30s, {max_workers} workers)")

    # Parallel extraction
    t0 = time.time()
    paths, seconds = extract_frames_parallel(source, timestamps, tmp_dir, max_workers)

    if not paths:
        print("  No frames extracted.")
        return []

    elapsed_extract = time.time() - t0
    print(f"  Extracted {len(paths)} frames in {elapsed_extract:.0f}s")

    # CLIP inference (serial, GPU-bound)
    t1 = time.time()
    scores = clf.predict_batch([str(p) for p in paths])
    elapsed_predict = time.time() - t1
    print(f"  CLIP inference: {elapsed_predict:.1f}s "
          f"({len(paths)/max(elapsed_predict, 0.01):.0f} fps)")

    # Filter candidates
    candidates = [(s, sc) for s, sc in zip(seconds, scores) if sc >= coarse_threshold]
    print(f"  Candidates above {coarse_threshold}: {len(candidates)}")

    for s, sc in candidates:
        print(f"    {fmt_time(s)}  score={sc:.3f}")

    return candidates


def stage1_fine_parallel(
    source: str, clf, tmp_dir: Path,
    windows: list[tuple[int, int]], fine_threshold: float,
    confirm_frames: int, duration: float = float('inf'),
    max_workers: int = 8,
) -> list[dict]:
    """Fine pass with parallel frame extraction within each window."""
    if not windows:
        return []

    print(f"\n  Fine pass: {len(windows)} window(s), every 5s, "
          f"threshold={fine_threshold}, {max_workers} workers")

    segments = []

    for w_start, w_end in windows:
        w_start = max(w_start, 0)
        w_end = min(w_end, int(duration) - 1)
        if w_end < w_start:
            continue
        timestamps = list(range(w_start, w_end + 1, 5))

        # Parallel extraction
        paths, seconds = extract_frames_parallel(source, timestamps, tmp_dir, max_workers)

        if not paths:
            continue

        scores = clf.predict_batch([str(p) for p in paths])

        # Find consecutive runs of confirm_frames+ above fine_threshold
        above = [sc >= fine_threshold for sc in scores]
        run_start = None
        run_len = 0

        for idx, is_above in enumerate(above):
            if is_above:
                if run_start is None:
                    run_start = idx
                run_len += 1
            else:
                if run_len >= confirm_frames:
                    seg_scores = scores[run_start:run_start + run_len]
                    seg = {
                        "start": seconds[run_start],
                        "end": seconds[run_start + run_len - 1],
                        "avg_score": sum(seg_scores) / len(seg_scores),
                        "max_score": max(seg_scores),
                        "n_frames": run_len,
                    }
                    segments.append(seg)
                    print(f"    {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  "
                          f"avg={seg['avg_score']:.3f}  max={seg['max_score']:.3f}  "
                          f"({seg['n_frames']} frames)")
                run_start = None
                run_len = 0

        # Handle run at end of window
        if run_len >= confirm_frames:
            seg_scores = scores[run_start:run_start + run_len]
            seg = {
                "start": seconds[run_start],
                "end": seconds[run_start + run_len - 1],
                "avg_score": sum(seg_scores) / len(seg_scores),
                "max_score": max(seg_scores),
                "n_frames": run_len,
            }
            segments.append(seg)
            print(f"    {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  "
                  f"avg={seg['avg_score']:.3f}  max={seg['max_score']:.3f}  "
                  f"({seg['n_frames']} frames)")

    return segments


# ─────────────────────────────────────────────────────────────────────
# Remote GLM client
# ─────────────────────────────────────────────────────────────────────

def glm_predict_remote(endpoint: str, frame_path: Path) -> tuple[str, str]:
    """Send a frame to the remote GLM API (GPU Pod). Returns (YES/NO, raw_text)."""
    b64 = base64.b64encode(frame_path.read_bytes()).decode()
    resp = requests.post(
        f"{endpoint}/predict",
        json={"image_base64": b64},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["result"], data.get("raw", "")


def check_glm_health(endpoint: str) -> bool:
    """Check if GLM API is reachable (GPU Pod)."""
    try:
        resp = requests.get(f"{endpoint}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def glm_predict_runpod(endpoint_id: str, api_key: str, frame_path: Path) -> tuple[str, str]:
    """Send a frame to RunPod Serverless endpoint. Returns (YES/NO, raw_text)."""
    b64 = base64.b64encode(frame_path.read_bytes()).decode()
    resp = requests.post(
        f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"input": {"image_base64": b64}},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "COMPLETED":
        error = data.get("error", data.get("status", "unknown"))
        raise RuntimeError(f"RunPod job failed: {error}")
    output = data["output"]
    if "error" in output:
        raise RuntimeError(f"Handler error: {output['error']}")
    return output["result"], output.get("raw", "")


def check_runpod_health(endpoint_id: str, api_key: str) -> bool:
    """Check if RunPod Serverless endpoint is reachable."""
    try:
        resp = requests.get(
            f"https://api.runpod.ai/v2/{endpoint_id}/health",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# Parallel GLM Stage 2
# ─────────────────────────────────────────────────────────────────────

def _glm_one_segment(seg, source, tmp_dir, args):
    """Process one segment through GLM. Returns (seg, result, raw) or (seg, error_str, None)."""
    mid_sec = (seg["start"] + seg["end"]) // 2
    frame_path = extract_frame(source, mid_sec, tmp_dir)

    if not frame_path:
        return seg, "SKIP", None

    use_runpod = bool(args.runpod_endpoint)
    if use_runpod:
        result, raw = glm_predict_runpod(args.runpod_endpoint, args.runpod_key, frame_path)
    else:
        result, raw = glm_predict_remote(args.gpu_endpoint, frame_path)
    return seg, result, raw


def stage2_glm_parallel(segments, source, tmp_dir, args, max_workers=4):
    """Run GLM confirmation on all segments in parallel. Returns confirmed list."""
    use_runpod = bool(args.runpod_endpoint)
    backend = "RunPod Serverless" if use_runpod else "GPU Pod"
    print(f"\n  Stage 2: GLM confirmation via {backend} "
          f"({len(segments)} segment(s), {max_workers} workers)")
    t_stage2 = time.time()

    confirmed = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_glm_one_segment, seg, source, tmp_dir, args): seg
            for seg in segments
        }
        # Collect results, print in segment order after all complete
        results = {}
        for fut in as_completed(futures):
            seg = futures[fut]
            try:
                _, result, raw = fut.result()
                results[id(seg)] = (result, raw)
            except Exception as e:
                results[id(seg)] = ("ERROR", str(e))

    # Print in original order and build confirmed list
    for seg in segments:
        result, raw = results[id(seg)]

        if result == "SKIP":
            print(f"    {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → SKIP (no frame)")
            seg["glm_result"] = "SKIP"
        elif result == "ERROR":
            print(f"    ✗ {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → ERROR: {raw}")
            seg["glm_result"] = "ERROR"
            seg["glm_error"] = raw
        else:
            seg["glm_result"] = result
            seg["glm_raw"] = raw
            if result == "YES":
                confirmed.append(seg)
                print(f"    ✓ {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → YES")
            else:
                print(f"    ✗ {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → {result}")

    elapsed_s2 = time.time() - t_stage2
    print(f"\n  Stage 2 完成: {len(confirmed)}/{len(segments)} 确认 ({elapsed_s2:.0f}s)")
    return confirmed


# ─────────────────────────────────────────────────────────────────────
# Pre-resolve m3u8 URLs
# ─────────────────────────────────────────────────────────────────────

def _resolve_one(url):
    """Resolve a single URL to m3u8. Returns (url, m3u8_url or None)."""
    if url.endswith(".m3u8"):
        return url, url
    if url.startswith("http"):
        m3u8 = resolve_m3u8(url)
        return url, m3u8
    return url, url  # local file


def preresolve_m3u8(urls: list[str], progress: dict, max_workers: int = 4) -> dict:
    """Resolve all m3u8 URLs in parallel. Returns {url: m3u8_url}."""
    # Only resolve URLs that need it and aren't already done
    to_resolve = []
    resolved = {}
    for url in urls:
        code = extract_video_code(url)
        if code in progress and progress[code].get("status") == "done":
            continue
        if url.endswith(".m3u8") or not url.startswith("http"):
            resolved[url] = url
        else:
            to_resolve.append(url)

    if not to_resolve:
        return resolved

    print(f"\n  Pre-resolving {len(to_resolve)} m3u8 URLs ({max_workers} workers)...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_resolve_one, url): url for url in to_resolve}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                _, m3u8 = fut.result()
                code = extract_video_code(url)
                if m3u8:
                    resolved[url] = m3u8
                    print(f"    ✓ {code}")
                else:
                    print(f"    ✗ {code} (resolve failed)")
            except Exception as e:
                code = extract_video_code(url)
                print(f"    ✗ {code} (error: {e})")

    elapsed = time.time() - t0
    print(f"  Resolved {len(resolved)}/{len(to_resolve) + len(resolved)} in {elapsed:.0f}s")
    return resolved


# ─────────────────────────────────────────────────────────────────────
# Per-video processing
# ─────────────────────────────────────────────────────────────────────

def process_video(
    url: str,
    clf,
    args,
    output_dir: Path,
    progress: dict,
    progress_path: Path,
    m3u8_cache: dict,
):
    """Process a single video through the two-stage pipeline."""
    video_code = extract_video_code(url)

    # Skip if already done
    if video_code in progress and progress[video_code].get("status") == "done":
        print(f"\n  SKIP {video_code} (already done)")
        return

    print(f"\n{'═' * 60}")
    print(f"  {video_code}")
    print(f"  {url[:70]}{'...' if len(url) > 70 else ''}")
    print(f"{'═' * 60}")

    # ── Resolve m3u8 (from cache or fresh) ──
    m3u8_url = m3u8_cache.get(url)
    if not m3u8_url and url.startswith("http") and not url.endswith(".m3u8"):
        # Cache miss — resolve now
        m3u8_url = resolve_m3u8(url)

    source = m3u8_url or url

    if url.startswith("http") and not url.endswith(".m3u8") and not m3u8_url:
        print(f"  ERROR: Could not resolve m3u8 for {video_code}")
        progress[video_code] = {
            "status": "error",
            "error": "m3u8_resolve_failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_progress(progress_path, progress)
        return

    # ── Duration ──
    print(f"  Probing duration...")
    duration = get_duration(source)
    if duration is None:
        print(f"  ERROR: Could not determine duration for {video_code}")
        progress[video_code] = {
            "status": "error",
            "error": "duration_probe_failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_progress(progress_path, progress)
        return
    print(f"  Duration: {fmt_time(int(duration))} ({int(duration)}s)")

    # ── Temp dir ──
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"talon_{video_code}_"))

    try:
        # ══════════════════════════════════════════════════════════════
        # Stage 1: CLIP (parallel frame extraction)
        # ══════════════════════════════════════════════════════════════
        print(f"\n  Stage 1: CLIP Coarse Scan")
        t_stage1 = time.time()

        candidates = stage1_coarse_parallel(
            source, clf, tmp_dir,
            args.start_time, duration, args.coarse_threshold,
            max_workers=args.parallel)

        windows = merge_windows(candidates, margin=60)

        segments = stage1_fine_parallel(
            source, clf, tmp_dir,
            windows, args.fine_threshold, args.confirm_frames,
            duration=duration, max_workers=args.parallel)

        elapsed_s1 = time.time() - t_stage1
        print(f"\n  Stage 1 完成: {len(segments)} 候选段 ({elapsed_s1:.0f}s)")

        # ══════════════════════════════════════════════════════════════
        # Stage 2: GLM (parallel API calls)
        # ══════════════════════════════════════════════════════════════
        confirmed = []

        if args.skip_glm or not segments:
            confirmed = segments
            if segments:
                print(f"  (Stage 2 跳过)")
        else:
            glm_workers = min(4, len(segments))
            confirmed = stage2_glm_parallel(
                segments, source, tmp_dir, args, max_workers=glm_workers)

        # ══════════════════════════════════════════════════════════════
        # Save results
        # ══════════════════════════════════════════════════════════════
        timestamp = datetime.now(timezone.utc).isoformat()

        result = {
            "video_code": video_code,
            "source": url,
            "m3u8": m3u8_url,
            "duration": int(duration),
            "stage1_segments": segments,
            "stage2_confirmed": confirmed,
            "timestamp": timestamp,
        }
        save_result(output_dir, video_code, result)

        progress[video_code] = {
            "status": "done",
            "segments": len(segments),
            "confirmed": len(confirmed),
            "timestamp": timestamp,
        }
        save_progress(progress_path, progress)

        # Summary
        if confirmed:
            print(f"\n  结果: {len(confirmed)} 个 handjob 时间段")
            for i, seg in enumerate(confirmed, 1):
                print(f"    {i}. {fmt_time(seg['start'])} - {fmt_time(seg['end'])}")
        else:
            print(f"\n  结果: 无 handjob 时间段")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TALON batch two-stage pipeline (CLIP local + GLM remote)")

    parser.add_argument("--video_list", required=True,
                        help="Path to file with URLs (one per line, # comments)")
    parser.add_argument("--model_clip", default="models/talon_best_v6.pt",
                        help="CLIP model path (default: models/talon_best_v6.pt)")
    parser.add_argument("--gpu_endpoint", default=None,
                        help="GPU Pod GLM endpoint URL (e.g. https://xxx-8000.proxy.runpod.net)")
    parser.add_argument("--runpod_endpoint", default=None,
                        help="RunPod Serverless endpoint ID")
    parser.add_argument("--runpod_key", default=None,
                        help="RunPod API key (required with --runpod_endpoint)")
    parser.add_argument("--output_dir", default="data/pipeline_results/",
                        help="Output directory for JSON reports (default: data/pipeline_results/)")
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
    parser.add_argument("--parallel", type=int, default=8,
                        help="Parallel ffmpeg workers for frame extraction (default: 8)")
    parser.add_argument("--device", default=None,
                        help="Force device (cpu/mps/cuda)")
    args = parser.parse_args()

    # ── Load video list ──
    video_list_path = Path(args.video_list)
    if not video_list_path.is_absolute():
        video_list_path = TALON_DIR / video_list_path
    if not video_list_path.exists():
        print(f"ERROR: Video list not found: {video_list_path}")
        sys.exit(1)

    urls = load_video_list(video_list_path)
    if not urls:
        print("ERROR: No URLs found in video list.")
        sys.exit(1)

    # ── Output dir ──
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = TALON_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Progress ──
    progress_path = output_dir / "progress.json"
    progress = load_progress(progress_path)

    done_count = sum(1 for u in urls
                     if extract_video_code(u) in progress
                     and progress[extract_video_code(u)].get("status") == "done")

    # ── GLM health check ──
    # Priority: --runpod_endpoint > --gpu_endpoint > --skip_glm
    if not args.skip_glm:
        if args.runpod_endpoint:
            if not args.runpod_key:
                print("ERROR: --runpod_key required with --runpod_endpoint")
                sys.exit(1)
            print(f"  Checking RunPod Serverless endpoint: {args.runpod_endpoint}")
            if not check_runpod_health(args.runpod_endpoint, args.runpod_key):
                print(f"  WARNING: RunPod endpoint health check failed (may still work if workers are scaling up)")
        elif args.gpu_endpoint:
            print(f"  Checking GPU Pod endpoint: {args.gpu_endpoint}")
            if not check_glm_health(args.gpu_endpoint):
                print(f"  ERROR: GLM endpoint not reachable. Start the server first or use --skip_glm")
                sys.exit(1)
            print(f"  GLM endpoint OK")
        else:
            print("ERROR: --runpod_endpoint or --gpu_endpoint required when not using --skip_glm")
            sys.exit(1)

    # ── Pre-resolve m3u8 URLs ──
    m3u8_cache = preresolve_m3u8(urls, progress, max_workers=4)

    # ── Header ──
    print(f"\n{'═' * 60}")
    print(f"  TALON Batch Pipeline")
    print(f"{'═' * 60}")
    print(f"  Videos: {len(urls)} ({done_count} already done)")
    print(f"  CLIP model: {args.model_clip}")
    if args.skip_glm:
        stage2_label = "skip"
    elif args.runpod_endpoint:
        stage2_label = f"RunPod Serverless ({args.runpod_endpoint})"
    else:
        stage2_label = args.gpu_endpoint
    print(f"  Stage 2: {stage2_label}")
    print(f"  Parallel extraction workers: {args.parallel}")
    print(f"  Output: {output_dir}")
    print(f"{'═' * 60}")

    # ── Load CLIP ──
    model_path = Path(args.model_clip)
    if not model_path.is_absolute():
        model_path = TALON_DIR / model_path

    from inference.clip_classifier import TalonClassifier
    clf = TalonClassifier(str(model_path), device=args.device)
    if not clf.ready:
        print("ERROR: CLIP model failed to load.")
        sys.exit(1)

    # ── Process videos ──
    t_total = time.time()

    for i, url in enumerate(urls, 1):
        code = extract_video_code(url)
        print(f"\n  [{i}/{len(urls)}] {code}")

        try:
            process_video(url, clf, args, output_dir, progress, progress_path, m3u8_cache)
        except KeyboardInterrupt:
            print(f"\n  Interrupted. Progress saved to {progress_path}")
            sys.exit(0)
        except Exception as e:
            print(f"\n  ERROR processing {code}: {e}")
            progress[code] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_progress(progress_path, progress)
            continue

    # ── Final summary ──
    elapsed = time.time() - t_total
    done = sum(1 for v in progress.values() if v.get("status") == "done")
    errors = sum(1 for v in progress.values() if v.get("status") == "error")
    total_confirmed = sum(v.get("confirmed", 0) for v in progress.values()
                         if v.get("status") == "done")

    print(f"\n{'═' * 60}")
    print(f"  Batch 完成 ({elapsed:.0f}s)")
    print(f"{'═' * 60}")
    print(f"  Done: {done}  Errors: {errors}  Total confirmed: {total_confirmed}")
    print(f"  Results: {output_dir}")
    print(f"  Progress: {progress_path}")


if __name__ == "__main__":
    main()
