#!/usr/bin/env python3
"""
TALON — Batch two-stage pipeline for multiple videos.

Runs CLIP Stage 1 locally (Mac/MPS), calls a remote GLM API for
Stage 2 confirmation. Supports resume via progress.json.

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
    stage1_coarse,
    stage1_fine,
)

import requests


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def parse_round2(response: str) -> str:
    """Parse YES/NO response, stripping <think> tags."""
    import re
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    text = text.upper().strip()
    if "YES" in text:
        return "YES"
    if "NO" in text:
        return "NO"
    return text


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
# Per-video processing
# ─────────────────────────────────────────────────────────────────────

def process_video(
    url: str,
    clf,
    args,
    output_dir: Path,
    progress: dict,
    progress_path: Path,
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

    # ── Resolve m3u8 ──
    source = url
    m3u8_url = None
    if source.startswith("http") and not source.endswith(".m3u8"):
        m3u8_url = resolve_m3u8(source)
        if not m3u8_url:
            print(f"  ERROR: Could not resolve m3u8 for {video_code}")
            progress[video_code] = {
                "status": "error",
                "error": "m3u8_resolve_failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_progress(progress_path, progress)
            return
        source = m3u8_url
    elif source.endswith(".m3u8"):
        m3u8_url = source

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
        # Stage 1: CLIP
        # ══════════════════════════════════════════════════════════════
        print(f"\n  Stage 1: CLIP Coarse Scan")
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
        print(f"\n  Stage 1 完成: {len(segments)} 候选段 ({elapsed_s1:.0f}s)")

        # ══════════════════════════════════════════════════════════════
        # Stage 2: GLM (remote or skip)
        # ══════════════════════════════════════════════════════════════
        confirmed = []

        if args.skip_glm or not segments:
            confirmed = segments
            if segments:
                print(f"  (Stage 2 跳过)")
        else:
            use_runpod = bool(args.runpod_endpoint)
            backend = "RunPod Serverless" if use_runpod else "GPU Pod"
            print(f"\n  Stage 2: GLM confirmation via {backend} ({len(segments)} segment(s))")
            t_stage2 = time.time()

            for seg in segments:
                mid_sec = (seg["start"] + seg["end"]) // 2
                frame_path = extract_frame(source, mid_sec, tmp_dir)

                if not frame_path:
                    print(f"    {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → SKIP (no frame)")
                    seg["glm_result"] = "SKIP"
                    continue

                try:
                    if use_runpod:
                        result, raw = glm_predict_runpod(
                            args.runpod_endpoint, args.runpod_key, frame_path)
                    else:
                        result, raw = glm_predict_remote(args.gpu_endpoint, frame_path)
                    seg["glm_result"] = result
                    seg["glm_raw"] = raw

                    if result == "YES":
                        confirmed.append(seg)
                        print(f"    ✓ {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → YES")
                    else:
                        print(f"    ✗ {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → {result}")

                except Exception as e:
                    print(f"    ✗ {fmt_time(seg['start'])} - {fmt_time(seg['end'])}  → ERROR: {e}")
                    seg["glm_result"] = "ERROR"
                    seg["glm_error"] = str(e)

            elapsed_s2 = time.time() - t_stage2
            print(f"\n  Stage 2 完成: {len(confirmed)}/{len(segments)} 确认 ({elapsed_s2:.0f}s)")

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
            process_video(url, clf, args, output_dir, progress, progress_path)
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
