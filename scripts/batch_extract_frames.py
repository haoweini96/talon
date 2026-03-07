"""
TALON — Batch frame extraction for CLIP training data.

Reads URLs from data/saved_urls.md, runs scene_analyzer on each,
extracts individual frames from scan windows, saves to data/raw_frames/.

Usage:
    python batch_extract_frames.py [--workers N]
"""

import sys
import os
import json
import time
import shutil
import argparse
import re
import concurrent.futures
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent          # talon/scripts/
TALON_DIR = SCRIPT_DIR.parent                          # projects/talon/
PROJECTS_DIR = TALON_DIR.parent                        # projects/
MEGA_DIR = PROJECTS_DIR / "mega"                       # projects/mega/
RECOMMENDER_DIR = MEGA_DIR / "recommender"

DATA_DIR = TALON_DIR / "data"
RAW_FRAMES_DIR = DATA_DIR / "raw_frames"
SAVED_URLS = DATA_DIR / "saved_urls.md"
PROCESSING_LOG = DATA_DIR / "processing_log.json"

# Add paths for scene_analyzer import
sys.path.insert(0, str(RECOMMENDER_DIR))
sys.path.insert(0, str(MEGA_DIR))

from dotenv import load_dotenv
load_dotenv(MEGA_DIR / ".env")

from scene_analyzer import analyze_video, WORK_DIR, _parse_code, ASSETS_DIR as SA_ASSETS_DIR


def load_urls() -> list[str]:
    """Read URLs from saved_urls.md, skip blank lines and comments."""
    if not SAVED_URLS.exists():
        return []
    urls = []
    for line in SAVED_URLS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def load_log() -> dict:
    """Load processing log."""
    if PROCESSING_LOG.exists():
        try:
            return json.loads(PROCESSING_LOG.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def save_log(log: dict):
    """Save processing log."""
    PROCESSING_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False))


def code_from_url(url: str) -> str:
    """Extract video code from 24av URL."""
    m = re.search(r"/v/([^/?#]+)", url)
    if not m:
        return "UNKNOWN"
    return _parse_code(m.group(1))


def is_already_processed(code: str, log: dict) -> bool:
    """Check if a video is already processed (has frames in raw_frames/{code}/)."""
    if code in log and log[code].get("status") == "done":
        return True
    code_dir = RAW_FRAMES_DIR / code
    if code_dir.exists():
        existing = list(code_dir.glob("*.jpg"))
        return len(existing) > 10
    return False


def _get_scan_start(code: str, scan_idx: int) -> int:
    """Get scan window start time in seconds from analysis JSON."""
    for path in [
        SA_ASSETS_DIR / f"{code}_analysis.json",
        RECOMMENDER_DIR / "outputs" / f"{code}_analysis.json",
        WORK_DIR / code / f"{code}_analysis.json",
    ]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for w in data.get("scan_summary", {}).get("windows", []):
                    if w.get("index") == scan_idx:
                        return int(w["start_s"])
            except Exception:
                pass
    return 0


def extract_individual_frames(code: str) -> dict:
    """Copy individual frames from /tmp work dir to raw_frames/{code}/ with proper naming."""
    work = WORK_DIR / code
    if not work.exists():
        return {"scan_count": 0, "frame_count": 0, "scans": []}

    code_dir = RAW_FRAMES_DIR / code
    code_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    scans = []
    scan_dirs = sorted(work.glob("scan_*_frames"), key=lambda p: int(p.name.split("_")[1]))

    for scan_dir in scan_dirs:
        scan_idx = int(scan_dir.name.split("_")[1])
        frames = sorted(scan_dir.glob("frame_*.jpg"))
        if not frames:
            continue

        scan_start_s = _get_scan_start(code, scan_idx)
        scan_frame_count = 0
        timestamps = []

        for frame_path in frames:
            frame_num = int(frame_path.stem.split("_")[1]) - 1
            total_s = scan_start_s + frame_num * 5

            h, rem = divmod(total_s, 3600)
            m, s = divmod(rem, 60)
            ts = f"{h}_{m:02d}_{s:02d}"

            dst_name = f"{code}_scan{scan_idx}_{ts}.jpg"
            dst = code_dir / dst_name
            shutil.copy2(frame_path, dst)
            scan_frame_count += 1
            timestamps.append(f"{h}:{m:02d}:{s:02d}")

        total_frames += scan_frame_count
        scans.append({
            "scan_idx": scan_idx,
            "frame_count": scan_frame_count,
            "time_range": f"{timestamps[0]}-{timestamps[-1]}" if timestamps else "",
        })

    return {"scan_count": len(scans), "frame_count": total_frames, "scans": scans}


def _load_existing_analysis(code: str) -> dict | None:
    """Try to load an existing analysis JSON (skip re-running scene_analyzer)."""
    for path in [
        SA_ASSETS_DIR / f"{code}_analysis.json",
        RECOMMENDER_DIR / "outputs" / f"{code}_analysis.json",
        WORK_DIR / code / f"{code}_analysis.json",
    ]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if data.get("finish_points") and data.get("scan_summary", {}).get("windows"):
                    return data
            except Exception:
                pass
    return None


def _has_scan_frames(code: str) -> bool:
    """Check if scan frame directories exist in /tmp work dir."""
    work = WORK_DIR / code
    if not work.exists():
        return False
    scan_dirs = list(work.glob("scan_*_frames"))
    return len(scan_dirs) >= 1 and any(
        len(list(d.glob("frame_*.jpg"))) > 0 for d in scan_dirs
    )


def process_one_video(url: str, index: int, total: int) -> dict:
    """Process a single video: full scene_analyzer pipeline + frame extraction."""
    code = code_from_url(url)

    print(f"\n{'='*70}")
    print(f"  [{index}/{total}] Processing {code}")
    print(f"  URL: {url}")
    print(f"{'='*70}")

    t0 = time.time()

    existing = _load_existing_analysis(code)
    if existing and _has_scan_frames(code):
        print(f"  Using existing analysis ({len(existing.get('finish_points', []))} finish points)")
        analysis = existing
    else:
        try:
            analysis = analyze_video(url)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ERROR processing {code}: {e}")
            return {
                "code": code, "url": url, "status": "error",
                "error": str(e), "elapsed_s": round(elapsed, 1),
            }

    if not analysis:
        elapsed = time.time() - t0
        print(f"\n  No results for {code}")
        return {
            "code": code, "url": url, "status": "no_results",
            "elapsed_s": round(elapsed, 1),
        }

    print(f"\n  Extracting individual frames to data/raw_frames/{code}/...")
    frame_info = extract_individual_frames(code)

    elapsed = time.time() - t0

    has_hj = any(
        fp.get("has_handjob_finish", False)
        for fp in analysis.get("finish_points", [])
    )
    hj_count = sum(
        1 for fp in analysis.get("finish_points", [])
        if fp.get("has_handjob_finish", False)
    )

    result = {
        "code": code,
        "url": url,
        "status": "done",
        "has_handjob_finish": has_hj,
        "hj_finish_count": hj_count,
        "total_scans": analysis.get("scan_summary", {}).get("total_scans", 0),
        "total_finish_points": len(analysis.get("finish_points", [])),
        "frames_extracted": frame_info["frame_count"],
        "scan_details": frame_info["scans"],
        "elapsed_s": round(elapsed, 1),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"\n  {code}: {frame_info['scan_count']} scans, {frame_info['frame_count']} frames extracted")
    print(f"  has_handjob_finish: {has_hj} ({hj_count} windows)")
    print(f"  Time: {elapsed:.0f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="TALON — Batch frame extraction for CLIP training")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of videos to process in parallel (default: 1)")
    args = parser.parse_args()

    RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    urls = load_urls()
    if not urls:
        print("No URLs in data/saved_urls.md")
        return

    log = load_log()

    to_process = []
    skipped = []
    for url in urls:
        code = code_from_url(url)
        if is_already_processed(code, log):
            skipped.append(code)
        else:
            to_process.append(url)

    print(f"{'='*70}")
    print(f"  TALON Batch Frame Extraction")
    print(f"{'='*70}")
    print(f"  Total URLs: {len(urls)}")
    print(f"  Already processed (skipping): {len(skipped)} {skipped if skipped else ''}")
    print(f"  To process: {len(to_process)}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*70}")

    if not to_process:
        print("\nAll URLs already processed. Nothing to do.")
        return

    results = []
    total = len(to_process)

    if args.workers <= 1:
        for i, url in enumerate(to_process, 1):
            result = process_one_video(url, i, total)
            results.append(result)
            log[result["code"]] = result
            save_log(log)
    else:
        def _run(item):
            i, url = item
            return process_one_video(url, i, total)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run, (i, url)): url
                for i, url in enumerate(to_process, 1)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    log[result["code"]] = result
                    save_log(log)
                except Exception as e:
                    url = futures[future]
                    code = code_from_url(url)
                    print(f"\n  FATAL ERROR: {code}: {e}")
                    log[code] = {
                        "code": code, "url": url, "status": "fatal_error",
                        "error": str(e),
                        "processed_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    save_log(log)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  BATCH SUMMARY")
    print(f"{'='*70}")

    total_frames = 0
    for r in results:
        status = r.get("status", "?")
        code = r.get("code", "?")
        if status == "done":
            hj = "HJ" if r.get("has_handjob_finish") else "--"
            frames = r.get("frames_extracted", 0)
            scans = r.get("total_scans", 0)
            total_frames += frames
            print(f"  {code:15s} | {status:6s} | {scans:2d} scans | {frames:4d} frames | {hj} | {r.get('elapsed_s',0):.0f}s")
        else:
            print(f"  {code:15s} | {status:10s} | {r.get('error', '')[:50]}")

    print(f"\n  Total frames extracted: {total_frames}")
    print(f"  Output directory: {RAW_FRAMES_DIR}")

    disk_frames = len(list(RAW_FRAMES_DIR.glob("**/*.jpg")))
    print(f"  Total frames on disk: {disk_frames}")


if __name__ == "__main__":
    main()
