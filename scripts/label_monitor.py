#!/usr/bin/env python3
"""Label data monitor — scans talon label folders, shows progress toward targets."""

import os
import re
import sys
import time
import signal
from collections import defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent / "data" / "labels"
NEG_DIR = BASE / "negative"
POS_DIR = BASE / "handjob"

# (vector, label, folder_name, target)
# folder_name is the full directory name under negative/, or "handjob" for positive
CATEGORIES = [
    ("[0,0,0]", "easy negative",   "[0,0,0] easy negative",   100),
    ("[0,0,1]", "只有动作",        "[0,0,1] 只有动作",        250),
    ("[0,1,0]", "只有胸",          "[0,1,0] 只有胸",          250),
    ("[0,1,1]", "有胸动作没有脸",  "[0,1,1] 有胸动作没有脸",  400),
    ("[1,0,0]", "只有脸",          "[1,0,0] 只有脸",          200),
    ("[1,0,1]", "有脸动作没有胸",  "[1,0,1] 有脸动作没有胸",  300),
    ("[1,1,0]", "有脸和胸",        "[1,1,0] 有脸和胸",        500),
    ("[1,1,1]", "positive",        "handjob",                  565),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BANGOU_RE = re.compile(r"^([A-Z]+-\d+)")  # extract JAV code from start of filename
BAR_WIDTH = 30

# ── ANSI helpers ────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"

def color_for_pct(pct: float) -> str:
    if pct >= 100:
        return GREEN
    if pct > 50:
        return YELLOW
    return RED

def progress_bar(current: int, target: int) -> str:
    pct = min(current / target, 1.0) if target else 0
    filled = int(BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    c = color_for_pct(pct * 100)
    return f"{c}{bar}{RESET}"

# ── Scanning ────────────────────────────────────────────────────────────────

def extract_bangou(filename: str) -> str:
    m = BANGOU_RE.match(filename)
    return m.group(1) if m else "(other)"

def scan_folder(folder: Path) -> dict:
    """Return {bangou: count} for image files in folder, grouped by bangou prefix."""
    by_bangou = defaultdict(int)
    if not folder.is_dir():
        return by_bangou
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            bangou = extract_bangou(f.name)
            by_bangou[bangou] += 1
    return dict(by_bangou)

def resolve_folder(folder_name: str) -> Path:
    if folder_name == "handjob":
        return POS_DIR
    return NEG_DIR / folder_name

# ── Rendering ───────────────────────────────────────────────────────────────

def render():
    lines = []
    lines.append(f"\n{BOLD}{CYAN}  ⚔  Talon Label Monitor{RESET}")
    lines.append(f"{DIM}  {'─' * 72}{RESET}\n")

    total_frames = 0
    total_target = 0

    for vec, label, folder_name, target in CATEGORIES:
        folder = resolve_folder(folder_name)
        by_bangou = scan_folder(folder)
        count = sum(by_bangou.values())
        total_frames += count
        total_target += target
        pct = (count / target * 100) if target else 0
        gap = max(target - count, 0)
        c = color_for_pct(pct)

        bar = progress_bar(count, target)
        status = f"{GREEN}✓ DONE{RESET}" if pct >= 100 else f"{c}{gap:>4} left{RESET}"

        lines.append(
            f"  {BOLD}{vec}{RESET}  {label:<16s}  "
            f"{bar}  {c}{count:>5}{RESET}/{target:<5}  "
            f"({c}{pct:5.1f}%{RESET})  {status}"
        )

        # bangou breakdown (top 5 by count)
        if by_bangou:
            sorted_b = sorted(by_bangou.items(), key=lambda x: -x[1])
            top = sorted_b[:5]
            parts = [f"{DIM}{name}{RESET}:{n}" for name, n in top]
            extra = len(sorted_b) - 5
            if extra > 0:
                parts.append(f"{DIM}+{extra} more{RESET}")
            lines.append(f"         {DIM}└─{RESET} {', '.join(parts)}")

    lines.append(f"\n{DIM}  {'─' * 72}{RESET}")
    overall_pct = (total_frames / total_target * 100) if total_target else 0
    oc = color_for_pct(overall_pct)
    lines.append(
        f"  {BOLD}TOTAL{RESET}  {oc}{total_frames}{RESET}/{total_target}  "
        f"({oc}{overall_pct:.1f}%{RESET})   "
        f"gap: {oc}{max(total_target - total_frames, 0)}{RESET}"
    )
    lines.append("")
    return "\n".join(lines)

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    watch = "--watch" in sys.argv or "-w" in sys.argv

    if not watch:
        print(render())
        return

    stop = False
    def on_sigint(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)

    print(f"{DIM}Watch mode — refreshing every 3s. Ctrl+C to exit.{RESET}")
    while not stop:
        os.system("clear")
        print(render())
        print(f"{DIM}  (watching… Ctrl+C to stop){RESET}")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            break

    print(f"\n{CYAN}Stopped.{RESET}")

if __name__ == "__main__":
    main()
