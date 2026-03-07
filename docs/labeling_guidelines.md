# TALON Labeling Guidelines

## What is Handjob?

A frame is labeled **handjob** when it shows the female performer using her hand to service the male performer, with the following elements **all visible in the same frame**:

1. **Female's face visible** (front or side profile)
2. **Female's upper body visible** (not a close-up crop of just a hand or body part)
3. **Female's arm extending downward** from her body, with a gripping/stroking motion

### Typical handjob frame
- Female is sitting or kneeling
- Facing the camera (or side-facing)
- Upper body exposed
- One or both arms reaching down toward the male's lower body area
- Hand in a gripping position with visible up-down motion

## What is NOT Handjob

- **Male touching female** — the male's hand on the female's body is not handjob
- **Foreplay / mutual touching** — general caressing or fondling is not handjob
- **Hand on thigh or other body part** — hand must be actively servicing
- **Close-up body crop** — if you can't see the female's face, it's not handjob (even if a hand is visible)
- **Blowjob** — head positioned near the male's lower area (oral, not manual)
- **Sex positions** — penetrative positions (missionary, doggy, cowgirl, etc.)
- **Titjob** — using chest to service

## Uncertain

Label a frame as **uncertain** when:
- The image is blurry or too dark to clearly identify the action
- The camera angle makes it impossible to tell what's happening
- Multiple actions are happening simultaneously and it's ambiguous
- You genuinely can't decide — don't force a label

## Labeling Process

1. Browse `data/raw_frames/` in Finder
2. Frames that clearly show handjob → drag to `data/labels/handjob/`
3. Frames you specifically want to mark as "definitely NOT handjob" (hard negatives) → drag to `data/labels/not_handjob/`
4. Frames you're unsure about → drag to `data/labels/uncertain/`
5. Frames left in `raw_frames/` are automatically treated as negative samples

### Tips
- Use QuickLook (spacebar) in Finder to preview frames quickly
- You can sort by filename to review frames from the same video/scan together
- Focus on labeling the handjob positives first — the negatives are auto-generated from unlabeled frames
- Hard negatives (not_handjob/) are optional but help the model learn tricky cases

## Naming Convention

Frames follow the pattern: `{CODE}_scan{N}_{H}_{MM}_{SS}.jpg`

Example: `MIDE-993_scan3_0_45_05.jpg` = MIDE-993, scan window 3, timestamp 0:45:05
