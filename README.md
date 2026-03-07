# TALON

**T**raining **A**nd **L**abeling for **O**ptimal **N**SFW classification.

A CLIP LoRA fine-tune pipeline for detecting handjob finish scenes in JAV videos. Replaces GPT-based classification in scene_analyzer.py with a local model for faster, cheaper, and more consistent inference.

## Architecture

```
Video URL → scene_analyzer.py (coarse scan + HLS extraction)
         → Individual frames → CLIP ViT-B/32 + LoRA → P(handjob)
```

## Pipeline

### 1. Data Collection
```bash
# Add URLs to data/saved_urls.md (one per line)
# Run batch extraction
python scripts/batch_extract_frames.py
```

### 2. Labeling
- Browse `data/raw_frames/` in Finder
- Drag handjob frames to `data/labels/handjob/`
- Optionally: hard negatives to `data/labels/not_handjob/`, uncertain to `data/labels/uncertain/`
- See `docs/labeling_guidelines.md` for detailed rules

### 3. Prepare Dataset
```bash
python scripts/prepare_dataset.py
# Generates data/splits/{train,val,test}.json
```

### 4. Train (on Colab/Kaggle)
```bash
pip install -r training/requirements.txt
python training/train_clip.py --data_dir ./data --epochs 10
```

### 5. Evaluate
```bash
python scripts/evaluate.py --model models/clip_handjob_v1.pt
```

### 6. Deploy
The trained model integrates into scene_analyzer.py via `inference/clip_classifier.py`.

## Project Structure

```
talon/
├── data/
│   ├── saved_urls.md           # URL list for batch processing
│   ├── raw_frames/             # All extracted frames (auto-negative)
│   ├── labels/
│   │   ├── handjob/            # Confirmed positive frames
│   │   ├── not_handjob/        # Hard negatives (optional)
│   │   └── uncertain/          # Ambiguous frames (excluded)
│   ├── splits/                 # Auto-generated train/val/test
│   └── processing_log.json     # Extraction history
├── docs/
│   └── labeling_guidelines.md
├── scripts/
│   ├── batch_extract_frames.py # Video → individual frames
│   ├── prepare_dataset.py      # Labels → train/val/test splits
│   └── evaluate.py             # Model evaluation
├── training/
│   ├── train_clip.py           # CLIP LoRA fine-tune
│   └── requirements.txt
├── models/                     # Trained checkpoints
└── inference/
    └── clip_classifier.py      # Inference API for scene_analyzer
```

## Current Status

- [x] Batch frame extraction pipeline
- [x] Dataset preparation with stratified splits
- [ ] Manual labeling (in progress)
- [ ] CLIP LoRA training script
- [ ] Model evaluation
- [ ] Integration with scene_analyzer.py
