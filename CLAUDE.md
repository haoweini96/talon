# CLAUDE.md — TALON

## Project Overview

**TALON** = **T**raining **A**nd **L**abeling for **O**ptimal **N**SFW classification.

A CLIP LoRA fine-tune pipeline for detecting handjob finish scenes in JAV videos. Replaces GPT-based classification in the parent `mega` project's `scene_analyzer.py` with a local model for faster, cheaper, and more consistent inference.

This is a **self-contained ML pipeline** — no web framework, no build tools, no deployment system. Pure Python scripts for data collection, labeling, training, and inference.

## Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Vision backbone | OpenAI CLIP + SigLIP2 via `open-clip-torch>=3.3.0` | ViT-B/32, ViT-L/14 (CLIP), ViT-SO400M-14-SigLIP2-378 (SigLIP2) |
| Fine-tuning | Manual LoRA (no peft) | CLIP: replace `nn.MultiheadAttention` modules. SigLIP2: `LoRALinear` on `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` |
| Training engine | PyTorch >=2.0 | AdamW, CrossEntropyLoss with positive class weighting |
| Metrics | scikit-learn | F0.5 (precision-weighted), stratified splits |
| Pre-screening | GPT-5.2 via OpenAI API | Contact sheet → binary classification |
| Frame extraction | ffmpeg + HLS/m3u8 | Via `mega/recommender/scene_analyzer.py` |
| Visualization | TensorBoard | Training curves |
| Image I/O | Pillow >=10.0 | Frame loading and contact sheets |

### Dependencies

Install via `pip install -r training/requirements.txt`. Key pins:
- `open-clip-torch>=3.3.0` — required for SigLIP2 support. CLIP models use `force_quick_gelu=True` + `pretrained="openai"`; SigLIP2 uses `pretrained="webli"` (no quick_gelu). Architecture and normalize constants are stored in checkpoint `model_config`.
- `timm>=0.9.12` — implicit dependency for SigLIP2's `VisionTransformer`
- `torch>=2.0`, `torchvision>=0.15`
- `scikit-learn>=1.3`, `numpy>=1.24`, `tqdm`, `tensorboard`

External (from parent `mega` project):
- `openai`, `python-dotenv` — for GPT-based pre-screening
- `scene_analyzer.py` — video analysis and HLS frame extraction

## Directory Structure

```
talon/
├── CLAUDE.md                          # This file
├── README.md                          # Quick-start guide
├── docs/
│   └── labeling_guidelines.md         # Visual criteria for positive/negative labeling
│
├── data/
│   ├── saved_urls.md                  # Input URLs (one per line, # for comments)
│   ├── raw_frames/                    # Extracted frames organized by video code
│   │   └── {CODE}/                    # e.g. MIDE-993/, FPRE-161/
│   ├── labels/                        # Manual labeling (drag in Finder)
│   │   ├── handjob/                   # Positive samples (~660 frames)
│   │   └── negative/                  # Negative samples (~1500 frames)
│   │       ├── [0,0,0] easy negative/
│   │       ├── [0,0,1] 只有动作/
│   │       ├── [0,1,0] 只有胸/
│   │       ├── [1,0,0] 只有脸/
│   │       └── ...                    # 3-tuple: [face, chest, action]
│   ├── splits/                        # Auto-generated train/val/test JSON
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── coarse_scan_results/           # GPT-5.2 pre-screening output
│   ├── coarse_positive/               # Frames flagged positive by GPT
│   ├── handjob_review/                # Manual false-positive review
│   └── processing_log.json            # Frame extraction history
│
├── scripts/                           # Data pipeline scripts
│   ├── batch_extract_frames.py        # Video URLs → raw frames (via scene_analyzer)
│   ├── prepare_dataset.py             # Label dirs → stratified train/val/test splits
│   ├── coarse_scan_frames.py          # Contact sheets → GPT-5.2 pre-screening
│   ├── gpt_prescreening.py            # Legacy GPT pre-screening
│   ├── extract_handjob_clips.py       # Extract short video clips from timestamps
│   ├── extract_coarse_positive.py     # Copy GPT-positive frames to coarse_positive/
│   ├── export_merged.py               # Merge LoRA weights into base CLIP for deployment
│   ├── evaluate.py                    # Model evaluation stub
│   └── label_monitor.py               # Watch label dirs, auto-sync to splits
│
├── training/
│   ├── train_clip.py                  # Core CLIP LoRA fine-tuning (880 lines)
│   └── requirements.txt               # Pinned dependencies
│
├── inference/
│   └── clip_classifier.py             # Unified inference API (merged + LoRA checkpoints)
│
├── models/                            # Trained checkpoints (gitignored)
│   ├── talon_best.pt                  # Best LoRA checkpoint (by F0.5)
│   └── talon_v1_merged.pt             # Merged model for deployment
│
├── diagnose_fp.py                     # False positive analysis tool
└── extract_frames.py                  # Legacy frame extractor (use batch_extract instead)
```

## Key Files

### Entry Points (in pipeline order)

| Step | Script | What it does |
|------|--------|-------------|
| 1. Collect | `scripts/batch_extract_frames.py` | Reads URLs from `data/saved_urls.md`, calls scene_analyzer, extracts frames to `data/raw_frames/{CODE}/` |
| 2. Pre-screen (optional) | `scripts/coarse_scan_frames.py` | Builds contact sheets, sends to GPT-5.2 for binary handjob detection |
| 3. Label | Manual in Finder | Drag frames into `data/labels/handjob/` or `data/labels/negative/[X,Y,Z] .../` |
| 4. Split | `scripts/prepare_dataset.py` | Scans label dirs, creates stratified train/val/test JSON splits |
| 5. Train | `training/train_clip.py` | CLIP/SigLIP2 ViT + manual LoRA, trains classification head |
| 6. Export | `scripts/export_merged.py` | Merges LoRA into base weights (~19% faster inference) |
| 7. Deploy | `inference/clip_classifier.py` | `TalonClassifier` class for integration with scene_analyzer |

### Critical Implementation Details

**`training/train_clip.py`** — The core training script:
- Two LoRA strategies: `LoRAMultiheadAttention` for CLIP, `LoRALinear` for SigLIP2
- Config-driven: `MODEL_CONFIGS` dict stores architecture, pretrained source, normalize constants
- `dry_run_check()` verifies LoRA gradients flow before committing to full training
- Early stopping on F0.5 score (precision-weighted — false positives are costly)
- Positive class weight default: 3.0x to handle imbalanced dataset
- Gradient clipping: `clip_grad_norm_(1.0)` per batch
- Checkpoints save full model config (incl. architecture, norm values), library versions, and LoRA method

**`inference/clip_classifier.py`** — Inference API:
- Auto-detects merged vs LoRA checkpoint format and CLIP vs SigLIP2 architecture
- Config-driven transforms: reads `input_size`, `norm_mean`, `norm_std` from checkpoint
- Methods: `predict_frame(path) → float`, `predict_batch(paths) → list[float]`, `predict_sheet(path)`
- Mirrors `LoRAMultiheadAttention` and `LoRALinear` classes from training

## Running Commands

```bash
# 1. Extract frames from saved URLs
cd projects/mega && venv/bin/python3 ../talon/scripts/batch_extract_frames.py

# 2. GPT pre-screening (optional, requires OPENAI_API_KEY in mega/.env)
cd projects/mega && venv/bin/python3 ../talon/scripts/coarse_scan_frames.py

# 3. Prepare dataset splits after labeling
python3 scripts/prepare_dataset.py [--train-ratio 0.6] [--val-ratio 0.2] [--seed 42]

# 4. Train on GPU (Colab/Kaggle) — CLIP ViT-L/14
python3 training/train_clip.py \
    --data_dir ./data --output_dir ./models \
    --model_name ViT-L-14 --epochs 40 --batch_size 32 \
    --grad_accum 4 --lr 5e-5

# 4. Train on GPU (Colab/Kaggle) — SigLIP2
python3 training/train_clip.py \
    --data_dir ./data --output_dir ./models \
    --model_name ViT-SO400M-14-SigLIP2-378 \
    --epochs 40 --batch_size 16 --grad_accum 8 --lr 1e-5 \
    --lora_rank 32 --lora_alpha 64 --lora_dropout 0.25

# 4a. Dry run (verify LoRA gradients without full training)
python3 training/train_clip.py --data_dir ./data --dry_run
python3 training/train_clip.py --data_dir ./data --dry_run --model_name ViT-SO400M-14-SigLIP2-378

# 4b. Resume from checkpoint
python3 training/train_clip.py --data_dir ./data --resume ./models/checkpoint_epoch10.pt

# 5. Export merged model
python3 scripts/export_merged.py \
    --model models/talon_best.pt --output models/talon_v1_merged.pt

# 6. Diagnose false positives
python3 diagnose_fp.py
```

## Development Conventions

### Code Patterns

- **Pure Python scripts** — no build tools, no bundlers, no package manager
- **Path resolution**: Always use `Path(__file__).resolve().parent` chains to navigate relative to script location
- **Cross-project imports**: `sys.path.insert(0, str(RECOMMENDER_DIR))` to import scene_analyzer from mega
- **Environment**: API keys loaded from `../mega/.env` via `python-dotenv`
- **Logging**: Structured print with `"=" * 70` separator headers; no logging module
- **Error handling**: Graceful degradation — log failures, don't raise. Processing continues on individual video failures

### Naming Conventions

- **Frame files**: `{CODE}_scan{N}_{H}_{MM}_{SS}.jpg` (e.g. `MIDE-993_scan3_0_45_05.jpg`)
- **Video codes**: Uppercase alphanumeric with hyphen (e.g. `FPRE-161`, `MIDE-068`)
- **Checkpoints**: `checkpoint_epoch{N}.pt`, `talon_best.pt`, `talon_v1_merged.pt`
- **Negative categories**: `[face, chest, action]` 3-tuple prefix (e.g. `[0,1,1] 有胸动作没有脸`)

### Data Structures

**Sample JSON** (in splits):
```json
{"path": "data/labels/handjob/MIDE-993_scan3_0_45_05.jpg", "label": 1, "video": "MIDE-993", "category": "[1,1,1]"}
```

**Checkpoint metadata**:
```python
{
    "model_state_dict": {...},
    "epoch": int,
    "val_f05": float,
    "model_config": {
        "clip_name": str, "pretrained": str, "embed_dim": int,
        "input_size": int, "lora_rank": int, "lora_alpha": int,
        "lora_dropout": float, "lora_targets": list[str],  # SigLIP2 only
        "norm_mean": tuple, "norm_std": tuple,
        "force_quick_gelu": bool, "architecture": str,  # "clip" or "siglip2"
    },
    "lora_method": "manual_module_replacement" | "lora_linear_proj",
    "library_versions": {"open_clip": str, "torch": str},
}
```

### Key Technical Decisions

1. **Manual LoRA over peft**: open_clip's `nn.MultiheadAttention` calls `F.multi_head_attention_forward()` internally, which bypasses `nn.Linear.forward()`. This makes peft-style LoRA completely inert. We replace entire MHA modules instead. For SigLIP2 (timm's `Attention`/`Mlp`), calls go through `nn.Linear.forward()` normally, so `LoRALinear` wrapper on `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` works.

2. **F0.5 over F1**: Precision is prioritized over recall — a false positive (marking non-handjob as handjob) is worse than missing a true positive in the downstream recommendation system.

3. **Video-stratified splits**: All frames from the same video stay in the same split to prevent data leakage across train/val/test.

4. **Version pinning**: `open-clip-torch>=3.3.0` required for SigLIP2 support. Architecture differences (GELU vs QuickGELU, normalize constants) are stored in checkpoint `model_config` so inference always matches training. CLIP models use `force_quick_gelu=True`, SigLIP2 does not.

### What NOT to Do

- Don't use peft/LoRA libraries — they're inert on open_clip CLIP models (see above); SigLIP2 uses LoRALinear instead
- Don't mix CLIP and SigLIP2 checkpoints — architectures differ (MHA vs timm Attention, different normalize constants)
- Don't mix frames from the same video across train/val/test splits
- Don't run training scripts on CPU — they're designed for GPU (Colab A100)
- Don't delete `processing_log.json` — it tracks which videos have been processed to avoid re-extraction
