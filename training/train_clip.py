"""
TALON — CLIP/SigLIP2 fine-tune for handjob finish detection.

Supports two vision backbones:
  - CLIP ViT-L/14 (768-dim, 224px) — manual LoRA on nn.MultiheadAttention
  - SigLIP2 ViT-SO400M-14-378 (1152-dim, 378px) — LoRALinear on attn.qkv/proj + mlp.fc1/fc2

CLIP LoRA: replaces entire nn.MultiheadAttention with LoRAMultiheadAttention.
  (open_clip's MHA calls F.multi_head_attention_forward(), bypassing nn.Linear.forward())
SigLIP2 LoRA: wraps target Linear layers (attn.qkv, attn.proj, mlp.fc1, mlp.fc2) with LoRALinear.
  (timm's Attention/Mlp call self.xxx(x) normally, so a Linear wrapper works)

  Setup:
    pip install -r requirements.txt  # open-clip-torch>=3.3.0

  Train (A100, ViT-L/14):
    python train_clip.py \
        --data_dir /content/talon_data --output_dir /content/models \
        --model_name ViT-L-14 --epochs 40 --batch_size 32 --grad_accum 4 --lr 5e-5

  Train (A100, SigLIP2):
    python train_clip.py \
        --data_dir /content/talon_data --output_dir /content/models \
        --model_name ViT-SO400M-14-SigLIP2-378 --epochs 40 --batch_size 16 --grad_accum 8 --lr 1e-5 \
        --lora_rank 32 --lora_alpha 64 --lora_dropout 0.25

  Dry-run (verify LoRA gradients):
    python train_clip.py --data_dir ./data --dry_run
    python train_clip.py --data_dir ./data --dry_run --model_name ViT-SO400M-14-SigLIP2-378

  Resume from checkpoint:
    python train_clip.py ... --resume ./models/checkpoint_epoch10.pt
"""

import json
import math
import random
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    print("Please install: pip install open-clip-torch>=3.3.0")
    raise

# Optional: tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

# ─────────────────────────────────────────────────────────────────────
# Model configs
# ─────────────────────────────────────────────────────────────────────

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

MODEL_CONFIGS = {
    "ViT-B-32": {
        "clip_name": "ViT-B-32",
        "pretrained": "openai",
        "embed_dim": 512,
        "input_size": 224,
        "lora_rank": 8,
        "lora_alpha": 16,
        "norm_mean": CLIP_MEAN,
        "norm_std": CLIP_STD,
        "force_quick_gelu": True,
        "architecture": "clip",
    },
    "ViT-L-14": {
        "clip_name": "ViT-L-14",
        "pretrained": "openai",
        "embed_dim": 768,
        "input_size": 224,
        "lora_rank": 16,
        "lora_alpha": 32,
        "norm_mean": CLIP_MEAN,
        "norm_std": CLIP_STD,
        "force_quick_gelu": True,
        "architecture": "clip",
    },
    "ViT-SO400M-14-SigLIP2-378": {
        "clip_name": "ViT-SO400M-14-SigLIP2-378",
        "pretrained": "webli",
        "embed_dim": 1152,
        "input_size": 378,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.25,
        "lora_targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        "norm_mean": (0.5, 0.5, 0.5),
        "norm_std": (0.5, 0.5, 0.5),
        "force_quick_gelu": False,
        "architecture": "siglip2",
    },
}

# ─────────────────────────────────────────────────────────────────────
# Manual LoRA — two strategies depending on backbone architecture
#
# CLIP (ViT-B-32, ViT-L-14): LoRAMultiheadAttention
#   Replaces entire nn.MultiheadAttention. open_clip's MHA uses
#   F.multi_head_attention_forward() which bypasses nn.Linear.forward(),
#   making peft LoRA inert. We reimplement attention + add LoRA delta.
#
# SigLIP2 (ViT-SO400M-14-SigLIP2-378): LoRALinear
#   Wraps target Linear layers (attn.qkv, attn.proj, mlp.fc1, mlp.fc2).
#   timm's Attention/Mlp call self.xxx(x) which goes through nn.Linear.forward()
#   normally, so a simple Linear wrapper with LoRA delta works fine.
# ─────────────────────────────────────────────────────────────────────

class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with LoRA on out_proj.

    Reuses the original MHA's in_proj_weight/bias and out_proj module
    (frozen), adding trainable lora_A and lora_B parameters.
    """

    def __init__(self, orig_mha: nn.MultiheadAttention,
                 rank: int, alpha: float, dropout: float = 0.05):
        super().__init__()

        assert getattr(orig_mha, '_qkv_same_embed_dim', True), \
            "LoRA only supports self-attention with same Q/K/V embed dim"

        self.embed_dim = orig_mha.embed_dim
        self.num_heads = orig_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.attn_dropout = orig_mha.dropout
        self.batch_first = getattr(orig_mha, 'batch_first', False)

        # Reuse base parameters (stay frozen)
        self.in_proj_weight = orig_mha.in_proj_weight
        self.in_proj_bias = orig_mha.in_proj_bias
        self.out_proj = orig_mha.out_proj

        # LoRA parameters (trainable)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, self.embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.embed_dim, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None):
        # Handle batch_first: convert to (L, N, E) internally
        if self.batch_first:
            query = query.transpose(0, 1)

        L, N, E = query.shape
        H = self.num_heads
        D = self.head_dim

        # ── QKV in-projection ──
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape: (L, N, E) -> (N*H, L, D)
        q = q.contiguous().view(L, N * H, D).transpose(0, 1)
        k = k.contiguous().view(L, N * H, D).transpose(0, 1)
        v = v.contiguous().view(L, N * H, D).transpose(0, 1)

        # ── Scaled dot-product attention ──
        scale = D ** -0.5
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_weights = attn_weights + attn_mask.unsqueeze(0)
            else:
                attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(N, H, L, -1)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_weights = attn_weights.view(N * H, L, -1)

        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.attn_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=True)

        # ── Value aggregation ──
        out = torch.bmm(attn_weights, v)  # (N*H, L, D)
        out = out.transpose(0, 1).contiguous().view(L, N, E)  # (L, N, E)

        # ── Out projection: base linear + LoRA delta ──
        base_out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
        lora_out = (self.lora_dropout(out) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        result = base_out + lora_out

        # Convert back to batch_first if needed
        if self.batch_first:
            result = result.transpose(0, 1)

        return result, None


def _apply_lora(clip_model, rank, alpha, dropout=0.05):
    """Replace all MultiheadAttention modules in the vision transformer with LoRA variants."""
    n_patched = 0
    for block in clip_model.visual.transformer.resblocks:
        orig_attn = block.attn
        lora_attn = LoRAMultiheadAttention(orig_attn, rank=rank, alpha=alpha, dropout=dropout)
        block.attn = lora_attn
        n_patched += 1
    return n_patched


class LoRALinear(nn.Module):
    """nn.Linear wrapper with LoRA delta for SigLIP2's timm Attention.proj.

    forward(x) = base(x) + (dropout(x) @ A^T) @ B^T * scaling
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.05):
        super().__init__()
        self.base = base
        in_features = base.in_features
        out_features = base.out_features

        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def _apply_lora_siglip2(clip_model, rank, alpha, dropout=0.05,
                        targets=("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")):
    """Wrap target Linear layers in each timm VisionTransformer block with LoRALinear.

    Supported targets: attn.qkv, attn.proj, mlp.fc1, mlp.fc2
    """
    n_patched = 0
    for block in clip_model.visual.trunk.blocks:
        for target in targets:
            parts = target.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            orig = getattr(parent, attr)
            if not isinstance(orig, nn.Linear):
                raise RuntimeError(f"Expected nn.Linear at block.{target}, got {type(orig).__name__}")
            setattr(parent, attr, LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout))
            n_patched += 1
    return n_patched


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class TalonDataset(Dataset):
    """Binary classification dataset: handjob (1) vs not (0)."""

    def __init__(self, samples: list[dict], data_dir: Path, transform=None):
        self.samples = samples
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rel_path = s["path"]
        if rel_path.startswith("data/"):
            img_path = self.data_dir / rel_path[5:]
        else:
            img_path = self.data_dir / rel_path

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, s["label"], s.get("video", "")



# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

class TalonClassifier(nn.Module):
    """CLIP vision encoder + LoRA + classification head."""

    def __init__(self, clip_model, embed_dim: int):
        super().__init__()
        self.visual = clip_model.visual
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x):
        features = self.visual(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.head(features)


def build_model(cfg: dict, device: str):
    """Load CLIP/SigLIP2, freeze weights, add LoRA, add classification head."""
    arch = cfg.get("architecture", "clip")
    pretrained = cfg.get("pretrained", "openai")
    force_quick_gelu = cfg.get("force_quick_gelu", True)

    create_kwargs = {"model_name": cfg["clip_name"], "pretrained": pretrained}
    if force_quick_gelu:
        create_kwargs["force_quick_gelu"] = True
    clip_model = open_clip.create_model(**create_kwargs)
    clip_model = clip_model.float()

    # Freeze all base parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    # Apply LoRA — dispatch based on architecture
    lora_dropout = cfg.get("lora_dropout", 0.05)
    if arch == "siglip2":
        lora_targets = cfg.get("lora_targets", ["attn.proj"])
        n_lora = _apply_lora_siglip2(clip_model, rank=cfg["lora_rank"], alpha=cfg["lora_alpha"],
                                     dropout=lora_dropout, targets=lora_targets)
        lora_type_name = "LoRALinear"
        print(f"  LoRA applied to {n_lora} layers across {len(clip_model.visual.trunk.blocks)} blocks "
              f"(targets={lora_targets}, rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']}, "
              f"dropout={lora_dropout}, type={lora_type_name})")
    else:
        n_lora = _apply_lora(clip_model, rank=cfg["lora_rank"], alpha=cfg["lora_alpha"],
                             dropout=lora_dropout)
        lora_type_name = "LoRAMultiheadAttention"
        print(f"  LoRA applied to {n_lora} attention layers (rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']}, "
              f"dropout={lora_dropout}, type={lora_type_name})")

    lora_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in clip_model.parameters())
    print(f"  LoRA params: {lora_params:,} / {total_params:,} ({100*lora_params/total_params:.2f}%)")

    model = TalonClassifier(clip_model, cfg["embed_dim"]).to(device)
    for param in model.head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable (LoRA + head): {trainable:,}")

    # Verify LoRA params are reachable
    lora_a_count = sum(1 for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n)
    lora_b_count = sum(1 for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)
    print(f"  LoRA layers visible to model: lora_A={lora_a_count}, lora_B={lora_b_count}")

    if lora_a_count == 0 or lora_b_count == 0:
        raise RuntimeError(
            f"FATAL: LoRA parameters not reachable! "
            f"lora_A={lora_a_count}, lora_B={lora_b_count}."
        )

    # Verify the replacement actually happened
    if arch == "siglip2":
        for i, block in enumerate(model.visual.trunk.blocks):
            for target in lora_targets:
                parts = target.split(".")
                obj = block
                for part in parts:
                    obj = getattr(obj, part)
                actual_type = type(obj).__name__
                if actual_type != "LoRALinear":
                    raise RuntimeError(
                        f"FATAL: block[{i}].{target} is {actual_type}, expected LoRALinear"
                    )
    else:
        for i, block in enumerate(model.visual.transformer.resblocks):
            attn_type = type(block.attn).__name__
            if attn_type != "LoRAMultiheadAttention":
                raise RuntimeError(
                    f"FATAL: resblock[{i}].attn is {attn_type}, expected LoRAMultiheadAttention"
                )

    return model


def verify_lora_training(model, epoch: int):
    """Check that LoRA B weights are not all-zero (i.e., actually being updated)."""
    all_zero = True
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            if param.data.abs().max().item() > 1e-8:
                all_zero = False
                break

    if all_zero and epoch >= 1:
        print(f"\n  WARNING: All lora_B weights are still zero after epoch {epoch}!")
        print(f"           LoRA is NOT learning — only the head is being trained.")
        if epoch >= 3:
            raise RuntimeError(
                f"FATAL: lora_B still all-zero after epoch {epoch}. "
                f"LoRA adapters are not being updated by the optimizer."
            )
        return False
    return True


def dry_run_check(model, cfg, device):
    """Forward+backward 2 steps, verify LoRA gradients are non-zero.

    Step 1: lora_B is zero-init, so only B gets gradients (A's gradient
    depends on B which is zero -> dL/dA = 0). This is mathematically expected.
    Step 2: After one optimizer step, B is non-zero, so both A and B get gradients.
    """
    arch = cfg.get("architecture", "clip")
    print(f"\n{'='*70}")
    print(f"  DRY RUN — gradient verification (2 steps, arch={arch})")
    print(f"{'='*70}")

    # Verify module types
    if arch == "siglip2":
        blocks = list(model.visual.trunk.blocks)
        lora_targets = cfg.get("lora_targets", ["attn.proj"])
        for i, block in enumerate(blocks):
            for target in lora_targets:
                parts = target.split(".")
                obj = block
                for part in parts:
                    obj = getattr(obj, part)
                actual_type = type(obj).__name__
                if i == 0:
                    print(f"  block[0].{target} type: {actual_type}")
                if actual_type != "LoRALinear":
                    print(f"  FAIL: block[{i}].{target} is {actual_type}, not LoRALinear!")
                    return False
    else:
        blocks = list(model.visual.transformer.resblocks)
        for i, block in enumerate(blocks):
            attn_type = type(block.attn).__name__
            if i == 0:
                print(f"  resblock[0].attn type: {attn_type}")
            if attn_type != "LoRAMultiheadAttention":
                print(f"  FAIL: resblock[{i}].attn is {attn_type}, not LoRAMultiheadAttention!")
                return False
    n_blocks = len(blocks)
    if arch == "siglip2":
        n_lora_modules = n_blocks * len(lora_targets)
    else:
        n_lora_modules = n_blocks

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    batch = torch.randn(4, 3, cfg["input_size"], cfg["input_size"], device=device)
    labels = torch.tensor([0, 1, 0, 1], device=device)

    # Step 1: B gets gradients, A=0 (expected with zero-init B)
    optimizer.zero_grad()
    logits = model(batch)
    loss = criterion(logits, labels)
    loss.backward()
    print(f"  Step 1 loss: {loss.item():.4f}, logits shape: {logits.shape}")

    b_grads_step1 = sum(
        1 for n, p in model.named_parameters()
        if "lora_B" in n and p.grad is not None and p.grad.norm().item() > 0
    )
    print(f"  Step 1: lora_B with grad > 0: {b_grads_step1}/{n_lora_modules} (B zero-init -> A grad=0, expected)")

    if b_grads_step1 == 0:
        print(f"  FAIL: lora_B has no gradients even at step 1!")
        return False

    # Optimizer step: B becomes non-zero
    optimizer.step()

    lora_b_max = max(
        p.data.abs().max().item()
        for n, p in model.named_parameters() if "lora_B" in n
    )
    print(f"  After step 1: lora_B max weight = {lora_b_max:.6f} (should be > 0)")

    # Step 2: Both A and B should now get gradients
    optimizer.zero_grad()
    logits = model(batch)
    loss = criterion(logits, labels)
    loss.backward()
    print(f"  Step 2 loss: {loss.item():.4f}")

    lora_a_grads = []
    lora_b_grads = []
    head_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "lora_A" in name:
                lora_a_grads.append((name, grad_norm))
            elif "lora_B" in name:
                lora_b_grads.append((name, grad_norm))
            elif "head" in name:
                head_grads.append((name, grad_norm))
        elif param.requires_grad:
            print(f"  BUG: {name} requires_grad=True but grad is None")

    print(f"\n  LoRA A gradients ({len(lora_a_grads)} layers):")
    all_a_ok = len(lora_a_grads) > 0 and all(g > 0 for _, g in lora_a_grads)
    print(f"    All non-zero: {'YES' if all_a_ok else 'NO'}")
    if lora_a_grads:
        print(f"    Range: [{min(g for _, g in lora_a_grads):.6f}, "
              f"{max(g for _, g in lora_a_grads):.6f}]")

    print(f"\n  LoRA B gradients ({len(lora_b_grads)} layers):")
    all_b_ok = len(lora_b_grads) > 0 and all(g > 0 for _, g in lora_b_grads)
    print(f"    All non-zero: {'YES' if all_b_ok else 'NO'}")
    if lora_b_grads:
        print(f"    Range: [{min(g for _, g in lora_b_grads):.6f}, "
              f"{max(g for _, g in lora_b_grads):.6f}]")

    print(f"\n  Head gradients ({len(head_grads)} params):")
    all_h_ok = len(head_grads) > 0 and all(g > 0 for _, g in head_grads)
    print(f"    All non-zero: {'YES' if all_h_ok else 'NO'}")

    frozen_nonzero = sum(
        1 for n, p in model.named_parameters()
        if not p.requires_grad and p.grad is not None and p.grad.norm().item() > 0
    )
    print(f"\n  Frozen params with non-zero grad: {frozen_nonzero} (should be 0)")

    ok = all_a_ok and all_b_ok and all_h_ok and frozen_nonzero == 0
    print(f"\n  {'PASS' if ok else 'FAIL'}: "
          f"{'LoRA gradients flowing correctly!' if ok else 'LoRA training will NOT work.'}")

    return ok


# ─────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────


def get_train_transform(input_size: int, mean: tuple = CLIP_MEAN, std: tuple = CLIP_STD):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transform(input_size: int, mean: tuple = CLIP_MEAN, std: tuple = CLIP_STD):
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, grad_accum_steps: int = 1):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    optimizer.zero_grad()
    for step, (images, labels, _) in enumerate(tqdm(loader, desc="  train", leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Handle remaining gradients
    if len(loader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if scheduler:
        scheduler.step()

    n = len(all_labels)
    return (
        total_loss / n if n else 0,
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, zero_division=0),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_videos = [], [], []

    for images, labels, videos in tqdm(loader, desc="  eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_videos.extend(videos)

    n = len(all_labels)
    return {
        "loss": total_loss / n if n else 0,
        "acc": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "f05": fbeta_score(all_labels, all_preds, beta=0.5, zero_division=0),
        "prec": precision_score(all_labels, all_preds, zero_division=0),
        "rec": recall_score(all_labels, all_preds, zero_division=0),
        "preds": all_preds,
        "labels": all_labels,
        "videos": all_videos,
    }


def print_per_video_breakdown(preds, labels, videos):
    """Print accuracy/F1 breakdown by video."""
    by_video = defaultdict(lambda: {"preds": [], "labels": []})
    for p, l, v in zip(preds, labels, videos):
        by_video[v]["preds"].append(p)
        by_video[v]["labels"].append(l)

    print(f"\n  {'Video':15s} {'Total':>6s} {'Pos':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print(f"  {'─'*15} {'─'*6} {'─'*5} {'─'*4} {'─'*4} {'─'*4} {'─'*6} {'─'*6} {'─'*6}")

    for v in sorted(by_video):
        d = by_video[v]
        n = len(d["labels"])
        n_pos = sum(d["labels"])
        tp = sum(p == 1 and l == 1 for p, l in zip(d["preds"], d["labels"]))
        fp = sum(p == 1 and l == 0 for p, l in zip(d["preds"], d["labels"]))
        fn = sum(p == 0 and l == 1 for p, l in zip(d["preds"], d["labels"]))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {v:15s} {n:6d} {n_pos:5d} {tp:4d} {fp:4d} {fn:4d} {prec:6.3f} {rec:6.3f} {f1:6.3f}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TALON — CLIP LoRA fine-tune")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to talon/data/ (contains splits/ and raw_frames/)")
    parser.add_argument("--model_name", type=str, default="ViT-L-14",
                        help="Model name: ViT-B-32, ViT-L-14, or ViT-SO400M-14-SigLIP2-378")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_pct", type=float, default=0.02,
                        help="Fraction of total steps for linear LR warmup")
    parser.add_argument("--class_weight", type=float, default=3.0,
                        help="Positive class weight for CrossEntropyLoss")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (increase if OOM)")
    parser.add_argument("--early_stop", type=int, default=15,
                        help="Stop if val F0.5 doesn't improve for N epochs")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="Override LoRA rank (default from MODEL_CONFIGS)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="Override LoRA alpha (default from MODEL_CONFIGS)")
    parser.add_argument("--lora_dropout", type=float, default=None,
                        help="Override LoRA dropout (default from MODEL_CONFIGS)")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load model, run 1 batch forward+backward, verify gradients, exit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_key = args.model_name
    if model_key not in MODEL_CONFIGS:
        print(f"  ERROR: Unknown model '{model_key}'. Supported: {list(MODEL_CONFIGS.keys())}")
        return
    cfg = MODEL_CONFIGS[model_key].copy()

    # CLI overrides for LoRA hyperparameters
    if args.lora_rank is not None:
        cfg["lora_rank"] = args.lora_rank
    if args.lora_alpha is not None:
        cfg["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        cfg["lora_dropout"] = args.lora_dropout

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    arch = cfg.get("architecture", "clip")
    lora_method = "lora_linear_proj" if arch == "siglip2" else "manual_module_replacement"

    print(f"\n{'='*70}")
    print(f"  TALON LoRA Fine-tune ({arch} backbone)")
    print(f"{'='*70}")
    print(f"  Device:       {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))
    print(f"  Model:        {cfg['clip_name']} (pretrained={cfg.get('pretrained', 'openai')})")
    print(f"  Architecture: {arch}")
    lora_dropout_val = cfg.get("lora_dropout", 0.05)
    lora_targets_val = cfg.get("lora_targets", ["attn.proj"] if arch == "siglip2" else ["out_proj"])
    print(f"  LoRA:         rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']}, dropout={lora_dropout_val} ({lora_method})")
    if arch == "siglip2":
        print(f"  LoRA targets: {lora_targets_val}")
    print(f"  Embed dim:    {cfg['embed_dim']}")
    print(f"  Input size:   {cfg['input_size']}px")
    print(f"  Normalize:    mean={cfg.get('norm_mean', CLIP_MEAN)}, std={cfg.get('norm_std', CLIP_STD)}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size} (accum={args.grad_accum} -> effective {args.batch_size * args.grad_accum})")
    print(f"  LR:           {args.lr} (warmup {args.warmup_pct*100:.0f}%)")
    print(f"  Precision:    fp32")
    print(f"  Early stop:   {args.early_stop} epochs (F0.5)")
    print(f"  Class weight: {args.class_weight}")
    print(f"  open_clip:    {open_clip.__version__}")
    print(f"  torch:        {torch.__version__}")

    # ── Build model ──
    print(f"\n  Loading {cfg['clip_name']}...")
    model = build_model(cfg, device)

    # ── Dry-run mode ──
    if args.dry_run:
        ok = dry_run_check(model, cfg, device)
        return 0 if ok else 1

    # ── Load splits ──
    data_dir = Path(args.data_dir)
    train_samples = json.loads((data_dir / "splits" / "train.json").read_text())
    val_samples = json.loads((data_dir / "splits" / "val.json").read_text())
    test_samples = json.loads((data_dir / "splits" / "test.json").read_text())

    for name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        pos = sum(1 for s in samples if s["label"] == 1)
        print(f"  {name:5s}: {len(samples):6d} ({pos} pos / {len(samples)-pos} neg)")

    # ── Print all trainable parameter names ──
    print(f"\n  Trainable parameters:")
    n_lora = 0
    n_head = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            group = "LoRA" if "lora_" in name else "head"
            if "lora_" in name:
                n_lora += param.numel()
            else:
                n_head += param.numel()
            print(f"    [{group:4s}] {name}: {tuple(param.shape)}")
    print(f"  Total: {n_lora:,} LoRA + {n_head:,} head = {n_lora + n_head:,}")

    # ── Transforms ──
    norm_mean = cfg.get("norm_mean", CLIP_MEAN)
    norm_std = cfg.get("norm_std", CLIP_STD)
    train_transform = get_train_transform(cfg["input_size"], mean=norm_mean, std=norm_std)
    val_transform = get_val_transform(cfg["input_size"], mean=norm_mean, std=norm_std)

    # ── Datasets & loaders ──
    train_ds = TalonDataset(train_samples, data_dir, transform=train_transform)
    val_ds = TalonDataset(val_samples, data_dir, transform=val_transform)
    test_ds = TalonDataset(test_samples, data_dir, transform=val_transform)

    effective = len(train_ds)
    print(f"  Samples/epoch: {effective} (full dataset, shuffle=True)")

    num_workers = min(4, len(train_samples) // 100 + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Loss with class weights ──
    class_weights = torch.tensor([1.0, args.class_weight], device=device)
    print(f"  Class weights: [neg=1.0, pos={args.class_weight:.1f}]")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer (explicit about what's included) ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Optimizer params: {len(trainable_params)} tensors, "
          f"{sum(p.numel() for p in trainable_params):,} values")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # ── LR scheduler: linear warmup + cosine decay ──
    steps_per_epoch = math.ceil(effective / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Tensorboard ──
    tb_writer = None
    if HAS_TB:
        tb_dir = Path(args.output_dir) / "tb_logs"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"  Tensorboard: {tb_dir}")

    # ── Resume ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_val_f05 = 0
    best_epoch = 0
    no_improve_count = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f05 = ckpt.get("val_f05", ckpt.get("val_f1", 0))
        best_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch - 1}, best val F0.5={best_val_f05:.4f}")

    # ── Checkpoint metadata ──
    # Ensure lora_targets is in model_config for inference/export
    if arch == "siglip2" and "lora_targets" not in cfg:
        cfg["lora_targets"] = ["attn.proj"]
    ckpt_meta = {
        "args": vars(args),
        "model_config": cfg,
        "lora_method": lora_method,
        "library_versions": {
            "open_clip": open_clip.__version__,
            "torch": torch.__version__,
        },
    }

    # ── Training loop ──
    print(f"\n{'='*70}")
    print(f"  Training ({args.epochs} epochs, early_stop={args.early_stop})")
    print(f"{'='*70}")
    hdr = f"  {'Ep':>3s} | {'TrLoss':>7s} {'TrAcc':>6s} {'TrF1':>5s} | {'VLoss':>7s} {'VAcc':>5s} {'VF05':>5s} {'VF1':>5s} {'VP':>5s} {'VR':>5s} | {'LR':>8s}"
    print(hdr)
    print(f"  {'─'*3}─┼─{'─'*7}─{'─'*6}─{'─'*5}─┼─{'─'*7}─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─┼─{'─'*8}")

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, grad_accum_steps=args.grad_accum,
        )
        val_res = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        star = ""

        # Tensorboard
        if tb_writer:
            tb_writer.add_scalars("loss", {"train": train_loss, "val": val_res["loss"]}, epoch)
            tb_writer.add_scalars("f1", {"train": train_f1, "val": val_res["f1"]}, epoch)
            tb_writer.add_scalar("f05/val", val_res["f05"], epoch)
            tb_writer.add_scalar("lr", current_lr, epoch)

        # ── LoRA health check ──
        verify_lora_training(model, epoch)

        # Best model (by val F0.5 — precision-weighted)
        if val_res["f05"] > best_val_f05:
            best_val_f05 = val_res["f05"]
            best_epoch = epoch
            no_improve_count = 0
            star = " *"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f05": val_res["f05"],
                "val_f1": val_res["f1"],
                "val_acc": val_res["acc"],
                **ckpt_meta,
            }, output_dir / "talon_best.pt")
        else:
            no_improve_count += 1

        # Checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f05": val_res["f05"],
            "val_f1": val_res["f1"],
            **ckpt_meta,
        }, output_dir / f"checkpoint_epoch{epoch}.pt")

        print(f"  {epoch:3d} | {train_loss:7.4f} {train_acc:6.4f} {train_f1:5.3f} | "
              f"{val_res['loss']:7.4f} {val_res['acc']:5.3f} {val_res['f05']:5.3f} "
              f"{val_res['f1']:5.3f} {val_res['prec']:5.3f} {val_res['rec']:5.3f} | "
              f"{current_lr:8.2e}{star}")

        # Early stopping
        if no_improve_count >= args.early_stop:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.early_stop} epochs)")
            break

    elapsed = time.time() - t_start
    print(f"\n  Training complete in {elapsed/60:.1f} min")
    print(f"  Best val F0.5: {best_val_f05:.4f} at epoch {best_epoch}")

    # ── Final LoRA weight summary ──
    print(f"\n  LoRA weight summary:")
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            norm = param.data.norm().item()
            maxv = param.data.abs().max().item()
            print(f"    {name}: norm={norm:.6f}, max={maxv:.6f}")
            break
    lora_b_norms = [
        p.data.norm().item()
        for n, p in model.named_parameters()
        if "lora_B" in n and p.requires_grad
    ]
    print(f"  lora_B norms: min={min(lora_b_norms):.6f}, max={max(lora_b_norms):.6f}, "
          f"mean={sum(lora_b_norms)/len(lora_b_norms):.6f}")

    # ── Test evaluation ──
    print(f"\n{'='*70}")
    print(f"  Test Set Evaluation (best model from epoch {best_epoch})")
    print(f"{'='*70}")

    ckpt = torch.load(output_dir / "talon_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_res = evaluate(model, test_loader, criterion, device)

    print(f"  Accuracy:  {test_res['acc']:.4f}")
    print(f"  Precision: {test_res['prec']:.4f}")
    print(f"  Recall:    {test_res['rec']:.4f}")
    print(f"  F1 Score:  {test_res['f1']:.4f}")
    print(f"  F0.5:      {test_res['f05']:.4f}")

    cm = confusion_matrix(test_res["labels"], test_res["preds"])
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Neg    Pos")
    if cm.shape == (2, 2):
        print(f"  Actual Neg  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"  Actual Pos  {cm[1][0]:5d}  {cm[1][1]:5d}")
    else:
        print(f"  {cm}")

    print(f"\n  Classification Report:")
    print(classification_report(
        test_res["labels"], test_res["preds"],
        target_names=["negative", "handjob"], zero_division=0,
    ))

    print_per_video_breakdown(test_res["preds"], test_res["labels"], test_res["videos"])

    print(f"\n  Val Set Breakdown:")
    val_res = evaluate(model, val_loader, criterion, device)
    print_per_video_breakdown(val_res["preds"], val_res["labels"], val_res["videos"])

    if tb_writer:
        tb_writer.close()

    # Clean up old checkpoints, keep best + last 3
    checkpoints = sorted(output_dir.glob("checkpoint_epoch*.pt"),
                         key=lambda p: int(p.stem.split("epoch")[1]))
    for ckpt_path in checkpoints[:-3]:
        if ckpt_path.name != "talon_best.pt":
            ckpt_path.unlink()
            print(f"  Cleaned up: {ckpt_path.name}")

    print(f"\n  Done! Best model: {output_dir / 'talon_best.pt'}")


if __name__ == "__main__":
    main()
