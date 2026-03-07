"""
TALON — CLIP/SigLIP2 inference for handjob finish detection.

Supports two model formats:
  1. Merged model (_merged.pt) — preferred, faster, no extra dependency
  2. Manual LoRA checkpoint — requires open_clip only (no peft)

Supports two vision backbones:
  - CLIP ViT-L/14 (768-dim, 224px) — LoRAMultiheadAttention
  - SigLIP2 ViT-SO400M-14-378 (1152-dim, 378px) — LoRALinear on attn.qkv/proj + mlp.fc1/fc2

Auto-detects format and architecture from checkpoint metadata.

Usage:
    from talon.inference.clip_classifier import TalonClassifier

    clf = TalonClassifier("talon/models/talon_v1_merged.pt")
    score = clf.predict_frame("path/to/frame.jpg")   # float 0.0-1.0
    scores = clf.predict_batch(["a.jpg", "b.jpg"])    # list[float]
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

try:
    import open_clip
except ImportError:
    open_clip = None


# Default CLIP normalization constants (used for backward compat)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ─────────────────────────────────────────────────────────────────────
# Manual LoRA (mirrors training/train_clip.py)
# ─────────────────────────────────────────────────────────────────────

class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with LoRA on out_proj."""

    def __init__(self, orig_mha: nn.MultiheadAttention, rank: int, alpha: float):
        super().__init__()
        self.embed_dim = orig_mha.embed_dim
        self.num_heads = orig_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.attn_dropout = orig_mha.dropout
        self.batch_first = getattr(orig_mha, 'batch_first', False)

        self.in_proj_weight = orig_mha.in_proj_weight
        self.in_proj_bias = orig_mha.in_proj_bias
        self.out_proj = orig_mha.out_proj

        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, self.embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.embed_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None):
        if self.batch_first:
            query = query.transpose(0, 1)

        L, N, E = query.shape
        H = self.num_heads
        D = self.head_dim

        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.contiguous().view(L, N * H, D).transpose(0, 1)
        k = k.contiguous().view(L, N * H, D).transpose(0, 1)
        v = v.contiguous().view(L, N * H, D).transpose(0, 1)

        scale = D ** -0.5
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_weights = attn_weights + attn_mask.unsqueeze(0)
            else:
                attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.bmm(attn_weights, v)
        out = out.transpose(0, 1).contiguous().view(L, N, E)

        base_out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
        lora_out = (out @ self.lora_A.T) @ self.lora_B.T * self.scaling
        result = base_out + lora_out

        if self.batch_first:
            result = result.transpose(0, 1)

        return result, None


def _apply_lora(clip_model, rank, alpha):
    """Replace all MHA modules with LoRA variants (CLIP architecture)."""
    for block in clip_model.visual.transformer.resblocks:
        orig_attn = block.attn
        lora_attn = LoRAMultiheadAttention(orig_attn, rank=rank, alpha=alpha)
        block.attn = lora_attn


class LoRALinear(nn.Module):
    """nn.Linear wrapper with LoRA delta for SigLIP2 (inference-only, no dropout)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        in_features = base.in_features
        out_features = base.out_features

        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def _apply_lora_siglip2(clip_model, rank, alpha,
                        targets=("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")):
    """Wrap target Linear layers in each timm VisionTransformer block with LoRALinear."""
    for block in clip_model.visual.trunk.blocks:
        for target in targets:
            parts = target.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            orig = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(orig, rank=rank, alpha=alpha))


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

class _TalonModel(nn.Module):
    """CLIP vision encoder + classification head."""

    def __init__(self, visual, embed_dim: int):
        super().__init__()
        self.visual = visual
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.0),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x):
        features = self.visual(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.head(features)


def _create_clip_model(clip_name: str, pretrained: str = "openai", force_quick_gelu: bool = True):
    """Create CLIP/SigLIP2 model with config-driven params."""
    create_kwargs = {"model_name": clip_name, "pretrained": pretrained}
    if force_quick_gelu:
        create_kwargs["force_quick_gelu"] = True
    return open_clip.create_model(**create_kwargs)


class TalonClassifier:
    """CLIP-based handjob finish classifier for inference."""

    def __init__(self, model_path: str | Path, threshold: float = 0.5, device: str | None = None):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model = None
        self.transform = None
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.ready = False
        self.load_method = None

        if not self.model_path.exists():
            merged_path = self.model_path.with_name(
                self.model_path.stem.replace("_v1", "_v1_merged") + self.model_path.suffix
            )
            if merged_path.exists():
                self.model_path = merged_path
            else:
                print(f"  TALON: Model not found at {self.model_path}")
                return

        self._load_model()

    def _load_model(self):
        """Load model — auto-detects merged vs LoRA format."""
        print(f"  TALON: Loading {self.model_path.name}...")
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if ckpt.get("merged"):
            self._load_merged(ckpt)
        elif "lora_method" in ckpt:
            self._load_manual_lora(ckpt)
        else:
            print(f"  TALON: Unknown checkpoint format.")
            return

        # Config-driven transforms from checkpoint metadata
        model_config = ckpt.get("model_config", {})
        input_size = model_config.get("input_size", 224)
        norm_mean = tuple(model_config.get("norm_mean", CLIP_MEAN))
        norm_std = tuple(model_config.get("norm_std", CLIP_STD))

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def _load_merged(self, ckpt: dict):
        """Load merged model (no LoRA wrappers, faster)."""
        if open_clip is None:
            print("  TALON: Missing open-clip-torch. pip install open-clip-torch")
            return

        model_config = ckpt.get("model_config", {})
        clip_name = model_config.get("clip_name", "ViT-L-14")
        embed_dim = model_config.get("embed_dim", 768)
        pretrained = model_config.get("pretrained", "openai")
        force_quick_gelu = model_config.get("force_quick_gelu", True)

        clip_model = _create_clip_model(clip_name, pretrained=pretrained, force_quick_gelu=force_quick_gelu)
        clip_model = clip_model.float()

        model = _TalonModel(clip_model.visual, embed_dim)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.load_method = "merged"

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  TALON: Ready (merged, {clip_name}, {total_params:,} params, {self.device})")
        self.ready = True

    def _load_manual_lora(self, ckpt: dict):
        """Load manual LoRA checkpoint — dispatches CLIP vs SigLIP2."""
        if open_clip is None:
            print("  TALON: Missing open-clip-torch. pip install open-clip-torch")
            return

        model_config = ckpt.get("model_config", {})
        clip_name = model_config.get("clip_name", "ViT-L-14")
        embed_dim = model_config.get("embed_dim", 768)
        lora_rank = model_config.get("lora_rank", 16)
        lora_alpha = model_config.get("lora_alpha", 32)
        arch = model_config.get("architecture", "clip")
        pretrained = model_config.get("pretrained", "openai")
        force_quick_gelu = model_config.get("force_quick_gelu", True)

        clip_model = _create_clip_model(clip_name, pretrained=pretrained, force_quick_gelu=force_quick_gelu)
        clip_model = clip_model.float()
        for param in clip_model.parameters():
            param.requires_grad = False

        if arch == "siglip2":
            lora_targets = model_config.get("lora_targets", ["attn.proj"])
            _apply_lora_siglip2(clip_model, rank=lora_rank, alpha=lora_alpha,
                                targets=lora_targets)
        else:
            _apply_lora(clip_model, rank=lora_rank, alpha=lora_alpha)

        model = _TalonModel(clip_model.visual, embed_dim)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.load_method = "manual_lora"

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  TALON: Ready ({arch} LoRA, {clip_name}, {total_params:,} params, {self.device})")
        self.ready = True

    def predict_frame(self, image_path: str | Path) -> float:
        """Classify a single frame. Returns P(handjob) as float 0.0-1.0."""
        if not self.ready:
            return -1.0

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  TALON: Failed to open {image_path}: {e}")
            return -1.0

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            return probs[0, 1].item()

    def predict_batch(self, image_paths: list[str | Path], batch_size: int = 32) -> list[float]:
        """Classify multiple frames. Returns list of P(handjob) scores."""
        if not self.ready:
            return [-1.0] * len(image_paths)

        scores = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            valid_indices = []

            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.transform(img))
                    valid_indices.append(j)
                except Exception:
                    pass

            batch_scores = [-1.0] * len(batch_paths)

            if tensors:
                batch_tensor = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    logits = self.model(batch_tensor)
                    probs = torch.softmax(logits, dim=1)
                    for k, idx in enumerate(valid_indices):
                        batch_scores[idx] = probs[k, 1].item()

            scores.extend(batch_scores)

        return scores

    def predict_sheet(self, sheet_path: str | Path, n_frames: int,
                      cols: int = 6, frame_w: int = 240, frame_h: int = 135,
                      header_h: int = 28, label_h: int = 20, margin: int = 3) -> list[float]:
        """Classify all frames in a scan sheet image."""
        if not self.ready:
            return [-1.0] * n_frames

        try:
            sheet = Image.open(sheet_path).convert("RGB")
        except Exception as e:
            print(f"  TALON: Failed to open sheet {sheet_path}: {e}")
            return [-1.0] * n_frames

        tensors = []
        for i in range(n_frames):
            r, c = divmod(i, cols)
            x = margin + c * (frame_w + margin)
            y = header_h + margin + r * (frame_h + label_h + margin)
            crop = sheet.crop((x, y, x + frame_w, y + frame_h))
            tensors.append(self.transform(crop))

        if not tensors:
            return [-1.0] * n_frames

        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            return [probs[i, 1].item() for i in range(len(tensors))]
