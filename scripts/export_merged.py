"""
TALON — Export merged model (LoRA baked into base weights).

Supports both CLIP (LoRAMultiheadAttention) and SigLIP2 (LoRALinear) checkpoints.
Merges LoRA adapters into base weights and saves a clean state_dict that loads
without LoRA wrappers. ~19% faster inference.

Usage:
    python export_merged.py
    python export_merged.py --model talon/models/talon_best.pt --output talon/models/talon_v1_merged.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    import open_clip
except ImportError:
    sys.exit("Missing: pip install open-clip-torch>=3.3.0")


SCRIPT_DIR = Path(__file__).resolve().parent
TALON_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL = TALON_DIR / "models" / "talon_best.pt"
DEFAULT_OUTPUT = TALON_DIR / "models" / "talon_v1_merged.pt"


class TalonModel(nn.Module):
    """CLIP visual encoder + classification head (no LoRA wrappers)."""

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


def _create_clip_model(clip_name, pretrained="openai", force_quick_gelu=True):
    """Create CLIP/SigLIP2 model with config-driven params."""
    create_kwargs = {"model_name": clip_name, "pretrained": pretrained}
    if force_quick_gelu:
        create_kwargs["force_quick_gelu"] = True
    return open_clip.create_model(**create_kwargs)


def _merge_clip_lora(sd, scaling):
    """Merge LoRAMultiheadAttention weights into base out_proj weights.

    State dict pattern:
        visual.transformer.resblocks.X.attn.out_proj.weight  (base)
        visual.transformer.resblocks.X.attn.lora_A            (LoRA)
        visual.transformer.resblocks.X.attn.lora_B            (LoRA)
    """
    merged_sd = {}
    n_merged = 0
    skip_keys = set()

    for key in sd:
        if key.endswith(".lora_A"):
            prefix = key[:-len(".lora_A")]
            lora_a_key = key
            lora_b_key = prefix + ".lora_B"
            out_proj_key = prefix + ".out_proj.weight"

            if lora_b_key not in sd or out_proj_key not in sd:
                print(f"  WARNING: Missing pair for {key}")
                continue

            lora_a = sd[lora_a_key]   # (rank, dim)
            lora_b = sd[lora_b_key]   # (dim, rank)
            base_w = sd[out_proj_key]  # (dim, dim)

            merged_sd[out_proj_key] = base_w + (lora_b @ lora_a) * scaling
            n_merged += 1

            skip_keys.update([lora_a_key, lora_b_key, out_proj_key])

    for key, value in sd.items():
        if key in skip_keys or "lora_" in key:
            continue
        merged_sd[key] = value

    return merged_sd, n_merged


def _merge_siglip2_lora(sd, scaling):
    """Merge LoRALinear weights into base weights for all wrapped layers.

    LoRALinear wraps any nn.Linear, producing state dict keys like:
        {prefix}.base.weight  (base)
        {prefix}.base.bias    (base)
        {prefix}.lora_A       (LoRA)
        {prefix}.lora_B       (LoRA)

    Merged keys become:
        {prefix}.weight
        {prefix}.bias
    """
    merged_sd = {}
    n_merged = 0
    skip_keys = set()

    for key in sd:
        if key.endswith(".lora_A") and ".trunk.blocks." in key:
            prefix = key[:-len(".lora_A")]
            lora_a_key = key
            lora_b_key = prefix + ".lora_B"
            base_w_key = prefix + ".base.weight"
            base_b_key = prefix + ".base.bias"

            if lora_b_key not in sd or base_w_key not in sd:
                print(f"  WARNING: Missing pair for {key}")
                continue

            lora_a = sd[lora_a_key]   # (rank, in_feat)
            lora_b = sd[lora_b_key]   # (out_feat, rank)
            base_w = sd[base_w_key]   # (out_feat, in_feat)

            # Merge: W_merged = W_base + B @ A * scaling
            merged_key = prefix + ".weight"
            merged_sd[merged_key] = base_w + (lora_b @ lora_a) * scaling
            n_merged += 1

            skip_keys.update([lora_a_key, lora_b_key, base_w_key])

            # Rename .base.bias → .bias
            if base_b_key in sd:
                merged_sd[prefix + ".bias"] = sd[base_b_key]
                skip_keys.add(base_b_key)

    for key, value in sd.items():
        if key in skip_keys or "lora_" in key:
            continue
        # Rename remaining .base. keys from LoRALinear wrappers
        if ".base." in key and ".trunk.blocks." in key:
            new_key = key.replace(".base.", ".", 1)
            # Only replace .base. that's at a LoRA target position
            # (the .base. comes from LoRALinear wrapping the original Linear)
            merged_sd[new_key] = value
        else:
            merged_sd[key] = value

    return merged_sd, n_merged


def main():
    parser = argparse.ArgumentParser(description="TALON — Export merged model")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    print(f"\n{'='*60}")
    print(f"  TALON Merged Model Export")
    print(f"{'='*60}")

    print(f"  Loading checkpoint: {model_path.name}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model_config = ckpt.get("model_config", {})
    clip_name = model_config.get("clip_name", "ViT-L-14")
    embed_dim = model_config.get("embed_dim", 768)
    lora_rank = model_config.get("lora_rank", 16)
    lora_alpha = model_config.get("lora_alpha", 32)
    lora_method = ckpt.get("lora_method", "unknown")
    arch = model_config.get("architecture", "clip")
    pretrained = model_config.get("pretrained", "openai")
    force_quick_gelu = model_config.get("force_quick_gelu", True)
    input_size = model_config.get("input_size", 224)
    scaling = lora_alpha / lora_rank

    print(f"  Model: {clip_name}, embed_dim={embed_dim}, arch={arch}")
    print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}, scaling={scaling:.2f}, method={lora_method}")
    print(f"  Epoch: {ckpt.get('epoch')}, Val F1: {ckpt.get('val_f1', 0):.4f}")

    sd = ckpt["model_state_dict"]

    # Check LoRA B weights are not zero
    lora_b_keys = [k for k in sd if "lora_B" in k]
    if lora_b_keys:
        all_zero = all(sd[k].abs().max().item() < 1e-8 for k in lora_b_keys)
        if all_zero:
            print(f"\n  ERROR: All lora_B weights are ZERO — LoRA was never trained!")
            sys.exit(1)
        max_norm = max(sd[k].norm().item() for k in lora_b_keys)
        print(f"  LoRA B max norm: {max_norm:.6f} (OK, non-zero)")

    # ── Merge LoRA weights into base weights ──
    print(f"  Merging LoRA weights ({arch} architecture)...")
    if arch == "siglip2":
        merged_sd, n_merged = _merge_siglip2_lora(sd, scaling)
    else:
        merged_sd, n_merged = _merge_clip_lora(sd, scaling)

    print(f"  Merged {n_merged} LoRA layers into base weights")
    print(f"  Merged state dict: {len(merged_sd)} keys")

    # Build clean model (no LoRA) and load merged weights
    clip_model = _create_clip_model(clip_name, pretrained=pretrained, force_quick_gelu=force_quick_gelu)
    clip_model = clip_model.float()
    clean_model = TalonModel(clip_model.visual, embed_dim)
    clean_model.load_state_dict(merged_sd)
    clean_model.eval()

    # Save — pass full model_config so inference can reconstruct transforms
    save_data = {
        "model_state_dict": merged_sd,
        "model_config": model_config,
        "merged": True,
        "source_checkpoint": model_path.name,
        "epoch": ckpt.get("epoch"),
        "val_f1": ckpt.get("val_f1"),
        "library_versions": ckpt.get("library_versions", {
            "open_clip": open_clip.__version__,
        }),
    }
    torch.save(save_data, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path.name} ({size_mb:.1f} MB)")

    # Verify: reload and compare
    print(f"\n  Verifying consistency...")
    dummy = torch.randn(4, 3, input_size, input_size)
    with torch.no_grad():
        merged_out = clean_model(dummy)

    ckpt2 = torch.load(output_path, map_location="cpu", weights_only=False)
    clip2 = _create_clip_model(clip_name, pretrained=pretrained, force_quick_gelu=force_quick_gelu)
    clip2 = clip2.float()
    reload_model = TalonModel(clip2.visual, embed_dim)
    reload_model.load_state_dict(ckpt2["model_state_dict"])
    reload_model.eval()

    with torch.no_grad():
        reload_out = reload_model(dummy)

    max_diff = torch.max(torch.abs(merged_out - reload_out)).item()
    print(f"  Max diff (merged vs reloaded): {max_diff:.8f}")
    print(f"  Consistency: {'OK' if max_diff < 1e-5 else 'WARNING: outputs differ!'}")

    # Verify LoRA was baked in (compare against base model)
    clip_base = _create_clip_model(clip_name, pretrained=pretrained, force_quick_gelu=force_quick_gelu)
    clip_base = clip_base.float()
    base_sd = clip_base.visual.state_dict()
    reload_visual_sd = reload_model.visual.state_dict()

    common = set(base_sd.keys()) & set(reload_visual_sd.keys())
    n_diff = sum(
        1 for k in common
        if torch.max(torch.abs(base_sd[k].float() - reload_visual_sd[k].float())).item() > 1e-6
    )
    print(f"  Visual keys differing from base: {n_diff}/{len(common)}")
    if n_diff == 0:
        print(f"  WARNING: Merged model is identical to base — LoRA merge had no effect!")

    print(f"\n{'='*60}")
    print(f"  Done! Merged model: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
