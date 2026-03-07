#!/usr/bin/env python3
"""
VLM handjob scene classification test script.
Designed to run on Colab A100 with fp16.

Supports (fallback order):
  - Qwen3-VL-8B    (default, requires transformers >= 4.57.0)
  - GLM-4.6V-Flash (--model glm, requires transformers >= 5.0.0)
  - Qwen2.5-VL-7B  (--model qwen2.5, requires transformers >= 4.45.0)

Usage:
    python test_qwen3vl.py                              # Qwen3-VL, 5 per cat
    python test_qwen3vl.py --model glm                  # GLM-4.6V-Flash
    python test_qwen3vl.py --model qwen2.5              # fallback to Qwen2.5-VL
    python test_qwen3vl.py --image path/to/img.jpg      # single image
    python test_qwen3vl.py --folder path/to/dir          # all images in folder
    python test_qwen3vl.py --samples 10                  # 10 per category

Install (Colab):
    pip install torch torchvision Pillow
    # Qwen3-VL (transformers >= 4.57.0):
    pip install git+https://github.com/huggingface/transformers
    # GLM-4.6V-Flash (transformers >= 5.0.0):
    pip install git+https://github.com/huggingface/transformers
    # Qwen2.5-VL (transformers >= 4.45.0):
    pip install transformers>=4.45.0
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODELS = {
    "qwen3": {
        "id": "Qwen/Qwen3-VL-8B-Instruct",
        "min_transformers": "4.57.0",
    },
    "glm": {
        "id": "zai-org/GLM-4.6V-Flash",
        "min_transformers": "5.0.0",
    },
    "qwen2.5": {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "min_transformers": "4.45.0",
    },
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "labels"

# Ignored top-level dirs under data/labels/ (legacy/temp folders)
IGNORE_TOPLEVEL = {"positive", "not_handjob", "easy negative", "uncertain"}

ROUND1_PROMPT = """看这张图片，分别判断以下三个要素是否存在：

1. 能看到女性的脸（face）
2. 女性胸部裸露可见（chest）
3. 女性的手在握住男性生殖器（action）

只用 JSON 格式回答：{"face": 0或1, "chest": 0或1, "action": 0或1}"""

ROUND2_PROMPT = """这张图片是否属于 handjob 场景（女性用手给男性手淫）？
注意排除乳交（胸部夹住生殖器）和口交。
只回答 YES 或 NO"""


# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------
def check_transformers_version(model_key: str):
    import transformers
    from packaging.version import Version

    current = Version(transformers.__version__)
    required = MODELS[model_key]["min_transformers"]

    # Dev versions like "4.58.0.dev0" satisfy >= 4.57.0
    if current < Version(required):
        print(f"ERROR: {MODELS[model_key]['id']} requires transformers >= {required}")
        print(f"  Installed: {transformers.__version__}")
        print("  Install from git: pip install git+https://github.com/huggingface/transformers")
        if model_key == "qwen3":
            print("  Or use --model glm or --model qwen2.5 as fallback")
        elif model_key == "glm":
            print("  Or use --model qwen3 or --model qwen2.5 as fallback")
        sys.exit(1)

    print(f"  transformers {transformers.__version__} (>= {required} ✓)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_key: str):
    model_id = MODELS[model_key]["id"]
    print("=" * 70)
    print(f"Loading {model_id} ...")
    check_transformers_version(model_key)
    print("=" * 70)

    from transformers import AutoProcessor

    if model_key == "qwen3":
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
        )
    elif model_key == "glm":
        from transformers import Glm4vForConditionalGeneration
        model = Glm4vForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
        )
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(model_id)

    print_gpu_usage()
    return model, processor


def print_gpu_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_mem / 1024**3
            print(f"  GPU {i}: {mem:.1f} / {total:.1f} GB used")
    else:
        print("  (No CUDA device)")


# ---------------------------------------------------------------------------
# Inference — two code paths
# ---------------------------------------------------------------------------
def ask_qwen3(model, processor, image_path: str, prompt: str) -> str:
    """Qwen3-VL: single-step apply_chat_template with PIL Image."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    trimmed = [
        out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)
    ]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text[0].strip()


def ask_glm(model, processor, image_path: str, prompt: str) -> str:
    """GLM-4.6V-Flash: apply_chat_template with url key for images."""
    abs_path = str(Path(image_path).resolve())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": abs_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def ask_qwen25(model, processor, image_path: str, prompt: str) -> str:
    """Qwen2.5-VL: two-step processor(text=..., images=...)."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_round1(response: str):
    match = re.search(r'\{[^}]+\}', response)
    if match:
        try:
            data = json.loads(match.group())
            return [int(data.get("face", 0)), int(data.get("chest", 0)), int(data.get("action", 0))]
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def parse_round2(response: str):
    text = response.upper().strip()
    if "YES" in text:
        return "YES"
    if "NO" in text:
        return "NO"
    return text


# ---------------------------------------------------------------------------
# Sampling — dynamic directory scanning
# ---------------------------------------------------------------------------
def _list_images(folder: Path) -> list[Path]:
    return [f for f in folder.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]


def _extract_tag(dirname: str) -> str | None:
    """Extract [x,x,x] tag from folder name like '[1,0,1] 有脸动作没有胸'."""
    m = re.match(r'\[([01],[01],[01])\]', dirname)
    return f"[{m.group(1)}]" if m else None


def scan_labels_dir(labels_dir: Path) -> list[tuple[Path, str, bool]]:
    """Scan data/labels/ and return [(folder, tag, is_positive), ...]."""
    categories = []

    # handjob/ → positive [1,1,1]
    hj = labels_dir / "handjob"
    if hj.is_dir():
        categories.append((hj, "[1,1,1]", True))

    # negative/ subfolders → extract [x,x,x] from name
    neg = labels_dir / "negative"
    if neg.is_dir():
        for sub in sorted(neg.iterdir()):
            if not sub.is_dir():
                continue
            tag = _extract_tag(sub.name)
            if tag:
                categories.append((sub, tag, False))

    return categories


def sample_images(data_dir: Path, samples_per_cat: int) -> list[tuple[str, str, bool]]:
    """Return list of (image_path, tag, is_positive)."""
    categories = scan_labels_dir(data_dir)
    if not categories:
        print("  ERROR: no label categories found")
        return []

    result = []
    for folder, tag, is_pos in categories:
        images = _list_images(folder)
        if not images:
            print(f"  WARNING: no images in {folder.name}")
            continue
        chosen = random.sample(images, min(samples_per_cat, len(images)))
        for img in chosen:
            result.append((str(img), tag, is_pos))
        label = "handjob" if is_pos else f"negative/{tag}"
        print(f"  {label}: sampled {len(chosen)} / {len(images)}")
    return result


def collect_from_path(path: str) -> list[tuple[str, str, bool]]:
    """Collect from --image or --folder. If folder is a labels dir, scan it properly."""
    p = Path(path).resolve()

    if p.is_file():
        return [(str(p), "unknown", False)]

    if not p.is_dir():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    # If the folder looks like a labels dir (has handjob/ or negative/), scan it
    if (p / "handjob").is_dir() or (p / "negative").is_dir():
        print(f"  Detected labels directory: {p}")
        return sample_images(p, 999999)  # all images

    # Otherwise treat as flat folder of images
    images = sorted(_list_images(p))
    return [(str(img), "unknown", False) for img in images]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test Qwen VL on handjob classification")
    parser.add_argument("--model", choices=["qwen3", "glm", "qwen2.5"], default="qwen3",
                        help="Model to use (default: qwen3)")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--folder", type=str, help="Folder of images")
    parser.add_argument("--samples", type=int, default=5, help="Samples per category (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Pick inference function
    ask_fn = {"qwen3": ask_qwen3, "glm": ask_glm, "qwen2.5": ask_qwen25}[args.model]

    # Collect test images
    if args.image:
        test_images = collect_from_path(args.image)
    elif args.folder:
        test_images = collect_from_path(args.folder)
    else:
        print(f"\nSampling {args.samples} images per category from {DATA_DIR}")
        test_images = sample_images(DATA_DIR, args.samples)

    if not test_images:
        print("No images to test.")
        sys.exit(1)

    print(f"\nTotal images to test: {len(test_images)}")

    # Load model
    model, processor = load_model(args.model)

    # Run inference
    print("\n" + "=" * 70)
    print(f"{'File':<40} | {'三要素 [f,c,a]':<16} | {'判断':<6} | 真实标签")
    print("-" * 70)

    results = []
    correct = 0
    total = 0

    for img_path, gt_tag, gt_positive in test_images:
        filename = Path(img_path).name

        r1_raw = ask_fn(model, processor, img_path, ROUND1_PROMPT)
        elements = parse_round1(r1_raw)
        elements_str = str(elements) if elements else f"?({r1_raw[:30]})"

        r2_raw = ask_fn(model, processor, img_path, ROUND2_PROMPT)
        judgment = parse_round2(r2_raw)

        is_positive_pred = judgment == "YES"
        gt_label = f"handjob {gt_tag}" if gt_positive else f"negative {gt_tag}"
        if gt_tag == "unknown":
            marker = " "
        else:
            match = gt_positive == is_positive_pred
            correct += int(match)
            total += 1
            marker = "✓" if match else "✗"

        print(f"{filename:<40} | {elements_str:<16} | {judgment:<6} | {gt_label} {marker}")

        results.append({
            "file": filename,
            "path": img_path,
            "elements": elements,
            "elements_raw": r1_raw,
            "judgment": judgment,
            "judgment_raw": r2_raw,
            "ground_truth_tag": gt_tag,
            "ground_truth_positive": gt_positive,
            "correct": match if gt_tag != "unknown" else None,
        })

    # Summary
    print("=" * 70)
    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")

        tp = sum(1 for r in results if r["ground_truth_positive"] and r["judgment"] == "YES")
        fn = sum(1 for r in results if r["ground_truth_positive"] and r["judgment"] == "NO")
        fp = sum(1 for r in results if not r["ground_truth_positive"] and r["ground_truth_tag"] != "unknown" and r["judgment"] == "YES")
        tn = sum(1 for r in results if not r["ground_truth_positive"] and r["ground_truth_tag"] != "unknown" and r["judgment"] == "NO")

        print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  Precision: {precision:.2%}  Recall: {recall:.2%}")

    print_gpu_usage()

    # Save results
    out_path = Path(__file__).resolve().parent.parent / "data" / f"{args.model}_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
