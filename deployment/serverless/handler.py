"""
TALON — RunPod Serverless handler for GLM-4.6V-Flash.

Model loads once at container start. Each request receives a base64 image,
runs GLM inference, and returns YES/NO classification.
"""

import base64
import re
import tempfile
from pathlib import Path

import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration

import runpod


# ── Model loading (runs once when container starts) ──────────────

MODEL_ID = "zai-org/GLM-4.6V-Flash"
LOCAL_MODEL_DIR = "/models/GLM-4.6V-Flash"

ROUND2_PROMPT = """仔细观察这张图片，判断是否同时满足以下所有条件：
1. 女性的手接触或包裹男性生殖器（手可能遮挡大部分，仅露出部分）
2. 能完整看到女性的胸部（包括乳头）
3. 不是乳交场景 — 乳交特征：胸部紧贴或夹住生殖器两侧，双手可能在推压胸部，生殖器位于两侧胸部之间
4. 不是口交场景

如果胸部紧贴生殖器两侧，即使手也在触碰生殖器，也应判断为 NO（这是乳交不是手交）。
不要过度分析，根据视觉证据直接判断。
只回答 YES 或 NO，不要解释。"""

# Load from baked-in local dir, fall back to HF hub download
model_path = LOCAL_MODEL_DIR if Path(LOCAL_MODEL_DIR).exists() else MODEL_ID
print(f"Loading {model_path} ...")
MODEL = Glm4vForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto",
)
PROCESSOR = AutoProcessor.from_pretrained(model_path)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {mem:.1f} / {total:.1f} GB used")

print("Model loaded.")


# ── Inference ────────────────────────────────────────────────────

def predict_image(image_path: str) -> tuple[str, str]:
    """Run GLM on a single image. Returns (YES/NO, raw_text)."""
    abs_path = str(Path(image_path).resolve())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": abs_path},
                {"type": "text", "text": ROUND2_PROMPT},
            ],
        }
    ]

    inputs = PROCESSOR.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(MODEL.device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        output_ids = MODEL.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    raw = PROCESSOR.decode(generated, skip_special_tokens=True).strip()

    # Strip <think>...</think> tags (including truncated unclosed tags)
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = text.upper().strip()
    if "YES" in text:
        result = "YES"
    elif "NO" in text:
        result = "NO"
    else:
        result = text

    return result, raw


# ── RunPod handler ───────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler. Expects {"image_base64": "..."}."""
    job_input = job["input"]
    image_base64 = job_input.get("image_base64")

    if not image_base64:
        return {"error": "Missing 'image_base64' in input"}

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        return {"error": "Invalid base64 image data"}

    # Save to temp file (GLM processor expects a file path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name

    try:
        result, raw = predict_image(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"result": result, "raw": raw}


runpod.serverless.start({"handler": handler})
