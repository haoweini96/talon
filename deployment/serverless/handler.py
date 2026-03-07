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

ROUND2_PROMPT = """这张图片是否属于 handjob 场景（女性用手给男性手淫）？
注意排除乳交（胸部夹住生殖器）和口交。
只回答 YES 或 NO"""

print(f"Loading {MODEL_ID} ...")
MODEL = Glm4vForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
)
PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_mem / 1024**3
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
        output_ids = MODEL.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    raw = PROCESSOR.decode(generated, skip_special_tokens=True).strip()

    # Strip <think>...</think> tags and parse YES/NO
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
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
