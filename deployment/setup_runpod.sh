#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# TALON — GLM-4.6V-Flash API setup for RunPod
#
# Paste this entire script into a RunPod terminal.
# It installs dependencies, writes glm_api.py, and starts the server.
#
# After startup, your endpoint is:
#   https://<pod-id>-8000.proxy.runpod.net
#
# Test:
#   curl https://<pod-id>-8000.proxy.runpod.net/health
# ─────────────────────────────────────────────────────────────────────
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  TALON — Installing dependencies"
echo "═══════════════════════════════════════════════════════════════"

pip install fastapi uvicorn transformers torch accelerate Pillow

echo "═══════════════════════════════════════════════════════════════"
echo "  Writing /workspace/glm_api.py"
echo "═══════════════════════════════════════════════════════════════"

cat > /workspace/glm_api.py << 'GLMAPI_EOF'
"""
TALON GLM-4.6V-Flash API — FastAPI service for Stage 2 confirmation.

Loads GLM-4.6V-Flash once at startup, exposes /predict for base64 images.
"""

import base64
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ── Model loading ──────────────────────────────────────────────────

MODEL_ID = "zai-org/GLM-4.6V-Flash"

ROUND2_PROMPT = """这张图片是否属于 handjob 场景（女性用手给男性手淫）？
注意排除乳交（胸部夹住生殖器）和口交。
只回答 YES 或 NO"""


def load_glm():
    """Load GLM-4.6V-Flash with fp16 + auto device map."""
    from transformers import AutoProcessor, Glm4vForConditionalGeneration

    print(f"Loading {MODEL_ID} ...")
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {mem:.1f} / {total:.1f} GB used")

    print("Model loaded.")
    return model, processor


def predict_image(model, processor, image_path: str) -> tuple[str, str]:
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
    raw = processor.decode(generated, skip_special_tokens=True).strip()

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


# ── FastAPI app ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    model, processor = load_glm()
    app.state.model = model
    app.state.processor = processor
    yield

app = FastAPI(title="TALON GLM API", lifespan=lifespan)


class PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    result: str
    raw: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # Save to temp file (GLM processor expects a file path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name

    try:
        result, raw = predict_image(app.state.model, app.state.processor, tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return PredictResponse(result=result, raw=raw)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
GLMAPI_EOF

echo "═══════════════════════════════════════════════════════════════"
echo "  Starting GLM API server on port 8000"
echo "═══════════════════════════════════════════════════════════════"

python /workspace/glm_api.py
