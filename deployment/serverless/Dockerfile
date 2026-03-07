FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    accelerate \
    Pillow

# Bake model weights into image (no download at cold start)
RUN pip install --no-cache-dir "huggingface_hub[cli]" && \
    huggingface-cli download zai-org/GLM-4.6V-Flash \
        --local-dir /models/GLM-4.6V-Flash

COPY deployment/serverless/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
