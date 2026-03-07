FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ARG CACHEBUST=1

# Install Python dependencies + bake model weights into image
RUN pip install --no-cache-dir \
    runpod transformers accelerate Pillow \
    huggingface_hub sentencepiece tiktoken einops torchvision && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-4.6V-Flash', local_dir='/models/GLM-4.6V-Flash', ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'])"

COPY deployment/serverless/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
