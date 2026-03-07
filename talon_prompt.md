请先读 CLAUDE.md，然后创建 RunPod Serverless 部署文件。
目标：代码推到 GitHub 后，RunPod 直接从 GitHub 自动构建，本地不需要 Docker。

## 创建 deployment/serverless/ 目录，包含以下文件：

### Dockerfile
- 基于 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- pip install runpod transformers accelerate Pillow huggingface_hub
- 用 huggingface-cli download 把 GLM-4.6V-Flash bake 进镜像
  （model ID 从 CLAUDE.md 或现有代码里确认）
- COPY handler.py /
- CMD ["python", "-u", "/handler.py"]

### handler.py
- 模块级加载 GLM 模型（只在容器启动时跑一次）
- handler(job): 接收 base64 图片 → GLM 推理 → 返回 YES/NO
- 复用项目已有的 ROUND2_PROMPT
- runpod.serverless.start({"handler": handler})

### .dockerignore
忽略 data/, outputs/, *.pt, *.pth, __pycache__, .git 等大文件

### README.md
说明部署方式是 GitHub → RunPod 自动构建，不需要本地 Docker，
包含：推送步骤、RunPod 控制台操作、GPU 选型（A40 48GB）、参数配置

### 同时更新 scripts/batch_pipeline.py
添加 --runpod_endpoint 和 --runpod_key 参数，
调用 POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync