# TALON — RunPod Serverless Deployment

RunPod Serverless endpoint for GLM-4.6V-Flash inference. Auto-scales to zero when idle, pay-per-use.

## Deployment

1. Push this repo to GitHub
2. In RunPod console: **Serverless** → **New Endpoint**
3. Select **GitHub Repo** as source, connect your repo
4. Set **Dockerfile Path** to `deployment/serverless/Dockerfile`
5. GPU: **A40 (48GB)** recommended for GLM-4.6V-Flash 10B fp16
6. Set **Max Workers** and **Idle Timeout** as needed
7. Deploy — RunPod auto-builds the Docker image

No environment variables needed (model weights are baked into the image).

## Request Format

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_base64": "'$(base64 -i test.jpg)'"
    }
  }'
```

### Response

```json
{
  "id": "...",
  "status": "COMPLETED",
  "output": {
    "result": "YES",
    "raw": "YES"
  }
}
```

## Health Check

```bash
curl "https://api.runpod.ai/v2/{ENDPOINT_ID}/health" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

## Integration with batch_pipeline.py

```bash
python scripts/batch_pipeline.py \
  --video_list data/saved_urls.md \
  --runpod_endpoint YOUR_ENDPOINT_ID \
  --runpod_key YOUR_API_KEY
```

This uses the RunPod Serverless API instead of the GPU Pod FastAPI endpoint.
Priority: `--runpod_endpoint` > `--gpu_endpoint` > `--skip_glm`.
