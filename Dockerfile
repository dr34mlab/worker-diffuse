# worker-diffuse — SD-1.5 + LCM-LoRA img2img server for a dedicated RunPod pod.
# Weights are baked in so the pod needs no network volume and runs in any DC.
# Build on a real Docker host (local Mac / mother), NOT jimi:
#   docker buildx build --platform linux/amd64 --push -t ghcr.io/dr34mlab/worker-diffuse:latest .
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    DIFFUSE_MODEL=Lykon/dreamshaper-8 \
    DIFFUSE_LCM_LORA=latent-consistency/lcm-lora-sdv1-5

WORKDIR /app

RUN pip install --no-cache-dir \
    "diffusers==0.31.0" \
    "transformers==4.44.2" \
    "accelerate==0.34.2" \
    "peft==0.13.2" \
    "safetensors" \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.30.6" \
    "pillow" \
    "huggingface_hub[hf_transfer]"

COPY server.py /app/server.py
COPY bake.py /app/bake.py

# Bake the checkpoint + LCM-LoRA into the image at build time.
RUN HF_XET_HIGH_PERFORMANCE=1 python /app/bake.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
