#!/usr/bin/env python3
"""Pre-download model weights into the image cache at build time.

Downloads the SD-1.5 checkpoint and the LCM-LoRA so the running container
never touches HuggingFace at request time and the pod boots weight-hot.
"""
import os

from diffusers import AutoPipelineForImage2Image
from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("DIFFUSE_MODEL", "Lykon/dreamshaper-8")
LCM_LORA = os.environ.get("DIFFUSE_LCM_LORA", "latent-consistency/lcm-lora-sdv1-5")

print(f"baking {MODEL_ID} + {LCM_LORA}")
# Pull the full pipeline snapshot (fp16 variant when present) into cache.
AutoPipelineForImage2Image.from_pretrained(MODEL_ID)
snapshot_download(LCM_LORA)
print("bake complete")
