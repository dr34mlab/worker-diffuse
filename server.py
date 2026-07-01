#!/usr/bin/env python3
"""worker-diffuse — fast SD-1.5 + LCM-LoRA img2img HTTP server.

Runs on a dedicated RunPod GPU pod. Keeps the pipeline warm in VRAM and
answers /img2img with a styled JPEG. Built for continuous ~5fps streaming
from the dreamdiffuse service on jimi, so per-request overhead is kept low:
JSON in, raw image/jpeg bytes out (no base64 on the hot return path).

Model weights are baked into the Docker image (see Dockerfile) so the pod
needs no network volume and can run in any (low-RTT US) datacenter.
"""
from __future__ import annotations

import base64
import io
import os
import time
import threading

import torch
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

MODEL_ID = os.environ.get("DIFFUSE_MODEL", "Lykon/dreamshaper-8")
LCM_LORA = os.environ.get("DIFFUSE_LCM_LORA", "latent-consistency/lcm-lora-sdv1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="worker-diffuse")

# The pipeline is not thread-safe; serialize inference. The dreamdiffuse loop
# is single-flight anyway, but a lock keeps us safe under concurrent probes.
_lock = threading.Lock()
_pipe = None
_ready = False
_meta: dict = {}


def _load():
    global _pipe, _ready, _meta
    from diffusers import AutoPipelineForImage2Image, LCMScheduler

    t0 = time.time()
    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA)
    pipe.fuse_lora()
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    _pipe = pipe

    # Warm the kernels with a throwaway inference so the first real frame is fast.
    warm = Image.new("RGB", (512, 512), (32, 32, 40))
    with _lock:
        _pipe(
            prompt="warmup",
            image=warm,
            num_inference_steps=6,
            strength=0.5,
            guidance_scale=1.5,
        )
    _meta = {
        "model": MODEL_ID,
        "lcm_lora": LCM_LORA,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "load_secs": round(time.time() - t0, 1),
    }
    _ready = True
    print(f"[worker-diffuse] ready in {_meta['load_secs']}s on {DEVICE}", flush=True)


class Img2ImgReq(BaseModel):
    image_b64: str
    prompt: str = "a photograph"
    negative: str = ""
    strength: float = 0.5
    steps: int = 6
    cfg: float = 1.5
    seed: int = -1
    size: int = 512
    quality: int = 80


@app.get("/health")
def health():
    return {"ok": _ready, "ready": _ready, **_meta}


@app.post("/img2img")
def img2img(req: Img2ImgReq):
    if not _ready:
        return JSONResponse({"error": "warming up"}, status_code=503)

    raw = base64.b64decode(req.image_b64)
    src = Image.open(io.BytesIO(raw)).convert("RGB")

    # Square-ish center crop to the target size keeps SD-1.5 happy and fast.
    size = max(256, min(768, int(req.size)))
    src = _fit(src, size)

    gen = None
    if req.seed is not None and req.seed >= 0:
        gen = torch.Generator(device=DEVICE).manual_seed(int(req.seed))

    steps = max(2, min(12, int(req.steps)))
    strength = max(0.1, min(1.0, float(req.strength)))
    cfg = max(0.0, min(4.0, float(req.cfg)))

    t0 = time.time()
    with _lock:
        out = _pipe(
            prompt=req.prompt,
            negative_prompt=req.negative or None,
            image=src,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=cfg,
            generator=gen,
        ).images[0]
    infer_ms = int((time.time() - t0) * 1000)

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=max(40, min(95, int(req.quality))))
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"X-Infer-Ms": str(infer_ms)},
    )


def _fit(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to square then resize to size x size."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load, daemon=True).start()
