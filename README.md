# worker-diffuse

Fast SD-1.5 + LCM-LoRA **img2img** HTTP server for a dedicated RunPod GPU pod.
Paired with the [`dreamdiffuse`](../../../.local/dreamdiffuse) service on jimi,
which streams the lounge camera through it at ~5fps.

## Design

- `AutoPipelineForImage2Image` on `Lykon/dreamshaper-8` (SD-1.5) with
  `latent-consistency/lcm-lora-sdv1-5` fused in → 4–8 step img2img.
- Weights are **baked into the image** (`bake.py` at build time), so the pod
  needs **no network volume** and can run in any low-RTT US datacenter.
- fp16, `safety_checker=None`, warm dummy inference on startup.
- JSON in, raw `image/jpeg` out (no base64 on the return path). `X-Infer-Ms`
  header reports GPU time.

## Endpoints

- `GET /health` → `{ok, ready, model, load_secs, ...}`
- `POST /img2img` → JPEG bytes. Body:
  `{image_b64, prompt, negative, strength, steps, cfg, seed, size, quality}`

## Build & deploy

```bash
bash deploy/build_push.sh            # build linux/amd64 + push to GHCR (local Mac)
python3 deploy/provision.py          # create dedicated RTX 4090 pod (US DC)
python3 deploy/provision.py --stop   # terminate the pod (stop billing)
```

`provision.py` stores the pod id + proxy URL in the earl keychain
(`runpod_diffuse_pod_id`, `runpod_diffuse_url`) and prints the URL to point
dreamdiffuse at.

## Cost

Dedicated RTX 4090: ~$0.34–0.44/hr community, ~$0.69/hr secure. Billed
continuously while the pod runs — stop it when idle (`provision.py --stop`,
or dreamdiffuse's auto-idle pauses the stream so no frames are sent).
