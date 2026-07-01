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

Normally you don't run `provision.py` at all — the dreamdiffuse service on jimi
owns the pod lifecycle (spins up on a viewer, stops on idle). `provision.py` is
just the manual/admin fallback for the build host.

## Gotcha: Cloudflare proxy + User-Agent

The pod's public HTTP endpoint (`https://<podid>-8000.proxy.runpod.net`) is
fronted by Cloudflare. Its WAF **403s large POST bodies sent with the default
`Python-urllib` User-Agent** (small requests pass, but a ~160KB base64 image POST
gets blocked). Send a browser-like `User-Agent` header and it goes through — the
dreamdiffuse client does. (A direct `<port>/tcp` mapping would bypass the proxy
entirely — lower latency, no WAF — at the cost of a dynamic IP:port.)

## Cost

Dedicated RTX 4090: ~$0.34–0.44/hr community, ~$0.69/hr secure (whichever has
stock). Billed continuously while RUNNING — a STOPPED pod releases the GPU (≈$0,
only pennies of disk) and resumes in ~15-40s. dreamdiffuse auto-stops it when no
one is viewing; terminate to free even the disk.
