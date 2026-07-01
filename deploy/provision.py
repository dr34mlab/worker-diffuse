#!/usr/bin/env python3
"""Manual RunPod pod control for worker-diffuse (rest.runpod.io/v1).

Normally the dreamdiffuse service on jimi owns the pod lifecycle (spins up on a
viewer, stops on idle — see gpu.py). This script is the manual/admin equivalent
for the build host: create, stop, status, terminate.

Usage:
  python3 deploy/provision.py --create      # create a dedicated pod (US, 4090)
  python3 deploy/provision.py --status
  python3 deploy/provision.py --stop        # stop (release GPU, keep disk)
  python3 deploy/provision.py --terminate   # delete the pod entirely
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"
IMAGE = "ghcr.io/dr34mlab/worker-diffuse:latest"
POD_NAME = "dreamdiffuse-worker"
HTTP_PORT = 8000
REGISTRY_AUTH_ID = "cmr1ip6kl00br119lj838zsca"  # ghcr-dr34mlab
GPU_TYPES = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4080", "NVIDIA RTX A5000"]
US_DCS = ["US-KS-2", "US-CA-2", "US-TX-3", "US-IL-1", "US-GA-1", "US-NC-1", "US-WA-1"]


def earl(account: str) -> str:
    return subprocess.check_output(
        ["security", "find-generic-password", "-s", "earl", "-a", account, "-w"],
        timeout=5).decode().strip()


def req(path: str, key: str, body: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"{REST}/{path}", data=data, method=method,
                               headers={"Content-Type": "application/json",
                                        "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(r, timeout=45) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {"_ok": True}
    except urllib.error.HTTPError as e:
        try:
            return {"_error": json.loads(e.read()).get("error", str(e.code)), "_code": e.code}
        except Exception:
            return {"_error": str(e.code)}


def proxy_url(pod_id: str) -> str:
    return f"https://{pod_id}-{HTTP_PORT}.proxy.runpod.net"


def find_pod(key: str) -> dict | None:
    pods = req("pods", key)
    items = pods if isinstance(pods, list) else pods.get("pods", pods.get("data", []))
    for p in (items or []):
        if p.get("name") == POD_NAME:
            return p
    return None


def create(key: str) -> None:
    for cloud in ("COMMUNITY", "SECURE"):
        print(f"  creating {POD_NAME} ({cloud}, US) ...")
        r = req("pods", key, {
            "name": POD_NAME, "imageName": IMAGE, "gpuTypeIds": GPU_TYPES, "gpuCount": 1,
            "cloudType": cloud, "computeType": "GPU", "containerDiskInGb": 25, "volumeInGb": 0,
            "ports": [f"{HTTP_PORT}/http", "22/tcp"], "dataCenterIds": US_DCS,
            "containerRegistryAuthId": REGISTRY_AUTH_ID, "env": {"PYTHONUNBUFFERED": "1"},
        }, "POST")
        pid = r.get("id")
        if pid:
            mach = r.get("machine") or {}
            print(f"  pod {pid} on {mach.get('gpuTypeId')} @ {mach.get('dataCenterId')} "
                  f"= ${r.get('costPerHr')}/hr")
            print(f"  url: {proxy_url(pid)}")
            wait_health(proxy_url(pid))
            return
        print(f"    {cloud}: {r.get('_error')}")
    sys.exit("  create failed on both clouds")


def wait_health(url: str, minutes: int = 12) -> None:
    print(f"  polling {url}/health (image pull + warm, up to {minutes} min) ...")
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=8) as r:
                h = json.loads(r.read())
                if h.get("ready"):
                    print(f"  READY: {json.dumps(h)}")
                    return
        except Exception:
            pass
        time.sleep(10)
    print("  health never went ready in time.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    args = ap.parse_args()
    key = earl("RUNPOD_API_KEY_CLI")

    if args.create:
        create(key)
        return

    pod = find_pod(key)
    if not pod:
        print("  no dreamdiffuse-worker pod found.")
        return
    pid = pod["id"]
    if args.stop:
        print(req(f"pods/{pid}/stop", key, method="POST"))
    elif args.terminate:
        print(req(f"pods/{pid}", key, method="DELETE"))
        print(f"  terminated {pid}")
    else:  # status
        print(f"  pod {pid}: {pod.get('desiredStatus')}  "
              f"{(pod.get('machine') or {}).get('gpuTypeId')} @ "
              f"{(pod.get('machine') or {}).get('dataCenterId')}  ${pod.get('costPerHr')}/hr")
        print(f"  url: {proxy_url(pid)}")


if __name__ == "__main__":
    main()
