#!/usr/bin/env python3
"""Provision (or stop) a dedicated RunPod GPU pod running worker-diffuse.

A dedicated pod (not serverless) is used because continuous ~5fps img2img
streaming needs a warm GPU with no cold-start / queue delay. The image bakes
its own weights, so no network volume is needed and we can pick a low-RTT US
datacenter.

Usage:
  python3 deploy/provision.py                 # create pod, wait for /health, store in earl
  python3 deploy/provision.py --gpu A5000     # cheaper GPU
  python3 deploy/provision.py --stop          # terminate the pod (stop billing)
  python3 deploy/provision.py --status        # show pod + health
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

RUNPOD_GQL = "https://api.runpod.io/graphql"
RUNPOD_REST = "https://api.runpod.io/v1"
IMAGE = "ghcr.io/dr34mlab/worker-diffuse:latest"
POD_NAME = "dreamdiffuse-worker"
HTTP_PORT = 8000

# GPU display names accepted by pod create (pods still use display names).
GPUS = {
    "4090": "NVIDIA GeForce RTX 4090",
    "A5000": "NVIDIA RTX A5000",
    "4080": "NVIDIA GeForce RTX 4080",
    "A4000": "NVIDIA RTX A4000",
}
# US datacenters, tried in order for lowest RTT to jimi (US home).
US_DCS = ["US-KS-2", "US-CA-2", "US-TX-3", "US-IL-1", "US-GA-1", "US-NC-1", "US-WA-1"]


def earl(account: str) -> str:
    return subprocess.check_output(
        ["security", "find-generic-password", "-s", "earl", "-a", account, "-w"],
        timeout=5,
    ).decode().strip()


def store_earl(account: str, value: str) -> None:
    subprocess.run(["security", "delete-generic-password", "-s", "earl", "-a", account],
                   capture_output=True)
    subprocess.check_call(["security", "add-generic-password", "-s", "earl",
                           "-a", account, "-w", value])
    print(f"  stored earl key: {account} = {value}")


def _req(url: str, key: str, body: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode()}


def gql(query: str, variables: dict, key: str) -> dict:
    return _req(RUNPOD_GQL, key, {"query": query, "variables": variables}, "POST")


def proxy_url(pod_id: str) -> str:
    return f"https://{pod_id}-{HTTP_PORT}.proxy.runpod.net"


def create_pod(key: str, gpu: str) -> str:
    gpu_name = GPUS[gpu]
    for dc in US_DCS:
        for cloud in ("COMMUNITY", "SECURE"):
            print(f"  trying {gpu_name} in {dc} ({cloud}) ...")
            resp = _req(f"{RUNPOD_REST}/pods", key, {
                "name": POD_NAME,
                "imageName": IMAGE,
                "gpuTypeId": gpu_name,
                "gpuCount": 1,
                "cloudType": cloud,
                "dataCenterId": dc,
                "containerDiskInGb": 25,
                "volumeInGb": 0,
                "ports": f"{HTTP_PORT}/http,22/tcp",
                "env": [{"key": "PYTHONUNBUFFERED", "value": "1"}],
            }, "POST")
            pod_id = resp.get("id", "")
            if pod_id:
                print(f"  pod created: {pod_id} in {dc} ({cloud})")
                return pod_id
            err = resp.get("_body", resp)
            print(f"    unavailable: {err}")
    print("  FAILED: no US datacenter had the GPU available.", file=sys.stderr)
    sys.exit(1)


def wait_runtime(key: str, pod_id: str, minutes: int = 8) -> None:
    print(f"  waiting for pod runtime (up to {minutes} min) ...")
    for i in range(minutes * 6):
        time.sleep(10)
        info = gql(
            "query Pod($id: String!) { pod(input: {podId: $id}) { id runtime { uptimeInSeconds } } }",
            {"id": pod_id}, key)
        try:
            rt = info["data"]["pod"]["runtime"]
        except (KeyError, TypeError):
            rt = None
        if rt and rt.get("uptimeInSeconds", 0) > 0:
            print(f"  runtime up after ~{(i + 1) * 10}s")
            return
        if i % 6 == 5:
            print(f"  ... still booting ({(i + 1) * 10}s)")
    print("  WARNING: runtime not reported; continuing to health poll anyway.")


def wait_health(url: str, minutes: int = 10) -> bool:
    print(f"  polling {url}/health (model bake + warm, up to {minutes} min) ...")
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
                h = json.loads(r.read())
                if h.get("ready"):
                    print(f"  HEALTHY: {json.dumps(h)}")
                    return True
                print(f"  ... {h}")
        except Exception as e:
            print(f"  ... not up yet ({type(e).__name__})")
        time.sleep(10)
    return False


def find_pod(key: str) -> str:
    try:
        return earl("runpod_diffuse_pod_id")
    except subprocess.CalledProcessError:
        return ""


def stop(key: str) -> None:
    pod_id = find_pod(key)
    if not pod_id:
        print("  no stored pod id; nothing to stop.")
        return
    gql("mutation Terminate($input: PodTerminateInput!) { podTerminate(input: $input) }",
        {"input": {"podId": pod_id}}, key)
    print(f"  terminated pod {pod_id}. (billing stopped)")


def status(key: str) -> None:
    pod_id = find_pod(key)
    print(f"  pod_id: {pod_id or '(none)'}")
    if not pod_id:
        return
    url = proxy_url(pod_id)
    print(f"  url:    {url}")
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
            print(f"  health: {r.read().decode()}")
    except Exception as e:
        print(f"  health: unreachable ({type(e).__name__})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="4090", choices=list(GPUS))
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    key = earl("RUNPOD_API_KEY_CLI")

    if args.stop:
        stop(key)
        return
    if args.status:
        status(key)
        return

    pod_id = create_pod(key, args.gpu)
    url = proxy_url(pod_id)
    store_earl("runpod_diffuse_pod_id", pod_id)
    store_earl("runpod_diffuse_url", url)

    wait_runtime(key, pod_id)
    ok = wait_health(url)

    print()
    print("Done." if ok else "Pod created but health never went ready — check logs.")
    print(f"  pod_id: {pod_id}")
    print(f"  url:    {url}")
    print(f"  point dreamdiffuse at: {url}")


if __name__ == "__main__":
    main()
