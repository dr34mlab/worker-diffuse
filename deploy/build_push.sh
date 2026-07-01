#!/usr/bin/env bash
# Build and push worker-diffuse to GHCR.
# Run on a machine with Docker + GHCR auth (local Mac / mother), NOT jimi.
set -euo pipefail

IMAGE="ghcr.io/dr34mlab/worker-diffuse:latest"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building $IMAGE (linux/amd64) ..."
docker buildx build \
    --platform linux/amd64 \
    --push \
    -t "$IMAGE" \
    "$ROOT"

echo "Pushed $IMAGE"
