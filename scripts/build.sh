#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="ghcr.io/aminatorex/fifth-endpoint"
VERSION="v1.0-experimental"

docker build -t "${IMAGE_NAME}:${VERSION}" .
docker tag "${IMAGE_NAME}:${VERSION}" "${IMAGE_NAME}:latest"
echo "Build complete: ${IMAGE_NAME}:${VERSION}"
