#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="ghcr.io/aminatorex/fifth-endpoint"
VERSION="v1.0-experimental"

docker push "${IMAGE_NAME}:${VERSION}"
docker push "${IMAGE_NAME}:latest"
echo "Push complete: ${IMAGE_NAME}:${VERSION}"
