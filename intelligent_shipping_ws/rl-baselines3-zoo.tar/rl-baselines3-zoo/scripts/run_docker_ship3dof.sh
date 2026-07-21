#!/bin/bash
# 在 Ship3DOF 基础训练镜像中执行命令
# 用法:
#   ./scripts/run_docker_ship3dof.sh python3 scripts/ship_3dof/run_known_mmg_pipeline.py --help
#   ./scripts/run_docker_ship3dof.sh bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

IMAGE="${IMAGE:-ship3dof:latest}"
CMD="${*:-bash}"

if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
    echo "镜像 ${IMAGE} 不存在，正在构建..."
    "${SCRIPT_DIR}/build_docker_ship3dof.sh" base
fi

echo ">>> 在 ${IMAGE} 中执行: ${CMD}"

docker run --rm -it \
    --gpus all \
    --shm-size=8g \
    --ipc=host \
    -v "${PROJECT_ROOT}:/workspace/rl-zoo" \
    -w /workspace/rl-zoo \
    -e PYTHONPATH=/workspace/rl-zoo \
    "${IMAGE}" \
    bash -c "${CMD}"
