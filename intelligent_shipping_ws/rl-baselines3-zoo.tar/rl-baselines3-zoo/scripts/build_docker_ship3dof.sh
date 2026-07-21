#!/bin/bash
# 构建 Ship3DOF Docker 镜像
# 用法:
#   ./scripts/build_docker_ship3dof.sh              # 构建基础训练镜像 (ship3dof:latest)
#   ./scripts/build_docker_ship3dof.sh remote       # 构建远程桌面镜像 (ship3dof-remote:latest)
#   ./scripts/build_docker_ship3dof.sh all          # 构建全部镜像
#   ./scripts/build_docker_ship3dof.sh isaac-ssh   # 基于本地 isaac 镜像构建 SSH 层
#   USE_GPU=False ./scripts/build_docker_ship3dof.sh remote  # CPU 版远程镜像

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET="${1:-base}"
USE_GPU="${USE_GPU:-True}"
ISAAC_BASE_IMAGE="${ISAAC_BASE_IMAGE:-isaac-lab-base-koopman:v3.0}"
ISAAC_SSH_IMAGE="${ISAAC_SSH_IMAGE:-isaac-lab-ssh:v3.0}"

if [[ "${USE_GPU}" == "True" ]]; then
    BASE_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
else
    BASE_IMAGE="python:3.11-slim"
fi

build_base() {
    echo ">>> 构建基础训练镜像 ship3dof:latest (BASE_IMAGE=${BASE_IMAGE})"
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        -t ship3dof:latest \
        -f docker/Dockerfile.ship3dof \
        .
    echo ">>> ship3dof:latest 构建完成"
}

build_remote() {
    echo ">>> 构建远程桌面镜像 ship3dof-remote:latest (BASE_IMAGE=${BASE_IMAGE})"
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        -t ship3dof-remote:latest \
        -f docker/Dockerfile.remote \
        .
    echo ">>> ship3dof-remote:latest 构建完成"
}

build_isaac_ssh() {
    if ! docker image inspect "${ISAAC_BASE_IMAGE}" > /dev/null 2>&1; then
        echo ">>> 基础镜像 ${ISAAC_BASE_IMAGE} 不存在，尝试导入..."
        "${SCRIPT_DIR}/load_isaac_image.sh"
    fi
    echo ">>> 构建 Isaac SSH 镜像 ${ISAAC_SSH_IMAGE} (BASE_IMAGE=${ISAAC_BASE_IMAGE})"
    docker build \
        --build-arg BASE_IMAGE="${ISAAC_BASE_IMAGE}" \
        -t "${ISAAC_SSH_IMAGE}" \
        -f docker/Dockerfile.isaac-ssh \
        .
    echo ">>> ${ISAAC_SSH_IMAGE} 构建完成"
}

build_zoo() {
    echo ">>> 构建 RL Zoo 官方镜像"
    if [[ "${USE_GPU}" == "True" ]]; then
        USE_GPU=True "${SCRIPT_DIR}/build_docker.sh"
    else
        "${SCRIPT_DIR}/build_docker.sh"
    fi
}

case "${TARGET}" in
    base)
        build_base
        ;;
    remote)
        build_remote
        ;;
    isaac-ssh)
        build_isaac_ssh
        ;;
    zoo)
        build_zoo
        ;;
    all)
        build_base
        build_remote
        ;;
    *)
        echo "用法: $0 [base|remote|isaac-ssh|zoo|all]"
        exit 1
        ;;
esac

echo ""
echo "构建完成。运行方式:"
echo "  SSH 容器:  ./scripts/create_ssh_container.sh create"
echo "  远程桌面:  ./scripts/run_docker_remote.sh"
echo "  训练任务:  ./scripts/run_docker_ship3dof.sh <command>"
echo "  Compose:   docker compose up -d ship3dof-remote"
