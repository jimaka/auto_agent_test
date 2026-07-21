#!/bin/bash
# 启动 Ship3DOF 远程桌面开发环境
# 用法:
#   ./scripts/run_docker_remote.sh                    # 启动远程桌面
#   ./scripts/run_docker_remote.sh --detach           # 后台运行
#   VNC_PASSWORD=mypass ./scripts/run_docker_remote.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

IMAGE="${IMAGE:-ship3dof-remote:latest}"
DETACH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detach|-d)
            DETACH="-d"
            shift
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# 检查镜像是否存在
if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
    echo "镜像 ${IMAGE} 不存在，正在构建..."
    "${SCRIPT_DIR}/build_docker_ship3dof.sh" remote
fi

NOVNC_PORT="${NOVNC_PORT:-6080}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
SSH_PORT="${SSH_PORT:-2222}"
VNC_PASSWORD="${VNC_PASSWORD:-ship3dof}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-ship3dof}"

echo ">>> 启动远程开发环境: ${IMAGE}"
echo ""

docker run ${DETACH} --rm \
    --name ship3dof-remote \
    --gpus all \
    --shm-size=8g \
    -p "${NOVNC_PORT}:6080" \
    -p "${JUPYTER_PORT}:8888" \
    -p "${TENSORBOARD_PORT}:6006" \
    -p "${SSH_PORT}:22" \
    -e VNC_PASSWORD="${VNC_PASSWORD}" \
    -e JUPYTER_TOKEN="${JUPYTER_TOKEN}" \
    -e SSH_PASSWORD="${SSH_PASSWORD:-ship3dof}" \
    -e VNC_RESOLUTION="${VNC_RESOLUTION:-1920x1080}" \
    -v "${PROJECT_ROOT}:/workspace/rl-zoo" \
    "${IMAGE}" \
    "${EXTRA_ARGS[@]}"

if [[ -n "${DETACH}" ]]; then
    HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost')"
    echo ""
    echo "=============================================="
    echo "  远程开发环境已在后台启动"
    echo "=============================================="
    echo "  远程桌面:    http://${HOST_IP}:${NOVNC_PORT}/vnc.html"
    echo "  Jupyter Lab: http://${HOST_IP}:${JUPYTER_PORT}/?token=${JUPYTER_TOKEN}"
    echo "  TensorBoard: http://${HOST_IP}:${TENSORBOARD_PORT}"
    echo "  SSH:         ssh developer@${HOST_IP} -p ${SSH_PORT}"
    echo "  默认密码:    ${VNC_PASSWORD}"
    echo "=============================================="
fi
