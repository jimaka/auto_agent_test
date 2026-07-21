#!/usr/bin/env bash
# 创建/管理支持 SSH 远程访问的开发容器
#
# 默认基于本地 isaac-lab-base-koopman:v3.0 构建（无需外网拉取 pytorch 镜像）
#
# 用法:
#   ./scripts/create_ssh_container.sh create          # 创建并启动容器
#   ./scripts/create_ssh_container.sh stop            # 停止容器
#   ./scripts/create_ssh_container.sh restart         # 重启
#   ./scripts/create_ssh_container.sh status          # 查看状态与连接信息
#   ./scripts/create_ssh_container.sh ssh             # 打印 SSH 连接命令
#   ./scripts/create_ssh_container.sh logs            # 查看容器日志
#   ./scripts/create_ssh_container.sh destroy         # 删除容器
#   ./scripts/create_ssh_container.sh shell           # 直接 exec 进入容器
#
# 环境变量（可选）:
#   PROJECT_DIR=/home/lijiming/rl-baselines3-zoo
#   IMAGE=isaac-lab-ssh:v3.0          # 默认；有外网可用 ship3dof-remote:latest
#   ISAAC_TAR=/path/to/isaac.tar      # 基础镜像 tar 包路径
#   SSH_PORT=2222
#   SSH_PASSWORD=your_password
#   CONTAINER_NAME=ship3dof-ssh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IMAGE="${IMAGE:-isaac-lab-ssh:v3.0}"
ISAAC_BASE_IMAGE="${ISAAC_BASE_IMAGE:-isaac-lab-base-koopman:v3.0}"
CONTAINER_NAME="${CONTAINER_NAME:-ship3dof-ssh}"
CONTAINER_WORKDIR="/workspace/rl-zoo"

SSH_PORT="${SSH_PORT:-2222}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

# isaac 镜像默认 root；ship3dof-remote 默认 developer
if [[ "${IMAGE}" == ship3dof-remote* ]]; then
    SSH_USER="${SSH_USER:-developer}"
else
    SSH_USER="${SSH_USER:-root}"
fi

SSH_PASSWORD="${SSH_PASSWORD:-ship3dof}"
VNC_PASSWORD="${VNC_PASSWORD:-ship3dof}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-ship3dof}"

SHM_SIZE="${SHM_SIZE:-8g}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[ERROR] $*" >&2; exit 1; }

is_isaac_ssh_image() {
    [[ "${IMAGE}" == isaac-lab-ssh* ]]
}

is_remote_desktop_image() {
    [[ "${IMAGE}" == ship3dof-remote* ]]
}

get_host_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1"
}

print_access_info() {
    local ip
    ip="$(get_host_ip)"
    echo ""
    echo "=============================================="
    echo "  容器: ${CONTAINER_NAME}"
    echo "  镜像: ${IMAGE}"
    echo "=============================================="
    echo "  SSH 连接:"
    echo "    ssh ${SSH_USER}@${ip} -p ${SSH_PORT}"
    echo "    密码: ${SSH_PASSWORD}"
    echo ""
    echo "  进入后工作目录:"
    echo "    cd ${CONTAINER_WORKDIR}"
    echo ""
    echo "  其他访问方式:"
    if is_remote_desktop_image; then
        echo "    远程桌面:    http://${ip}:${NOVNC_PORT}/vnc.html"
    fi
    echo "    Jupyter:     http://${ip}:${JUPYTER_PORT}/?token=${JUPYTER_TOKEN}"
    echo "    TensorBoard: http://${ip}:${TENSORBOARD_PORT}"
    echo "=============================================="
}

check_prerequisites() {
    command -v docker >/dev/null 2>&1 || err "未安装 docker"

    [[ -d "${PROJECT_DIR}" ]] \
        || err "项目目录不存在: ${PROJECT_DIR}"

    [[ -f "${PROJECT_DIR}/scripts/ship_3dof/run_known_mmg_pipeline.py" ]] \
        || err "项目目录不正确，找不到 pipeline 脚本: ${PROJECT_DIR}"
}

ensure_isaac_base_image() {
    if docker image inspect "${ISAAC_BASE_IMAGE}" > /dev/null 2>&1; then
        log "基础镜像已存在: ${ISAAC_BASE_IMAGE}"
        return 0
    fi
    log "基础镜像 ${ISAAC_BASE_IMAGE} 不存在，尝试从 tar 导入..."
    ISAAC_TAR="${ISAAC_TAR:-}" "${SCRIPT_DIR}/load_isaac_image.sh"
}

build_image_if_missing() {
    if docker image inspect "${IMAGE}" > /dev/null 2>&1; then
        log "镜像已存在: ${IMAGE}"
        return 0
    fi

    log "镜像 ${IMAGE} 不存在，开始构建..."

    if is_isaac_ssh_image; then
        ensure_isaac_base_image
        docker build \
            --build-arg BASE_IMAGE="${ISAAC_BASE_IMAGE}" \
            -t "${IMAGE}" \
            -f "${PROJECT_DIR}/docker/Dockerfile.isaac-ssh" \
            "${PROJECT_DIR}"
    elif is_remote_desktop_image; then
        local build_script="${PROJECT_DIR}/scripts/build_docker_ship3dof.sh"
        if [[ -x "${build_script}" ]]; then
            "${build_script}" remote
        else
            docker build -t "${IMAGE}" -f "${PROJECT_DIR}/docker/Dockerfile.remote" "${PROJECT_DIR}"
        fi
    else
        err "未知镜像 ${IMAGE}。请使用 isaac-lab-ssh:v3.0 或 ship3dof-remote:latest"
    fi

    log "镜像构建完成: ${IMAGE}"
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

container_running() {
    docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

cmd_create() {
    check_prerequisites
    build_image_if_missing

    if container_running; then
        log "容器已在运行: ${CONTAINER_NAME}"
        print_access_info
        return 0
    fi

    if container_exists; then
        log "启动已有容器: ${CONTAINER_NAME}"
        docker start "${CONTAINER_NAME}"
        print_access_info
        return 0
    fi

    log "创建新容器: ${CONTAINER_NAME}"

    local -a PORT_ARGS=(
        -p "${SSH_PORT}:22"
        -p "${JUPYTER_PORT}:8888"
        -p "${TENSORBOARD_PORT}:6006"
    )
    if is_remote_desktop_image; then
        PORT_ARGS+=(-p "${NOVNC_PORT}:6080")
    fi

    local -a ENV_ARGS=(
        -e ACCEPT_EULA=Y
        -e SSH_PASSWORD="${SSH_PASSWORD}"
        -e JUPYTER_TOKEN="${JUPYTER_TOKEN}"
        -e ENABLE_SSH=true
        -e PYTHONPATH="${CONTAINER_WORKDIR}"
        -e WORKSPACE="${CONTAINER_WORKDIR}"
    )
    if is_remote_desktop_image; then
        ENV_ARGS+=(-e VNC_PASSWORD="${VNC_PASSWORD}")
    fi

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --hostname "${CONTAINER_NAME}" \
        --gpus all \
        --shm-size="${SHM_SIZE}" \
        --restart unless-stopped \
        "${PORT_ARGS[@]}" \
        "${ENV_ARGS[@]}" \
        -v "${PROJECT_DIR}:${CONTAINER_WORKDIR}" \
        "${IMAGE}"

    log "等待 SSH 服务启动..."
    sleep 8

    if container_running; then
        log "容器启动成功"
        print_access_info
    else
        err "容器启动失败，查看日志: docker logs ${CONTAINER_NAME}"
    fi
}

cmd_stop() {
    if container_running; then
        docker stop "${CONTAINER_NAME}"
        log "容器已停止: ${CONTAINER_NAME}"
    else
        log "容器未在运行: ${CONTAINER_NAME}"
    fi
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_create
}

cmd_status() {
    if ! container_exists; then
        echo "容器不存在: ${CONTAINER_NAME}"
        echo "运行: $0 create"
        exit 0
    fi

    docker ps -a --filter "name=${CONTAINER_NAME}" \
        --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    if container_running; then
        print_access_info
    fi
}

cmd_ssh() {
    local ip
    ip="$(get_host_ip)"
    echo "ssh ${SSH_USER}@${ip} -p ${SSH_PORT}"
    echo "# 密码: ${SSH_PASSWORD}"
}

cmd_logs() {
    container_exists || err "容器不存在"
    docker logs -f "${CONTAINER_NAME}"
}

cmd_destroy() {
    if container_exists; then
        docker rm -f "${CONTAINER_NAME}"
        log "容器已删除: ${CONTAINER_NAME}"
    else
        log "容器不存在"
    fi
}

cmd_shell() {
    container_running || err "容器未运行，先执行: $0 create"
    docker exec -it "${CONTAINER_NAME}" bash
}

ACTION="${1:-create}"
shift || true

case "${ACTION}" in
    create)   cmd_create ;;
    stop)     cmd_stop ;;
    restart)  cmd_restart ;;
    status)   cmd_status ;;
    ssh)      cmd_ssh ;;
    logs)     cmd_logs ;;
    destroy)  cmd_destroy ;;
    shell)    cmd_shell ;;
    *)
        echo "用法: $0 {create|stop|restart|status|ssh|logs|destroy|shell}"
        exit 1
        ;;
esac
