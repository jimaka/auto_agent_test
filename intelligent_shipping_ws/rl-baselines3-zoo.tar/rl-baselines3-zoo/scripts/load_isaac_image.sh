#!/usr/bin/env bash
# 从 tar 包导入 isaac-lab-base-koopman 镜像（离线服务器使用）
#
# 用法:
#   ./scripts/load_isaac_image.sh
#   ISAAC_TAR=/path/to/isaac-lab-base-koopman_v3.0.tar ./scripts/load_isaac_image.sh
#
# 环境变量:
#   ISAAC_TAR          - tar 包路径（默认自动搜索）
#   ISAAC_BASE_IMAGE   - 目标基础镜像名（默认 isaac-lab-base-koopman:v3.0）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ISAAC_BASE_IMAGE="${ISAAC_BASE_IMAGE:-isaac-lab-base-koopman:v3.0}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[ERROR] $*" >&2; exit 1; }

if docker image inspect "${ISAAC_BASE_IMAGE}" > /dev/null 2>&1; then
    log "基础镜像已存在: ${ISAAC_BASE_IMAGE}"
    exit 0
fi

# 自动搜索 tar 包
if [[ -z "${ISAAC_TAR:-}" ]]; then
    for candidate in \
        "${PROJECT_ROOT}/isaac-lab-base-koopman_v3.0.tar" \
        "${PROJECT_ROOT}/isaac-lab-base-koopman:v3.0.tar" \
        "${HOME}/isaac-lab-base-koopman_v3.0.tar" \
        "/home/lijiming/isaac-lab-base-koopman_v3.0.tar" \
        "/home/lijiming/rl-baselines3-zoo/isaac-lab-base-koopman_v3.0.tar"
    do
        if [[ -f "${candidate}" ]]; then
            ISAAC_TAR="${candidate}"
            break
        fi
    done
fi

[[ -n "${ISAAC_TAR:-}" && -f "${ISAAC_TAR}" ]] \
    || err "找不到 tar 包。请设置 ISAAC_TAR=/path/to/isaac-lab-base-koopman_v3.0.tar"

log "导入镜像: ${ISAAC_TAR}"
docker load -i "${ISAAC_TAR}"

# 导入后检查目标 tag，若不存在则尝试从常见 tag 重命名
if docker image inspect "${ISAAC_BASE_IMAGE}" > /dev/null 2>&1; then
    log "导入成功: ${ISAAC_BASE_IMAGE}"
    exit 0
fi

for alt_tag in \
    "isaac-lab-base-koopman:latest" \
    "isaac-lab-base-koopman_v3.0:latest" \
    "isaac-lab-base-koopman-v3.0:latest"
do
    if docker image inspect "${alt_tag}" > /dev/null 2>&1; then
        log "将 ${alt_tag} 标记为 ${ISAAC_BASE_IMAGE}"
        docker tag "${alt_tag}" "${ISAAC_BASE_IMAGE}"
        log "导入成功: ${ISAAC_BASE_IMAGE}"
        exit 0
    fi
done

# 列出最近导入的镜像帮助排查
log "已导入镜像列表:"
docker images --format "  {{.Repository}}:{{.Tag}}" | head -10
err "导入完成但未找到 ${ISAAC_BASE_IMAGE}，请手动 docker tag"
