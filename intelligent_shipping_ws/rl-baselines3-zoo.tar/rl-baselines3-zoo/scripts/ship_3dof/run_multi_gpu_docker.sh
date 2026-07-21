#!/usr/bin/env bash
# 在 isaac-lab-base-koopman:v3.0 容器中启动多 GPU 并行训练
#
# 用法:
#   GPUS=0,1 ./scripts/ship_3dof/run_multi_gpu_docker.sh --smoke
#   GPUS=0,1,2,3 MODE=pipeline ./scripts/ship_3dof/run_multi_gpu_docker.sh --detach
#   PROJECT_DIR=/home/lijiming/rl-baselines3-zoo GPUS=0,1 ./scripts/ship_3dof/run_multi_gpu_docker.sh
#
# 注意: 容器需要 --gpus all 才能看到全部 GPU，任务分配由 multi_gpu_train.py 通过
#       CUDA_VISIBLE_DEVICES 完成。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
IMAGE="${IMAGE:-isaac-lab-base-koopman:v3.0}"
CONTAINER_WORKDIR="/workspace/rl-zoo"
PYTHON="${PYTHON:-/isaac-sim/python.sh}"

GPUS="${GPUS:-}"
MODE="${MODE:-sac}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/ship_3dof/multi_gpu}"
LOG_ROOT="${LOG_ROOT:-logs_multi_gpu}"
MMG_PARAMS="${MMG_PARAMS:-scripts/ship_3dof/mmg_params_example.json}"
PHASE_STEPS="${PHASE_STEPS:-600000 800000 1100000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NET_ARCH="${NET_ARCH:-512,512,512}"

DETACH=false
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detach|-d) DETACH=true; shift ;;
        --smoke)     EXTRA_FLAGS="${EXTRA_FLAGS} --smoke"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
    echo "[ERROR] 镜像不存在: ${IMAGE}"
    exit 1
fi

SEEDS_ARG=""
if [[ -n "${SEEDS}" ]]; then
    SEEDS_ARG="--seeds ${SEEDS}"
fi

MULTI_GPU_CMD="
cd ${CONTAINER_WORKDIR} && \
${PYTHON} scripts/ship_3dof/multi_gpu_train.py \
  --python ${PYTHON} \
  --mode ${MODE} \
  --gpus '${GPUS}' \
  --jobs-per-gpu ${JOBS_PER_GPU} \
  --max-parallel ${MAX_PARALLEL} \
  --base-seed ${SEED} \
  ${SEEDS_ARG} \
  --output-root ${OUTPUT_ROOT} \
  --log-root ${LOG_ROOT} \
  --mmg-params ${MMG_PARAMS} \
  --generate-trials-if-missing \
  --device cuda \
  --phase-steps ${PHASE_STEPS} \
  --batch-size ${BATCH_SIZE} \
  --net-arch ${NET_ARCH} \
  ${EXTRA_FLAGS}
"

LOG_FILE="/tmp/ship3dof_multi_gpu.log"

echo "=============================================="
echo "  Ship3DOF Multi-GPU (Docker)"
echo "=============================================="
echo "  镜像:      ${IMAGE}"
echo "  项目目录:  ${PROJECT_DIR}"
echo "  模式:      ${MODE}"
echo "  GPU:       ${GPUS:-auto}"
echo "=============================================="

if [[ "${DETACH}" == "true" ]]; then
    echo "[INFO] 后台运行，日志: ${LOG_FILE}"
    nohup docker run --rm --gpus all --shm-size=8g \
        -e ACCEPT_EULA=Y \
        -v "${PROJECT_DIR}:${CONTAINER_WORKDIR}" \
        --entrypoint bash "${IMAGE}" -lc "${MULTI_GPU_CMD}" \
        > "${LOG_FILE}" 2>&1 &
    echo "[OK] PID=$!  查看日志: tail -f ${LOG_FILE}"
else
    docker run --rm -it --gpus all --shm-size=8g \
        -e ACCEPT_EULA=Y \
        -v "${PROJECT_DIR}:${CONTAINER_WORKDIR}" \
        --entrypoint bash "${IMAGE}" -lc "${MULTI_GPU_CMD}"
fi
