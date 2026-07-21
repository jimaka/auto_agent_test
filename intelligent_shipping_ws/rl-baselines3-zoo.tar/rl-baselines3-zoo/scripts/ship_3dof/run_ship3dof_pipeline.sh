#!/usr/bin/env bash
# 在 isaac-lab-base-koopman:v3.0 容器中启动 Ship3DOF pipeline
#
# 用法:
#   ./scripts/ship_3dof/run_ship3dof_pipeline.sh
#   ./scripts/ship_3dof/run_ship3dof_pipeline.sh --detach
#   ./scripts/ship_3dof/run_ship3dof_pipeline.sh --run-tag my_exp
#   PROJECT_DIR=/home/lijiming/rl-baselines3-zoo ./scripts/ship_3dof/run_ship3dof_pipeline.sh --detach
#
# 环境变量（可选）:
#   PROJECT_DIR, IMAGE, RUN_TAG, BATCH_SIZE, PHASE_STEPS, SEED, DEVICE 等

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
IMAGE="${IMAGE:-isaac-lab-base-koopman:v3.0}"
CONTAINER_WORKDIR="/workspace/rl-zoo"
PYTHON="${PYTHON:-/isaac-sim/python.sh}"

RUN_TAG="${RUN_TAG:-exp_seed42}"
LOG_FOLDER="${LOG_FOLDER:-logs_pipeline}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/ship_3dof/runs}"
MMG_PARAMS="${MMG_PARAMS:-scripts/ship_3dof/mmg_params_example.json}"

PHASE_STEPS="${PHASE_STEPS:-600000 800000 1100000}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
TRAIN_FREQ="${TRAIN_FREQ:-4}"
GRADIENT_STEPS="${GRADIENT_STEPS:-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NET_ARCH="${NET_ARCH:-512,512,512}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
SHADOW_EPISODES="${SHADOW_EPISODES:-100}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

DETACH=false
LOG_FILE="${LOG_FILE:-/tmp/ship3dof_${RUN_TAG}.log}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detach|-d)   DETACH=true; shift ;;
        --run-tag)     RUN_TAG="$2"; LOG_FILE="/tmp/ship3dof_${RUN_TAG}.log"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "用法: $0 [--detach] [--run-tag TAG] [--project-dir PATH]"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

PIPELINE_SCRIPT="${PROJECT_DIR}/scripts/ship_3dof/run_known_mmg_pipeline.py"
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
    echo "[ERROR] 找不到 pipeline 脚本: ${PIPELINE_SCRIPT}"
    echo "        请检查 PROJECT_DIR=${PROJECT_DIR}"
    exit 1
fi

if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
    echo "[ERROR] 镜像不存在: ${IMAGE}"
    exit 1
fi

echo "=============================================="
echo "  Ship3DOF Pipeline"
echo "=============================================="
echo "  镜像:      ${IMAGE}"
echo "  项目目录:  ${PROJECT_DIR}"
echo "  Run Tag:   ${RUN_TAG}"
echo "  设备:      ${DEVICE}"
echo "=============================================="

PIPELINE_CMD="
cd ${CONTAINER_WORKDIR} && \
${PYTHON} scripts/ship_3dof/run_known_mmg_pipeline.py \
  --python ${PYTHON} \
  --mmg-params ${MMG_PARAMS} \
  --generate-trials-if-missing \
  --run-tag ${RUN_TAG} \
  --output-dir ${OUTPUT_DIR} \
  --log-folder ${LOG_FOLDER} \
  --phase-steps ${PHASE_STEPS} \
  --learning-starts ${LEARNING_STARTS} \
  --learning-rate ${LEARNING_RATE} \
  --train-freq ${TRAIN_FREQ} \
  --gradient-steps ${GRADIENT_STEPS} \
  --batch-size ${BATCH_SIZE} \
  --net-arch ${NET_ARCH} \
  --eval-episodes ${EVAL_EPISODES} \
  --shadow-episodes ${SHADOW_EPISODES} \
  --seed ${SEED} \
  --device ${DEVICE}
"

DOCKER_ARGS=(
    docker run --rm
    --gpus all
    --shm-size=8g
    -e ACCEPT_EULA=Y
    -v "${PROJECT_DIR}:${CONTAINER_WORKDIR}"
    --entrypoint bash
    "${IMAGE}"
    -lc "${PIPELINE_CMD}"
)

if [[ "${DETACH}" == "true" ]]; then
    echo "[INFO] 后台运行，日志: ${LOG_FILE}"
    nohup "${DOCKER_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
    echo "[OK] PID=$!  查看日志: tail -f ${LOG_FILE}"
else
    DOCKER_ARGS=(docker run --rm -it --gpus all --shm-size=8g -e ACCEPT_EULA=Y
        -v "${PROJECT_DIR}:${CONTAINER_WORKDIR}"
        --entrypoint bash "${IMAGE}" -lc "${PIPELINE_CMD}")
    "${DOCKER_ARGS[@]}"
fi
