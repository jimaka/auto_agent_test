#!/usr/bin/env bash
# 多 GPU 并行独立训练（每张卡一个 SAC 或完整 pipeline 任务）
#
# 原理: 通过 CUDA_VISIBLE_DEVICES 将独立任务分配到不同 GPU，非单模型数据并行。
#
# 示例:
#   ./scripts/ship_3dof/run_multi_gpu.sh --smoke
#   GPUS=0,1,2,3 SEEDS=42,43,44,45 ./scripts/ship_3dof/run_multi_gpu.sh --mode sac
#   MODE=pipeline GPUS=0,1 ./scripts/ship_3dof/run_multi_gpu.sh
#   GPUS=0,1 JOBS_PER_GPU=2 ./scripts/ship_3dof/run_multi_gpu.sh --mode pipeline --detach
#
# 环境变量:
#   GPUS, MODE, JOBS_PER_GPU, MAX_PARALLEL, SEEDS, SEED, BATCH_SIZE, PHASE_STEPS 等

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

DETACH=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detach|-d) DETACH=true; shift ;;
        --smoke)     EXTRA_ARGS+=("--smoke"); shift ;;
        --mode)      MODE="$2"; shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

MODE="${MODE:-sac}"
LOG_FILE="${LOG_FILE:-/tmp/ship3dof_multi_gpu.log}"

CMD=(
    python3 scripts/ship_3dof/multi_gpu_train.py
    --mode "${MODE}"
    --gpus "${GPUS:-}"
    --jobs-per-gpu "${JOBS_PER_GPU:-1}"
    --max-parallel "${MAX_PARALLEL:-0}"
    --base-seed "${SEED:-42}"
    --output-root "${OUTPUT_ROOT:-artifacts/ship_3dof/multi_gpu}"
    --log-root "${LOG_ROOT:-logs_multi_gpu}"
    --mmg-params "${MMG_PARAMS:-scripts/ship_3dof/mmg_params_example.json}"
    --device "${DEVICE:-cuda}"
    --phase-steps ${PHASE_STEPS:-600000 800000 1100000}
    --learning-starts "${LEARNING_STARTS:-5000}"
    --learning-rate "${LEARNING_RATE:-3e-4}"
    --train-freq "${TRAIN_FREQ:-4}"
    --gradient-steps "${GRADIENT_STEPS:-4}"
    --batch-size "${BATCH_SIZE:-512}"
    --net-arch "${NET_ARCH:-512,512,512}"
    --eval-episodes "${EVAL_EPISODES:-30}"
    --shadow-episodes "${SHADOW_EPISODES:-50}"
)

if [[ -n "${SEEDS:-}" ]]; then
    CMD+=(--seeds ${SEEDS})
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "=============================================="
echo "  Ship3DOF Multi-GPU Training"
echo "=============================================="
echo "  模式:      ${MODE}"
echo "  GPU:       ${GPUS:-auto}"
echo "  每卡任务:  ${JOBS_PER_GPU:-1}"
echo "  项目目录:  ${PROJECT_DIR}"
echo "=============================================="

if [[ "${DETACH}" == "true" ]]; then
    echo "[INFO] 后台运行，日志: ${LOG_FILE}"
    nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    echo "[OK] PID=$!  查看日志: tail -f ${LOG_FILE}"
else
    "${CMD[@]}"
fi
