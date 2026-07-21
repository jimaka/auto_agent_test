#!/bin/bash
set -euo pipefail
LOGDIR="${WORKSPACE:-/workspace/rl-zoo}"
PORT="${TENSORBOARD_PORT:-6006}"
PREFIX="${TENSORBOARD_PATH_PREFIX:-}"
PYTHON="${PYTHON:-}"

ARGS=(--logdir "${LOGDIR}" --host 0.0.0.0 --port "${PORT}")
if [[ -n "${PREFIX}" ]]; then
    ARGS+=(--path_prefix "${PREFIX}")
fi

if [[ -n "${PYTHON}" ]]; then
    exec "${PYTHON}" -m tensorboard.main "${ARGS[@]}"
fi

exec tensorboard "${ARGS[@]}"
