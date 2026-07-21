#!/bin/bash
# Isaac Lab SSH 开发环境入口（SSH + Jupyter + TensorBoard，无远程桌面）
set -euo pipefail

export CONTAINER_USER="${CONTAINER_USER:-root}"
export WORKSPACE="${WORKSPACE:-/workspace/rl-zoo}"
export ENABLE_SSH="${ENABLE_SSH:-true}"
export PYTHON="${PYTHON:-/isaac-sim/python.sh}"

export SSH_PASSWORD="${SSH_PASSWORD:-ship3dof}"
export JUPYTER_TOKEN="${JUPYTER_TOKEN:-ship3dof}"

export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
export TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
export SSH_PORT="${SSH_PORT:-22}"
export JUPYTER_BASE_URL="${JUPYTER_BASE_URL:-/}"
export TENSORBOARD_PATH_PREFIX="${TENSORBOARD_PATH_PREFIX:-}"

mkdir -p "${HOME}/.jupyter" "${WORKSPACE}"

if [[ "${ENABLE_SSH}" == "true" ]]; then
    echo "${CONTAINER_USER}:${SSH_PASSWORD}" | chpasswd
    mkdir -p /var/run/sshd
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
    sed -i 's/PermitRootLogin no/PermitRootLogin yes/' /etc/ssh/sshd_config 2>/dev/null || true
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
fi

cat > "${HOME}/.jupyter/jupyter_lab_config.py" <<EOF
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = ${JUPYTER_PORT}
c.ServerApp.open_browser = False
c.ServerApp.token = "${JUPYTER_TOKEN}"
c.ServerApp.root_dir = "${WORKSPACE}"
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_origin = "*"
c.ServerApp.base_url = "${JUPYTER_BASE_URL}"
EOF

echo "=============================================="
echo "  Ship3DOF Isaac SSH 开发环境"
echo "=============================================="
echo "  工作目录:    ${WORKSPACE}"
echo "  Jupyter:     http://<host>:${JUPYTER_PORT}/?token=${JUPYTER_TOKEN}"
echo "  TensorBoard: http://<host>:${TENSORBOARD_PORT}"
if [[ "${ENABLE_SSH}" == "true" ]]; then
    echo "  SSH:         ssh ${CONTAINER_USER}@<host> -p ${SSH_PORT}"
fi
echo "=============================================="

if [[ $# -gt 0 ]]; then
    cd "${WORKSPACE}"
    exec "$@"
fi

exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord-isaac-ssh.conf
