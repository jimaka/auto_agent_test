#!/bin/bash
# 远程开发环境入口脚本
# 支持 Docker / Docker Compose / Kubernetes 三种运行方式
set -euo pipefail

export USER="${CONTAINER_USER:-developer}"
export CONTAINER_USER="${CONTAINER_USER:-developer}"
export HOME="/home/${USER}"
export WORKSPACE="${WORKSPACE:-/workspace/rl-zoo}"
export DISPLAY="${DISPLAY:-:1}"
export VNC_RESOLUTION="${VNC_RESOLUTION:-1920x1080}"
export ENABLE_SSH="${ENABLE_SSH:-true}"

# 默认凭据（生产环境请通过环境变量或 K8s Secret 覆盖）
export VNC_PASSWORD="${VNC_PASSWORD:-ship3dof}"
export JUPYTER_TOKEN="${JUPYTER_TOKEN:-ship3dof}"
export SSH_PASSWORD="${SSH_PASSWORD:-ship3dof}"

# 服务端口
export VNC_PORT="${VNC_PORT:-5901}"
export NOVNC_PORT="${NOVNC_PORT:-6080}"
export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
export TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
export TENSORBOARD_PATH_PREFIX="${TENSORBOARD_PATH_PREFIX:-}"
export SSH_PORT="${SSH_PORT:-22}"
export JUPYTER_BASE_URL="${JUPYTER_BASE_URL:-/}"

mkdir -p "${HOME}/.vnc" "${HOME}/.jupyter" "${WORKSPACE}"
chown -R "${USER}:${USER}" "${HOME}" 2>/dev/null || true

# VNC 密码文件
echo "${VNC_PASSWORD}" | vncpasswd -f > "${HOME}/.vnc/passwd"
chmod 600 "${HOME}/.vnc/passwd"
chown "${USER}:${USER}" "${HOME}/.vnc/passwd" 2>/dev/null || true

# SSH 基础配置
if [[ "${ENABLE_SSH}" == "true" ]]; then
    echo "${USER}:${SSH_PASSWORD}" | chpasswd
    mkdir -p /var/run/sshd
    sed -i 's/#PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
fi

# Jupyter 配置（支持 Ingress 子路径，通过 JUPYTER_BASE_URL 设置）
JUPYTER_BASE_URL="${JUPYTER_BASE_URL:-/}"
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
chown "${USER}:${USER}" "${HOME}/.jupyter/jupyter_lab_config.py" 2>/dev/null || true

# XFCE 桌面启动脚本（VNC 会话内执行）
cat > "${HOME}/.vnc/xstartup" <<'XSTART'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XKL_XMODMAP_DISABLE=1
[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
xsetroot -solid "#2e3440"
startxfce4 &
XSTART
chmod +x "${HOME}/.vnc/xstartup"
chown "${USER}:${USER}" "${HOME}/.vnc/xstartup" 2>/dev/null || true

echo "=============================================="
echo "  Ship3DOF RL 远程开发环境"
echo "=============================================="
echo "  工作目录:  ${WORKSPACE}"
echo "  远程桌面:  http://<host>:${NOVNC_PORT}/vnc.html"
echo "  Jupyter:   http://<host>:${JUPYTER_PORT}/?token=${JUPYTER_TOKEN}"
echo "  TensorBoard: http://<host>:${TENSORBOARD_PORT}"
if [[ "${ENABLE_SSH:-true}" == "true" ]]; then
    echo "  SSH:       ssh ${USER}@<host> -p ${SSH_PORT}"
fi
echo "=============================================="

# 如果传入了命令，直接执行（适合 K8s Job / 一次性训练任务）
if [[ $# -gt 0 ]]; then
    cd "${WORKSPACE}"
    exec "$@"
fi

# 默认：启动 supervisor 管理所有远程访问服务
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf
