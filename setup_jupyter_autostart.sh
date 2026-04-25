#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-jupyterlab-originbot}"
ENV_NAME="${ENV_NAME:-train113}"
CONDA_DIR="${CONDA_DIR:-/root/miniconda3}"
PROJECT_DIR="${PROJECT_DIR:-/root/originbot}"
JUPYTER_IP="${JUPYTER_IP:-0.0.0.0}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
JUPYTER_PASSWORD="${JUPYTER_PASSWORD:-}"
AUTO_ALLOW_UFW="${AUTO_ALLOW_UFW:-true}"
MODE="${MODE:-auto}" # auto | systemd | nohup
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/jupyter.log}"
PID_FILE="${PID_FILE:-${PROJECT_DIR}/jupyter.pid}"
STARTUP_CMD_FILE="${STARTUP_CMD_FILE:-${PROJECT_DIR}/jupyter_startup_command.sh}"

if [ "$(id -u)" -ne 0 ]; then
  echo "[ERROR] Please run as root."
  exit 1
fi

if [ ! -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  echo "[ERROR] Missing conda.sh: ${CONDA_DIR}/etc/profile.d/conda.sh"
  exit 1
fi

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "[ERROR] Missing project directory: ${PROJECT_DIR}"
  exit 1
fi

CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

jupyter_args=(
  "--ServerApp.ip=${JUPYTER_IP}"
  "--ServerApp.port=${JUPYTER_PORT}"
  "--ServerApp.open_browser=False"
  "--ServerApp.allow_root=True"
  "--ServerApp.allow_remote_access=True"
  "--notebook-dir=${PROJECT_DIR}"
)

if [ -n "${JUPYTER_TOKEN}" ]; then
  jupyter_args+=("--IdentityProvider.token=${JUPYTER_TOKEN}")
fi

if [ -n "${JUPYTER_PASSWORD}" ]; then
  jupyter_args+=("--ServerApp.password=${JUPYTER_PASSWORD}")
fi

is_systemd_ready() {
  command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]
}

open_firewall_if_needed() {
  if [ "${AUTO_ALLOW_UFW}" = "true" ] && command -v ufw >/dev/null 2>&1; then
    if ufw status | grep -q "Status: active"; then
      ufw allow "${JUPYTER_PORT}/tcp" || true
    fi
  fi
}

start_with_systemd() {
  cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Jupyter Lab for OriginBot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${PROJECT_DIR}
ExecStart=/bin/bash -lc 'source "${CONDA_SH}" && conda activate "${ENV_NAME}" && exec jupyter lab ${jupyter_args[*]}'
Restart=always
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable "${SERVICE_NAME}"
  systemctl restart "${SERVICE_NAME}"

  open_firewall_if_needed

  echo "[DONE] systemd service installed: ${SERVICE_FILE}"
  echo "[CHECK] systemctl status ${SERVICE_NAME} --no-pager"
  echo "[LOG]   journalctl -u ${SERVICE_NAME} -n 80 --no-pager"
  echo "[URL]   http://<your-server-ip>:${JUPYTER_PORT}/lab"
}

start_with_nohup() {
  mkdir -p "${PROJECT_DIR}"
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    kill "$(cat "${PID_FILE}")" || true
    sleep 1
  fi

  nohup /bin/bash -lc "source '${CONDA_SH}' && conda activate '${ENV_NAME}' && exec jupyter lab ${jupyter_args[*]}" > "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"

  open_firewall_if_needed

  cat > "${STARTUP_CMD_FILE}" <<EOF
bash -lc "source ${CONDA_SH} && conda activate ${ENV_NAME} && nohup jupyter lab ${jupyter_args[*]} > ${LOG_FILE} 2>&1 &"
EOF
  chmod +x "${STARTUP_CMD_FILE}"

  echo "[DONE] Jupyter started with nohup (container-friendly)."
  echo "[PID]  $(cat "${PID_FILE}")"
  echo "[LOG]  tail -n 30 ${LOG_FILE}"
  echo "[PORT] ss -lntp | grep ${JUPYTER_PORT}"
  echo "[URL]  http://<your-server-ip>:${JUPYTER_PORT}/lab"
  echo "[BOOT] Startup command saved to: ${STARTUP_CMD_FILE}"
}

selected_mode="${MODE}"
if [ "${selected_mode}" = "auto" ]; then
  if is_systemd_ready; then
    selected_mode="systemd"
  else
    selected_mode="nohup"
  fi
fi

case "${selected_mode}" in
  systemd)
    if ! is_systemd_ready; then
      echo "[ERROR] systemd mode requested but systemd is not available."
      exit 1
    fi
    start_with_systemd
    ;;
  nohup)
    start_with_nohup
    ;;
  *)
    echo "[ERROR] Unsupported MODE: ${MODE}. Use auto/systemd/nohup."
    exit 1
    ;;
esac

if [ -z "${JUPYTER_TOKEN}" ] && [ -z "${JUPYTER_PASSWORD}" ]; then
  echo "[WARN] Token/password are empty. Anyone who can access this IP:port can open Jupyter."
fi
