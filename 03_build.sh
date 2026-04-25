#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

CONFIG_FILE="${CONFIG_FILE:-./resnet18_224x224_nv12.yaml}"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "[ERROR] Missing config file: ${CONFIG_FILE}"
  exit 1
fi

if ! command -v hb_mapper >/dev/null 2>&1; then
  echo "[ERROR] hb_mapper not found in PATH."
  echo "[HINT] Please run this inside Horizon OpenExplorer container/environment."
  exit 1
fi

EXTRA_ARGS=()
if hb_mapper makertbin --help 2>/dev/null | grep -q -- "--model-type"; then
  EXTRA_ARGS+=(--model-type onnx)
fi

echo "[INFO] Running hb_mapper with ${CONFIG_FILE}"
hb_mapper makertbin --config "${CONFIG_FILE}" "${EXTRA_ARGS[@]}"

echo "[DONE] 03_build.sh finished"
