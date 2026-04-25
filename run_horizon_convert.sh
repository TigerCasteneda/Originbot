#!/usr/bin/env bash
set -euo pipefail

MAPPER_DIR="${1:-}"
if [ -z "${MAPPER_DIR}" ]; then
  echo "Usage: bash run_horizon_convert.sh <mapper_dir>"
  exit 1
fi

if [ ! -d "${MAPPER_DIR}" ]; then
  echo "[ERROR] mapper_dir not found: ${MAPPER_DIR}"
  exit 1
fi

cd "${MAPPER_DIR}"

if [ ! -f "02_preprocess.sh" ]; then
  echo "[ERROR] Missing 02_preprocess.sh in ${MAPPER_DIR}"
  exit 1
fi

if [ ! -f "03_build.sh" ]; then
  echo "[ERROR] Missing 03_build.sh in ${MAPPER_DIR}"
  exit 1
fi

if [ ! -f "resnet18_224x224_nv12.yaml" ]; then
  echo "[ERROR] Missing YAML config: ${MAPPER_DIR}/resnet18_224x224_nv12.yaml"
  exit 1
fi

if [ ! -f "best_line_follower_model_xy.onnx" ]; then
  echo "[ERROR] Missing ONNX model: ${MAPPER_DIR}/best_line_follower_model_xy.onnx"
  exit 1
fi

if [ ! -d "image_dataset" ]; then
  echo "[ERROR] Missing dataset folder: ${MAPPER_DIR}/image_dataset"
  exit 1
fi

echo "[STEP] 1/2 preprocess calibration data"
sh 02_preprocess.sh

echo "[STEP] 2/2 build bin model"
sh 03_build.sh

BIN_FILE="$(find model_output -maxdepth 1 -name '*.bin' | head -n 1 || true)"
if [ -z "${BIN_FILE}" ]; then
  echo "[ERROR] Conversion finished but no .bin found under ${MAPPER_DIR}/model_output"
  exit 1
fi

echo "[DONE] BIN generated: ${MAPPER_DIR}/${BIN_FILE}"
