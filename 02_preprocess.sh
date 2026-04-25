#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

PY_BIN="${PY_BIN:-python3}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  PY_BIN="python"
fi

SRC_DIR="${SRC_DIR:-./image_dataset}"
DST_DIR="${DST_DIR:-./calibration_data_bgr_f32}"
PIC_EXT="${PIC_EXT:-.rgb}"
INPUT_WIDTH="${INPUT_WIDTH:-224}"
INPUT_HEIGHT="${INPUT_HEIGHT:-224}"

echo "[INFO] Preprocess images for calibration"
echo "[INFO] src_dir=${SRC_DIR}"
echo "[INFO] dst_dir=${DST_DIR}"
echo "[INFO] size=${INPUT_WIDTH}x${INPUT_HEIGHT}"

"${PY_BIN}" ./horizon_preprocess.py \
  --src_dir "${SRC_DIR}" \
  --dst_dir "${DST_DIR}" \
  --pic_ext "${PIC_EXT}" \
  --read_mode opencv \
  --width "${INPUT_WIDTH}" \
  --height "${INPUT_HEIGHT}"

echo "[DONE] 02_preprocess.sh finished"
