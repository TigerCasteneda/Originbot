#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-train113}"
PYTHON_VER="${PYTHON_VER:-3.10}"
CONDA_DIR="${CONDA_DIR:-/root/miniconda3}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

echo "[INFO] Target env: ${ENV_NAME} (python ${PYTHON_VER})"

if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] apt-get installation requires root. Please run this script as root."
    exit 1
  fi
  echo "[INFO] Installing OS dependencies"
  apt-get update -y
  apt-get install -y \
    wget \
    curl \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg
elif ! command -v wget >/dev/null 2>&1; then
  echo "[ERROR] apt-get not available and wget is missing."
  echo "[ERROR] Please install wget manually first."
  exit 1
fi

if [ ! -x "${CONDA_DIR}/bin/conda" ]; then
  echo "[INFO] Installing Miniconda to ${CONDA_DIR}"
  wget -O /tmp/miniconda.sh "${MINICONDA_URL}"
  bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
fi

eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
conda config --set always_yes yes --set changeps1 no >/dev/null

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda env exists: ${ENV_NAME}"
else
  echo "[INFO] Creating conda env: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" pip
fi

conda activate "${ENV_NAME}"

echo "[INFO] Installing pinned packages (A16 / CUDA 11.4-11.6 friendly stack)"
python -m pip install --upgrade pip setuptools wheel
# Keep numpy pinned; also install via pip to avoid missing-package edge cases.
conda install -y "numpy=1.26.4"
python -m pip install --no-cache-dir \
  numpy==1.26.4 \
  torch==1.12.1+cu113 \
  torchvision==0.13.1+cu113 \
  torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install --no-cache-dir \
  pillow \
  scipy \
  pandas \
  matplotlib \
  tqdm \
  pyyaml \
  scikit-learn \
  opencv-python \
  onnx \
  onnxruntime \
  onnxsim \
  jupyterlab \
  notebook \
  ipykernel \
  ipywidgets

echo "[INFO] Enforcing numpy<2 for torch 1.12 compatibility"
python -m pip uninstall -y numpy || true
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
python -m pip check || true

echo "[INFO] Verifying runtime"
python - <<'PY'
import torch
import torchvision
import numpy
import cv2
import PIL
import onnx
import onnxruntime
import sklearn
import pandas
import matplotlib
import tqdm
major = int(numpy.__version__.split(".")[0])
if major >= 2:
    raise RuntimeError(f"numpy must be <2, got {numpy.__version__}")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("Pillow:", PIL.__version__)
print("onnx:", onnx.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("scikit-learn:", sklearn.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)
print("tqdm:", tqdm.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "[DONE] Setup complete."
echo "[NEXT] source /root/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
echo "[NEXT] jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=originbot123 --ServerApp.allow_origin='*'"
echo "[NEXT] MODE=nohup JUPYTER_TOKEN='your_token_here' bash setup_jupyter_autostart.sh   # container-safe auto-start"
