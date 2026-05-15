# OriginBot Training & Conversion

This repository provides a full workflow for:
- line/lane image data usage
- PyTorch `xy` regression training
- ONNX export
- Horizon `hb_mapper` conversion to `.bin`

## Key Files

- `originbot_train.py`: training script (`best_line_follower_model_xy.pth`)
- `originbot_onnx.py`: export `.pth` to `.onnx`
- `01 train.ipynb`: training notebook
- `02 model convert.ipynb`: Docker/OpenExplorer conversion notebook (Jupyter under `/data` friendly)
- `setup_train113.sh`: environment bootstrap (Miniconda + pinned deps)
- `setup_jupyter_autostart.sh`: Jupyter launcher (`MODE=auto/systemd/nohup`)
- `prepare_horizon_mapper.py`: mapper preparation helper
- `02_preprocess.sh`: calibration data preprocessing
- `03_build.sh`: `hb_mapper` build entry
- `run_horizon_convert.sh`: preprocess + build wrapper
- `upload_all_to_server.ps1`: Windows PowerShell uploader

## Recommended Versions

For A16 + older driver stacks:
- Python `3.10`
- `torch==1.12.1+cu113`
- `torchvision==0.13.1+cu113`
- `torchaudio==0.12.1`
- `numpy==1.26.4` (`numpy<2` required)

## 1) Server Environment Setup

Run on server (prefer root):

```bash
cd /data/originbot
chmod +x setup_train113.sh
ENV_NAME=train113 CONDA_DIR=/root/miniconda3 bash setup_train113.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate train113
```

## 2) Training

Multi-dataset training example:

```bash
python originbot_train.py \
  --dataset-dir /data/originbot/image_dataset \
  --dataset-dir /data/originbot/image_dataset_0424 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 3 \
  --prefetch-factor 2 \
  --val-every 3 \
  --amp \
  --expected-gpu-name "NVIDIA A16"
```

Export ONNX:

```bash
python originbot_onnx.py
```

Output:
- `best_line_follower_model_xy.onnx`

## 3) Jupyter (Container-safe)

If no `systemd`, use `nohup` mode:

```bash
cd /data/originbot
MODE=nohup JUPYTER_PORT=8888 JUPYTER_TOKEN='your_token_here' PROJECT_DIR=/data/originbot bash setup_jupyter_autostart.sh
```

Local tunnel:

```powershell
ssh -N -L 8888:127.0.0.1:8888 -p <SSH_PORT> root@<HOST>
```

Open:
- `http://127.0.0.1:8888/lab?token=your_token_here`

## 4) Convert ONNX to BIN

Notebook path (recommended):
- Run `02 model convert.ipynb`.
- The notebook now detects the current project directory automatically, which fits Jupyter roots like `/data`.
- It copies required files into an OpenExplorer sample workspace and runs `hb_mapper` inside Docker.
- Optional overrides:
  ```bash
  export PROJECT_DIR=/data/originbot
  export OPEN_EXPLORER_ROOT=/data/horizon_x5_open_explorer_v1.2.8-py310_20240926
  export WORK_NAME=originbot_convert
  export DOCKER_IMAGE=openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310
  ```

CLI path:

```bash
cd /data/originbot
python prepare_horizon_mapper.py --mapper-dir /data/originbot --onnx /data/originbot/best_line_follower_model_xy.onnx --dataset-dir /data/originbot/image_dataset
bash run_horizon_convert.sh /data/originbot
ls -lh /data/originbot/model_output/*.bin
```

CLI note:
- `run_horizon_convert.sh` still assumes `hb_mapper` is already available in the current shell.
- If your current runtime image does not provide `hb_mapper`, use `02 model convert.ipynb` instead.

Note:
- `image_dataset_0424` is useful for training, but not required for `.bin` conversion.
- If your current runtime image is `cv-cuda+pytorch2.4+python3.10`, training and ONNX export scripts are compatible; `setup_train113.sh` is only for the older pinned `torch1.12` environment.

## 5) Upload from Windows (PowerShell)

In local repo directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\upload_all_to_server.ps1
```

Skip second dataset upload:

```powershell
.\upload_all_to_server.ps1 -SkipDataset0424
```

## Troubleshooting

- `torchvision` / NumPy error:
  ```bash
  python -m pip install --force-reinstall "numpy==1.26.4"
  ```
- `conda` not found:
  ```bash
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate train113
  ```
- `hb_mapper` not found:
  - enter Horizon OpenExplorer environment/container first, then rerun conversion.
