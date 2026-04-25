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
- `02 model convert.ipynb`: conversion notebook (default mapper: `/root/originbot`)
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
cd /root/originbot
chmod +x setup_train113.sh
ENV_NAME=train113 CONDA_DIR=/root/miniconda3 bash setup_train113.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate train113
```

## 2) Training

Multi-dataset training example:

```bash
python originbot_train.py \
  --dataset-dir /root/originbot/image_dataset \
  --dataset-dir /root/originbot/image_dataset_0424 \
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
cd /root/originbot
MODE=nohup JUPYTER_PORT=8888 JUPYTER_TOKEN='your_token_here' PROJECT_DIR=/root/originbot bash setup_jupyter_autostart.sh
```

Local tunnel:

```powershell
ssh -N -L 8888:127.0.0.1:8888 -p <SSH_PORT> root@<HOST>
```

Open:
- `http://127.0.0.1:8888/lab?token=your_token_here`

## 4) Convert ONNX to BIN

`hb_mapper` must be available in current environment:

```bash
which hb_mapper
hb_mapper --help
```

Notebook path (recommended):
- Run `02 model convert.ipynb`.
- Default mapper workspace: `/root/originbot`.
- Optional override:
  ```bash
  export MAPPER_DIR=/your/mapper/path
  ```

CLI path:

```bash
cd /root/originbot
python prepare_horizon_mapper.py --mapper-dir /root/originbot --onnx /root/originbot/best_line_follower_model_xy.onnx --dataset-dir /root/originbot/image_dataset
bash run_horizon_convert.sh /root/originbot
ls -lh /root/originbot/model_output/*.bin
```

Note:
- `image_dataset_0424` is useful for training, but not required for `.bin` conversion.

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
