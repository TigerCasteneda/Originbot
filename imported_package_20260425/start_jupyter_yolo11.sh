#!/usr/bin/env bash
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"
exec conda run -n yolo11 jupyter lab --config="$HOME/.jupyter/jupyter_server_config.py"
