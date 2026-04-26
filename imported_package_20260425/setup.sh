set -e

echo "====== [1/4] 安装 Miniconda ======"
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source $HOME/miniconda3/etc/profile.d/conda.sh
echo 'source $HOME/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc

echo "====== [2/4] 创建 yolo11 环境（Python 3.10）======"
conda create -n yolo11 python=3.10 -y
conda activate yolo11

echo "====== [3/4] 安装 PyTorch + YOLO + ONNX ======"
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install onnx==1.18.0 onnxslim onnxruntime-gpu ultralytics jupyterlab -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "====== [4/4] 下载地平线工具链 ======"
mkdir -p /data/horizon && cd /data/horizon
wget -c "ftp://x5ftp@vrftp.horizon.ai/OpenExplorer/v1.2.8_release/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz" \
--ftp-password='x5ftp@123$%' \
-O horizon_oe.tar.gz
wget -c "ftp://x5ftp@vrftp.horizon.ai/OpenExplorer/v1.2.8_release/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz" \
--ftp-password='x5ftp@123$%' \
-O horizon_docker.tar.gz
echo ">>> 解压 OE 包..."
tar -xvf horizon_oe.tar.gz
echo ">>> 装载 Docker 镜像..."
sudo docker load -i horizon_docker.tar.gz

echo "======================================================"
echo "✅ 全部完成！"
echo "======================================================"
