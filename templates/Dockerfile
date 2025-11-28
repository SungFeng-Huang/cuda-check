# Dockerfile for Kimi-Audio
# 基於 NVIDIA NGC PyTorch - 自動處理 CUDA 版本匹配

FROM nvcr.io/nvidia/pytorch:24.10-py3

# Metadata
LABEL maintainer="your-team@example.com"
LABEL version="2.1"
LABEL description="Kimi-Audio TTS with automatic GPU support"

WORKDIR /app

# ========================================
# 安裝系統依賴
# ========================================
RUN apt-get update && apt-get install -y \
    sox \
    ffmpeg \
    libgl1-mesa-glx \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# ========================================
# 安裝 Python 依賴
# ========================================

# 複製 requirements.txt（假設在 Kimi-Audio 目錄）
COPY requirements.txt /app/

# 安裝基礎套件
RUN pip install --no-cache-dir -r requirements.txt

# 安裝 flash-attention（如果需要）
# 注意：這個編譯較慢，可能需要 10-15 分鐘
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# ========================================
# 複製專案代碼
# ========================================

# 複製整個專案（使用 .dockerignore 排除不必要的文件）
COPY . /app/

# ========================================
# 設定環境變數
# ========================================

ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_HOME=/usr/local/cuda
ENV HF_HOME=/hfcache

# ========================================
# 創建必要的目錄
# ========================================

RUN mkdir -p /app/output \
             /app/checkpoints \
             /hfcache

# ========================================
# 設定 Python 符號連結（如果需要）
# ========================================

RUN ln -sf /usr/bin/python3 /usr/bin/python || true

# ========================================
# 健康檢查（可選）
# ========================================

# 驗證關鍵套件是否正確安裝
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import torchaudio; print(f'Torchaudio: {torchaudio.__version__}')"

# ========================================
# 設定工作目錄和啟動命令
# ========================================

WORKDIR /workspace

CMD ["/bin/bash"]

# ========================================
# 構建說明
# ========================================
# 
# 構建命令：
#   docker build -t kimi-audio:v2.1 -f Dockerfile.template /path/to/Kimi-Audio/
#
# 測試命令：
#   docker run --gpus all -it kimi-audio:v2.1 nvidia-smi
#   docker run --gpus all -it kimi-audio:v2.1 python -c "import torch; print(torch.cuda.is_available())"
#
# 保存為 tar：
#   docker save kimi-audio:v2.1 -o kimi-audio-v2.1.tar
#
# 壓縮（可選）：
#   gzip kimi-audio-v2.1.tar
#
# 上傳到集群：
#   scp kimi-audio-v2.1.tar.gz cluster:/path/to/project/
#
# 在集群上轉換：
#   bash convert_docker_to_sqsh.sh kimi-audio-v2.1.tar.gz kimi-audio-v2.1.sqsh
#

