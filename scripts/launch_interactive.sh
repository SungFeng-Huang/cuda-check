#!/usr/bin/env bash

# Use NVIDIA official PyTorch container (easiest solution)
# This automatically handles all GPU mounting and CUDA libraries
# 
# Pros:
#   - Automatic GPU support (no manual mounting needed)
#   - CUDA/cuDNN/cuFFT all pre-configured
#   - PyTorch pre-installed
#   - Versions tested by NVIDIA
#
# Cons:
#   - Need to install project-specific packages on first run
#   - Larger image size

IMG='--container-image=nvcr.io/nvidia/pytorch:24.10-py3'
MOUNTS="--container-mounts=/path/to/project/Kimi-Audio:/workspace,/lustre:/lustre,/path/to/cache/huggingface:/hfcache"
NGPU=1

echo "========================================================"
echo "  Using NVIDIA Official PyTorch Container"
echo "========================================================"
echo ""
echo "Container: nvcr.io/nvidia/pytorch:24.10-py3"
echo "  - PyTorch: Pre-installed (latest)"
echo "  - CUDA: Auto-configured"
echo "  - GPU Support: Automatic"
echo ""
echo "On first run, you may need to install project dependencies:"
echo "  pip install -r /workspace/requirements.txt"
echo ""
echo "========================================================"

srun --account=your-account --partition=interactive --job-name=kimi-audio-nvidia --time=4:00:00 --gpus-per-node=$NGPU --ntasks-per-node=1 $IMG $MOUNTS --no-container-remap-root --container-workdir="/workspace" --pty /bin/bash

