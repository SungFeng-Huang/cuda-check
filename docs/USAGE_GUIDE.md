# Usage Guide

Complete guide for using CUDA & Container Toolkit on HPC clusters.

---

## 🎯 Quick Start

### 1. Test Your Environment

```bash
# Quick check
bash scripts/check_cuda.sh

# Full diagnostic (includes cuFFT tests)
python3 scripts/check_cuda_pytorch.py

# Test NGC official container
bash scripts/test_nvidia_container.sh
```

### 2. Launch Interactive Session

```bash
# Launches NVIDIA PyTorch container with automatic GPU support
bash scripts/launch_interactive.sh
```

**Inside the container**:
```bash
# Verify GPU
nvidia-smi

# Test CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Test cuFFT
python3 -c "import torch; x=torch.randn(1000).cuda(); print(torch.fft.rfft(x).shape)"
```

---

## 🔍 Diagnostic Tools

### check_cuda.sh - Quick Check

Shows essential version information:
- NVIDIA Driver version
- CUDA Runtime version
- PyTorch version and CUDA support
- Quick cuFFT test

**Usage**:
```bash
bash scripts/check_cuda.sh
```

### check_cuda_pytorch.py - Full Diagnostic

Comprehensive testing including:
1. NVIDIA Driver & GPU Info
2. CUDA Version (runtime & toolkit)
3. PyTorch Info (version, CUDA, cuDNN)
4. PyTorch CUDA Test (actual GPU computation)
5. **cuFFT Test** (FFT/IFFT, STFT/ISTFT) - Critical for audio processing
6. **Audio Processing** (Mel-Spectrogram) - TTS essential
7. Compatibility Analysis

**Usage**:
```bash
python3 scripts/check_cuda_pytorch.py
```

### test_nvidia_container.sh - Container Validation

Tests NVIDIA official container:
- GPU access
- PyTorch CUDA functionality
- cuFFT operations
- Audio processing capabilities

**Usage**:
```bash
bash scripts/test_nvidia_container.sh
```

---

## 🐳 Using Docker Registry

### Why Docker Registry?

**Pros**:
- ✅ Automatic GPU mounting (no manual configuration)
- ✅ CUDA version auto-matching
- ✅ Easy version control
- ✅ Team collaboration

**Cons**:
- ⚠️ Requires registry setup (one-time)
- ⚠️ Need Docker on build machine

### Workflow

#### On Local Machine (with Docker)

```bash
# 1. Build image based on NGC PyTorch
cd /path/to/project
docker build -t my-registry.com/project:v1.0 -f templates/Dockerfile .

# 2. Test locally
docker run --gpus all -it my-registry.com/project:v1.0 nvidia-smi

# 3. Push to registry
docker push my-registry.com/project:v1.0
```

#### On Cluster

```bash
# Configure registry credentials (one-time)
cat >> ~/.config/enroot/.credentials << EOF
machine my-registry.com login myuser password mytoken
EOF

# Use directly - no conversion needed!
srun --account=your-account --partition=interactive --gpus=1 \
     --container-image=my-registry.com/project:v1.0 \
     --container-mounts=/your/workspace:/workspace \
     --pty /bin/bash
```

### Supported Registries

- **NGC** (nvcr.io) - NVIDIA official containers
- **Docker Hub** (docker.io)
- **GitHub Container Registry** (ghcr.io)
- **Harbor** - Enterprise registry
- **Private Registry** - Any Docker Registry v2 API

See [DOCKER_REGISTRY_GUIDE.md](DOCKER_REGISTRY_GUIDE.md) for detailed setup.

---

## 🚨 Common Issues

### Issue 1: `torch.cuda.is_available()` returns False

**Diagnosis**:
```bash
# Check GPU visibility
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(torch.version.cuda)"
```

**Solutions**:
1. Ensure `--gpus` flag is set in srun
2. Use NGC container (auto-configures GPU)
3. Add `--no-container-remap-root` flag

### Issue 2: cuFFT Test Failed

**Symptoms**: Audio processing fails, STFT/Mel-spectrogram errors

**Diagnosis**:
```bash
python3 scripts/check_cuda_pytorch.py
# Check Section 5 (cuFFT Test) results
```

**Solution**:
- Use NGC official container with matching CUDA version
- Tested to work: `nvcr.io/nvidia/pytorch:24.10-py3`

### Issue 3: CUDA Version Mismatch

**Symptoms**:
```
RuntimeError: CUDA error: no kernel image is available
```

**Diagnosis**:
```bash
bash scripts/check_cuda.sh
# Compare Driver CUDA vs PyTorch CUDA versions
```

**Solution**:
- Use NGC containers - they auto-adjust CUDA versions
- Example: Cluster CUDA 12.2 → Container auto-adjusts to 12.6 ✅

---

## 📚 Understanding Enroot Containers

### Why Can't I Use Local .tar Files?

**Enroot Limitation**: `enroot import` only supports:
- `docker://` - Docker Registry
- `dockerd://` - Docker Daemon (requires Docker running)
- `podman://` - Podman

**Does NOT support**:
- ❌ Local .tar files
- ❌ docker-archive:// format
- ❌ Direct filesystem archives

### Architecture: Login Node vs Compute Node

```
Login Node                     Compute Node
├─ enroot: ❌ Not available   ├─ enroot: ✅ Available
├─ Submit jobs via srun       ├─ Pyxis handles containers
└─ .credentials: ✅ Set up    └─ Auto GPU mounting ✅
```

**This is normal!** You don't need enroot command on login node. Use `srun --container-image=...` and Pyxis handles everything.

---

## 🎓 Best Practices

### 1. Use NGC Official Containers

**Recommended**: Start with NVIDIA's official containers
```bash
bash scripts/launch_interactive.sh
```

**Why?**:
- ✅ Zero configuration
- ✅ All CUDA libraries pre-installed
- ✅ Versions tested by NVIDIA
- ✅ Auto GPU mounting

### 2. Build Custom Containers on NGC Base

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3
# Add your dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt
```

**Why NGC base?**:
- ✅ CUDA auto-matching
- ✅ All GPU libraries included
- ✅ cuFFT guaranteed to work

### 3. Use Docker Registry for Team Work

Push to registry instead of managing .tar/.sqsh files:
- ✅ Easy version control
- ✅ Team collaboration
- ✅ No manual conversion needed

---

## 📖 Additional Documentation

- **[CHECK_CUDA_README.md](CHECK_CUDA_README.md)** - Detailed diagnostic tool usage
- **[DOCKER_REGISTRY_GUIDE.md](DOCKER_REGISTRY_GUIDE.md)** - Registry setup and authentication

---

## 🆘 Getting Help

1. Check [QUICK_REFERENCE.md](../QUICK_REFERENCE.md) for common commands
2. Run diagnostic tools to identify issues
3. Review error messages from `check_cuda_pytorch.py`
4. Consult documentation guides

---

**Last Updated**: 2025-11-28

