# CUDA & Container Toolkit

Comprehensive CUDA, PyTorch, and cuFFT diagnostic toolkit for HPC clusters with SLURM + Pyxis + Enroot.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Features

- ✅ **CUDA Diagnostics** - Verify CUDA, PyTorch, driver compatibility
- ✅ **cuFFT Testing** - Essential for audio/speech processing (STFT, Mel-Spectrogram)
- ✅ **Auto GPU Mounting** - Using NVIDIA NGC containers
- ✅ **One-Command Setup** - Launch development environment instantly
- ✅ **HPC Optimized** - Designed for SLURM + Pyxis + Enroot

---

## 📁 Structure

```
cuda-check/
├── README.md                  # This file
├── QUICK_REFERENCE.md         # Command cheat sheet
│
├── scripts/                   # Executable tools
│   ├── check_cuda.sh               # Quick CUDA check
│   ├── check_cuda_pytorch.py       # Full diagnostic (cuFFT included)
│   ├── test_nvidia_container.sh    # Container validation
│   └── launch_interactive.sh       # Launch development environment
│
├── templates/                 # Dockerfile templates
│   ├── Dockerfile                  # NGC-based template
│   └── .dockerignore.template      # Docker build optimization
│
└── docs/                      # Documentation
    ├── USAGE_GUIDE.md              # Complete usage guide
    ├── CHECK_CUDA_README.md        # Diagnostic tools reference
    └── DOCKER_REGISTRY_GUIDE.md    # Registry setup guide
```

---

## 🚀 Quick Start (30 seconds)

### Test Your Environment

```bash
git clone https://github.com/YOUR_USERNAME/cuda-check.git
cd cuda-check

# Quick check
bash scripts/check_cuda.sh

# Full diagnostic
python3 scripts/check_cuda_pytorch.py
```

### Launch Development Environment

```bash
# Update account in scripts/launch_interactive.sh first
bash scripts/launch_interactive.sh
```

This launches NVIDIA PyTorch container with:
- ✅ Automatic GPU mounting
- ✅ CUDA/cuDNN/cuFFT pre-configured
- ✅ PyTorch pre-installed

---

## 📖 Documentation

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Essential commands
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Complete usage guide
- **[docs/CHECK_CUDA_README.md](docs/CHECK_CUDA_README.md)** - Diagnostic tools reference
- **[docs/DOCKER_REGISTRY_GUIDE.md](docs/DOCKER_REGISTRY_GUIDE.md)** - Docker Registry setup

---

## 🔍 Diagnostic Tools

### Quick Check
```bash
bash scripts/check_cuda.sh
```
Shows: Driver, CUDA, PyTorch versions, GPU count, quick cuFFT test

### Full Diagnostic
```bash
python3 scripts/check_cuda_pytorch.py
```
Tests: GPU, CUDA, PyTorch, cuFFT (FFT/STFT), Mel-Spectrogram, compatibility

### Container Test
```bash
bash scripts/test_nvidia_container.sh
```
Validates NGC container functionality

---

## 🛠️ Use Cases

### Scenario 1: Check CUDA Compatibility

```bash
bash scripts/check_cuda.sh
```

If versions mismatch → Use NGC container (auto-adjusts)

### Scenario 2: cuFFT Test Failed

**Symptoms**: Audio processing errors (STFT, Mel-spectrogram)

**Solution**:
```bash
# Diagnose
python3 scripts/check_cuda_pytorch.py

# Fix: Use NGC container
bash scripts/launch_interactive.sh
```

### Scenario 3: Custom Container Deployment

**Build locally**:
```bash
# Use provided template
docker build -t my-registry.com/project:v1.0 -f templates/Dockerfile .
docker push my-registry.com/project:v1.0
```

**Use on cluster**:
```bash
srun --account=your-account --partition=interactive --gpus=1 \
     --container-image=my-registry.com/project:v1.0 \
     --pty /bin/bash
```

See [docs/DOCKER_REGISTRY_GUIDE.md](docs/DOCKER_REGISTRY_GUIDE.md) for details.

---

## 💡 Key Insights

### Why NGC Containers?

✅ **CUDA Auto-Matching**: Container CUDA adapts to cluster driver  
Example: Cluster CUDA 12.2 → Container uses 12.6 (tested ✅)

✅ **Auto GPU Mounting**: No manual library mounting needed  
NGC containers include proper Docker metadata for Pyxis/Enroot

✅ **Fully Configured**: cuFFT, cuDNN, NCCL all pre-installed  
Audio processing (STFT, Mel-spectrogram) works out-of-the-box

### Why Docker Registry?

✅ **No Local Conversion**: Direct usage from registry  
❌ Enroot doesn't support local .tar files

✅ **Team Collaboration**: Push once, use everywhere  
Version control and easy updates

---

## 🔧 Requirements

- **Cluster**: SLURM + Pyxis + Enroot
- **GPU**: NVIDIA GPUs with drivers
- **Optional**: Docker (for building custom containers)
- **Optional**: Docker Registry access

---

## 📊 Tested Environment

| Component | Version | Status |
|-----------|---------|--------|
| SLURM | 23.02+ | ✅ |
| GPU | NVIDIA A100 | ✅ |
| Driver | 535.x | ✅ |
| CUDA | 12.2 (cluster) → 12.6 (container) | ✅ |
| PyTorch | 2.5-2.6 | ✅ |

---

## 🤝 Contributing

Contributions welcome! Please submit Pull Requests.

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- NVIDIA NGC for optimized containers
- Enroot and Pyxis teams
- SLURM community

---

**Version**: 2.1  
**Last Updated**: 2025-11-28
