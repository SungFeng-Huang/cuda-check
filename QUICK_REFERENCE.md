# Quick Reference

Essential commands for CUDA & Container Toolkit.

---

## ⚡ One-Liners

### Test Environment
```bash
bash scripts/check_cuda.sh
```

### Full Diagnostic
```bash
python3 scripts/check_cuda_pytorch.py
```

### Launch Development Environment
```bash
bash scripts/launch_interactive.sh
```

### Test NGC Container
```bash
bash scripts/test_nvidia_container.sh
```

---

## 🔍 Diagnostic Commands

### Inside Container

```bash
# Check GPU
nvidia-smi

# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python3 -c "import torch; print(f'Available: {torch.cuda.is_available()}')"

# Test cuFFT
python3 -c "import torch; x=torch.randn(1000).cuda(); print(torch.fft.rfft(x).shape)"

# Test STFT (audio processing)
python3 -c "
import torch
audio = torch.randn(48000).cuda()
stft = torch.stft(audio, n_fft=1024, hop_length=256, return_complex=True)
print(f'STFT: {audio.shape} -> {stft.shape}')
"
```

---

## 🐳 Docker Registry Workflow

### Build & Push (Local Machine)

```bash
# Build
docker build -t registry.com/project:v1.0 -f templates/Dockerfile .

# Test
docker run --gpus all -it registry.com/project:v1.0 nvidia-smi

# Push
docker push registry.com/project:v1.0
```

### Use on Cluster

```bash
# Configure credentials (one-time)
cat >> ~/.config/enroot/.credentials << EOF
machine registry.com login user password token
EOF

# Launch
srun --account=your-account --partition=interactive --gpus=1 \
     --container-image=registry.com/project:v1.0 \
     --container-mounts=/workspace:/workspace \
     --pty /bin/bash
```

---

## 🔧 Troubleshooting

### Quick Checks

| Issue | Command | Fix |
|-------|---------|-----|
| GPU not visible | `nvidia-smi` | Add `--gpus=1` to srun |
| CUDA unavailable | `check_cuda.sh` | Use NGC container |
| cuFFT failed | `check_cuda_pytorch.py` | Use NGC container |
| Container error | Check logs | Verify registry credentials |

---

## 📚 Documentation

- [README.md](README.md) - Overview
- [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) - Complete guide
- [docs/DOCKER_REGISTRY_GUIDE.md](docs/DOCKER_REGISTRY_GUIDE.md) - Registry setup

---

## 🎯 Recommended Workflow

### For Users

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/cuda-check.git
cd cuda-check

# 2. Update account in scripts/launch_interactive.sh
vim scripts/launch_interactive.sh
# Change: --account=your-account

# 3. Launch
bash scripts/launch_interactive.sh

# 4. Inside container, install dependencies
pip install -r /workspace/requirements.txt
```

### For Developers

```bash
# 1. Build custom container
docker build -t registry.com/project:v1.0 -f templates/Dockerfile .

# 2. Push to registry
docker push registry.com/project:v1.0

# 3. Team members use directly
srun --container-image=registry.com/project:v1.0 ...
```

---

**Version**: 2.1
