#!/bin/bash
# Quick CUDA and PyTorch compatibility check

echo "========================================================"
echo "  Quick CUDA & PyTorch Version Check"
echo "========================================================"

echo ""
echo "1. NVIDIA Driver Version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "❌ nvidia-smi not available"

echo ""
echo "2. CUDA Runtime Version:"
nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "❌ Could not detect"

echo ""
echo "3. PyTorch Version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch not installed"

echo ""
echo "4. PyTorch CUDA Version:"
python3 -c "import torch; print(f'Built with CUDA: {torch.version.cuda}')" 2>/dev/null || echo "❌ Could not detect"

echo ""
echo "5. CUDA Available in PyTorch:"
python3 -c "import torch; print('✅ CUDA Available' if torch.cuda.is_available() else '❌ CUDA NOT Available')" 2>/dev/null || echo "❌ Error"

echo ""
echo "6. GPU Count:"
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('N/A')" 2>/dev/null

echo ""
echo "7. Quick cuFFT Test (for Audio Processing):"
python3 -c "
import torch
if torch.cuda.is_available():
    try:
        x = torch.randn(1000).cuda()
        y = torch.fft.rfft(x)
        print('✅ cuFFT working (FFT test passed)')
    except Exception as e:
        print(f'❌ cuFFT error: {e}')
else:
    print('⚠️  CUDA not available, skipping test')
" 2>/dev/null || echo "❌ Test failed"

echo ""
echo "========================================================"
echo "Run 'python3 check_cuda_pytorch.py' for detailed check"
echo "  (includes STFT/ISTFT and Mel-Spectrogram tests)"
echo "========================================================"

