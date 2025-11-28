#!/bin/bash
# Quick test to verify NVIDIA official container works

echo "========================================================"
echo "  Testing NVIDIA Official PyTorch Container"
echo "========================================================"
echo ""
echo "This will:"
echo "  1. Pull nvcr.io/nvidia/pytorch:24.10-py3 (if not cached)"
echo "  2. Test GPU access"
echo "  3. Test PyTorch CUDA"
echo "  4. Test cuFFT"
echo ""
echo "First run may take a few minutes to download the container."
echo "========================================================"
echo ""

srun --account=your-account --partition=interactive --gpus=1 --time=5 \
  --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
  bash -c '
echo "=== 1. GPU Info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "=== 2. CUDA Version ==="
python3 -c "import torch; print(f\"Driver CUDA: {torch.version.cuda}\"); print(f\"PyTorch version: {torch.__version__}\")"

echo ""
echo "=== 3. PyTorch CUDA Test ==="
python3 -c "
import torch
if torch.cuda.is_available():
    print(f\"✅ CUDA Available: {torch.cuda.get_device_name(0)}\")
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f\"✅ GPU computation successful\")
else:
    print(\"❌ CUDA not available\")
    exit(1)
"

echo ""
echo "=== 4. cuFFT Test ==="
python3 -c "
import torch
if torch.cuda.is_available():
    try:
        # Test 1D FFT
        signal = torch.randn(16000).cuda()
        fft_result = torch.fft.rfft(signal)
        reconstructed = torch.fft.irfft(fft_result, n=16000)
        error = torch.abs(signal - reconstructed).mean()
        print(f\"✅ 1D FFT successful (error: {error:.2e})\")
        
        # Test STFT
        audio = torch.randn(48000).cuda()
        stft_result = torch.stft(audio, n_fft=1024, hop_length=256, return_complex=True)
        audio_recon = torch.istft(stft_result, n_fft=1024, hop_length=256, length=48000)
        stft_error = torch.abs(audio - audio_recon).mean()
        print(f\"✅ STFT/ISTFT successful (error: {stft_error:.2e})\")
        
        print(\"\\n🎉 All tests passed! Container is ready for audio processing.\")
    except Exception as e:
        print(f\"❌ cuFFT test failed: {e}\")
        exit(1)
else:
    print(\"❌ CUDA not available\")
    exit(1)
"

echo ""
echo "========================================================"
echo "  ✅ Container verification complete!"
echo "========================================================"
echo ""
echo "You can now use interactive_nvidia_official.sh for your work."
'

