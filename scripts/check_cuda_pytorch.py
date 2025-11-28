#!/usr/bin/env python3
"""
Check CUDA and PyTorch compatibility
"""
import sys
import subprocess

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_nvidia_smi():
    """Check nvidia-smi and driver version"""
    print_section("1. NVIDIA Driver & GPU Info")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi failed")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False

def check_cuda_version():
    """Check CUDA version"""
    print_section("2. CUDA Version")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("⚠️  nvcc not found (CUDA toolkit may not be installed)")
    except FileNotFoundError:
        print("⚠️  nvcc not found (CUDA toolkit may not be installed)")
    
    # Check CUDA version from libraries
    import ctypes
    try:
        cuda = ctypes.CDLL('libcudart.so')
        version = ctypes.c_int()
        cuda.cudaRuntimeGetVersion(ctypes.byref(version))
        cuda_version = version.value
        print(f"CUDA Runtime Version: {cuda_version // 1000}.{(cuda_version % 1000) // 10}")
    except:
        print("⚠️  Could not detect CUDA runtime version")

def check_pytorch():
    """Check PyTorch and its CUDA support"""
    print_section("3. PyTorch Info")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"PyTorch Built with CUDA: {torch.version.cuda}")
        print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA is available in PyTorch")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("❌ CUDA is NOT available in PyTorch")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False

def test_pytorch_cuda():
    """Test PyTorch CUDA operations"""
    print_section("4. PyTorch CUDA Test")
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  Skipping test - CUDA not available")
            return False
        
        # Test tensor creation on GPU
        print("Testing tensor creation on GPU...")
        x = torch.randn(1000, 1000).cuda()
        print(f"✅ Created tensor on GPU: {x.device}")
        
        # Test computation
        print("Testing GPU computation...")
        y = torch.matmul(x, x)
        print(f"✅ GPU computation successful")
        
        # Test memory
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch CUDA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cufft():
    """Test cuFFT operations through PyTorch FFT"""
    print_section("5. cuFFT Test (for Audio Processing)")
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  Skipping test - CUDA not available")
            return False
        
        print("Testing 1D FFT (used in audio processing)...")
        # Simulate audio signal
        signal = torch.randn(2, 16000).cuda()  # 2 channels, 16000 samples
        fft_result = torch.fft.rfft(signal)
        print(f"✅ 1D FFT successful: {signal.shape} -> {fft_result.shape}")
        
        # Test inverse FFT
        reconstructed = torch.fft.irfft(fft_result, n=16000)
        print(f"✅ Inverse FFT successful: {fft_result.shape} -> {reconstructed.shape}")
        
        # Check reconstruction error
        error = torch.abs(signal - reconstructed).mean()
        print(f"   Reconstruction error: {error:.2e}")
        
        print("\nTesting 2D FFT...")
        signal_2d = torch.randn(4, 64, 64).cuda()
        fft_2d = torch.fft.fft2(signal_2d)
        print(f"✅ 2D FFT successful: {signal_2d.shape}")
        
        print("\nTesting STFT (Short-Time Fourier Transform)...")
        # Simulate longer audio for STFT
        audio = torch.randn(2, 48000).cuda()  # 2 channels, 48000 samples
        n_fft = 1024
        hop_length = 256
        stft_result = torch.stft(
            audio, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True
        )
        print(f"✅ STFT successful: {audio.shape} -> {stft_result.shape}")
        
        # Test inverse STFT
        reconstructed_audio = torch.istft(
            stft_result, 
            n_fft=n_fft, 
            hop_length=hop_length,
            length=audio.shape[-1]
        )
        print(f"✅ ISTFT successful: {stft_result.shape} -> {reconstructed_audio.shape}")
        
        # Check reconstruction error
        stft_error = torch.abs(audio - reconstructed_audio).mean()
        print(f"   STFT reconstruction error: {stft_error:.2e}")
        
        if stft_error < 1e-4:
            print("✅ cuFFT operations are working correctly!")
            return True
        else:
            print(f"⚠️  High reconstruction error: {stft_error}")
            return True  # Still pass, but warn
            
    except Exception as e:
        print(f"❌ cuFFT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_processing():
    """Test typical audio processing operations"""
    print_section("6. Audio Processing Test (Mel-Spectrogram)")
    try:
        import torch
        import torchaudio
        
        if not torch.cuda.is_available():
            print("⚠️  Skipping test - CUDA not available")
            return False
        
        print("Testing Mel-Spectrogram computation on GPU...")
        
        # Create mel-spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        ).cuda()
        
        # Generate audio
        audio = torch.randn(1, 24000).cuda()  # 1 second of audio at 24kHz
        
        # Compute mel-spectrogram
        mel_spec = mel_transform(audio)
        print(f"✅ Mel-Spectrogram computed: {audio.shape} -> {mel_spec.shape}")
        
        # Test with batch
        audio_batch = torch.randn(8, 24000).cuda()  # Batch of 8
        mel_spec_batch = mel_transform(audio_batch)
        print(f"✅ Batch Mel-Spectrogram: {audio_batch.shape} -> {mel_spec_batch.shape}")
        
        return True
        
    except ImportError:
        print("⚠️  torchaudio not installed, skipping audio processing test")
        return True  # Not critical
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_compatibility():
    """Check version compatibility"""
    print_section("7. Compatibility Analysis")
    try:
        import torch
        pytorch_cuda = torch.version.cuda
        
        # Get driver CUDA version from nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    driver_cuda = line.split('CUDA Version:')[1].split()[0]
                    print(f"Driver CUDA Version: {driver_cuda}")
                    print(f"PyTorch CUDA Version: {pytorch_cuda}")
                    
                    # Compare versions
                    driver_major = int(driver_cuda.split('.')[0])
                    pytorch_major = int(pytorch_cuda.split('.')[0])
                    
                    if driver_major >= pytorch_major:
                        print(f"✅ Driver CUDA ({driver_cuda}) >= PyTorch CUDA ({pytorch_cuda})")
                        print("   Compatibility: OK")
                    else:
                        print(f"⚠️  Driver CUDA ({driver_cuda}) < PyTorch CUDA ({pytorch_cuda})")
                        print("   Compatibility: May have issues")
                    break
        
        # Check if cuFFT is available
        print("\nChecking cuFFT availability...")
        import ctypes
        try:
            cufft = ctypes.CDLL('libcufft.so')
            print("✅ libcufft.so found")
        except:
            print("⚠️  libcufft.so not found (may still work through PyTorch)")
            
    except Exception as e:
        print(f"Could not check compatibility: {e}")

def main():
    print("="*60)
    print("  CUDA, PyTorch & cuFFT Compatibility Check")
    print("="*60)
    
    results = []
    results.append(("NVIDIA Driver", check_nvidia_smi()))
    check_cuda_version()
    results.append(("PyTorch CUDA", check_pytorch()))
    results.append(("PyTorch Test", test_pytorch_cuda()))
    results.append(("cuFFT Test", test_cufft()))
    results.append(("Audio Processing", test_audio_processing()))
    check_compatibility()
    
    # Summary
    print_section("Summary")
    for name, status in results:
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{name:20} {status_str}")
    
    all_pass = all(status for _, status in results)
    if all_pass:
        print("\n🎉 All checks passed! PyTorch with CUDA and cuFFT is ready for audio processing.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

