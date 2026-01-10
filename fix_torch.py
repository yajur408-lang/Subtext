"""
Quick script to fix PyTorch installation on Windows
Run this if you're getting DLL errors
"""

import subprocess
import sys

print("=" * 60)
print("PyTorch Installation Fixer for Windows")
print("=" * 60)
print()

# Step 1: Uninstall existing PyTorch
print("Step 1: Uninstalling existing PyTorch...")
try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                  check=False)
    print("✓ Uninstalled existing PyTorch")
except Exception as e:
    print(f"⚠ Warning: {e}")

print()

# Step 2: Install CPU-only version
print("Step 2: Installing CPU-only PyTorch (more stable on Windows)...")
try:
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ], check=True)
    print("✓ Installed CPU-only PyTorch")
except Exception as e:
    print(f"✗ Error installing PyTorch: {e}")
    print("\nPlease try manually:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

print()

# Step 3: Verify installation
print("Step 3: Verifying installation...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} installed successfully!")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nYou may need to:")
    print("1. Install Visual C++ Redistributables:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Restart your computer")
    sys.exit(1)

print()
print("=" * 60)
print("Installation complete! You can now run the viewer.")
print("=" * 60)



