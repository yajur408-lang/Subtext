"""
Alternative PyTorch installation script that avoids long path issues
Uses a shorter installation path and CPU-only version
"""

import subprocess
import sys
import os

print("=" * 70)
print("Alternative PyTorch Installation (Avoids Long Path Issues)")
print("=" * 70)
print()

# Method 1: Try installing to user site-packages (shorter path)
print("Method 1: Installing PyTorch to user site-packages...")
try:
    # Use --user flag to install to shorter path
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "--user",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Successfully installed PyTorch to user directory!")
        print("\nVerifying installation...")
        try:
            import torch
            print(f"✓ PyTorch {torch.__version__} is working!")
            sys.exit(0)
        except ImportError as e:
            print(f"⚠ Installed but import failed: {e}")
            print("Trying alternative method...")
    else:
        print("⚠ Method 1 failed, trying alternative...")
        print(result.stderr)
except Exception as e:
    print(f"⚠ Error: {e}")
    print("Trying alternative method...")

print()
print("=" * 70)
print("Method 2: Using pip cache and shorter paths...")
print("=" * 70)
print()

# Method 2: Set shorter temp directory
try:
    # Create a shorter temp directory
    short_temp = os.path.join(os.environ.get('TEMP', 'C:\\temp'), 'pip_short')
    os.makedirs(short_temp, exist_ok=True)
    
    env = os.environ.copy()
    env['TMP'] = short_temp
    env['TEMP'] = short_temp
    
    print(f"Using temporary directory: {short_temp}")
    
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--no-cache-dir"  # Avoid cache issues
    ], env=env, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Successfully installed PyTorch!")
        print("\nVerifying installation...")
        try:
            import torch
            print(f"✓ PyTorch {torch.__version__} is working!")
            sys.exit(0)
        except ImportError as e:
            print(f"✗ Installed but import failed: {e}")
    else:
        print("✗ Installation failed")
        print(result.stderr)
        
except Exception as e:
    print(f"✗ Error: {e}")

print()
print("=" * 70)
print("If both methods failed, try these manual steps:")
print("=" * 70)
print("1. Enable Long Path support (run enable_long_paths.ps1 as Administrator)")
print("2. Restart your computer")
print("3. Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
print()
print("OR use the app without PyTorch (dataset viewer will still work)")

