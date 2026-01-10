# Quick Fix for PyTorch Installation

## The Problem
Windows has a 260-character path limit, and PyTorch has very long file paths that exceed this limit.

## Easiest Solution (No Admin Required)

### Option 1: Use the Batch Script (Easiest)
Just double-click this file:
```
INSTALL_PYTORCH_SIMPLE.bat
```

This installs PyTorch to your user directory (shorter path) and avoids the long path issue.

### Option 2: Manual Installation (User Directory)
Open PowerShell or Command Prompt and run:
```bash
pip uninstall torch torchvision torchaudio -y
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

The `--user` flag installs to a shorter path that avoids the 260-character limit.

## Permanent Solution (Requires Admin)

If the above doesn't work, enable Windows Long Path support:

1. **Right-click PowerShell** → **Run as Administrator**

2. **Run this command**:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWord -Force
   ```

3. **Restart your computer**

4. **Then install PyTorch normally**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## Verify Installation

After installation, test it:
```bash
python -c "import torch; print('PyTorch', torch.__version__, 'works!')"
```

If you see a version number, you're good to go!

## What Works Without PyTorch

Remember: **The Dataset Viewer works perfectly without PyTorch!**
- View tweets ✅
- Filter by sentiment, topic, target ✅
- Search tweets ✅
- Export data ✅

Only the Sentiment Analyzer needs PyTorch.

