# Troubleshooting Guide

## Issue 1: Streamlit Command Not Recognized

**Error**: `streamlit : The term 'streamlit' is not recognized`

**Solution**: Use Python module execution instead:

```bash
python -m streamlit run viewer.py
```

Or use the provided batch file:
```bash
run_viewer.bat
```

## Issue 2: Windows Long Path Support Error

**Error**: `OSError: [Errno 2] No such file or directory` with very long paths

**Solution Options**:

### Option A: Enable Long Path Support (Recommended)

1. **Run PowerShell as Administrator**:
   - Right-click PowerShell
   - Select "Run as Administrator"

2. **Run the enable script**:
   ```powershell
   .\enable_long_paths.ps1
   ```

3. **Or manually enable**:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWord -Force
   ```

4. **Restart your computer**

### Option B: Alternative Installation (No Admin Required)

Run the alternative installation script:
```bash
python install_pytorch_alternative.py
```

This installs PyTorch to a shorter path to avoid the issue.

### Option C: Use Without PyTorch

The app works without PyTorch:
- Dataset Viewer: Fully functional
- Sentiment Analyzer: Shows installation instructions

## Issue 3: PyTorch DLL Error

**Error**: `ImportError: DLL load failed while importing _C`

**Solutions**:

1. **Install Visual C++ Redistributables**:
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart

2. **Reinstall PyTorch (CPU version)**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **The app now handles this gracefully** - it will work without PyTorch

## Quick Start Commands

### Run the Viewer:
```bash
python -m streamlit run viewer.py
```

### Fix PyTorch (if needed):
```bash
python install_pytorch_alternative.py
```

### Enable Long Paths (Admin required):
```powershell
# Run PowerShell as Administrator, then:
.\enable_long_paths.ps1
```

## Still Having Issues?

1. **Check Python version**: `python --version` (should be 3.8+)
2. **Check pip version**: `python -m pip --version`
3. **Update pip**: `python -m pip install --upgrade pip`
4. **Try virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Feature Availability

| Feature | Requires PyTorch | Works Without PyTorch |
|---------|------------------|----------------------|
| Dataset Viewer | ❌ No | ✅ Yes |
| Data Filtering | ❌ No | ✅ Yes |
| Sentiment Analyzer | ✅ Yes | ❌ No (shows instructions) |

The app is designed to work even if PyTorch isn't available!

