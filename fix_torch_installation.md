# Fixing PyTorch DLL Error on Windows

If you're getting the error: `ImportError: DLL load failed while importing _C`, follow these steps:

## Solution 1: Install Visual C++ Redistributables (Recommended)

1. Download and install Microsoft Visual C++ Redistributable:
   - **Direct link**: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Or search for "Visual C++ Redistributable 2015-2022" on Microsoft's website

2. After installing, restart your computer

3. Try running the app again

## Solution 2: Reinstall PyTorch (CPU-only version)

If Solution 1 doesn't work, reinstall PyTorch:

```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Install CPU-only version (more stable on Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Solution 3: Use Alternative Installation Method

Try installing PyTorch using conda (if you have Anaconda/Miniconda):

```bash
conda install pytorch cpuonly -c pytorch
```

## Solution 4: Use Without PyTorch (Limited Functionality)

The app will still work for viewing the dataset, but the sentiment analyzer will be disabled.

You can use the dataset viewer tab which doesn't require PyTorch.

## Verification

After fixing, verify the installation:

```python
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

If this works without errors, you're good to go!

## Alternative: Use Online Sentiment Analysis

If PyTorch continues to cause issues, you can:
1. Use the dataset viewer (no PyTorch needed)
2. Use online sentiment analysis tools
3. Run the analysis on a different machine/cloud service

