@echo off
REM Simple batch script to install PyTorch avoiding long path issues
REM This installs to user directory which has shorter paths

echo ========================================
echo PyTorch Installation (Avoiding Long Paths)
echo ========================================
echo.

echo Step 1: Uninstalling existing PyTorch...
python -m pip uninstall torch torchvision torchaudio -y
echo.

echo Step 2: Installing PyTorch to user directory (shorter path)...
python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo.

echo Step 3: Verifying installation...
python -c "import torch; print('SUCCESS: PyTorch', torch.__version__, 'is installed!')"
echo.

if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo Installation successful!
    echo ========================================
    echo.
    echo You can now use the Sentiment Analyzer in the viewer.
) else (
    echo ========================================
    echo Installation may have issues.
    echo ========================================
    echo.
    echo Try enabling Long Path support:
    echo 1. Run PowerShell as Administrator
    echo 2. Run: .\enable_long_paths.ps1
    echo 3. Restart your computer
    echo 4. Try installing again
)

pause

