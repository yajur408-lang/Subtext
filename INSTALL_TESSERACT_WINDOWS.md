# Install Tesseract OCR on Windows - Step by Step

## Quick Installation Guide

### Step 1: Download Tesseract OCR

1. Go to: **https://github.com/UB-Mannheim/tesseract/wiki**
2. Scroll down to find the **Windows** section
3. Download: **`tesseract-ocr-w64-setup-5.x.x.exe`** (64-bit version)
   - Direct link (check for latest version): https://digi.bib.uni-mannheim.de/tesseract/

### Step 2: Install Tesseract

1. **Run the installer** (`tesseract-ocr-w64-setup-5.x.x.exe`)
2. **Follow the installation wizard**
3. **Important**: Note the installation path (usually `C:\Program Files\Tesseract-OCR`)
4. **Check the box** to add Tesseract to PATH (if available)
5. Complete the installation

### Step 3: Verify Installation

Open PowerShell and run:
```powershell
tesseract --version
```

If you see a version number, Tesseract is installed correctly!

### Step 4: Configure Python (If Auto-Detection Fails)

If the app still can't find Tesseract, add this to `viewer.py` after the imports:

```python
import pytesseract
# Set the path to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Or use the quick fix script:**
```bash
python quick_fix_tesseract.py
```

This will find Tesseract and give you the exact code to add.

### Step 5: Restart Streamlit

After installation, restart your Streamlit app:
```bash
# Stop the current app (Ctrl+C)
# Then restart:
python -m streamlit run viewer.py
```

## Alternative: Add to PATH Manually

If Tesseract is installed but not in PATH:

1. **Find Tesseract installation folder** (usually `C:\Program Files\Tesseract-OCR`)
2. **Copy the full path**
3. **Add to Windows PATH:**
   - Press `Win + X` → System → Advanced system settings
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and paste: `C:\Program Files\Tesseract-OCR`
   - Click OK on all dialogs
   - **Restart your computer** (or restart PowerShell/Command Prompt)

## Troubleshooting

### "tesseract is not installed or it's not in your PATH"

**Solution 1**: Install Tesseract (see steps above)

**Solution 2**: Manually set the path in viewer.py:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Solution 3**: Add Tesseract to PATH (see above)

### "TesseractNotFoundError"

Same as above - Tesseract is not found. Follow installation steps.

### After Installation, Still Getting Errors

1. **Restart your computer** (ensures PATH is updated)
2. **Restart Streamlit app**
3. **Check the path** - run `python quick_fix_tesseract.py` to verify

## Quick Test

After installation, test OCR:
```python
import pytesseract
from PIL import Image

# Test
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
print(pytesseract.get_tesseract_version())
```

If this works, OCR is ready!

