"""
Quick fix script to configure Tesseract OCR path
Run this to find and configure Tesseract on Windows
"""

import os
import platform

print("=" * 70)
print("Tesseract OCR Configuration Helper")
print("=" * 70)
print()

if platform.system() != 'Windows':
    print("This script is for Windows. On other systems, ensure Tesseract is in PATH.")
    print("Mac: brew install tesseract")
    print("Linux: sudo apt-get install tesseract-ocr")
    exit()

# Common installation paths
common_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
]

print("Searching for Tesseract OCR...")
found_path = None

for path in common_paths:
    if os.path.exists(path):
        found_path = path
        print(f"✓ Found Tesseract at: {path}")
        break

if not found_path:
    print("✗ Tesseract not found in common locations")
    print()
    print("Please install Tesseract OCR:")
    print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("2. Install it (note the installation path)")
    print("3. Run this script again or manually set the path")
    print()
    print("After installation, add this to viewer.py:")
    print("import pytesseract")
    print("pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
else:
    print()
    print("=" * 70)
    print("Configuration Code for viewer.py")
    print("=" * 70)
    print()
    print("Add this code after the pytesseract import in viewer.py:")
    print()
    print(f"import pytesseract")
    print(f"pytesseract.pytesseract.tesseract_cmd = r'{found_path}'")
    print()
    print("=" * 70)
    print("Or copy this complete block:")
    print("=" * 70)
    print()
    print(f"""# Configure Tesseract path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'{found_path}'
OCR_AVAILABLE = True
TESSERACT_CONFIGURED = True""")
    print()

