"""Test Tesseract OCR configuration"""

import pytesseract

# Set the path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract is working! Version: {version}")
    print("\n✅ OCR is ready to use in the Streamlit app!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease check:")
    print("1. Tesseract is installed at: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    print("2. If installed elsewhere, update the path in viewer.py")

