# OCR Setup Guide for Sentiment Analyzer

## Overview

The Sentiment Analyzer now supports OCR (Optical Character Recognition) to extract text from images. This allows you to upload images containing text and analyze their sentiment.

## Installation

### Step 1: Install Python Package

```bash
pip install pytesseract Pillow
```

### Step 2: Install Tesseract OCR Engine

pytesseract requires the Tesseract OCR engine to be installed on your system.

#### Windows

1. **Download Tesseract installer:**
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the latest Windows installer (e.g., `tesseract-ocr-w64-setup-5.x.x.exe`)

2. **Install Tesseract:**
   - Run the installer
   - **Important**: Note the installation path (usually `C:\Program Files\Tesseract-OCR`)
   - Add Tesseract to your PATH, or configure pytesseract to find it

3. **Configure pytesseract (if needed):**
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

#### macOS

```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### Linux (Fedora)

```bash
sudo dnf install tesseract
```

## Usage

1. **Open the Sentiment Analyzer tab** in the Streamlit viewer
2. **Select "📷 Image Upload (OCR)"** as the input method
3. **Upload an image** containing text (PNG, JPG, JPEG, GIF, BMP, TIFF)
4. **Review the extracted text** (you can edit it if needed)
5. **Click "Analyze Sentiment"** to get sentiment analysis

## Supported Image Formats

- PNG
- JPEG/JPG
- GIF
- BMP
- TIFF

## Tips for Better OCR Results

1. **Image Quality**: Use clear, high-resolution images
2. **Text Contrast**: Ensure good contrast between text and background
3. **Font Size**: Larger fonts are easier to recognize
4. **Orientation**: Text should be horizontal (not rotated)
5. **Lighting**: Avoid shadows and glare
6. **Language**: Tesseract supports multiple languages (default is English)

## Troubleshooting

### "pytesseract is not installed"
```bash
pip install pytesseract
```

### "TesseractNotFoundError"
- Make sure Tesseract OCR engine is installed
- On Windows, you may need to set the path manually:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

### Poor OCR Results
- Try preprocessing the image (increase contrast, resize)
- Ensure the image is clear and text is readable
- Check if the text language matches Tesseract's language data

### No Text Extracted
- The image might not contain readable text
- Try a different image or improve image quality
- Check if the text is too small or blurry

## Advanced: Multi-language Support

To use OCR with other languages:

1. Install language data packs for Tesseract
2. Specify language in pytesseract:
   ```python
   pytesseract.image_to_string(image, lang='spa')  # Spanish
   ```

## Example Use Cases

- Analyze sentiment from screenshots of social media posts
- Extract text from memes and analyze sentiment
- Process images of handwritten notes (if clear enough)
- Analyze text from photos of documents

