"""
Setup script to download required NLTK data for TextBlob
Run this once before using the main pipeline
"""

import nltk

print("Downloading NLTK data for TextBlob...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK data downloaded successfully!")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("Please run: python -m nltk.downloader punkt brown wordnet")

