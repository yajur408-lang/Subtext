"""
Streamlit Interactive Tweet Viewer
Filter and visualize tweets by sentiment and target labels
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
from scipy.sparse import hstack, csr_matrix
from PIL import Image
import io
import requests
from datetime import datetime, timedelta

# Hardcoded Google GenAI API Key
HARDCODED_GOOGLE_API_KEY = "AIzaSyBfnBHzwp3-JjCiPw_5VTOf6FHQyEq8RZw"

# Try to import pytesseract for OCR
OCR_AVAILABLE = False
TESSERACT_CONFIGURED = False

try:
    import pytesseract
    OCR_AVAILABLE = True
    
    # Manual Tesseract configuration (set your installation path here)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Try to configure Tesseract path for Windows (common installation location)
    import platform
    if platform.system() == 'Windows':
        # First check if manual path exists
        if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            TESSERACT_CONFIGURED = True
        else:
            # Try other common paths
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    TESSERACT_CONFIGURED = True
                    break
    
    # Test if Tesseract is accessible
    if TESSERACT_CONFIGURED or platform.system() != 'Windows':
        try:
            pytesseract.get_tesseract_version()
            TESSERACT_CONFIGURED = True
        except:
            TESSERACT_CONFIGURED = False
except ImportError:
    OCR_AVAILABLE = False
except Exception:
    # pytesseract installed but Tesseract engine not found
    OCR_AVAILABLE = True
    TESSERACT_CONFIGURED = False

# ML model imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False

# Google Generative AI imports
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    TORCH_AVAILABLE = False
    # Don't show warning on initial load - only show when needed
    # The app works fine without PyTorch for dataset viewing

# Try to import transformers
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    TRANSFORMERS_AVAILABLE = False
    # Don't show warning on initial load - only show when needed

# gTTS for text-to-speech tab (optional)
try:
    from gtts import gTTS
    from io import BytesIO
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="subtext",
    page_icon=None,
    layout="wide"
)

# Global monochrome theme: white backgrounds, black text, black borders, Lexend font, subtle shadows
def inject_monochrome_theme():
    # Comprehensive CSS rules to force white theme everywhere
    css_rules = [
        "*{font-family:'Lexend',sans-serif!important}",
        "h1{font-size:4rem!important;font-weight:bold!important;color:#006B75!important}",
        "[data-testid='stMarkdownContainer'] h1{font-size:4rem!important;font-weight:bold!important;color:#006B75!important}",
        "[data-testid='stSidebar'] h1,[data-testid='stSidebar'] h2,[data-testid='stSidebar'] h3,[data-testid='stSidebar'] h4,[data-testid='stSidebar'] h5,[data-testid='stSidebar'] h6{font-size:1.2rem!important;font-weight:normal!important;color:#000000!important}",
        "[data-testid='stSidebar'] *{font-size:0.9rem!important}",
        "[data-testid='stSidebar'] label{font-size:0.9rem!important}",
        "[data-testid='stSidebar'] .stMarkdown{font-size:0.9rem!important}",
        ":root{--background-color:#E1F4F7!important;--text-color:#000000!important;--primary-color:#000000!important;color-scheme:light!important}",
        "[data-theme='dark'],[data-baseweb='theme']{--background-color:#E1F4F7!important;--text-color:#000000!important}",
        "html{background-color:#E1F4F7!important;color:#000000!important}",
        "body{background-color:#E1F4F7!important;color:#000000!important}",
        "[data-testid='stAppViewContainer']{background-color:#E1F4F7!important;color:#000000!important}",
        "[data-testid='stHeader']{background-color:#E1F4F7!important;color:#000000!important}",
        "[data-testid='stSidebar']{background-color:#E1F4F7!important;color:#000000!important;border-right:1px solid #e4e4e7!important;box-shadow:2px 0 12px rgba(0,0,0,0.08)!important}",
        "[data-testid='stSidebar'] *{background-color:#E1F4F7!important;color:#000000!important}",
        ".block-container{background-color:#E1F4F7!important;color:#000000!important;padding:2rem!important;max-width:1200px!important}",
        ".main{background-color:#E1F4F7!important;color:#000000!important}",
        "[data-baseweb='base']{background-color:#E1F4F7!important;color:#000000!important}",
        "[data-baseweb='root']{background-color:#E1F4F7!important;color:#000000!important}",
        "[class*='stApp']{background-color:#E1F4F7!important;color:#000000!important}",
        "[class*='stAppViewContainer']{background-color:#E1F4F7!important;color:#000000!important}",
        "*{color:#000000!important}",
        "div{background-color:#E1F4F7!important;color:#000000!important}",
        "section{background-color:#E1F4F7!important;color:#000000!important}",
        "article{background-color:#E1F4F7!important;color:#000000!important}",
        "aside{background-color:#E1F4F7!important;color:#000000!important}",
        "header{background-color:#E1F4F7!important;color:#000000!important}",
        "footer{background-color:#E1F4F7!important;color:#000000!important}",
        "main{background-color:#E1F4F7!important;color:#000000!important}",
        "nav{background-color:#E1F4F7!important;color:#000000!important}",
        "[class*='st']{background-color:#E1F4F7!important;color:#000000!important}",
        "[id*='st']{background-color:#E1F4F7!important;color:#000000!important}",
        "textarea{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;padding:8px 12px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important;font-size:14px!important}",
        "textarea:focus{outline:none!important;border-color:#006B75!important;box-shadow:0 0 0 3px rgba(0,107,117,0.1)!important;ring:2px solid rgba(0,107,117,0.1)!important}",
        "textarea:hover{border-color:#a1a1aa!important}",
        "input{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;padding:8px 12px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important;font-size:14px!important}",
        "input:focus{outline:none!important;border-color:#006B75!important;box-shadow:0 0 0 3px rgba(0,107,117,0.1)!important;ring:2px solid rgba(0,107,117,0.1)!important}",
        "input:hover{border-color:#a1a1aa!important}",
        "select{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;padding:8px 12px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important;font-size:14px!important}",
        "select:focus{outline:none!important;border-color:#006B75!important;box-shadow:0 0 0 3px rgba(0,107,117,0.1)!important}",
        "select:hover{border-color:#a1a1aa!important}",
        "button{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;padding:8px 16px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important;font-size:14px!important;font-weight:500!important;cursor:pointer!important}",
        "button:hover{background-color:#fafafa!important;border-color:#a1a1aa!important;box-shadow:0 2px 4px rgba(0,0,0,0.08)!important;transform:translateY(-1px)!important}",
        "button:active{transform:translateY(0)!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important}",
        ".stTextInput>div>div>input{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;padding:8px 12px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important;font-size:14px!important}",
        ".stTextInput>div>div>input:focus{outline:none!important;border-color:#006B75!important;box-shadow:0 0 0 3px rgba(0,107,117,0.1)!important}",
        ".stTextInput>div>div>input:hover{border-color:#a1a1aa!important}",
        ".stTextArea textarea{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;padding:8px 12px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important;font-size:14px!important}",
        ".stTextArea textarea:focus{outline:none!important;border-color:#006B75!important;box-shadow:0 0 0 3px rgba(0,107,117,0.1)!important}",
        ".stTextArea textarea:hover{border-color:#a1a1aa!important}",
        "div[data-baseweb='select']{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important}",
        "div[data-baseweb='select']:focus-within{border-color:#006B75!important;box-shadow:0 0 0 3px rgba(0,107,117,0.1)!important}",
        "div[data-baseweb='select']:hover{border-color:#a1a1aa!important}",
        ".stButton>button{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;padding:8px 16px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important;font-size:14px!important;font-weight:500!important;cursor:pointer!important}",
        ".stButton>button:hover{background-color:#fafafa!important;border-color:#a1a1aa!important;box-shadow:0 2px 4px rgba(0,0,0,0.08)!important;transform:translateY(-1px)!important}",
        ".stButton>button:active{transform:translateY(0)!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important}",
        ".stSelectbox>div>div>select{background-color:#ffffff!important;color:#000000!important;border:1px solid #000000!important}",
        ".stMultiSelect>div>div>select{background-color:#ffffff!important;color:#000000!important;border:1px solid #000000!important}",
        ".stFileUploader>div{background-color:#ffffff!important;border:1px dashed #e4e4e7!important;border-radius:6px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;transition:all 0.2s ease!important;padding:16px!important}",
        ".stFileUploader>div:hover{border-color:#a1a1aa!important;background-color:#fafafa!important}",
        ".stFileUploader *{color:#000000!important}",
        "div[data-testid='stAlert']{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;box-shadow:0 2px 8px rgba(0,0,0,0.08)!important;padding:12px 16px!important}",
        "div[data-testid='stAlert'] *{color:#000000!important}",
        "div[data-testid='stMetricValue']{color:#000000!important}",
        "div[data-testid='stMetricLabel']{color:#000000!important}",
        "div[data-testid='stMetric']{background-color:#ffffff!important;border:1px solid #e4e4e7!important;border-radius:6px!important;box-shadow:0 1px 3px rgba(0,0,0,0.06)!important;padding:16px!important;transition:all 0.2s ease!important}",
        "div[data-testid='stMetric']:hover{box-shadow:0 4px 12px rgba(0,0,0,0.1)!important;transform:translateY(-2px)!important}",
        ".stDataFrame{border:1px solid #e4e4e7!important;border-radius:6px!important;box-shadow:0 1px 3px rgba(0,0,0,0.06)!important;overflow:hidden!important}",
        ".stTable{border:1px solid #e4e4e7!important;border-radius:6px!important;box-shadow:0 1px 3px rgba(0,0,0,0.06)!important;overflow:hidden!important}",
        ".stDataFrame *{color:#000000!important}",
        ".stTable *{color:#000000!important}",
        "pre{background-color:#ffffff!important;color:#000000!important;border:1px solid #000000!important;box-shadow:0 2px 4px rgba(0,0,0,0.1)!important}",
        "code{background-color:#ffffff!important;color:#000000!important;border:1px solid #000000!important;box-shadow:0 2px 4px rgba(0,0,0,0.1)!important}",
        ".stMarkdown{color:#000000!important}",
        ".stText{color:#000000!important}",
        ".stWrite{color:#000000!important}",
        ".stHeader{color:#000000!important}",
        ".stSubheader{color:#000000!important}",
        ".streamlit-expanderHeader{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:6px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;padding:12px 16px!important;transition:all 0.2s ease!important;cursor:pointer!important}",
        ".streamlit-expanderHeader:hover{background-color:#fafafa!important;border-color:#a1a1aa!important}",
        ".streamlit-expanderContent{background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important;border-radius:0 0 6px 6px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;padding:16px!important;margin-top:-1px!important}",
        ".stRadio>div{box-shadow:0 1px 3px rgba(0,0,0,0.08)!important}",
        ".stCheckbox>div{box-shadow:0 1px 3px rgba(0,0,0,0.08)!important}",
        "[data-testid='column']{box-shadow:0 1px 3px rgba(0,0,0,0.08)!important}",
        ".stTabs [data-baseweb='tab-list']{box-shadow:0 1px 3px rgba(0,0,0,0.08)!important}",
        ".stTabs [data-baseweb='tab']{border-radius:6px!important;transition:all 0.2s ease!important;margin:0 4px!important;padding:8px 16px!important;font-weight:500!important;font-size:14px!important}",
        ".stTabs [data-baseweb='tab']:hover{background-color:#fafafa!important;box-shadow:0 2px 4px rgba(0,0,0,0.08)!important;transform:translateY(-1px)!important}",
        ".stTabs [data-baseweb='tab'][aria-selected='true']{border-radius:6px!important;background-color:#006B75!important;color:#ffffff!important;transition:all 0.2s ease!important;box-shadow:0 2px 8px rgba(0,107,117,0.25)!important;font-weight:600!important}",
        ".stTabs [data-baseweb='tab'][aria-selected='false']{border-radius:6px!important;background-color:#ffffff!important;color:#09090b!important;border:1px solid #e4e4e7!important}",
        ".stTabs [data-baseweb='tab']:active{transform:translateY(0)!important;transition:all 0.1s ease!important}",
        "[style*='background-color']{background-color:#E1F4F7!important}",
        "[style*='background']{background-color:#E1F4F7!important}",
        "script{display:none!important;visibility:hidden!important;height:0!important;width:0!important;position:absolute!important;opacity:0!important}",
        "style{display:none!important;visibility:hidden!important}",
        "[class*='stMarkdown'] script{display:none!important;visibility:hidden!important;height:0!important;width:0!important;position:absolute!important;opacity:0!important}",
        "[class*='stMarkdown'] style{display:none!important;visibility:hidden!important}"
    ]
    css_content = ''.join(css_rules)
    
    # Inject via components.html to completely hide the code (height=0 makes it invisible)
    try:
        import streamlit.components.v1 as components
        
        # Create the HTML with CSS and JavaScript - use format() instead of f-string to avoid brace issues
        js_func = """
        (function(){
            document.documentElement.style.colorScheme="light";
            document.documentElement.style.backgroundColor="#E1F4F7";
            document.body.style.backgroundColor="#E1F4F7";
            document.body.style.color="#000000";
            
            function forceWhiteTheme(){
                var els=document.querySelectorAll("*");
                for(var i=0;i<els.length;i++){
                    var el=els[i];
                    if(el.tagName!=="SCRIPT"&&el.tagName!=="STYLE"&&el.tagName!=="LINK"&&el.tagName!=="META"){
                        var bg=window.getComputedStyle(el).backgroundColor;
                        var computedColor=window.getComputedStyle(el).color;
                        if(bg&&(bg.indexOf("rgb(14")>=0||bg.indexOf("rgb(19")>=0||bg.indexOf("rgb(0, 0, 0")>=0||bg==="rgb(0, 0, 0)"||bg.indexOf("rgba(0")>=0||bg.indexOf("#0")>=0||bg.indexOf("#1")>=0)){
                            el.style.setProperty("background-color","#E1F4F7","important");
                            el.style.setProperty("color","#000000","important");
                        }
                        if(computedColor&&(computedColor.indexOf("rgb(255")>=0||computedColor.indexOf("rgb(14")>=0||computedColor.indexOf("rgb(19")>=0)){
                            el.style.setProperty("color","#000000","important");
                        }
                    }
                }
            }
            
            forceWhiteTheme();
            setInterval(forceWhiteTheme,50);
            setTimeout(forceWhiteTheme,100);
            setTimeout(forceWhiteTheme,500);
            setTimeout(forceWhiteTheme,1000);
        })();
        """
        
        html_content = """
        <link href="https://fonts.googleapis.com/css2?family=Lexend:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
        <style>
        """ + css_content + """
        script{display:none!important;visibility:hidden!important;height:0!important;width:0!important;position:absolute!important;opacity:0!important}
        </style>
        <script>
        """ + js_func + """
        </script>
        """
        
        components.html(html_content, height=0, scrolling=False)
        
        # Also inject CSS directly via markdown for immediate effect
        st.markdown(
            f"""
            <link href="https://fonts.googleapis.com/css2?family=Lexend:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
            <style>
            {css_content}
            script{{display:none!important;visibility:hidden!important;height:0!important;width:0!important;position:absolute!important;opacity:0!important}}
            </style>
            <script>
            setTimeout(function(){{
                var scripts=document.querySelectorAll("script");
                for(var i=0;i<scripts.length;i++){{
                    if(scripts[i].textContent.indexOf("forceWhiteTheme")>=0||scripts[i].textContent.indexOf("colorScheme")>=0){{
                        scripts[i].style.display="none";
                        scripts[i].style.visibility="hidden";
                        scripts[i].style.height="0";
                        scripts[i].style.width="0";
                        scripts[i].style.position="absolute";
                        scripts[i].style.opacity="0";
                    }}
                }}
                var markdownDivs=document.querySelectorAll("[class*='stMarkdown']");
                for(var i=0;i<markdownDivs.length;i++){{
                    var scripts=markdownDivs[i].querySelectorAll("script");
                    for(var j=0;j<scripts.length;j++){{
                        scripts[j].style.display="none";
                        scripts[j].style.visibility="hidden";
                        scripts[j].style.height="0";
                        scripts[j].style.width="0";
                        scripts[j].style.position="absolute";
                        scripts[j].style.opacity="0";
                    }}
                }}
            }},10);
            </script>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        # Fallback: inject CSS directly and hide script tags
        # Use regular string instead of f-string to avoid brace escaping issues
        js_code = """
            document.documentElement.style.colorScheme="light";
            document.documentElement.style.backgroundColor="#E1F4F7";
            document.body.style.backgroundColor="#E1F4F7";
            document.body.style.color="#000000";
            function f(){
            var e=document.querySelectorAll("*");
            for(var i=0;i<e.length;i++){
            var el=e[i];
            if(el.tagName!=="SCRIPT"&&el.tagName!=="STYLE"&&el.tagName!=="LINK"&&el.tagName!=="META"){
            var bg=window.getComputedStyle(el).backgroundColor;
            if(bg&&(bg.indexOf("rgb(14")>=0||bg.indexOf("rgb(19")>=0||bg.indexOf("rgb(0")>=0||bg.indexOf("#0")>=0||bg.indexOf("#1")>=0)){
            el.style.setProperty("background-color","#E1F4F7","important");
            el.style.setProperty("color","#000000","important");
            }}}
            }
            f();
            setInterval(f,50);
            setTimeout(f,100);
            setTimeout(f,500);
            setTimeout(f,1000);
            """
        
        st.markdown(
            f"""
            <link href="https://fonts.googleapis.com/css2?family=Lexend:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
            <style>
            {css_content}
            script{{display:none!important;visibility:hidden!important;height:0!important;width:0!important;position:absolute!important;opacity:0!important}}
            </style>
            <script>
            {js_code}
            </script>
            """,
            unsafe_allow_html=True,
        )

# Color mapping for sentiments (monochrome theme)
SENTIMENT_COLORS = {
    'Positive': '#ffffff',
    'Negative': '#ffffff',
    'Neutral': '#ffffff',
    'Irrelevant': '#ffffff'
}

TARGET_COLORS = {
    1: '#ffffff',
    0: '#ffffff'
}

# Extended sentiment colors (all white background; borders/text handle emphasis)
EXTENDED_SENTIMENT_COLORS = {
    'Sarcastic': '#ffffff',
    'Playful': '#ffffff',
    'Funny': '#ffffff',
    'Flirty': '#ffffff',
    'Angry': '#ffffff',
    'Sadness': '#ffffff',
    'Dark Humour': '#ffffff',
    'Neutral': '#ffffff',
    'Positive': '#ffffff',
    'Negative': '#ffffff'
}


def extract_text_from_quotes(text):
    """Extract text inside quotation marks"""
    if pd.isna(text):
        return ""
    text = str(text)
    quoted_text = re.findall(r'"([^"]*)"', text)
    if quoted_text:
        return ' '.join(quoted_text)
    return text


def clean_text(text):
    """Clean text for display"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text.strip()


def clean_text_for_ml(text):
    """Clean text for ML processing (same as training)"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.lower().strip()


def extract_text_from_image(image):
    """Extract text from image using OCR"""
    if not OCR_AVAILABLE:
        return None, "pytesseract is not installed. Install with: pip install pytesseract"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(image)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            return None, "No text could be extracted from the image. Please ensure the image contains readable text."
        
        return extracted_text.strip(), None
    except Exception as e:
        return None, f"Error extracting text from image: {str(e)}"


# ML Feature Extractors (same as training script)
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for Word2Vec embeddings"""
    
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    
    def fit(self, X, y=None):
        """Train Word2Vec model"""
        tokenized = [simple_preprocess(text, deacc=True) for text in X]
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=0
        )
        return self
    
    def transform(self, X):
        """Generate embeddings"""
        tokenized = [simple_preprocess(text, deacc=True) for text in X]
        embeddings = []
        for tokens in tokenized:
            if tokens:
                word_vectors = [
                    self.model.wv[word] 
                    for word in tokens 
                    if word in self.model.wv
                ]
                if word_vectors:
                    embeddings.append(np.mean(word_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.vector_size))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)


class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract sentiment features using VADER and TextBlob"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract sentiment features"""
        features = []
        for text in X:
            # VADER scores
            vader_scores = self.vader.polarity_scores(str(text))
            # TextBlob scores
            blob = TextBlob(str(text))
            features.append([
                vader_scores['compound'],
                vader_scores['pos'],
                vader_scores['neu'],
                vader_scores['neg'],
                blob.sentiment.polarity,
                blob.sentiment.subjectivity
            ])
        return np.array(features)


def combine_features(tfidf_features, sentiment_features, w2v_features):
    """Combine different feature types"""
    # Convert sentiment and w2v to sparse matrices for efficient stacking
    sentiment_sparse = csr_matrix(sentiment_features)
    w2v_sparse = csr_matrix(w2v_features)
    
    # Combine all features
    combined = hstack([tfidf_features, sentiment_sparse, w2v_sparse]).toarray()
    return combined


@st.cache_resource
def load_sentiment_models():
    """Load sentiment analysis models"""
    models = {}
    
    if not TRANSFORMERS_AVAILABLE:
        return models  # Return empty dict, error will be shown in UI
    
    # Check if torch is actually available (transformers needs it)
    if not TORCH_AVAILABLE:
        st.warning("PyTorch is required for sentiment models. Models will not load.")
        return models
    
    # Determine device (use CPU if torch not available)
    device = -1  # Default to CPU (-1 means CPU in transformers)
    if TORCH_AVAILABLE:
        try:
            # Import torch locally to avoid NameError
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except (NameError, ImportError, AttributeError):
            device = -1
    
    try:
        with st.spinner("Loading Hugging Face Twitter-RoBERTa model..."):
            models['twitter_roberta'] = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True,
                device=device
            )
    except Exception as e:
        st.warning(f"Could not load Twitter-RoBERTa: {e}")
        models['twitter_roberta'] = None
    
    try:
        with st.spinner("Loading FinBERT model..."):
            models['finbert'] = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True,
                device=device
            )
    except Exception as e:
        st.warning(f"Could not load FinBERT: {e}")
        models['finbert'] = None
    
    try:
        with st.spinner("Loading BERT model..."):
            models['bert'] = hf_pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True,
                device=device
            )
    except Exception as e:
        st.warning(f"Could not load BERT: {e}")
        models['bert'] = None
    
    return models


def analyze_sentiment_with_model(text, model, model_name):
    """Analyze sentiment using a specific model"""
    if model is None:
        return None, None
    
    try:
        # Truncate text if too long
        max_length = 512
        text_truncated = text[:max_length] if len(text) > max_length else text
        
        results = model(text_truncated)
        
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                # Multiple scores format
                scores = {item['label']: item['score'] for item in results[0]}
            else:
                # Single result format
                scores = {results[0]['label']: results[0]['score']}
            
            return scores, results
        return None, None
    except Exception as e:
        st.error(f"Error with {model_name}: {e}")
        return None, None


def map_to_extended_sentiment(scores_dict, model_name):
    """Map model scores to extended sentiment categories"""
    if scores_dict is None:
        return "Neutral", 0.0
    
    # Normalize labels
    scores_lower = {k.lower(): v for k, v in scores_dict.items()}
    
    # Get dominant sentiment
    dominant_label = max(scores_dict.items(), key=lambda x: x[1])[0].lower()
    dominant_score = max(scores_dict.values())
    
    # Map based on model and scores
    if model_name == 'twitter_roberta':
        # Labels: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
        neg_score = scores_lower.get('label_0', scores_lower.get('negative', 0))
        pos_score = scores_lower.get('label_2', scores_lower.get('positive', 0))
        neu_score = scores_lower.get('label_1', scores_lower.get('neutral', 0))
        
        if neg_score > 0.6:
            if neg_score > 0.85:
                return "Angry", neg_score
            elif neg_score > 0.75:
                return "Sadness", neg_score
            elif neg_score > 0.65:
                # Check if it might be dark humour (negative but with some neutral/positive elements)
                if neu_score > 0.2 or pos_score > 0.15:
                    return "Dark Humour", neg_score
                return "Sarcastic", neg_score
            else:
                return "Sarcastic", neg_score
        elif pos_score > 0.6:
            if pos_score > 0.8:
                return "Playful", pos_score
            else:
                return "Funny", pos_score
        else:
            return "Neutral", neu_score
    
    elif model_name == 'finbert':
        # FinBERT: positive, negative, neutral
        pos_score = scores_lower.get('positive', 0)
        neg_score = scores_lower.get('negative', 0)
        neu_score = scores_lower.get('neutral', 0)
        
        if pos_score > 0.6:
            if pos_score > 0.75:
                return "Playful", pos_score
            else:
                return "Funny", pos_score
        elif neg_score > 0.6:
            if neg_score > 0.8:
                return "Angry", neg_score
            elif neg_score > 0.7:
                return "Sadness", neg_score
            elif neg_score > 0.65:
                # Check if it might be dark humour (negative but with some neutral/positive elements)
                if neu_score > 0.2 or pos_score > 0.15:
                    return "Dark Humour", neg_score
                return "Sarcastic", neg_score
            else:
                return "Sarcastic", neg_score
        else:
            return "Neutral", neu_score
    
    elif model_name == 'bert':
        # BERT multilingual: 1-5 star ratings
        star_5 = scores_lower.get('5 stars', scores_lower.get('5', 0))
        star_4 = scores_lower.get('4 stars', scores_lower.get('4', 0))
        star_3 = scores_lower.get('3 stars', scores_lower.get('3', 0))
        star_2 = scores_lower.get('2 stars', scores_lower.get('2', 0))
        star_1 = scores_lower.get('1 star', scores_lower.get('1', 0))
        
        if star_5 > 0.4:
            return "Funny", star_5
        elif star_4 > 0.4:
            return "Playful", star_4
        elif star_1 > 0.4:
            return "Angry", star_1
        elif star_2 > 0.4:
            # 2-star ratings could be sadness or dark humour
            # If star_3 is also high, might be dark humour (mixed sentiment)
            if star_3 > 0.3:
                return "Dark Humour", star_2
            return "Sadness", star_2
        else:
            return "Neutral", star_3
    
    return "Neutral", dominant_score


def detect_sentiment_category(text, scores_dict):
    """Detect specific sentiment categories from text and scores"""
    text_lower = text.lower()
    
    # Extended keyword-based detection
    sarcastic_keywords = ['sarcasm', 'sarcastic', 'obviously', 'sure', 'right', 'yeah right', 
                         'totally', 'of course', 'clearly', 'obviously', 'duh']
    playful_keywords = ['haha', 'lol', 'hehe', 'fun', 'play', 'joke', 'joking', 'teasing',
                       'prank', 'giggling', 'wink', '😉', '😜']
    funny_keywords = ['funny', 'hilarious', 'laugh', 'comedy', 'humor', 'lmao', 'rofl',
                     'hysterical', 'comical', 'amusing', '😂', '🤣']
    flirty_keywords = ['flirt', 'cute', 'hot', 'sexy', 'attractive', 'beautiful', 'gorgeous',
                      'handsome', 'charming', 'seductive', 'wink', '😘', '😍', '💋']
    angry_keywords = ['angry', 'mad', 'furious', 'rage', 'hate', 'stupid', 'idiot', 'damn',
                     'annoyed', 'irritated', 'frustrated', 'pissed', '😠', '😡']
    sadness_keywords = ['sad', 'sadness', 'depressed', 'depression', 'unhappy', 'upset', 'down',
                       'melancholy', 'gloomy', 'sorrow', 'grief', 'heartbroken', 'crying',
                       'tears', 'lonely', 'loneliness', 'disappointed', 'hopeless', '😢', '😭', '💔']
    dark_humour_keywords = ['dark humor', 'dark humour', 'dark comedy', 'gallows humor', 'gallows humour',
                           'morbid', 'morbid humor', 'twisted', 'twisted humor', 'cynical', 'cynical humor',
                           'black comedy', 'black humor', 'edgy', 'edgy humor', 'deadpan', 'macabre',
                           'sick humor', 'sick humour', 'grim humor', 'bleak humor', 'ironic death',
                           'death joke', 'tragedy joke', 'morbid joke', 'dark joke', 'twisted joke']
    
    # Check for keywords with priority (Dark Humour first as it's most specific)
    if any(kw in text_lower for kw in dark_humour_keywords):
        return "Dark Humour"
    elif any(kw in text_lower for kw in angry_keywords):
        return "Angry"
    elif any(kw in text_lower for kw in sadness_keywords):
        return "Sadness"
    elif any(kw in text_lower for kw in sarcastic_keywords):
        return "Sarcastic"
    elif any(kw in text_lower for kw in funny_keywords):
        return "Funny"
    elif any(kw in text_lower for kw in flirty_keywords):
        return "Flirty"
    elif any(kw in text_lower for kw in playful_keywords):
        return "Playful"
    
    # Fallback to model-based detection
    return None  # Will use model mapping


def analyze_urgency_servicenow(text, sentiment_results=None, consensus_sentiment=None):
    """
    Analyze urgency using ServiceNow-inspired priority framework.
    Returns urgency level, score (0-100), and recommended response time.
    """
    text_lower = text.lower()
    
    # Urgency keywords based on ServiceNow priority framework
    critical_keywords = [
        'urgent', 'emergency', 'critical', 'asap', 'immediately', 'now', 'urgently',
        'hurry', 'rushed', 'desperate', 'help now', 'dying', 'deadline passed',
        'deadline today', 'outage', 'down', 'broken', 'not working', 'crashed',
        'error', 'failed', 'fatal', 'disaster', 'catastrophe', 'life-threatening'
    ]
    
    high_keywords = [
        'important', 'soon', 'today', 'priority', 'need', 'required', 'must',
        'essential', 'critical', 'serious', 'significant', 'major', 'high priority',
        'need help', 'issue', 'problem', 'concern', 'worried', 'anxious', 'stressed'
    ]
    
    medium_keywords = [
        'whenever', 'soon', 'this week', 'later', 'please', 'would like',
        'could you', 'can you', 'if possible', 'when convenient', 'no rush'
    ]
    
    low_keywords = [
        'whenever', 'no rush', 'take your time', 'whenever possible', 'eventually',
        'sometime', 'just wondering', 'curious', 'informational', 'fyi', 'for info'
    ]
    
    # Calculate urgency score based on sentiment
    urgency_score = 50  # Base score (medium)
    
    if consensus_sentiment:
        sentiment_urgency_map = {
            'Angry': 85,
            'Sarcastic': 75,
            'Sadness': 70,
            'Dark Humour': 65,
            'Neutral': 50,
            'Funny': 45,
            'Playful': 40,
            'Flirty': 35,
            'Positive': 30,
            'Negative': 80
        }
        urgency_score = sentiment_urgency_map.get(consensus_sentiment, 50)
    
    # Adjust based on keywords
    critical_count = sum(1 for kw in critical_keywords if kw in text_lower)
    high_count = sum(1 for kw in high_keywords if kw in text_lower)
    medium_count = sum(1 for kw in medium_keywords if kw in text_lower)
    low_count = sum(1 for kw in low_keywords if kw in text_lower)
    
    # Calculate keyword impact
    if critical_count > 0:
        urgency_score = min(100, urgency_score + (critical_count * 10))
    elif high_count > 0:
        urgency_score = min(95, urgency_score + (high_count * 8))
    elif medium_count > 0:
        urgency_score = max(40, urgency_score - (medium_count * 5))
    elif low_count > 0:
        urgency_score = max(10, urgency_score - (low_count * 10))
    
    # Adjust based on text length (short messages might be more urgent)
    if len(text.split()) < 5:
        urgency_score = min(100, urgency_score + 5)
    
    # Adjust based on exclamation marks or caps
    if '!' in text:
        urgency_score = min(100, urgency_score + 5)
    if text.isupper() and len(text) > 3:
        urgency_score = min(100, urgency_score + 10)
    
    # Determine priority level (ServiceNow-style)
    if urgency_score >= 85:
        priority = "Critical"
        priority_numeric = 1
        response_time_minutes = 15
        response_time_text = "15 minutes"
    elif urgency_score >= 70:
        priority = "High"
        priority_numeric = 2
        response_time_minutes = 60
        response_time_text = "1 hour"
    elif urgency_score >= 40:
        priority = "Medium"
        priority_numeric = 3
        response_time_minutes = 240
        response_time_text = "4 hours"
    else:
        priority = "Low"
        priority_numeric = 4
        response_time_minutes = 1440
        response_time_text = "24 hours"
    
    # Calculate exact response deadline
    response_deadline = datetime.now() + timedelta(minutes=response_time_minutes)
    response_deadline_str = response_deadline.strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'priority': priority,
        'priority_numeric': priority_numeric,
        'urgency_score': min(100, max(0, int(urgency_score))),
        'response_time_minutes': response_time_minutes,
        'response_time_text': response_time_text,
        'response_deadline': response_deadline_str,
        'critical_keywords_found': critical_count,
        'high_keywords_found': high_count,
        'medium_keywords_found': medium_count,
        'low_keywords_found': low_count
    }


def analyze_sentiment_with_google_genai(text, api_key=None, model_name="gemini-2.5-pro"):
    """Analyze sentiment using Google Generative AI"""
    if not GOOGLE_GENAI_AVAILABLE:
        return None, "Google Generative AI is not installed. Install with: pip install google-generativeai"
    
    try:
        # Use API key from parameter, environment, or hardcoded default
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv('GOOGLE_API_KEY'):
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        elif HARDCODED_GOOGLE_API_KEY:
            genai.configure(api_key=HARDCODED_GOOGLE_API_KEY)
        else:
            return None, "Google API key not found. Set GOOGLE_API_KEY environment variable or provide api_key parameter."
        
        # Create the model with specified model name
        try:
            model = genai.GenerativeModel(model_name)
        except Exception as e:
            # If specified model fails, try alternatives
            fallback_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
            model = None
            for fallback_name in fallback_models:
                if fallback_name != model_name:
                    try:
                        model = genai.GenerativeModel(fallback_name)
                        model_name = fallback_name  # Update to the working model
                        break
                    except:
                        continue
            
            if model is None:
                return None, f"Could not create model '{model_name}'. Error: {str(e)}. Please try a different model name (gemini-2.5-pro, gemini-2.5-flash, gemini-1.5-pro, or gemini-1.5-flash)."
        
        # Create prompt for sentiment analysis
        prompt = f"""Analyze the sentiment of the following text and classify it into one of these categories:
- Sarcastic: Sarcastic or ironic content
- Playful: Light-hearted, fun content
- Funny: Humorous, comedic content
- Flirty: Flirtatious or romantic content
- Angry: Angry, frustrated, or negative content
- Sadness: Sad, depressed, or melancholic content
- Dark Humour: Dark, morbid, or twisted humor about serious topics
- Neutral: Neutral or balanced content

Text: "{text}"

Respond with ONLY the category name and a confidence score (0.0 to 1.0) in this format:
Category: [category name]
Confidence: [score]
"""
        
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse response
        category = "Neutral"
        confidence = 0.5
        
        lines = response_text.split('\n')
        for line in lines:
            if 'Category:' in line or 'category:' in line.lower():
                category = line.split(':')[-1].strip()
            elif 'Confidence:' in line or 'confidence:' in line.lower():
                try:
                    confidence = float(line.split(':')[-1].strip())
                except:
                    confidence = 0.5
        
        # Map to our sentiment categories
        category_map = {
            'sarcastic': 'Sarcastic',
            'playful': 'Playful',
            'funny': 'Funny',
            'flirty': 'Flirty',
            'angry': 'Angry',
            'sadness': 'Sadness',
            'sad': 'Sadness',
            'dark humour': 'Dark Humour',
            'dark humor': 'Dark Humour',
            'neutral': 'Neutral'
        }
        
        category = category_map.get(category.lower(), category)
        
        return {
            'sentiment': category,
            'confidence': confidence,
            'raw_response': response_text
        }, None
        
    except Exception as e:
        return None, f"Error analyzing with Google GenAI: {str(e)}"


def generate_response_with_gemini(original_text, detected_sentiment, training_examples=None, api_key=None, model_name="gemini-2.5-pro", style_instruction=""):
    """Generate a response with similar tone using Gemini, based on training data examples"""
    if not GOOGLE_GENAI_AVAILABLE:
        return None, "Google Generative AI is not installed. Install with: pip install google-generativeai"
    
    try:
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv('GOOGLE_API_KEY'):
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        else:
            return None, "Google API key not found. Set GOOGLE_API_KEY environment variable or provide api_key parameter."
        
        # Create the model
        try:
            model = genai.GenerativeModel(model_name)
        except Exception as e:
            # Try fallback models
            fallback_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
            model = None
            for fallback_name in fallback_models:
                if fallback_name != model_name:
                    try:
                        model = genai.GenerativeModel(fallback_name)
                        model_name = fallback_name
                        break
                    except:
                        continue
            
            if model is None:
                return None, f"Could not create model '{model_name}'. Error: {str(e)}"
        
        # Build examples text from training data
        examples_text = ""
        if training_examples is not None and len(training_examples) > 0:
            examples_text = "\n\nHere are some example texts with similar sentiment from the training data:\n"
            for i, example in enumerate(training_examples[:10], 1):  # Limit to 10 examples
                # Clean example text
                example_clean = str(example).replace('"', "'")[:300]  # Limit length
                examples_text += f"{i}. \"{example_clean}\"\n"
        
        # Add style instruction if provided
        style_text = ""
        if style_instruction:
            style_text = f"\nAdditional Style Instruction: {style_instruction}"
        
        # Create prompt for response generation
        prompt = f"""You are a response generator that crafts replies matching the tone and sentiment of the input text.

Original text: "{original_text}"
Detected sentiment: {detected_sentiment}
{examples_text}

Task: Generate a response that:
1. Matches the tone and sentiment of the original text ({detected_sentiment})
2. Is natural and conversational
3. Maintains the same style (casual, formal, playful, sarcastic, etc.)
4. Is appropriate for the context
5. Is similar in length and structure to the original
{style_text}

Generate ONLY the response text, without any explanations or labels. The response should feel like a natural reply that someone with the same tone would write.

Response:"""
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Extract text from response (handle different response formats)
        generated_text = ""
        try:
            # Try to get text directly
            if hasattr(response, 'text'):
                generated_text = response.text.strip()
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                # Newer API format
                if hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts'):
                        parts = response.candidates[0].content.parts
                        if parts and len(parts) > 0:
                            generated_text = parts[0].text.strip()
                    elif hasattr(response.candidates[0].content, 'text'):
                        generated_text = response.candidates[0].content.text.strip()
            elif hasattr(response, 'result'):
                generated_text = str(response.result).strip()
            else:
                # Fallback: convert to string
                generated_text = str(response).strip()
        except Exception as e:
            return None, f"Error extracting response text: {str(e)}. Response object: {type(response)}"
        
        if not generated_text or len(generated_text) == 0:
            return None, "Generated response is empty. Please try again or check your API key."
        
        # Clean up the response (remove any labels if model added them)
        if "Response:" in generated_text:
            generated_text = generated_text.split("Response:")[-1].strip()
        if "response:" in generated_text.lower():
            lines = generated_text.split('\n')
            for i, line in enumerate(lines):
                if "response:" in line.lower():
                    generated_text = '\n'.join(lines[i+1:]).strip()
                    break
        
        # Remove any markdown formatting that might interfere
        generated_text = generated_text.replace("```", "").strip()
        
        return {
            'response': generated_text,
            'original_sentiment': detected_sentiment,
            'model_used': model_name
        }, None
        
    except Exception as e:
        return None, f"Error generating response with Google GenAI: {str(e)}"


def simulate_conversation(original_text, detected_sentiment, training_examples=None, api_key=None, model_name="gemini-2.5-flash", num_exchanges=10):
    """Simulate a conversation with matching tone and sentiment"""
    if not GOOGLE_GENAI_AVAILABLE:
        return None, "Google Generative AI is not installed. Install with: pip install google-generativeai"
    
    try:
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv('GOOGLE_API_KEY'):
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        else:
            return None, "Google API key not found. Set GOOGLE_API_KEY environment variable or provide api_key parameter."
        
        # Create the model
        try:
            model = genai.GenerativeModel(model_name)
        except Exception as e:
            # Try fallback models
            fallback_models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
            model = None
            for fallback_name in fallback_models:
                if fallback_name != model_name:
                    try:
                        model = genai.GenerativeModel(fallback_name)
                        model_name = fallback_name
                        break
                    except:
                        continue
            
            if model is None:
                return None, f"Could not create model '{model_name}'. Error: {str(e)}"
        
        # Build examples text from training data
        examples_text = ""
        if training_examples is not None and len(training_examples) > 0:
            examples_text = "\n\nHere are some example texts with similar sentiment from the training data:\n"
            for i, example in enumerate(training_examples[:5], 1):  # Limit to 5 examples for conversation
                example_clean = str(example).replace('"', "'")[:200]
                examples_text += f"{i}. \"{example_clean}\"\n"
        
        # Create prompt for conversation simulation
        prompt = f"""You are a conversation simulator. Generate a natural conversation between two people that matches the tone and sentiment of the original text.

Original text: "{original_text}"
Detected sentiment: {detected_sentiment}
{examples_text}

Task: Generate a conversation with {num_exchanges} exchanges (back-and-forth messages) that:
1. Matches the tone and sentiment of the original text ({detected_sentiment})
2. Is natural and conversational
3. Maintains the same style (casual, formal, playful, sarcastic, etc.)
4. Each exchange should be realistic and flow naturally
5. The conversation should feel authentic to the sentiment

Format the output as:
User: [message]
Other: [response]
User: [message]
Other: [response]
... (continue for {num_exchanges} exchanges)

Generate ONLY the conversation, without any explanations or labels."""

        # Generate conversation
        response = model.generate_content(prompt)
        conversation_text = response.text.strip()
        
        # Parse the conversation into exchanges
        exchanges = []
        lines = conversation_text.split('\n')
        current_exchange = {"user": "", "other": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("User:") or line.startswith("user:"):
                if current_exchange["user"] or current_exchange["other"]:
                    exchanges.append(current_exchange)
                current_exchange = {"user": line.split(":", 1)[-1].strip(), "other": ""}
            elif line.startswith("Other:") or line.startswith("other:") or line.startswith("Person 2:") or line.startswith("Person2:"):
                current_exchange["other"] = line.split(":", 1)[-1].strip()
                exchanges.append(current_exchange)
                current_exchange = {"user": "", "other": ""}
        
        # If we have remaining exchange
        if current_exchange["user"] or current_exchange["other"]:
            exchanges.append(current_exchange)
        
        # Generate suggestions
        suggestions_prompt = f"""Based on this conversation context:
Original text: "{original_text}"
Sentiment: {detected_sentiment}

Generate:
1. 5 things the user could say next (matching the tone)
2. 5 things the other person could reply (matching the tone)

Format as:
USER_SUGGESTIONS:
1. [suggestion]
2. [suggestion]
...

OTHER_SUGGESTIONS:
1. [suggestion]
2. [suggestion]
..."""

        suggestions_response = model.generate_content(suggestions_prompt)
        suggestions_text = suggestions_response.text.strip()
        
        # Parse suggestions
        user_suggestions = []
        other_suggestions = []
        current_section = None
        
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if 'USER_SUGGESTIONS' in line.upper() or 'USER' in line.upper() and 'SUGGESTIONS' in line.upper():
                current_section = 'user'
                continue
            elif 'OTHER_SUGGESTIONS' in line.upper() or 'OTHER' in line.upper() and 'SUGGESTIONS' in line.upper():
                current_section = 'other'
                continue
            
            if line and (line[0].isdigit() or line.startswith('-')):
                suggestion = line.split('.', 1)[-1].strip() if '.' in line else line.split('-', 1)[-1].strip() if '-' in line else line
                if suggestion:
                    if current_section == 'user' and len(user_suggestions) < 5:
                        user_suggestions.append(suggestion)
                    elif current_section == 'other' and len(other_suggestions) < 5:
                        other_suggestions.append(suggestion)
        
        return {
            'exchanges': exchanges[:num_exchanges],  # Limit to requested number
            'user_suggestions': user_suggestions[:5],
            'other_suggestions': other_suggestions[:5],
            'original_sentiment': detected_sentiment,
            'model_used': model_name,
            'raw_conversation': conversation_text
        }, None
        
    except Exception as e:
        return None, f"Error simulating conversation with Google GenAI: {str(e)}"


@st.cache_resource
def load_bullying_detector():
    """Load or train a bullying detection model"""
    if not SKLEARN_AVAILABLE:
        return None, None, None, None, None
    
    # Try to load pre-trained bullying model
    try:
        if os.path.exists('bullying_detector.pkl'):
            with open('bullying_detector.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                model = saved_data.get('model')
                tfidf = saved_data.get('tfidf')
                if model and tfidf:
                    return model, tfidf, None, None, None
    except Exception as e:
        st.warning(f"Could not load saved bullying detector: {e}")
    
    # Train bullying detector
    try:
        # Load cyberbullying dataset - try multiple paths
        cyberbullying_df = None
        possible_paths = [
            '../synthetic_cyberbullying_tweets (1).csv',
            'synthetic_cyberbullying_tweets (1).csv',
            os.path.join('..', 'synthetic_cyberbullying_tweets (1).csv'),
            os.path.join('..', '..', 'synthetic_cyberbullying_tweets (1).csv'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'synthetic_cyberbullying_tweets (1).csv') if '__file__' in globals() else None
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    cyberbullying_df = pd.read_csv(path)
                    break
                except Exception:
                    continue
        
        if cyberbullying_df is None:
            return None, None, None, None, None
        
        # Prepare data
        cyberbullying_df['cleaned_text'] = cyberbullying_df['tweet'].apply(clean_text_for_ml)
        cyberbullying_df['is_bullying'] = (cyberbullying_df['label'] == 'Bullying').astype(int)
        
        X_train = cyberbullying_df['cleaned_text'].values
        y_train = cyberbullying_df['is_bullying'].values
        
        # Train TF-IDF and model
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_train_tfidf = tfidf.fit_transform(X_train)
        
        # Train Random Forest for bullying detection
        bullying_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        bullying_model.fit(X_train_tfidf, y_train)
        
        # Save model
        try:
            with open('bullying_detector.pkl', 'wb') as f:
                pickle.dump({
                    'model': bullying_model,
                    'tfidf': tfidf
                }, f)
        except Exception as e:
            st.warning(f"Could not save bullying detector: {e}")
        
        return bullying_model, tfidf, None, None, None
        
    except Exception as e:
        st.warning(f"Could not train bullying detector: {e}")
        return None, None, None, None, None


def detect_bullying(text, bullying_model, tfidf):
    """Detect if text contains bullying"""
    if not bullying_model or not tfidf:
        return False, 0.0
    
    try:
        cleaned_text = clean_text_for_ml(text)
        text_tfidf = tfidf.transform([cleaned_text])
        proba = bullying_model.predict_proba(text_tfidf)[0]
        is_bullying = bullying_model.predict(text_tfidf)[0]
        confidence = proba[1] if is_bullying else proba[0]
        
        return bool(is_bullying), float(confidence)
    except Exception as e:
        return False, 0.0


@st.cache_resource
def load_ml_models():
    """Load or train ML models (Google GenAI, Random Forest, XGBoost)"""
    if not SKLEARN_AVAILABLE:
        return None, None, None, None, None, None
    
    # Try to load pre-trained models
    models = {}
    feature_extractors = {}
    
    try:
        # Check if models are saved
        if os.path.exists('ml_models.pkl'):
            with open('ml_models.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                models = saved_data.get('models', {})
                feature_extractors = saved_data.get('extractors', {})
                if models and feature_extractors:
                    st.success("Loaded pre-trained ML models")
                    return (
                        models.get('Random Forest'),
                        models.get('XGBoost'),
                        feature_extractors.get('tfidf'),
                        feature_extractors.get('sentiment'),
                        feature_extractors.get('w2v')
                    )
    except Exception as e:
        st.warning(f"Could not load saved models: {e}")
    
    # If models not found, train them on the fly
    st.info("Training ML models on training data... This may take a few minutes.")
    
    try:
        # Load training data
        train_df = pd.read_csv('twitter_training.csv', header=None,
                              names=['id', 'topic', 'sentiment', 'tweet'])
        
        # Add cyberbullying dataset
        try:
            # Try multiple paths to find the cyberbullying dataset
            cyberbullying_df = None
            possible_paths = [
                '../synthetic_cyberbullying_tweets (1).csv',
                'synthetic_cyberbullying_tweets (1).csv',
                os.path.join('..', 'synthetic_cyberbullying_tweets (1).csv'),
                os.path.join('..', '..', 'synthetic_cyberbullying_tweets (1).csv')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        cyberbullying_df = pd.read_csv(path)
                        break
                    except Exception:
                        continue
            
            if cyberbullying_df is None:
                raise FileNotFoundError("Cyberbullying dataset not found")
            
            # Map labels: Bullying -> Negative, Not Bullying -> Positive, Moderate/Light Banter -> Neutral
            def map_bullying_label(label):
                if label == 'Bullying':
                    return 'Negative'
                elif label == 'Not Bullying':
                    return 'Positive'
                else:  # Moderate/Light Banter
                    return 'Neutral'
            
            cyberbullying_processed = pd.DataFrame({
                'id': range(len(train_df), len(train_df) + len(cyberbullying_df)),
                'topic': 'Cyberbullying',
                'sentiment': cyberbullying_df['label'].apply(map_bullying_label),
                'tweet': cyberbullying_df['tweet']
            })
            train_df = pd.concat([train_df, cyberbullying_processed], ignore_index=True)
            st.info(f"Added {len(cyberbullying_df)} cyberbullying examples to training data")
        except Exception as e:
            st.warning(f"Could not load cyberbullying dataset: {e}")
        
        # Process tweets
        train_df['tweet_text'] = train_df['tweet'].apply(extract_text_from_quotes)
        train_df['cleaned_text'] = train_df['tweet_text'].apply(clean_text_for_ml)
        
        # Encode target
        train_df['target'] = (train_df['sentiment'] == 'Positive').astype(int)
        
        # Prepare data
        X_train_text = train_df['cleaned_text'].values
        y_train = train_df['target'].values
        
        # Extract features
        with st.spinner("Extracting features..."):
            # TF-IDF
            tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, max_df=0.95)
            X_train_tfidf = tfidf.fit_transform(X_train_text)
            
            # Sentiment features
            sentiment_extractor = SentimentFeatureExtractor()
            X_train_sentiment = sentiment_extractor.fit_transform(X_train_text)
            
            # Word2Vec
            w2v = Word2VecTransformer(vector_size=100)
            X_train_w2v = w2v.fit_transform(X_train_text)
            
            # Combine features
            X_train_processed = combine_features(X_train_tfidf, X_train_sentiment, X_train_w2v)
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_processed)
        
        # Train models
        with st.spinner("Training models..."):
            # Random Forest
            rf_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ])
            rf_model.fit(X_train_scaled, y_train)
            models['Random Forest'] = rf_model
            
            # XGBoost
            xgb_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'))
            ])
            xgb_model.fit(X_train_scaled, y_train)
            models['XGBoost'] = xgb_model
        
        # Save models for future use
        try:
            with open('ml_models.pkl', 'wb') as f:
                pickle.dump({
                    'models': models,
                    'extractors': {
                        'tfidf': tfidf,
                        'sentiment': sentiment_extractor,
                        'w2v': w2v,
                        'scaler': scaler
                    }
                }, f)
        except Exception as e:
            st.warning(f"Could not save models: {e}")
        
        st.success("ML models trained successfully.")
        return (
            models['Random Forest'],
            models['XGBoost'],
            tfidf,
            sentiment_extractor,
            w2v
        )
        
    except Exception as e:
        st.error(f"Error training ML models: {e}")
        return None, None, None, None, None


def predict_with_ml_models(text, rf_model, xgb_model, tfidf, sentiment_extractor, w2v):
    """Predict sentiment using ML models"""
    if not all([rf_model, xgb_model, tfidf, sentiment_extractor, w2v]):
        return {}
    
    # Clean text
    cleaned_text = clean_text_for_ml(text)
    
    # Extract features
    try:
        # TF-IDF
        text_tfidf = tfidf.transform([cleaned_text])
        
        # Sentiment features
        text_sentiment = sentiment_extractor.transform([cleaned_text])
        
        # Word2Vec
        text_w2v = w2v.transform([cleaned_text])
        
        # Combine features
        text_features = combine_features(text_tfidf, text_sentiment, text_w2v)
        
        # Scale (scaler is in the pipeline, but we need to scale before)
        # Actually, the scaler is in the pipeline, so we don't need to scale here
        
        results = {}
        
        # Predict with each model
        for name, model in [('Random Forest', rf_model), 
                           ('XGBoost', xgb_model)]:
            try:
                # Get prediction probability
                proba = model.predict_proba(text_features)[0]
                prediction = model.predict(text_features)[0]
                
                # Map to sentiment categories
                if prediction == 1:
                    sentiment = "Positive"
                    confidence = proba[1]
                    # Further categorize based on confidence
                    if confidence > 0.8:
                        sentiment = "Playful"
                    elif confidence > 0.6:
                        sentiment = "Funny"
                else:
                    sentiment = "Negative"
                    confidence = proba[0]
                    # Further categorize
                    if confidence > 0.85:
                        sentiment = "Angry"
                    elif confidence > 0.7:
                        sentiment = "Sadness"
                    elif confidence > 0.6:
                        # Could be dark humour if there's some ambiguity
                        sentiment = "Dark Humour"
                    elif confidence > 0.55:
                        sentiment = "Sarcastic"
                    else:
                        sentiment = "Neutral"
                
                # Check for keyword overrides
                keyword_sentiment = detect_sentiment_category(text, None)
                if keyword_sentiment:
                    sentiment = keyword_sentiment
                
                results[name] = {
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'prediction': int(prediction),
                    'probabilities': {
                        'Negative': float(proba[0]),
                        'Positive': float(proba[1]) if len(proba) > 1 else 0.0
                    }
                }
            except Exception as e:
                st.warning(f"Error with {name}: {e}")
        
        return results
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return {}


def analyze_text_sentiment(text):
    """Analyze text sentiment using all available models"""
    if not text or len(text.strip()) == 0:
        return None
    
    results = {}
    
    # Transformer models
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        models = load_sentiment_models()
        for model_name, model in models.items():
            if model is not None:
                scores, raw_results = analyze_sentiment_with_model(text, model, model_name)
                if scores:
                    sentiment, confidence = map_to_extended_sentiment(scores, model_name)
                    
                    # Override with keyword detection if applicable
                    keyword_sentiment = detect_sentiment_category(text, scores)
                    if keyword_sentiment:
                        sentiment = keyword_sentiment
                    
                    results[model_name] = {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'scores': scores,
                        'raw': raw_results
                    }
    
    # ML models
    if SKLEARN_AVAILABLE:
        rf_model, xgb_model, tfidf, sentiment_extractor, w2v = load_ml_models()
        ml_results = predict_with_ml_models(text, rf_model, xgb_model, 
                                           tfidf, sentiment_extractor, w2v)
        results.update(ml_results)
    
    return results


@st.cache_data
def load_data():
    """Load and process data"""
    try:
        train_df = pd.read_csv('twitter_training.csv', header=None,
                              names=['id', 'topic', 'sentiment', 'tweet'])
        val_df = pd.read_csv('twitter_validation.csv', header=None,
                            names=['id', 'topic', 'sentiment', 'tweet'])
        
        # Process tweets
        train_df['tweet_text'] = train_df['tweet'].apply(extract_text_from_quotes)
        train_df['cleaned_text'] = train_df['tweet_text'].apply(clean_text)
        val_df['tweet_text'] = val_df['tweet'].apply(extract_text_from_quotes)
        val_df['cleaned_text'] = val_df['tweet_text'].apply(clean_text)
        
        # Add target column
        train_df['target'] = (train_df['sentiment'] == 'Positive').astype(int)
        val_df['target'] = (val_df['sentiment'] == 'Positive').astype(int)
        
        return train_df, val_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def display_tweet_row(row, show_cleaned=True):
    """Display a single tweet row with color coding"""
    sentiment = row['sentiment']
    target = row['target']
    tweet_text = row['cleaned_text'] if show_cleaned else row['tweet_text']
    
    # Monochrome styling
    bg_color = '#ffffff'
    text_color = '#000000'
    
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            color: {text_color};
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #000000;
            margin: 5px 0;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        ">
            <strong>Sentiment:</strong> {sentiment} | 
            <strong>Target:</strong> {target} | 
            <strong>Topic:</strong> {row.get('topic', 'N/A')}
            <br>
            <strong>Tweet:</strong> {tweet_text[:200]}{'...' if len(tweet_text) > 200 else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def acronym_explainer():
    """Acronym and Tone Explainer using Google Gemini"""
    st.header("Acronym and Tone Explainer")
    st.markdown("Enter an acronym, slang, or text to get its meaning, tone, and context.")
    
    # Get API key from sidebar or environment
    if GOOGLE_GENAI_AVAILABLE:
        # Check for API key in session state (from sentiment analyzer) or environment
        api_key = None
        if 'google_api_key' in st.session_state and st.session_state.google_api_key:
            api_key = st.session_state.google_api_key
        elif os.getenv('GOOGLE_API_KEY'):
            api_key = os.getenv('GOOGLE_API_KEY')
        else:
            api_key = HARDCODED_GOOGLE_API_KEY
        
        if not api_key:
            st.warning("Please enter your Google API key to use the Acronym Explainer.")
            st.info("""
            **How to get an API key:**
            1. Go to https://makersuite.google.com/app/apikey
            2. Create a new API key
            3. Enter it above or set it as an environment variable: `GOOGLE_API_KEY`
            
            **Note:** If you've already entered an API key in the Sentiment Analyzer tab, it will be used here automatically.
            """)
            return
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
            help="Select the Gemini model to use"
        )
        
        # Initialize client - try new API first, fallback to old API
        use_new_api = False
        client = None
        genai_old = None
        
        try:
            # Try new API first (from google import genai)
            try:
                from google import genai as genai_new
                client = genai_new.Client(api_key=api_key)
                use_new_api = True
            except (ImportError, AttributeError, TypeError):
                # Fallback to old API (google.generativeai)
                genai_old = genai  # Use the already imported genai
                genai_old.configure(api_key=api_key)
                use_new_api = False
        except Exception as e:
            st.error(f"Error initializing Gemini client: {str(e)}")
            return
        
        # Input field
        question = st.text_input(
            "Enter your acronym, slang, or text:",
            placeholder="e.g., LOL, SMH, YOLO, or any text",
            help="Enter an acronym, slang term, or any text to analyze",
            key="acronym_input"
        )
        
        if st.button("Explain", type="primary", key="explain_button"):
            if not question.strip():
                st.warning("No input provided. Please enter an acronym, slang, or text.")
            else:
                with st.spinner("Analyzing with Gemini..."):
                    try:
                        prompt = (
                            "You are precise. Provide one definitive answer and explain briefly. "
                            f"Text: {question}\n"
                            "What does it mean as an acronym and tell me the tone? and slang if any. and emojis if any"
                        )
                        
                        if use_new_api and client:
                            # New API (genai.Client)
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt
                            )
                            answer = response.text.strip()
                        else:
                            # Old API (google.generativeai)
                            model = genai_old.GenerativeModel(model_name)
                            response = model.generate_content(prompt)
                            answer = response.text.strip()
                        
                        if not answer:
                            st.info(f'"{question}" is not a commonly recognized acronym with a standardized meaning.')
                        else:
                            st.success("Analysis Complete!")
                            st.markdown("---")
                            st.markdown(f"**Q: {question}**")
                            st.markdown("**A:**")
                            
                            # Display answer in a nice box
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: #ffffff;
                                    border: 1px solid #000000;
                                    padding: 15px;
                                    margin: 10px 0;
                                    border-radius: 5px;
                                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                                ">
                                    <p style="font-size: 16px; margin: 0; white-space: pre-wrap;">{answer}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Also show in code block for easy copying
                            st.code(answer, language=None)
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Troubleshooting:")
                        st.markdown("""
                        - Check that your API key is correct
                        - Ensure you have API access enabled
                        - Try a different model
                        - Check your internet connection
                        - If using new API format, make sure you have the latest google-generativeai package
                        """)
    else:
        st.error("Google Generative AI is not installed.")
        st.code("pip install google-generativeai")
        st.info("After installing, restart the Streamlit app.")


def main():
    """Main Streamlit app"""
    # Apply global monochrome theme
    inject_monochrome_theme()
    
    # Initialize user management system
    if "users" not in st.session_state:
        # Format: {"username": "password"}
        st.session_state.users = {"admin": "1234"}
    
    if "logged" not in st.session_state:
        st.session_state.logged = False
    
    # Show login/signup if not logged in
    if not st.session_state.logged:
        st.title("subtext")
        st.markdown("User Management System")
        
        option = st.radio("Choose Action", ["Sign In", "Sign Up"], horizontal=True)
        
        # Sign Up
        if option == "Sign Up":
            new_u = st.text_input("Create Username")
            new_p = st.text_input("Create Password", type="password")
            
            if st.button("Register", use_container_width=True):
                if new_u in st.session_state.users:
                    st.error("Username already exists!")
                elif new_u == "":
                    st.warning("Username cannot be empty")
                else:
                    # Adding to the dictionary
                    st.session_state.users[new_u] = new_p
                    st.session_state.logged = True
                    st.session_state.current_user = new_u  # Store current username
                    st.success(f"Account created for {new_u}")
                    st.rerun()  # Refresh to show logged-in state
        
        # Sign In
        if option == "Sign In":
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            
            if st.button("Login", use_container_width=True):
                # Checking against the dictionary
                if u in st.session_state.users and st.session_state.users[u] == p:
                    st.session_state.logged = True
                    st.session_state.current_user = u  # Store current username
                    st.rerun()  # Refresh to show logged-in state immediately
                else:
                    st.error("Invalid username or password")
        
        return  # Don't show the rest of the app if not logged in
    
    # Logged in view - show main application
    # Top bar with username (left) and logout button (right)
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        username = st.session_state.get('current_user', 'User')
        st.markdown(f"**👤 {username}**")
    
    with col3:
        if st.button("Logout", use_container_width=True, key="top_logout_btn"):
            st.session_state.logged = False
            st.session_state.current_user = None
            st.rerun()
    
    st.markdown("---")  # Separator line
    
    st.title("subtext")
    
    # Navigation at the top using tabs
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Text Analysis (Text)"

    # Create tabs for navigation at the top
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Text Analysis (Text)",
        "Text Analysis (Image)",
        "Responses",
        "Acronym Explainer",
        "Text to Speech",
    ])

    # Display content based on active tab
    with tab1:
        st.session_state.current_page = "Text Analysis (Text)"
        text_analysis_text_page()
    
    with tab2:
        st.session_state.current_page = "Text Analysis (Image)"
        text_analysis_image_page()
    
    with tab3:
        st.session_state.current_page = "Responses"
        responses_page()
    
    with tab4:
        st.session_state.current_page = "Acronym Explainer"
        acronym_explainer()
    
    with tab5:
        st.session_state.current_page = "Text to Speech"
        text_to_speech()


def sentiment_analyzer(force_input_method=None, show_response_nav_button=False):
    """Sentiment analysis for custom text input or image OCR"""
    st.header("Multi-Model Sentiment Analysis")
    st.markdown("Analyze sentiment using Transformer models (Twitter-RoBERTa, FinBERT, BERT) and ML models (Google GenAI, Random Forest, XGBoost)")
    
    # Create unique key suffix based on input method to avoid duplicate element IDs
    key_suffix = "text" if force_input_method == "Text Input" else ("image" if force_input_method == "Image Upload (OCR)" else "default")
    
    # Check if at least one type of model is available
    if not TRANSFORMERS_AVAILABLE and not SKLEARN_AVAILABLE:
        st.error("""
        **Transformers library is not available.**
        
        To use the sentiment analyzer, please install the required packages.
        """)
        
        with st.expander("📋 Installation Instructions"):
            st.markdown("""
            ### Step 1: Install Visual C++ Redistributables (Required for PyTorch)
            
            Download and install from: https://aka.ms/vs/17/release/vc_redist.x64.exe
            
            Then restart your computer.
            
            ### Step 2: Install PyTorch and Transformers
            
            Run these commands in your terminal:
            ```bash
            pip uninstall torch torchvision torchaudio -y
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            pip install transformers
            ```
            
            ### Step 3: Verify Installation
            
            ```bash
            python -c "import torch; print('PyTorch works!')"
            ```
            
            ### Alternative: Use Without PyTorch
            
            You can continue using the application for basic features.
            """)
        
        return
    
    # Input method selection
    if force_input_method is None:
        input_method = st.radio(
            "Input Method:",
            ["Text Input", "Image Upload (OCR)"],
            horizontal=True,
        )
    else:
        input_method = force_input_method
    
    # Initialize session state for extracted text
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    
    text_input = ""
    
    if input_method == "Text Input":
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here...",
            help="Enter any text to get sentiment analysis from multiple models",
            key=f"text_input_area_{key_suffix}"
        )
        # Clear extracted text when switching to text input
        if st.session_state.extracted_text:
            st.session_state.extracted_text = ""
    else:
        # Image upload with OCR
        st.subheader("Upload Image for OCR")
        
        if not OCR_AVAILABLE:
            st.warning("OCR package (pytesseract) is not installed.")
            st.code("pip install pytesseract Pillow")
            st.info("""
            **Note**: You also need to install Tesseract OCR engine:
            - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
            - **Mac**: `brew install tesseract`
            - **Linux**: `sudo apt-get install tesseract-ocr`
            """)
        elif not TESSERACT_CONFIGURED:
            st.error("Tesseract OCR engine not found.")
            with st.expander("📋 How to Install Tesseract OCR"):
                st.markdown("""
                ### Windows Installation:
                
                1. **Download Tesseract:**
                   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
                   - Download: `tesseract-ocr-w64-setup-5.x.x.exe` (64-bit)
                
                2. **Install Tesseract:**
                   - Run the installer
                   - **Note the installation path** (usually `C:\\Program Files\\Tesseract-OCR`)
                   - Or run: `powershell -ExecutionPolicy Bypass -File install_tesseract_windows.ps1`
                
                3. **Configure (if auto-detection fails):**
                   Add this to the top of viewer.py after the imports:
                   ```python
                   import pytesseract
                   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                   ```
                
                4. **Restart Streamlit app**
                
                ### Quick Fix:
                Run: `powershell -ExecutionPolicy Bypass -File install_tesseract_windows.ps1`
                """)
        else:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
                help="Upload an image containing text to extract and analyze",
                key=f"image_uploader_{key_suffix}"
            )
            
            if uploaded_file is not None:
                try:
                    # Display the uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    # Extract text from image
                    with st.spinner("Extracting text from image using OCR..."):
                        extracted_text, error = extract_text_from_image(image)
                        
                        if error:
                            st.error(f"{error}")
                            text_input = ""
                            st.session_state.extracted_text = ""
                        elif extracted_text:
                            st.success("Text extracted successfully.")
                            # Store in session state
                            st.session_state.extracted_text = extracted_text
                            
                            # Make the text area editable so user can review/edit
                            edited_text = st.text_area(
                                "Extracted Text (you can edit this):",
                                value=extracted_text,
                                height=150,
                                key=f"extracted_text_area_{key_suffix}",
                                help="Review and edit the extracted text if needed"
                            )
                            # Update session state if user edited
                            st.session_state.extracted_text = edited_text
                            text_input = edited_text
                        else:
                            st.warning("No text could be extracted from the image.")
                            text_input = ""
                            st.session_state.extracted_text = ""
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    text_input = ""
                    st.session_state.extracted_text = ""
            else:
                # If no file uploaded, use session state if available
                if st.session_state.extracted_text:
                    text_input = st.text_area(
                        "Previously Extracted Text:",
                        value=st.session_state.extracted_text,
                        height=150,
                        key=f"extracted_text_area_{key_suffix}",
                        help="Text from previously uploaded image (you can edit this)"
                    )
                    st.session_state.extracted_text = text_input
                else:
                    text_input = ""
    
    # Enable all available models by default (no selection needed)
    # Transformer models
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        use_twitter_roberta = True
        use_finbert = True
        use_bert = True
    else:
        use_twitter_roberta = False
        use_finbert = False
        use_bert = False
    
    # Google GenAI
    if GOOGLE_GENAI_AVAILABLE:
        use_google_genai = True
        # No sidebar inputs (prevents duplicated widgets across tabs). Use stored/env/hardcoded key.
        google_api_key = st.session_state.get("google_api_key") or os.getenv("GOOGLE_API_KEY") or HARDCODED_GOOGLE_API_KEY
        st.session_state.google_api_key = google_api_key
        google_model_name = "gemini-2.5-pro"
    else:
        use_google_genai = False
        google_api_key = HARDCODED_GOOGLE_API_KEY
        google_model_name = "gemini-2.5-pro"
    
    # Traditional ML models
    if SKLEARN_AVAILABLE:
        use_rf = True
        use_xgb = True
    else:
        use_rf = False
        use_xgb = False
    
    # Show OCR status
    if input_method == "Image Upload (OCR)":
        if not OCR_AVAILABLE:
            st.error("OCR package (pytesseract) is not installed.")
            st.code("pip install pytesseract Pillow")
        elif not TESSERACT_CONFIGURED:
            st.error("Tesseract OCR engine not found.")
            with st.expander("📋 How to Install Tesseract OCR"):
                st.markdown("""
                ### Windows Installation:
                
                1. **Download Tesseract:**
                   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
                   - Download: `tesseract-ocr-w64-setup-5.x.x.exe` (64-bit)
                
                2. **Install Tesseract:**
                   - Run the installer
                   - **Note the installation path** (usually `C:\\Program Files\\Tesseract-OCR`)
                   - Or run: `powershell -ExecutionPolicy Bypass -File install_tesseract_windows.ps1`
                
                3. **Configure (if auto-detection fails):**
                   Add this to the top of viewer.py:
                   ```python
                   import pytesseract
                   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                   ```
                
                4. **Restart Streamlit app**
                
                ### Alternative: Quick Fix Script
                Run: `powershell -ExecutionPolicy Bypass -File install_tesseract_windows.ps1`
                """)
        else:
            st.success("OCR is ready. Upload an image to extract text.")
    
    # Initialize session state for results
    if 'sentiment_results' not in st.session_state:
        st.session_state.sentiment_results = None
    if 'analyzed_text' not in st.session_state:
        st.session_state.analyzed_text = ""
    if 'consensus_sentiment' not in st.session_state:
        st.session_state.consensus_sentiment = None
    
    if st.button("Analyze Sentiment", type="primary", key=f"analyze_sentiment_button_{key_suffix}"):
        # Get text from session state if using image upload
        if input_method == "Image Upload (OCR)" and st.session_state.extracted_text:
            text_input = st.session_state.extracted_text
        
        if not text_input or len(text_input.strip()) == 0:
            if input_method == "Image Upload (OCR)":
                st.warning("Please upload an image with text or switch to text input.")
            else:
                st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing sentiment..."):
            results = {}
            
            # Transformer models
            if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
                models = load_sentiment_models()
                
                if use_twitter_roberta and models.get('twitter_roberta'):
                    scores, raw = analyze_sentiment_with_model(
                        text_input, models['twitter_roberta'], 'twitter_roberta'
                    )
                    if scores:
                        sentiment, confidence = map_to_extended_sentiment(scores, 'twitter_roberta')
                        keyword_sentiment = detect_sentiment_category(text_input, scores)
                        if keyword_sentiment:
                            sentiment = keyword_sentiment
                        results['Twitter-RoBERTa'] = {
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'scores': scores
                        }
                
                if use_finbert and models.get('finbert'):
                    scores, raw = analyze_sentiment_with_model(
                        text_input, models['finbert'], 'finbert'
                    )
                    if scores:
                        sentiment, confidence = map_to_extended_sentiment(scores, 'finbert')
                        keyword_sentiment = detect_sentiment_category(text_input, scores)
                        if keyword_sentiment:
                            sentiment = keyword_sentiment
                        results['FinBERT'] = {
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'scores': scores
                        }
                
                if use_bert and models.get('bert'):
                    scores, raw = analyze_sentiment_with_model(
                        text_input, models['bert'], 'bert'
                    )
                    if scores:
                        sentiment, confidence = map_to_extended_sentiment(scores, 'bert')
                        keyword_sentiment = detect_sentiment_category(text_input, scores)
                        if keyword_sentiment:
                            sentiment = keyword_sentiment
                        results['BERT'] = {
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'scores': scores
                        }
            
            # Google GenAI
            if use_google_genai and GOOGLE_GENAI_AVAILABLE:
                genai_result, error = analyze_sentiment_with_google_genai(text_input, google_api_key, google_model_name)
                if error:
                    st.warning(f"Google GenAI: {error}")
                elif genai_result:
                    # Check for keyword overrides
                    keyword_sentiment = detect_sentiment_category(text_input, None)
                    if keyword_sentiment:
                        genai_result['sentiment'] = keyword_sentiment
                    results['Google GenAI'] = genai_result
            
            # ML models
            if SKLEARN_AVAILABLE:
                rf_model, xgb_model, tfidf, sentiment_extractor, w2v = load_ml_models()
                ml_results = predict_with_ml_models(text_input, rf_model, xgb_model,
                                                   tfidf, sentiment_extractor, w2v)
                
                # Filter by selection
                if use_rf and 'Random Forest' in ml_results:
                    results['Random Forest'] = ml_results['Random Forest']
                if use_xgb and 'XGBoost' in ml_results:
                    results['XGBoost'] = ml_results['XGBoost']
        
        # Store results in session state
        if results:
            st.session_state.sentiment_results = results
            st.session_state.analyzed_text = text_input
            # Calculate and store consensus
            sentiments = []
            for model_name, r in results.items():
                weight = 2 if model_name == 'Google GenAI' else 1
                sentiments.extend([r['sentiment']] * weight)
            sentiment_counts = pd.Series(sentiments).value_counts()
            if len(sentiment_counts) > 0:
                st.session_state.consensus_sentiment = sentiment_counts.index[0]
        
        # Check for bullying
        bullying_detected = False
        bullying_confidence = 0.0
        if SKLEARN_AVAILABLE:
            bullying_model, bullying_tfidf, _, _, _ = load_bullying_detector()
            if bullying_model and bullying_tfidf:
                bullying_detected, bullying_confidence = detect_bullying(text_input, bullying_model, bullying_tfidf)
        
        # Just show success message, results will be displayed below
        if results:
            st.success("Analysis complete!")
        else:
            st.error("No models were able to analyze the text. Please check model availability.")
    
    # Display stored results if they exist (single unified display)
    if st.session_state.sentiment_results:
        results = st.session_state.sentiment_results
        text_input = st.session_state.analyzed_text
        
        # Check for bullying
        bullying_detected = False
        bullying_confidence = 0.0
        if SKLEARN_AVAILABLE:
            bullying_model, bullying_tfidf, _, _, _ = load_bullying_detector()
            if bullying_model and bullying_tfidf:
                bullying_detected, bullying_confidence = detect_bullying(text_input, bullying_model, bullying_tfidf)
        
        # Display bullying warning if detected
        if bullying_detected:
            st.warning("""
            **Bullying Detected**
            
            This might be a rude message, maybe take a step back and set boundaries.
            
            The system detected potential bullying content with {:.1f}% confidence.
            """.format(bullying_confidence * 100))
        
        # Calculate consensus sentiment
        sentiments = []
        for model_name, r in results.items():
            weight = 2 if model_name == 'Google GenAI' else 1
            sentiments.extend([r['sentiment']] * weight)
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        if len(sentiment_counts) > 0:
            consensus = sentiment_counts.index[0]
            agreement = sentiment_counts.iloc[0] / len(results) * 100
            
            # Display consensus sentiment prominently
            st.subheader("Consensus Sentiment")
            st.markdown(
                f"""
                <div style="
                    background-color: #ffffff;
                    color: #000000;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #000000;
                    text-align: center;
                    font-size: 24px;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                ">
                    <strong>{consensus}</strong>
                    <br>
                    <small>Agreement: {agreement:.1f}% ({sentiment_counts.iloc[0]}/{len(results)} models)</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # ServiceNow Urgency Analysis
        st.subheader("ServiceNow Urgency Analysis")
        urgency_result = analyze_urgency_servicenow(text_input, results, consensus)
        
        # Color mapping for priority levels
        priority_colors = {
            'Critical': '#d32f2f',  # Red
            'High': '#f57c00',      # Orange
            'Medium': '#fbc02d',    # Yellow
            'Low': '#388e3c'        # Green
        }
        
        priority_color = priority_colors.get(urgency_result['priority'], '#757575')
        urgency_score = urgency_result['urgency_score']
        
        # Display urgency information in a card
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color: {priority_color};
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
                ">
                    <h3 style="margin: 0; font-size: 14px; font-weight: normal;">Priority</h3>
                    <h2 style="margin: 10px 0; font-size: 28px; font-weight: bold;">{urgency_result['priority']}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: #ffffff;
                    color: #000000;
                    padding: 20px;
                    border-radius: 10px;
                    border: 2px solid {priority_color};
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
                ">
                    <h3 style="margin: 0; font-size: 14px; font-weight: normal;">Urgency Score</h3>
                    <h2 style="margin: 10px 0; font-size: 28px; font-weight: bold; color: {priority_color};">{urgency_score}/100</h2>
                    <div style="margin-top: 10px; height: 8px; background-color: #e0e0e0; border-radius: 4px; overflow: hidden;">
                        <div style="height: 100%; width: {urgency_score}%; background-color: {priority_color}; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div style="
                    background-color: #ffffff;
                    color: #000000;
                    padding: 20px;
                    border-radius: 10px;
                    border: 2px solid {priority_color};
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
                ">
                    <h3 style="margin: 0; font-size: 14px; font-weight: normal;">Response Time</h3>
                    <h2 style="margin: 10px 0; font-size: 24px; font-weight: bold; color: {priority_color};">{urgency_result['response_time_text']}</h2>
                    <p style="margin: 5px 0; font-size: 12px; color: #666;">Deadline: {urgency_result['response_deadline']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Detailed urgency breakdown in expandable section
        with st.expander("View Detailed Urgency Breakdown", expanded=False):
            st.markdown(f"**Priority Level:** {urgency_result['priority']} (Priority {urgency_result['priority_numeric']})")
            st.markdown(f"**Urgency Score:** {urgency_score}/100")
            st.markdown(f"**Recommended Response Time:** {urgency_result['response_time_text']}")
            st.markdown(f"**Response Deadline:** {urgency_result['response_deadline']}")
            
            st.markdown("---")
            st.markdown("**Keyword Analysis:**")
            keyword_cols = st.columns(4)
            with keyword_cols[0]:
                st.metric("Critical Keywords", urgency_result['critical_keywords_found'], 
                         delta=None if urgency_result['critical_keywords_found'] == 0 else "+High Impact")
            with keyword_cols[1]:
                st.metric("High Keywords", urgency_result['high_keywords_found'],
                         delta=None if urgency_result['high_keywords_found'] == 0 else "+Medium Impact")
            with keyword_cols[2]:
                st.metric("Medium Keywords", urgency_result['medium_keywords_found'],
                         delta=None if urgency_result['medium_keywords_found'] == 0 else "-Low Impact")
            with keyword_cols[3]:
                st.metric("Low Keywords", urgency_result['low_keywords_found'],
                         delta=None if urgency_result['low_keywords_found'] == 0 else "-Reduced Urgency")
        
        # Individual model results in expandable section - display in rows to prevent overlap
        with st.expander("View Individual Model Results", expanded=False):
            st.subheader("Model Results")
            
            # Display models in a grid layout (2 columns) to prevent overlap
            num_models = len(results)
            num_cols = min(2, num_models)  # Use 2 columns max
            
            # Create rows of models
            model_items = list(results.items())
            for row_start in range(0, num_models, num_cols):
                cols = st.columns(num_cols)
                row_models = model_items[row_start:row_start + num_cols]
                
                for col_idx, (model_name, result) in enumerate(row_models):
                    with cols[col_idx]:
                        sentiment = result['sentiment']
                        confidence = result['confidence']
                        
                        st.markdown(
                            f"""
                            <div style="
                                    background-color: #ffffff;
                                    color: #000000;
                                    padding: 15px;
                                    border-radius: 10px;
                                    border: 1px solid #000000;
                                    text-align: center;
                                    margin: 10px 0;
                                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                                    min-height: 120px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: center;
                                ">
                                    <h3 style="margin: 5px 0; font-size: 14px; word-wrap: break-word;">{model_name}</h3>
                                    <h2 style="margin: 10px 0; font-size: 20px; word-wrap: break-word;">{sentiment}</h2>
                                    <p style="margin: 5px 0; font-size: 12px;">Confidence: {confidence:.2%}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            # Detailed scores section with proper spacing
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("Detailed Scores")
            
            for idx, (model_name, result) in enumerate(results.items()):
                st.markdown(f"**{model_name}**")
                
                # Handle different result formats (transformer vs ML)
                if 'scores' in result:
                    # Transformer model
                    scores_df = pd.DataFrame([
                        {'Label': k, 'Score': v} 
                        for k, v in result['scores'].items()
                    ])
                    st.dataframe(scores_df, use_container_width=True)
                    st.bar_chart(scores_df.set_index('Label'))
                elif 'probabilities' in result:
                    # ML model
                    prob_df = pd.DataFrame([
                        {'Label': k, 'Probability': v} 
                        for k, v in result['probabilities'].items()
                    ])
                    st.dataframe(prob_df, use_container_width=True)
                    st.bar_chart(prob_df.set_index('Label'))
                    st.write(f"Prediction: {result.get('prediction', 'N/A')}")
            
                # Add spacing between models
                if idx < len(results) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation button to Responses page (for text/image analysis pages)
    if show_response_nav_button:
        st.markdown("---")
        if st.button("Do you want to generate a response?", key=f"generate_response_nav_button_{key_suffix}"):
            st.session_state.current_page = "Responses"
            # Use st.rerun() for Streamlit >= 1.27
            try:
                st.rerun()
            except AttributeError:
                # Fallback for older versions
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
            
            # Response Generation Section (moved to separate Responses page)
            # Disabled here to keep this page focused on analysis only.
            if GOOGLE_GENAI_AVAILABLE and False:
                st.divider()
                st.markdown("---")
                st.subheader("Generate Response with Similar Tone")
                st.markdown("""
                Use **Google Gemini** to craft a response that matches the tone and sentiment of the analyzed text.
                
                The generator uses:
                - The detected sentiment from your analysis
                - Similar examples from your training data
                - Customizable response styles
                """)
                
                # Visual indicator box
                st.info("Location: The generated response will appear below this section after you click 'Generate Response'.")
                
                # Get consensus sentiment for response generation
                consensus_sentiment = st.session_state.consensus_sentiment if st.session_state.consensus_sentiment else consensus
                
                # Get similar examples from training data
                train_df, _ = load_data()
                similar_examples = []
                if train_df is not None:
                    # Map consensus sentiment to training data sentiments
                    sentiment_mapping = {
                        'Sarcastic': ['Negative'],
                        'Playful': ['Positive'],
                        'Funny': ['Positive'],
                        'Flirty': ['Positive'],
                        'Angry': ['Negative'],
                        'Sadness': ['Negative'],
                        'Dark Humour': ['Negative'],
                        'Neutral': ['Neutral'],
                        'Positive': ['Positive'],
                        'Negative': ['Negative']
                    }
                    
                    # Get matching sentiments
                    matching_sentiments = sentiment_mapping.get(consensus_sentiment, ['Neutral', 'Positive', 'Negative'])
                    similar_df = train_df[train_df['sentiment'].isin(matching_sentiments)]
                    
                    if len(similar_df) > 0:
                        # Get random examples
                        similar_examples = similar_df.sample(min(10, len(similar_df)))['tweet_text'].tolist()
                
                # Response generation options
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    generate_response = st.button("Generate Response", type="primary", use_container_width=True)
                
                with col2:
                    response_style = st.selectbox(
                        "Response Style",
                        ["Match Original", "More Casual", "More Formal", "More Playful", "More Professional", "Teenager"],
                        help="Adjust the style of the generated response"
                    )
                
                if generate_response:
                    # Use stored text input from session state
                    text_for_response = st.session_state.analyzed_text if st.session_state.analyzed_text else text_input
                    
                    # Validate inputs before generating
                    if not text_for_response or len(text_for_response.strip()) == 0:
                        st.warning("No text to generate response for. Please analyze some text first.")
                    elif not google_api_key and not os.getenv('GOOGLE_API_KEY') and not HARDCODED_GOOGLE_API_KEY:
                        st.error("Google API key is required. Please enter it in the sidebar.")
                    elif not consensus_sentiment:
                        st.warning("No sentiment detected. Please run sentiment analysis first.")
                    else:
                        with st.spinner("Generating response with Gemini..."):
                            # Adjust prompt based on style
                            style_instruction = ""
                            if response_style == "More Casual":
                                style_instruction = "Make the response more casual and relaxed."
                            elif response_style == "More Formal":
                                style_instruction = "Make the response more formal and professional."
                            elif response_style == "More Playful":
                                style_instruction = "Make the response more playful and lighthearted."
                            elif response_style == "More Professional":
                                style_instruction = "Make the response more professional and business-like."
                            elif response_style == "Teenager":
                                style_instruction = """You are a helpful assistant that helps people respond to scenarios using teen slang and acronyms. 
Use common teen slang like SMH, LOL, LMAO, BRUH, SUS, GOAT, FYP, WTF, BFF, IDK, ILY, TBH, GTG, FOMO, YOLO, TFW, ICYMI, NSFW, FML, BTW, HMU, BAE, IDC, ILYSM, TMI, WYD, WDYM, NVM, OMW, PFP, GG, AFK, FTW, IKR, JK, RN, HBU, GN, GM, etc.
Make the response sound like a teenager would text - use acronyms, slang, emojis where appropriate, and keep it casual and relatable. Match the tone and sentiment of the original message but express it in teen-speak."""
                            
                            # Generate response
                            try:
                                response_result, error = generate_response_with_gemini(
                                    text_for_response,
                                    consensus_sentiment,
                                    similar_examples,
                                    google_api_key,
                                    google_model_name,
                                    style_instruction
                                )
                                
                                if error:
                                    st.error(f"Error generating response: {error}")
                                    st.info("Troubleshooting tips:")
                                    st.markdown("""
                                    - Check that your Google API key is correct
                                    - Ensure you have API access enabled for Gemini
                                    - Try a different model (gemini-2.5-flash is faster)
                                    - Check your internet connection
                                    - Verify the API key has Gemini API access
                                    """)
                                elif response_result:
                                    if 'response' in response_result and response_result['response']:
                                        # Success message
                                        st.success("Response generated successfully.")
                                        
                                        # Visual separator
                                        st.markdown("---")
                                        
                                        # Display the generated response prominently
                                        st.markdown("### Generated Response")
                                        st.markdown("**Your response that matches the tone:**")
                                        
                                        # Store generated response in session state for conversation simulator
                                        st.session_state.generated_response = response_result['response']
                                        
                                        # Display in a highlighted box
                                        st.markdown(
                                            f"""
                                            <div style="
                                                background-color: #ffffff;
                                                border: 1px solid #000000;
                                                padding: 15px;
                                                margin: 10px 0;
                                                border-radius: 5px;
                                                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                                            ">
                                                <p style="font-size: 16px; margin: 0;">{response_result['response']}</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                        
                                        # Also show in code block for easy copying
                                        st.markdown("**Copy this response:**")
                                        st.code(response_result['response'], language=None)
                                        
                                        # Show metadata
                                        st.markdown("---")
                                        with st.expander("Response Details"):
                                            st.write(f"**Original Sentiment:** {response_result.get('original_sentiment', 'N/A')}")
                                            st.write(f"**Model Used:** {response_result.get('model_used', 'N/A')}")
                                            st.write(f"**Training Examples Used:** {len(similar_examples)}")
                                        
                                        # Show examples used (if any)
                                        if similar_examples:
                                            with st.expander("Similar Examples from Training Data"):
                                                st.write("These examples from your training data were used to guide the tone:")
                                                for i, example in enumerate(similar_examples[:5], 1):
                                                    st.write(f"{i}. {example[:200]}{'...' if len(example) > 200 else ''}")
                                    else:
                                        st.warning("Response was generated but appears to be empty. Please try again.")
                                        if 'raw_response' in response_result:
                                            st.code(f"Raw response: {response_result['raw_response']}")
                                else:
                                    st.error("No response was generated. Please check your API key and try again.")
                                    
                            except Exception as e:
                                st.error(f"Exception occurred: {str(e)}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
                
                # Conversation Simulator Section
                st.markdown("---")
                st.subheader("Conversation Simulator")
                st.markdown("""
                Simulate a 10-exchange conversation that matches the tone and sentiment of your analyzed text.
                The simulator will generate realistic back-and-forth exchanges and provide suggestions for what you or the other person could say next.
                """)
                
                # Conversation simulator button
                # Check if a generated response exists
                has_generated_response = 'generated_response' in st.session_state and st.session_state.generated_response
                
                if not has_generated_response:
                    st.info("💡 **Tip:** Generate a response first, then use the conversation simulator to see how the conversation might continue based on that generated response.")
                
                simulate_conversation_btn = st.button(
                    "Simulate Conversation (10 Exchanges)", 
                    type="secondary", 
                    use_container_width=True,
                    disabled=not has_generated_response
                )
                
                if simulate_conversation_btn:
                    # Use the generated response instead of original text
                    if 'generated_response' in st.session_state and st.session_state.generated_response:
                        text_for_simulation = st.session_state.generated_response
                    else:
                        # Fallback to original text if no generated response
                        text_for_simulation = st.session_state.analyzed_text if st.session_state.analyzed_text else text_input
                    
                    # Validate inputs
                    if not text_for_simulation or len(text_for_simulation.strip()) == 0:
                        st.warning("No generated response found. Please generate a response first, then try the conversation simulator.")
                    elif not google_api_key and not os.getenv('GOOGLE_API_KEY') and not HARDCODED_GOOGLE_API_KEY:
                        st.error("Google API key is required. Please enter it in the sidebar.")
                    elif not consensus_sentiment:
                        st.warning("No sentiment detected. Please run sentiment analysis first.")
                    else:
                        with st.spinner("Simulating conversation with Gemini based on your generated response... This may take a moment."):
                            try:
                                conversation_result, error = simulate_conversation(
                                    text_for_simulation,
                                    consensus_sentiment,
                                    similar_examples,
                                    google_api_key,
                                    google_model_name,
                                    num_exchanges=10
                                )
                                
                                if error:
                                    st.error(f"Error simulating conversation: {error}")
                                elif conversation_result:
                                    st.success("Conversation simulated successfully.")
                                    st.markdown("---")
                                    
                                    # Display the conversation
                                    st.markdown("### Simulated Conversation")
                                    st.markdown(f"**Sentiment:** {conversation_result['original_sentiment']}")
                                    st.markdown(f"**Model Used:** {conversation_result['model_used']}")
                                    st.markdown("---")
                                    
                                    # Display exchanges
                                    for i, exchange in enumerate(conversation_result['exchanges'], 1):
                                        st.markdown(f"**Exchange {i}:**")
                                        
                                        # User message
                                        if exchange.get('user'):
                                            st.markdown(
                                                f"""
                                                <div style="
                                                    background-color: #ffffff;
                                                    border: 1px solid #000000;
                                                    padding: 10px;
                                                    margin: 5px 0;
                                                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                                    border-radius: 5px;
                                                ">
                                                    <strong>You:</strong> {exchange['user']}
                                                </div>
                                                """,
                                                unsafe_allow_html=True
                                            )
                                        
                                        # Other person's message
                                        if exchange.get('other'):
                                            st.markdown(
                                                f"""
                                                <div style="
                                                    background-color: #ffffff;
                                                    border: 1px solid #000000;
                                                    padding: 10px;
                                                    margin: 5px 0;
                                                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                                    border-radius: 5px;
                                                ">
                                                    <strong>Other:</strong> {exchange['other']}
                                                </div>
                                                """,
                                                unsafe_allow_html=True
                                            )
                                        
                                        st.markdown("---")
                                    
                                    # Display suggestions
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("### 5 Things You Could Say Next:")
                                        if conversation_result.get('user_suggestions'):
                                            for i, suggestion in enumerate(conversation_result['user_suggestions'], 1):
                                                st.markdown(
                                                    f"""
                                                    <div style="
                                                        background-color: #ffffff;
                                                        padding: 8px;
                                                        margin: 5px 0;
                                                        border-radius: 3px;
                                                        border: 1px solid #000000;
                                                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                                    ">
                                                        {i}. {suggestion}
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True
                                                )
                                        else:
                                            st.info("No suggestions generated.")
                                    
                                    with col2:
                                        st.markdown("### 5 Things The Other Person Could Reply:")
                                        if conversation_result.get('other_suggestions'):
                                            for i, suggestion in enumerate(conversation_result['other_suggestions'], 1):
                                                st.markdown(
                                                    f"""
                                                    <div style="
                                                        background-color: #ffffff;
                                                        padding: 8px;
                                                        margin: 5px 0;
                                                        border-radius: 3px;
                                                        border: 1px solid #000000;
                                                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                                    ">
                                                        {i}. {suggestion}
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True
                                                )
                                        else:
                                            st.info("No suggestions generated.")
                                    
                                    # Show raw conversation in expander
                                    with st.expander("View Raw Conversation"):
                                        st.code(conversation_result.get('raw_conversation', 'N/A'), language=None)
                                    
                            except Exception as e:
                                st.error(f"Exception occurred: {str(e)}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
                else:
                    st.info("Enable Google GenAI in the sidebar and provide an API key to generate responses.")
        else:
            st.error("No models were able to analyze the text. Please check model availability.")


def dataset_viewer():
    """Original dataset viewer functionality"""
    # Load data
    train_df, val_df = load_data()
    
    if train_df is None or val_df is None:
        st.error("Failed to load data files. Please ensure twitter_training.csv and twitter_validation.csv are in the current directory.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Select Dataset",
        ["Training", "Validation", "Both"]
    )
    
    # Get selected dataset
    if dataset == "Training":
        df = train_df.copy()
    elif dataset == "Validation":
        df = val_df.copy()
    else:
        df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Sentiment filter
    sentiments = df['sentiment'].unique().tolist()
    selected_sentiments = st.sidebar.multiselect(
        "Filter by Sentiment",
        options=sentiments,
        default=sentiments
    )
    
    # Target filter
    targets = sorted(df['target'].unique().tolist())
    selected_targets = st.sidebar.multiselect(
        "Filter by Target",
        options=targets,
        default=targets,
        format_func=lambda x: "Positive" if x == 1 else "Negative/Neutral"
    )
    
    # Apply filters
    filtered_df = df[
        (df['sentiment'].isin(selected_sentiments)) &
        (df['target'].isin(selected_targets))
    ].copy()
    
    # Main area: show basic stats and preview
    st.subheader("Dataset Overview")
    st.write(f"Total rows after filters: **{len(filtered_df)}**")

    if len(filtered_df) == 0:
        st.warning("No rows match the current filters. Try changing the sentiment/target filters.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Tweets", len(filtered_df))
    with col2:
        sentiment_counts = filtered_df['sentiment'].value_counts()
        st.write("Sentiment counts:")
        st.bar_chart(sentiment_counts)

    st.subheader("Sample Tweets")
    # Show up to 100 rows
    st.dataframe(filtered_df.head(100))


def text_analysis_text_page():
    """Page: Text analysis using direct text input only"""
    # Force text input method and add navigation button to Responses
    sentiment_analyzer(force_input_method="Text Input", show_response_nav_button=True)


def text_analysis_image_page():
    """Page: Text analysis using image OCR input only"""
    # Force image input method and add navigation button to Responses
    sentiment_analyzer(force_input_method="Image Upload (OCR)", show_response_nav_button=True)


def responses_page():
    """Page: Response generator and conversation simulator"""
    st.header("Responses")
    st.markdown(
        "Generate replies and simulate conversations based on the **most recent sentiment analysis**."
    )

    # Ensure we have analysis results
    results = st.session_state.get("sentiment_results")
    text_input = st.session_state.get("analyzed_text", "")
    consensus_sentiment = st.session_state.get("consensus_sentiment")

    if not results or not text_input:
        st.warning(
            "No analysis found. Please analyze some text first on the **Text Analysis** pages."
        )
        return

    # Show the analyzed text and consensus
    st.subheader("Analyzed Text")
    st.info(text_input)

    if consensus_sentiment:
        consensus_color = EXTENDED_SENTIMENT_COLORS.get(consensus_sentiment, "#95a5a6")
        st.markdown(
            f"""
            <div style="
                background-color: {consensus_color};
                color: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 10px 0;
            ">
                <strong>Consensus Sentiment:</strong> {consensus_sentiment}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Google GenAI configuration: no UI inputs (prevents duplicated widgets across tabs).
    if not GOOGLE_GENAI_AVAILABLE:
        st.error("Google Generative AI is not installed. Install with: pip install google-generativeai")
        return

    api_key = st.session_state.get("google_api_key") or os.getenv("GOOGLE_API_KEY") or HARDCODED_GOOGLE_API_KEY
    st.session_state.google_api_key = api_key
    model_name = "gemini-2.5-pro"

    # Prepare similar examples based on consensus
    train_df, _ = load_data()
    similar_examples = []
    if train_df is not None and consensus_sentiment:
        sentiment_mapping = {
            "Sarcastic": ["Negative"],
            "Playful": ["Positive"],
            "Funny": ["Positive"],
            "Flirty": ["Positive"],
            "Angry": ["Negative"],
            "Sadness": ["Negative"],
            "Dark Humour": ["Negative"],
            "Neutral": ["Neutral"],
            "Positive": ["Positive"],
            "Negative": ["Negative"],
        }
        matching_sentiments = sentiment_mapping.get(
            consensus_sentiment, ["Neutral", "Positive", "Negative"]
        )
        similar_df = train_df[train_df["sentiment"].isin(matching_sentiments)]
        if len(similar_df) > 0:
            similar_examples = (
                similar_df.sample(min(10, len(similar_df)))["tweet_text"].tolist()
            )

    # Response generation UI
    st.markdown("---")
    st.subheader("Generate Response with Similar Tone")

    col1, col2 = st.columns([2, 1])
    with col1:
        generate_response = st.button(
            "Generate Response", type="primary", use_container_width=True
        )
    with col2:
        response_style = st.selectbox(
            "Response Style",
            ["Match Original", "More Casual", "More Formal", "More Playful", "More Professional", "Teenager"],
            help="Adjust the style of the generated response",
        )

    if generate_response:
        text_for_response = text_input

        if not text_for_response or len(text_for_response.strip()) == 0:
            st.warning("No text to generate response for. Please analyze some text first.")
        elif not (api_key or os.getenv("GOOGLE_API_KEY") or HARDCODED_GOOGLE_API_KEY):
            st.error("Google API key is required. Please enter it above or set GOOGLE_API_KEY.")
        elif not consensus_sentiment:
            st.warning("No sentiment detected. Please run sentiment analysis first.")
        else:
            style_instruction = ""
            if response_style == "More Casual":
                style_instruction = "Make the response more casual and relaxed."
            elif response_style == "More Formal":
                style_instruction = "Make the response more formal and professional."
            elif response_style == "More Playful":
                style_instruction = "Make the response more playful and lighthearted."
            elif response_style == "More Professional":
                style_instruction = "Make the response more professional and business-like."
            elif response_style == "Teenager":
                style_instruction = """You are a helpful assistant that helps people respond to scenarios using teen slang and acronyms. 
Use common teen slang like SMH, LOL, LMAO, BRUH, SUS, GOAT, FYP, WTF, BFF, IDK, ILY, TBH, GTG, FOMO, YOLO, TFW, ICYMI, NSFW, FML, BTW, HMU, BAE, IDC, ILYSM, TMI, WYD, WDYM, NVM, OMW, PFP, GG, AFK, FTW, IKR, JK, RN, HBU, GN, GM, etc.
Make the response sound like a teenager would text - use acronyms, slang, emojis where appropriate, and keep it casual and relatable. Match the tone and sentiment of the original message but express it in teen-speak."""

            with st.spinner("Generating response with Gemini..."):
                try:
                    response_result, error = generate_response_with_gemini(
                        text_for_response,
                        consensus_sentiment,
                        similar_examples,
                        api_key,
                        model_name,
                        style_instruction,
                    )

                    if error:
                        st.error(f"Error generating response: {error}")
                    elif response_result and response_result.get("response"):
                        st.success("Response generated successfully.")
                        st.markdown("---")
                        st.markdown("### Generated Response")
                        st.markdown("**Your response that matches the tone:**")
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #ffffff;
                                border: 1px solid #000000;
                                padding: 15px;
                                margin: 10px 0;
                                border-radius: 5px;
                                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                            ">
                                <p style="font-size: 16px; margin: 0;">{response_result['response']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        # Store generated response in session state for conversation simulator
                        st.session_state.generated_response = response_result['response']
                        st.markdown("**Copy this response:**")
                        st.code(response_result["response"], language=None)
                    else:
                        st.warning("No response was generated. Please try again.")
                except Exception as e:
                    st.error(f"Exception occurred: {str(e)}")

    # Conversation simulator
    st.markdown("---")
    st.subheader("Conversation Simulator")
    st.markdown(
        "Simulate a 10-exchange conversation based on your **generated response**. The simulator will continue the conversation from your generated response that matches the tone."
    )
    
    # Check if a generated response exists
    has_generated_response = 'generated_response' in st.session_state and st.session_state.generated_response
    
    if not has_generated_response:
        st.info("💡 **Tip:** Generate a response first, then use the conversation simulator to see how the conversation might continue based on that generated response.")

    simulate_conversation_btn = st.button(
        "Simulate Conversation (10 Exchanges)",
        type="secondary",
        use_container_width=True,
        disabled=not has_generated_response
    )

    if simulate_conversation_btn:
        # Use the generated response instead of original text
        if 'generated_response' in st.session_state and st.session_state.generated_response:
            text_for_simulation = st.session_state.generated_response
        else:
            # Fallback to original text if no generated response
            text_for_simulation = text_input

        if not text_for_simulation or len(text_for_simulation.strip()) == 0:
            st.warning("No generated response found. Please generate a response first, then try the conversation simulator.")
        elif not (api_key or os.getenv("GOOGLE_API_KEY") or HARDCODED_GOOGLE_API_KEY):
            st.error("Google API key is required. Please enter it above or set GOOGLE_API_KEY.")
        elif not consensus_sentiment:
            st.warning("No sentiment detected. Please run sentiment analysis first.")
        else:
            with st.spinner("Simulating conversation with Gemini based on your generated response... This may take a moment."):
                try:
                    conversation_result, error = simulate_conversation(
                        text_for_simulation,
                        consensus_sentiment,
                        similar_examples,
                        api_key,
                        model_name,
                        num_exchanges=10,
                    )

                    if error:
                        st.error(f"Error simulating conversation: {error}")
                    elif conversation_result:
                        st.success("Conversation simulated successfully.")
                        st.markdown("---")

                        st.markdown("### Simulated Conversation")
                        st.markdown(f"**Sentiment:** {conversation_result['original_sentiment']}")
                        st.markdown(f"**Model Used:** {conversation_result['model_used']}")
                        st.markdown("---")

                        for i, exchange in enumerate(conversation_result["exchanges"], 1):
                            st.markdown(f"**Exchange {i}:**")
                            if exchange.get("user"):
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color: #ffffff;
                                        border: 1px solid #000000;
                                        padding: 10px;
                                        margin: 5px 0;
                                        border-radius: 5px;
                                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                    ">
                                        <strong>You:</strong> {exchange['user']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            if exchange.get("other"):
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color: #ffffff;
                                        border: 1px solid #000000;
                                        padding: 10px;
                                        margin: 5px 0;
                                        border-radius: 5px;
                                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                    ">
                                        <strong>Other:</strong> {exchange['other']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            st.markdown("---")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 5 Things You Could Say Next:")
                            if conversation_result.get("user_suggestions"):
                                for i, suggestion in enumerate(
                                    conversation_result["user_suggestions"], 1
                                ):
                                    st.markdown(f"{i}. {suggestion}")
                        with col2:
                            st.markdown("### 5 Things The Other Person Could Reply:")
                            if conversation_result.get("other_suggestions"):
                                for i, suggestion in enumerate(
                                    conversation_result["other_suggestions"], 1
                                ):
                                    st.markdown(f"{i}. {suggestion}")
                except Exception as e:
                    st.error(f"Exception occurred while simulating conversation: {str(e)}")


def text_to_speech():
    """Text-to-speech tab using gTTS"""
    st.header("Text-to-Speech Converter")
    st.markdown("Convert text to speech using Google Text-to-Speech (gTTS).")
    
    if not GTTS_AVAILABLE:
        st.error("gTTS is not installed. Install it with:")
        st.code("pip install gtts")
        return
    
    # Text input
    text = st.text_area("Enter text to convert to speech:")
    
    # Button to generate speech
    if st.button("Play Speech"):
        if text.strip() != "":
            try:
                # Convert text to speech and store in memory
                tts = gTTS(text=text, lang='en', slow=False)
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)  # Move to the start of the BytesIO buffer
                
                # Play the audio in Streamlit
                st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                st.error(f"Error generating speech: {str(e)}")
        else:
            st.error("Please enter some text to convert.")

if __name__ == "__main__":
    main()
