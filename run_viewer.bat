@echo off
REM Batch script to run Streamlit viewer
REM This works even if 'streamlit' command is not in PATH

echo Starting Tweet Sentiment Viewer...
echo.

REM Try using python -m streamlit (works even if streamlit not in PATH)
python -m streamlit run viewer.py

if errorlevel 1 (
    echo.
    echo Error: Could not start Streamlit
    echo.
    echo Trying alternative method...
    python -c "import streamlit; import subprocess; subprocess.run(['streamlit', 'run', 'viewer.py'])"
)

pause

