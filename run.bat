@echo off
echo ============================================================
echo   Sentiment Analysis of User Reviews - Setup ^& Run
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       Done.

echo.
echo [2/3] Running training pipeline (downloads data + trains model)...
echo       This may take 5-10 minutes on first run.
python train_pipeline.py
if errorlevel 1 (
    echo [ERROR] Training pipeline failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Launching Streamlit app...
echo       Open http://localhost:8501 in your browser.
echo.
streamlit run app.py

pause
