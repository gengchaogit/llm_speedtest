@echo off
chcp 65001 >nul
echo ====================================
echo    LLM Speed Test Tool - Quick Start
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python first!
    pause
    exit /b 1
)

REM Change to script directory
cd /d "%~dp0"

echo [1/3] Checking dependencies...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing dependencies...
    pip install fastapi uvicorn httpx
)

echo [2/3] Starting Python backend server...
start "LLM Speed Test Backend" cmd /k "python llm_test_backend.py"

echo [3/3] Waiting for server to start...
timeout /t 3 /nobreak >nul

echo [DONE] Opening test page...
start "" "LLM_Speed_Test_v2_Python_Backend.html"

echo.
echo ====================================
echo    Startup Complete!
echo    Backend running at http://localhost:8000
echo    Close backend window to stop service
echo ====================================
echo.
pause
