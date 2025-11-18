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

REM Delete old port config file
if exist ".backend_port" del ".backend_port"

REM Start Python backend in new window (backend will write .backend_port file)
start "" python llm_test_backend.py

echo [INFO] Waiting for backend to start...
REM Wait for port config file to be created (max 10 seconds)
set /a count=0
:wait_port
timeout /t 1 /nobreak >nul
if exist ".backend_port" (
    set /p BACKEND_PORT=<.backend_port
    if not "%BACKEND_PORT%"=="" goto port_found
)
set /a count+=1
if %count% lss 10 goto wait_port

echo [WARNING] Could not detect backend port, using default port 18000
set BACKEND_PORT=18000
goto port_done

:port_found
echo [3/3] Detected backend port: %BACKEND_PORT%
goto port_done

:port_done

echo [DONE] Opening test page...
REM Open backend URL with default browser (backend will serve HTML)
start "" "http://localhost:%BACKEND_PORT%/"

echo.
echo ====================================
echo    Startup Complete
echo    Test page: http://localhost:%BACKEND_PORT%/
echo    Close backend window to stop service
echo ====================================
echo.
pause
