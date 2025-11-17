@echo off
chcp 65001 >nul
echo ====================================
echo    LLM速度测试工具 - 一键启动
echo ====================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python！
    pause
    exit /b 1
)

REM 获取当前脚本所在目录
cd /d "%~dp0"

echo [1/3] 检查依赖...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 正在安装依赖包...
    pip install fastapi uvicorn httpx -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo [2/3] 启动Python后端服务器...
start "LLM测速后端" cmd /k "python llm_test_backend.py"

echo [3/3] 等待服务器启动...
timeout /t 3 /nobreak >nul

echo [完成] 正在打开测试页面...
start "" "LLM_Speed_Test_v2_Python_Backend.html"

echo.
echo ====================================
echo    启动完成！
echo    后端运行在 http://localhost:8000
echo    关闭后端窗口即可停止服务
echo ====================================
echo.
pause
