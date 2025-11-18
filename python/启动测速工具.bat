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

REM 删除旧的端口配置文件
if exist ".backend_port" del ".backend_port"

REM 在新窗口启动Python后端（后端会自动写入.backend_port文件）
start "" python llm_test_backend.py

echo [提示] 等待后端启动...
REM 等待端口配置文件生成（最多等待10秒）
set /a count=0
:wait_port
timeout /t 1 /nobreak >nul
if exist ".backend_port" (
    set /p BACKEND_PORT=<.backend_port
    if not "%BACKEND_PORT%"=="" goto port_found
)
set /a count+=1
if %count% lss 10 goto wait_port

echo [警告] 无法检测到后端端口，使用默认端口18000
set BACKEND_PORT=18000
goto port_done

:port_found
echo [3/3] 检测到后端端口: %BACKEND_PORT%
goto port_done

:port_done

echo [完成] 正在打开测试页面...
REM 使用默认浏览器访问后端URL（后端会自动返回HTML页面）
start "" "http://localhost:%BACKEND_PORT%/"

echo.
echo ====================================
echo    启动完成
echo    测试页面: http://localhost:%BACKEND_PORT%/
echo    关闭后端窗口即可停止服务
echo ====================================
echo.
pause
