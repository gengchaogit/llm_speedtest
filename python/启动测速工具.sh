#!/bin/bash
echo "===================================="
echo "   LLM速度测试工具 - 一键启动"
echo "===================================="
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python，请先安装Python3！"
    exit 1
fi

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/3] 检查依赖..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "[提示] 正在安装依赖包..."
    pip3 install fastapi uvicorn httpx -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo "[2/3] 启动Python后端服务器..."

# 删除旧的端口配置文件
rm -f .backend_port

# 启动Python后端（在后台运行）
python3 llm_test_backend.py &
BACKEND_PID=$!

echo "[提示] 等待后端启动... (PID: $BACKEND_PID)"
# 等待端口配置文件生成（最多等待10秒）
count=0
while [ $count -lt 10 ]; do
    sleep 1
    if [ -f ".backend_port" ]; then
        BACKEND_PORT=$(cat .backend_port 2>/dev/null | tr -d '[:space:]')
        if [ -n "$BACKEND_PORT" ]; then
            echo "[3/3] 检测到后端端口: $BACKEND_PORT"
            break
        fi
    fi
    count=$((count + 1))
done

if [ -z "$BACKEND_PORT" ]; then
    echo "[警告] 无法检测到后端端口，使用默认端口18000"
    BACKEND_PORT=18000
fi

# 尝试打开浏览器（如果有的话）
if command -v xdg-open &> /dev/null; then
    echo "[完成] 正在打开测试页面..."
    xdg-open "http://localhost:$BACKEND_PORT/" &> /dev/null || true
elif command -v open &> /dev/null; then
    echo "[完成] 正在打开测试页面..."
    open "http://localhost:$BACKEND_PORT/" &> /dev/null || true
else
    echo "[提示] 未检测到图形界面，请在浏览器手动打开: http://localhost:$BACKEND_PORT/"
fi

echo ""
echo "===================================="
echo "   启动完成"
echo "   测试页面: http://localhost:$BACKEND_PORT/"
echo "   停止服务: kill $BACKEND_PID"
echo "===================================="
echo ""
echo "按Ctrl+C停止后端服务"

# 等待用户中断
wait $BACKEND_PID
