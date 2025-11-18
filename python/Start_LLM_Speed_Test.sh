#!/bin/bash
echo "===================================="
echo "   LLM Speed Test - Quick Start"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[Error] Python3 not found, please install Python3 first!"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/3] Checking dependencies..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "[Info] Installing dependencies..."
    pip3 install fastapi uvicorn httpx
fi

echo "[2/3] Starting Python backend server..."

# Remove old port configuration file
rm -f .backend_port

# Start Python backend (run in background)
python3 llm_test_backend.py &
BACKEND_PID=$!

echo "[Info] Waiting for backend to start... (PID: $BACKEND_PID)"
# Wait for port configuration file (max 10 seconds)
count=0
while [ $count -lt 10 ]; do
    sleep 1
    if [ -f ".backend_port" ]; then
        BACKEND_PORT=$(cat .backend_port 2>/dev/null | tr -d '[:space:]')
        if [ -n "$BACKEND_PORT" ]; then
            echo "[3/3] Detected backend port: $BACKEND_PORT"
            break
        fi
    fi
    count=$((count + 1))
done

if [ -z "$BACKEND_PORT" ]; then
    echo "[Warning] Cannot detect backend port, using default port 18000"
    BACKEND_PORT=18000
fi

# Try to open browser (if available)
if command -v xdg-open &> /dev/null; then
    echo "[Done] Opening test page..."
    xdg-open "http://localhost:$BACKEND_PORT/" &> /dev/null || true
elif command -v open &> /dev/null; then
    echo "[Done] Opening test page..."
    open "http://localhost:$BACKEND_PORT/" &> /dev/null || true
else
    echo "[Info] No GUI detected, please manually open in browser: http://localhost:$BACKEND_PORT/"
fi

echo ""
echo "===================================="
echo "   Startup Complete"
echo "   Test Page: http://localhost:$BACKEND_PORT/"
echo "   Stop Service: kill $BACKEND_PID"
echo "===================================="
echo ""
echo "Press Ctrl+C to stop backend service"

# Wait for user interrupt
wait $BACKEND_PID
