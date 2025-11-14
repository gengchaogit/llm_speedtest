# LLM速度测试工具 - Python后端版

基于 v1.9 的 Python 后端版本，突破浏览器并发限制，支持真正的高并发测试。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动后端
```bash
python llm_test_backend.py
```
看到 `Uvicorn running on http://0.0.0.0:8000` 说明启动成功。

### 3. 打开前端
在浏览器中访问: http://localhost:8000/

## 文件说明

```
python/
├── llm_test_backend.py                    # Python 后端服务器 (FastAPI + WebSocket)
├── LLM_Speed_Test_v2_Python_Backend.html  # 前端页面 (WebSocket版)
├── LLM_Speed_Test_Tool_v1.9_EN.html       # 原版页面 (浏览器版)
├── requirements.txt                        # Python 依赖
└── README_Backend.md                       # 本文档
```

## 核心改进

| 功能 | v1.9 (浏览器版) | v2.0 (Python后端版) |
|------|----------------|-------------------|
| 最大并发数 | 6 (浏览器限制) | 无限制 |
| 执行环境 | 浏览器 fetch | Python asyncio |
| 真实并发 | ❌ | ✅ |
| 网络延迟 | 包含浏览器开销 | 纯后端测量 |

## 功能特性

- ✅ 突破浏览器6并发限制，支持真正的高并发 (10/20/50+)
- ✅ WebSocket 实时推送测试进度和结果
- ✅ 完整支持 OpenAI 和 Ollama 接口
- ✅ 保留原版所有功能 (历史记录、导出、对比等)
- ✅ 支持 usage、reasoning_tokens、timing 等完整统计信息

## 使用方法

1. 配置 API 参数 (地址、模型、密钥)
2. 设置测试参数 (提示词长度、并发数等)
3. 点击"开始测试"
4. 实时查看结果，支持导出 CSV/Markdown/图表

## 技术架构

- **前端**: HTML + JavaScript + WebSocket 客户端
- **后端**: FastAPI + WebSocket + httpx.AsyncClient
- **通信**: WebSocket (ws://localhost:8000/ws/test)
- **并发**: Python asyncio.gather 实现真实并发

## 消息协议

### 前端 → 后端 (配置)
```json
{
  "api_url": "http://localhost:11434/v1/chat/completions",
  "model_name": "qwen2.5:7b",
  "api_type": "openai",
  "concurrency": 10,
  ...
}
```

### 后端 → 前端 (消息类型)
- `info`: 测试开始信息
- `progress`: 测试进度 (当前/总数)
- `result`: 单个测试点的结果
- `complete`: 测试完成
- `error`: 错误信息

## 故障排除

**连接失败**: 确保后端已启动，运行 `curl http://localhost:8000/`  
**依赖安装失败**: 尝试 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`  
**端口被占用**: 修改 `llm_test_backend.py` 最后一行的端口号

## 注意事项

- 高并发测试会对 API 服务器产生压力，请谨慎设置并发数
- 确保目标 API 服务器支持对应的并发数
- 建议先用小并发数 (5-10) 测试，再逐步提高
