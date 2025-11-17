# 🐍 LLM速度测试工具 - Python后端版 v2.1

## 简介

Python后端版本通过Python FastAPI后端发起HTTP请求，突破浏览器6并发限制，支持真正的高并发测试（50+并发）。

## 与浏览器版本对比

| 特性 | 浏览器版本 | Python后端版本 |
|------|-----------|---------------|
| 安装要求 | 无需安装 | 需要Python 3.7+ |
| 并发限制 | ≤6 (浏览器限制) | 50+ (无限制) |
| 测试准确性 | 一般 | 高 (真实并发) |
| 启动方式 | 双击HTML | 需启动后端服务 |
| 适用场景 | 快速测试 | 高并发压测 |

## 安装说明

### 系统要求
- Python 3.7 或更高版本
- Windows / Linux / macOS

### 安装依赖

```bash
cd python
pip install fastapi uvicorn httpx
```

**使用国内镜像加速（可选）：**
```bash
pip install fastapi uvicorn httpx -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 使用方法

### 方法一：一键启动（推荐）

#### Windows用户
双击以下任一脚本：
- `启动测速工具.bat` (中文界面)
- `Start_LLM_Speed_Test.bat` (英文界面)

脚本会自动：
1. 检查Python环境
2. 安装缺失的依赖包
3. 启动后端服务器
4. 在默认浏览器打开测试页面

#### Linux/macOS用户
创建启动脚本：
```bash
#!/bin/bash
cd python
python llm_test_backend.py &
sleep 2
open LLM_Speed_Test_v2_Python_Backend.html  # macOS
# xdg-open LLM_Speed_Test_v2_Python_Backend.html  # Linux
```

### 方法二：手动启动

#### 1. 启动后端服务器

```bash
cd python
python llm_test_backend.py
```

后端服务将运行在 `http://localhost:8000`

#### 2. 打开前端页面

在浏览器中打开 `LLM_Speed_Test_v2_Python_Backend.html`

## 配置说明

### 后端配置

编辑 `llm_test_backend.py` 中的配置（如需修改）：

```python
# 端口配置
port = 8000  # 默认端口

# CORS配置（跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议限制来源
    ...
)
```

### 前端配置

前端页面的所有功能与浏览器版本相同，包括：
- API地址、模型名称、并发数等参数配置
- 历史记录管理
- 图表导出
- CSV导出/导入

## 主要功能

### 1. 突破并发限制
- 浏览器版本受限于浏览器的6并发限制
- Python后端版本可支持50+并发，测试真实的系统并发能力

### 2. 更准确的吞吐量计算
- 使用真实的墙上时间计算总吞吐量
- 记录每个请求的精确时间戳（开始时间、首token时间、结束时间）
- 计算公式：
  - Prefill总吞吐 = 总prompt tokens / (最晚首token时间 - 最早开始时间)
  - Decode总吞吐 = 总output tokens / (最晚结束时间 - 最早首token时间)

### 3. Token来源追踪
- 显示token统计的来源（API/本地估算）
- 对本地估算值添加⚠警告标记

### 4. 百分位统计
- 显示P50/P90/P95百分位统计
- 更准确评估性能分布和稳定性

### 5. 存储优化
- 未勾选"保存测试详情"时不保存prompt/output文本
- 50并发测试存储占用减少约98%

## 技术架构

### 后端 (llm_test_backend.py)
- **框架**: FastAPI
- **WebSocket**: 实时测试进度推送
- **异步HTTP客户端**: httpx (支持真实高并发)
- **功能**:
  - 接收前端测试请求
  - 并发发起HTTP请求到LLM API
  - 记录精确时间戳
  - 解析响应流并统计tokens
  - 实时推送测试进度

### 前端 (LLM_Speed_Test_v2_Python_Backend.html)
- **通信**: WebSocket连接后端
- **功能**:
  - 配置测试参数
  - 接收实时测试结果
  - 计算真实并发吞吐量
  - 图表展示和数据导出

## 常见问题

### 1. 后端启动失败

**错误**: `Address already in use`

**解决方法**:
```bash
# 查找占用端口的进程
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/macOS

# 修改端口或关闭占用进程
```

### 2. 前端连接失败

**错误**: WebSocket连接失败

**检查步骤**:
1. 确认后端已启动且运行正常
2. 检查浏览器控制台(F12)查看错误信息
3. 确认后端URL正确（默认：`ws://localhost:8000/ws/test`）
4. 检查防火墙设置

### 3. 并发测试结果异常

**可能原因**:
- LLM服务器资源不足（GPU显存、CPU）
- 网络带宽限制
- LLM服务配置的最大batch size限制

**建议**:
- 从较低并发开始测试（如10），逐步增加
- 监控服务器资源使用情况
- 检查LLM服务器日志

### 4. 依赖安装失败

**错误**: `pip install` 失败

**解决方法**:
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi uvicorn httpx

# 或使用清华镜像
pip install -i https://pypi.mirrors.ustc.edu.cn/simple fastapi uvicorn httpx
```

## 停止服务

### Windows
关闭后端命令行窗口即可停止服务

### Linux/macOS
在后端终端按 `Ctrl+C` 停止服务

## 性能建议

### 并发数设置
- **测试开始**: 建议从10并发开始
- **逐步增加**: 每次增加10-20并发
- **监控观察**: 注意服务器负载和响应时间
- **找到上限**: 当延迟明显增加时即为系统瓶颈

### 提示词长度
- **短提示词测试** (128-1024): 测试Prefill性能
- **长提示词测试** (4096-32768): 测试长上下文处理能力
- **超长提示词** (32768+): 测试系统极限

### 输出长度
- **短输出** (128-512): 测试基础Decode性能
- **长输出** (1024-2048): 测试持续生成能力

## 更新日志

### v2.1 (当前版本)
- ✅ 初始发布Python后端版本
- ✅ 支持50+真实高并发测试
- ✅ 修复并发吞吐量计算错误
- ✅ 添加Token来源追踪
- ✅ 添加百分位统计(P50/P90/P95)
- ✅ 优化存储空间占用
- ✅ 提供一键启动脚本

## 技术支持

- **GitHub**: https://github.com/gengchaogit/llm_speedtest
- **QQ群**: 1028429001
- **Issues**: 欢迎在GitHub提交问题和建议

## 许可证

本项目基于原作者纸鸢随风（B站）的工作进行改进

---

**开发者**: chao (魔改版维护者)
**版本**: v2.1
**最后更新**: 2025-11-18
