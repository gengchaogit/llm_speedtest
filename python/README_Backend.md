# 🐍 LLM 速度测试与高并发压测套件 - Python 后端与 CLI 指南 (v3.5)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg?style=flat-square)](https://fastapi.tiangolo.com)
[![HTTP/2 Support](https://img.shields.io/badge/Protocol-HTTP%2F2%20Multiplexing-orange.svg?style=flat-square)](https://python-hyper.org/projects/h2/)
[![Prometheus Ready](https://img.shields.io/badge/Telemetry-Prometheus%20Scraper-E6522C.svg?style=flat-square)](https://prometheus.io/)

---

✨ **在线体验免安装纯前端版本：**
- ⚡ **矩阵测速引擎**: [**https://gengchaogit.github.io/llm_speedtest/**](https://gengchaogit.github.io/llm_speedtest/)
- 🏆 **全网天梯榜**: [**https://gengchaogit.github.io/llm_speedtest/LLM_Speed_Test_v3_Leaderboard.html**](https://gengchaogit.github.io/llm_speedtest/LLM_Speed_Test_v3_Leaderboard.html)

---

## 1. 概述与核心优势

`python/llm_test_backend.py` 是专为高并发压力测试与生产级运行时诊断设计的异步基准套件。它既可以作为本地无头 **CLI 压测工具** 嵌入自动化评测流水线，也可以作为 **FastAPI / WebSocket 本地服务端** 配合专用前端页面 `LLM_Speed_Test_v3_Python_Backend.html` 使用。

### 为什么选择 Python 后端与 CLI 套件？

| 特性维度 | 浏览器纯前端版 (index.html) | Python 高并发套件 (CLI / 服务端) |
|---|---|---|
| **最大并发能力** | 受浏览器 TCP 连接池限制 ($\le 6$) | **50 ～ 256+** 真实物理高并发 |
| **HTTP 传输协议** | 浏览器标准 HTTP/1.1 或 HTTP/2 | **HTTP/2 多路复用连接池** (`httpx` + `h2`) |
| **服务端状态遥测** | ❌ 仅客户端黑盒测速 | **✅ 实时抓取 Prometheus `/metrics`** |
| **KV Cache 监控** | ❌ 无法感知 | **✅ 实时监控 GPU KV Cache 占用率与等待队列** |
| **有效并发率探测** | 基于客户端时间戳推断 | **✅ 服务端排队探测并打标 `(X/Y)*`** |
| **输出死循环熔断** | 前端 JavaScript 滑动窗口 | **Python 异步双通道 Repetition Watchdog** |
| **自动化集成** | 需人工操作浏览器页面 | **支持 Shell / CI / Headless 无头批量测试** |
| **云端天梯榜上榜** | 浏览器一键上传 | **CLI 自动签名并直传 Supabase 云端** |

---

## 2. 环境准备与依赖安装

### 2.1 系统要求
- Python 3.10 或更高版本
- 支持 Windows 10/11、macOS (Intel / Apple Silicon)、Linux (x86_64 / aarch64)

### 2.2 安装依赖库

```bash
cd python
pip install -r requirements.txt
```

若在国内网络环境下，推荐使用清华大学镜像加速：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements.txt` 核心依赖说明：
- `fastapi` & `uvicorn`: 轻量高性能异步 Web 与 WebSocket 服务器；
- `httpx` & `h2`: 支持 HTTP/2 协议多路复用的超高性能异步 HTTP 客户端；
- `websockets`: 实时双向流式推送通信组件。

---

## 3. 运行模式与使用指南

套件提供 **无头 CLI 压测模式** 和 **图形化本地服务模式** 两种运行形态，并完整支持三大评测范式：**标准 1D 并发阶梯扫频 (`1d_sweep`)**、**2D 并发 × 上下文矩阵扫频 (`2d_matrix`)** 以及 **稳态持续解码基准 (`sustained_decode`)**。

### 模式 A：无头 CLI 压测模式 (推荐 CI/CD 与批量评估)

通过 `--benchmark` 或 `--cli` 标志直接在终端中运行，无需打开任何浏览器：

#### 示例 1：标准 1D 并发阶梯扫频 (`--benchmark-mode 1d_sweep`)
针对本地 vLLM 发起 1, 4, 16 阶梯并发测试，采集 Prometheus 服务端状态并导出报告：
```bash
python3 llm_test_backend.py --benchmark \
  --benchmark-mode 1d_sweep \
  --api-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,4,16 \
  --contexts 512,1024,2048 \
  --max-tokens 128 \
  --metrics-url http://localhost:8000/metrics \
  --output benchmark_1d_report.json
```

#### 示例 2：2D 并发 × 上下文矩阵扫频 (`--benchmark-mode 2d_matrix`)
基于探路遍历（Scout traversal）、KV Cache 容量动态超限保护（`--max-kv-tokens`）与单单元格故障隔离（带 1.0s 冷却与错误重试）：
```bash
python3 llm_test_backend.py --benchmark \
  --benchmark-mode 2d_matrix \
  --api-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --matrix-concurrency 1,2,4,8 \
  --matrix-context 512,1024,2048 \
  --matrix-max-kv-tokens 32768 \
  --max-tokens 128 \
  --hardware "NVIDIA RTX 4090 24GB" \
  --framework "vLLM" \
  --output benchmark_matrix_report.json
```

#### 示例 3：稳态持续解码压测 (`--benchmark-mode sustained_decode`)
模拟生产满载稳态，在固定持续时间（如 30s）内动态补充工作流，隔离单请求 TTFT 与持续稳态 TPS，并推算 `(X/Y)*` 有效并发率：
```bash
python3 llm_test_backend.py --benchmark \
  --benchmark-mode sustained_decode \
  --api-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 4 \
  --sustained-duration 30.0 \
  --sustained-prompt-length 128 \
  --max-tokens 128 \
  --output benchmark_sustained_report.json
```

#### 示例 4：测试完成自动签名直传全网天梯榜
```bash
python3 llm_test_backend.py --benchmark \
  --api-url http://localhost:8000/v1 \
  --model DeepSeek-R1-Distill-Qwen-14B \
  --concurrency 16 \
  --hardware "NVIDIA RTX 4090 24GB" \
  --framework "vLLM 0.6.3" \
  --quantization "AWQ-4bit" \
  --upload \
  --user-code "lab_runner_01" \
  --nickname "Performance-Team" \
  --notes "CUDA 12.4, FlashAttention-2"
```

---

### 模式 B：图形化本地服务模式 (Web GUI)

通过 `--server` 标志启动本地 FastAPI / WebSocket 服务端：

```bash
# 启动本地服务，默认绑定 18000 端口（若被占用自动递增寻找可用端口）
python3 llm_test_backend.py --server --port 18000
```

服务启动后，在浏览器中打开 `python/LLM_Speed_Test_v3_Python_Backend.html` 即可在可视化界面中：
1. **模式切换器**：自由在「标准 1D 阶梯扫频」、「2D 矩阵全景热力」与「稳态持续解码监控」三大面板间无缝切换；
2. **2D 矩阵热力图**：直观查看 Concurrency $\times$ Context 各单元格吞吐与延迟，支持点击单元格下钻详情弹窗；
3. **稳态持续解码大屏**：实时观测目标持续时间窗口、活跃工作流状态、稳态 TPS 曲线与 `(X/Y)*` 有效并发率指示；
4. **全套数据导出**：测试完成后支持一键导出 CSV、Markdown 表格，或一键带数字签名上传至 Supabase 全网天梯榜。

---

### 模式 C：跨平台一键启动脚本

在 `python/` 目录下准备了免配环境的一键式脚本，会自动检测虚拟环境、检查依赖并在默认浏览器打开测试界面：

- **macOS / Linux**:
  ```bash
  # 启动中文版
  bash python/run_for_linux_mac_zh.sh
  # 启动英文版
  bash python/run_for_linux_mac_en.sh
  ```
- **Windows**:
  - `python/run_for_windows_zh.bat` (中文界面，预设 UTF-8 控制台编码)
  - `python/run_for_windows_en.bat` (英文界面)
  - `python/run_for_windows_en_china_user.bat` (英文界面 + 国内 PyPI 镜像加速)

---

## 4. Prometheus 运行时遥测与服务端状态探测

在大并发或长上下文推理评测中，仅从客户端观察响应耗时无法定位真实的性能瓶颈。本套件内置了面向 **vLLM** 和 **SGLang** 的 Prometheus Exposition 格式遥测抓取器 (`PrometheusTelemetryScraper`)。

### 4.1 开启 Prometheus 遥测

在命令行中指定 `--metrics-url` 参数（例如目标推理引擎的 `/metrics` 接口）：
```bash
python3 llm_test_backend.py --benchmark \
  --api-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --concurrency 32 \
  --metrics-url http://localhost:8000/metrics
```

### 4.2 采集的关键度量指标

抓取器每 1.0 秒对服务端进行一次非侵入式后台采样，精确解析以下指标并剔除 Tensor Parallelism (TP) 多卡重复统计：

1. **GPU KV Cache 显存占用率**:
   - vLLM 指标: `vllm:gpu_cache_usage_factor` (0.0 ~ 1.0)
   - SGLang 指标: `sglang:utilization` 或 `sglang:token_usage`
2. **服务端排队等待请求数 (Waiting Queue)**:
   - vLLM 指标: `vllm:num_requests_waiting`
   - SGLang 指标: `sglang:num_queue_reqs`
3. **服务端当前正在处理请求数 (Running Requests)**:
   - vLLM 指标: `vllm:num_requests_running`
   - SGLang 指标: `sglang:num_running_reqs`
4. **服务端内部报告生成吞吐**:
   - vLLM 指标: `vllm:avg_generation_throughput_tok_per_s`
   - SGLang 指标: `sglang:gen_throughput`

### 4.3 有效并发率判定与 `(X/Y)*` 告警

当客户端配置高并发（如 $Y = 32$），但服务端内部调度槽位受限（例如 vLLM `--max-num-seqs 8` 或 llama.cpp `--parallel 1`）时，Prometheus 遥测会探测到 `num_requests_waiting > 0` 且 `num_requests_running < Y`。

此时，套件会自动计算推断的真实并行度 $X = P_{\text{inferred}}$，并在控制台输出与天梯榜结果中打上 **`(X/Y)*`** 标识（如 `(8/32)*`）。这明确告知评测人员：**并发吞吐未能线性提升并非网络瓶颈，而是服务端推理调度槽位已达饱和上限！**

### 4.4 容错回退机制 (Graceful Fallback)

若目标服务的 `/metrics` 接口未开启、返回 404 或网络不可达，套件会输出提示：
```text
[Telemetry] Prometheus endpoint unreachable; gracefully falling back to client-only timings.
```
测试流程将无缝切换为纯客户端物理时间戳推断模式，绝不会中断或导致评测失败。

---

## 5. 死循环熔断保护器 (Repetition Watchdog)

在对 DeepSeek-R1 等长思维链思考模型或低量化（AWQ / GGUF）模型进行压力测试时，偶发提示词可能诱发模型进入无限重复输出状态（如连续输出数千次相同短语或符号）。

### 5.1 运作机制
- **周期检测算法 (`exact_periodic_tail`)**: 以 256 字符重块为触发点，在最近 128 KiB 输出流中通过 64 字符锚点逆向寻找周期 $P \in [1, 32768]$，并通过二分法锁定最大重复跨度 $L^*$；
- **熔断条件**: 当循环周期数 $\ge 4$ 且重复跨度 $\ge 4096$ 字符时，判定为无限死循环；
- **熔断行为**: 立即关闭 HTTP/2 连接终止接收，在单元格与日志中标记 `ERROR: loop`，并赋予哨兵吞吐值 `aggregate_tps = -3.0`，有效保护显卡显存并剔除异常数据。

### 5.2 相关控制参数
- `--no-watchdog`: 禁用循环熔断保护器（默认处于开启状态）；
- `--cjk-watchdog`: 开启思考阶段非预期 CJK 汉字漂移拦截。

---

## 6. CLI 完整参数清单

```text
usage: llm_test_backend.py [-h] [--server] [--port PORT] [--host HOST]
                           [--benchmark]
                           [--benchmark-mode {1d_sweep,2d_matrix,sustained_decode}]
                           [--matrix-concurrency MATRIX_CONCURRENCY]
                           [--matrix-context MATRIX_CONTEXT]
                           [--matrix-max-kv-tokens MATRIX_MAX_KV_TOKENS]
                           [--sustained-duration SUSTAINED_DURATION]
                           [--sustained-prompt-length SUSTAINED_PROMPT_LENGTH]
                           [--model MODEL] [--api-url API_URL]
                           [--api-key API_KEY] [--api-type {openai,ollama}]
                           [--concurrency CONCURRENCY] [--contexts CONTEXTS]
                           [--max-tokens MAX_TOKENS]
                           [--temperature TEMPERATURE] [--top-p TOP_P]
                           [--metrics-url METRICS_URL] [--hardware HARDWARE]
                           [--framework FRAMEWORK]
                           [--quantization QUANTIZATION] [--upload]
                           [--user-code USER_CODE] [--nickname NICKNAME]
                           [--notes NOTES] [--run-command RUN_COMMAND]
                           [--output OUTPUT] [--no-watchdog]
                           [--timeout TIMEOUT]
```

### 选项分类速查

| 参数分类 | 参数选项 | 说明 |
|---|---|---|
| **模式控制** | `--benchmark` / `--cli` | 启动无头 CLI 基准压测模式 |
| | `--server` / `--serve` | 启动本地 Web / WebSocket 服务模式 |
| | `--port PORT` | 指定服务端口（默认 18000） |
| | `--host HOST` | 本地服务监听地址（默认 `0.0.0.0`） |
| | `--benchmark-mode`, `--mode` | 评测模式选择：`1d_sweep` (默认)、`2d_matrix`、`sustained_decode` |
| | `--task-type`, `--task` | 评测场景题库选择：`general` (默认通用)、`code` (代码生成)、`structured` (结构化JSON)、`prose` (散文深度问答)、`prefill_16k` (16K长上下文首字)、`reasoning` (思维链推理) |
| **2D 矩阵参数** | `--matrix-concurrency` | 2D 矩阵并发梯队，逗号分隔（默认 `'1,2,4,8'`） |
| | `--matrix-context` | 2D 矩阵上下文长度梯队，逗号分隔（默认 `'512,1024,2048'`） |
| | `--matrix-max-kv-tokens`, `--max-kv-tokens` | KV Cache 显存容量跳过上限（默认 `32768`，设为 0 取消限制） |
| **稳态解码参数** | `--sustained-duration`, `--duration` | 稳态持续解码测试窗口时长（秒，默认 `30.0`） |
| | `--sustained-prompt-length`, `--sustained-context` | 稳态解码单次请求输入 Prompt Token 长度（默认 `128`） |
| **API 配置** | `--api-url API_URL` | 推理服务根 URL（默认 `http://localhost:8000/v1`） |
| | `--api-key API_KEY` | API 鉴权令牌密钥 |
| | `--api-type` | 通信协议（`openai` 或 `ollama`） |
| | `--model MODEL` | 目标模型名称（CLI 模式必填） |
| **基础压测参数** | `--concurrency -c` | 1D 模式并发数或阶梯（如 `1` 或 `1,4,16,64`） |
| | `--contexts -l` | 1D 模式输入 Prompt 长度阶梯（如 `100,500,2000`） |
| | `--max-tokens -m` | 最大生成 Token 数量（默认 128） |
| | `--temperature` | 采样温度（默认 0.7） |
| | `--top-p` | 采样 Top-P（默认 0.9） |
| | `--timeout` | 请求超时时间（毫秒，默认 120000） |
| **服务端遥测** | `--metrics-url` | vLLM / SGLang 的 Prometheus `/metrics` 地址 |
| **守卫控制** | `--no-watchdog` | 关闭死循环熔断保护器 |
| **天梯榜上传** | `--upload` | 自动直传 Supabase 全网天梯榜 |
| | `--user-code` | 8-16 位用户身份唯一编码 |
| | `--nickname` | 上榜昵称或实验室标识 |
| | `--hardware` | 硬件名称（留空则自动检测 GPU/CPU） |
| | `--framework` | 推理框架（vLLM, SGLang, Ollama 等） |
| | `--quantization` | 量化精度（FP16, BF16, AWQ, GGUF 等） |
| | `--notes` | 运行说明备注 |
| | `--run-command` | 服务端启动命令记录 |
| | `--output -o` | 本地 JSON 评测报告输出路径 |

---

## 7. 常见问题排查 (Troubleshooting)

### Q1: 运行提示 `ImportError: No module named 'h2'`
**解决**：高并发连接池需使用 HTTP/2 模块支持，请运行：
```bash
pip install h2>=4.1.0
# 或安装完整依赖清单：
pip install -r python/requirements.txt
```

### Q2: 报错 `Address already in use` 端口冲突
**解决**：默认端口为 18000。若该端口被其他服务占用，可指定任意空闲端口：
```bash
python3 llm_test_backend.py --server --port 18888
```

### Q3: 压测并发超过 32 后大量请求报错超时
**解决**：
1. 检查被测服务启动参数，确认推理服务显存池（如 vLLM `--gpu-memory-utilization`）是否充足；
2. 确认 vLLM 的 `--max-num-seqs` 是否允许足够并发槽位；
3. 检查单请求超时设置，可通过 `--timeout 300000` 将超时延长至 5 分钟。

### Q4: 本地离线环境测试结果如何保存？
**解决**：指定 `--output report.json` 参数，基准测试结果将完整导出为结构化 JSON 文件（包含矩阵每个单元格的 TTFT、ITL P50/P90/P99、真实墙上吞吐与验证哈希），便于离线分析与归档。
