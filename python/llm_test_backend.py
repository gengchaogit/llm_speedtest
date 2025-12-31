"""
LLM Speed Test Backend
使用Python绕过浏览器并发限制，支持真正的高并发测试
"""
import asyncio
import time
import json
import random
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# 单词库用于生成提示词（避免cache命中）
WORD_LIST = [
    # 科技与创新
    "algorithm", "artificial", "automation", "blockchain", "compute", "digital", "innovation",
    "quantum", "robotics", "software", "technology", "virtual", "network", "database",
    # 自然与环境
    "mountain", "ocean", "forest", "planet", "climate", "wildlife", "ecosystem", "atmosphere",
    "renewable", "sustainable", "biological", "natural", "organic", "environment",
    # 社会与人文
    "community", "society", "culture", "tradition", "diversity", "equality", "justice",
    "democracy", "freedom", "humanity", "civilization", "education", "heritage", "philosophy",
    # 商业与经济
    "economy", "finance", "market", "investment", "enterprise", "commerce", "industry",
    "revenue", "strategy", "competition", "management", "resource", "capital", "prosperity",
    # 科学与知识
    "research", "science", "discovery", "experiment", "theory", "hypothesis", "evidence",
    "analysis", "knowledge", "wisdom", "intelligence", "learning", "academic", "scholarship",
    # 艺术与创造
    "creative", "artistic", "imagination", "aesthetic", "expression", "inspiration", "design",
    "architecture", "literature", "poetry", "painting", "sculpture", "performance", "melody",
    # 情感与心理
    "emotion", "passion", "empathy", "compassion", "mindfulness", "awareness", "consciousness",
    "perception", "intuition", "reflection", "meditation", "happiness", "serenity", "gratitude",
    # 时间与空间
    "moment", "eternal", "temporal", "spatial", "dimension", "horizon", "infinity",
    "universe", "cosmos", "reality", "existence", "journey", "destiny", "evolution",
    # 行动与发展
    "action", "progress", "development", "advancement", "achievement", "success", "excellence",
    "improvement", "transformation", "revolution", "growth", "expansion", "breakthrough", "pioneer",
    # 关系与连接
    "connection", "relationship", "interaction", "collaboration", "communication", "cooperation",
    "harmony", "unity", "solidarity", "partnership", "network", "community", "integration", "bond"
]

app = FastAPI(title="LLM Speed Test Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse
import os

current_port = 18000

@app.get("/")
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "LLM_Speed_Test_v2_Python_Backend.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Frontend HTML not found"}

@app.get("/api/port")
async def get_port():
    """返回当前后端端口号"""
    return {"port": current_port}


class TestConfig(BaseModel):
    api_url: str
    model_name: str
    api_key: str = ""
    api_type: str = "openai"
    min_length: int
    max_length: int
    step: int
    test_lengths: list = None  # 可选的测试点列表，如果提供则忽略min/max/step
    output_length: int
    concurrency: int
    timeout: int
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def estimate_token_count(text: str) -> int:
    """估算文本的token数量（当API不返回usage时使用）"""
    if not text or len(text) == 0:
        return 0

    # 统计中文字符
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fa5'])
    # 其他字符
    other_chars = len(text) - chinese_chars

    # 粗略估算：中文约1.5字符=1token，英文约4字符=1token
    estimated_tokens = round(chinese_chars / 1.5 + other_chars / 4)

    return max(1, estimated_tokens)


def calculate_dynamic_timeout(prompt_length: int, base_timeout: int) -> int:
    """
    根据prompt长度动态计算超时时间

    规则（以65K tokens = 30分钟为基准）：
    - 1K tokens: ~27.7秒
    - 2K tokens: ~55.4秒
    - 4K tokens: ~1.85分钟
    - 8K tokens: ~3.69分钟
    - 16K tokens: ~7.38分钟
    - 32K tokens: ~14.77分钟
    - 65K tokens: 30分钟
    - 128K (65K*2): 60分钟
    - 256K (65K*4): 120分钟

    对于 >= 1K tokens: 按比例动态计算
    对于 < 1K tokens: 使用配置的base_timeout
    """
    import math

    # 基准点：65K tokens = 30分钟
    REFERENCE_TOKENS = 65536
    REFERENCE_TIME_MS = 30 * 60 * 1000  # 1800000 ms

    if prompt_length >= 1024:  # >= 1K tokens
        if prompt_length <= REFERENCE_TOKENS:
            # 1K - 65K: 线性计算
            # timeout = (prompt_length / 65536) * 30分钟
            calculated_timeout = int((prompt_length / REFERENCE_TOKENS) * REFERENCE_TIME_MS)
            print(f"[Timeout] Prompt长度 {prompt_length} -> 动态超时: {calculated_timeout}ms ({calculated_timeout/1000:.1f}秒 / {calculated_timeout/60000:.2f}分钟)", flush=True)
            return calculated_timeout
        else:
            # > 65K: 按2的幂次翻倍
            ratio = prompt_length / REFERENCE_TOKENS
            power = math.ceil(math.log2(ratio))
            timeout_multiplier = 2 ** power
            calculated_timeout = int(REFERENCE_TIME_MS * timeout_multiplier)
            print(f"[Timeout] Prompt长度 {prompt_length} -> 动态超时: {calculated_timeout}ms ({calculated_timeout/60000:.1f}分钟)", flush=True)
            return calculated_timeout
    else:
        # < 1K tokens: 使用配置的超时
        print(f"[Timeout] Prompt长度 {prompt_length} -> 使用配置超时: {base_timeout}ms", flush=True)
        return base_timeout


def generate_prompt(length: int, seed: int = 0) -> str:
    """生成随机prompt，避免cache命中"""
    words = []
    
    # 添加唯一前缀（并发测试时避免cache）
    if seed > 0:
        words.append(f"[Request-{seed}-{int(time.time() * 1000)}]")
    
    # 随机选择单词
    for _ in range(length - 20):
        words.append(random.choice(WORD_LIST))
    
    prompt = " ".join(words)
    prompt += "\nBased on the words above, write a short philosophical essay discussing the meaning of existence, the nature of consciousness, and humanity's place in the universe. Use clear, coherent sentences."
    
    return prompt


async def execute_single_request(
    api_url: str,
    api_key: str,
    api_type: str,
    model_name: str,
    prompt_length: int,
    output_length: int,
    timeout: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """执行单个测试请求"""
    
    print(f"[Request] 开始请求 - Prompt长度: {prompt_length}, 输出长度: {output_length}, Seed: {seed}")
    
    # 使用随机单词生成prompt，避免cache
    prompt_text = generate_prompt(prompt_length, seed)
    
    if api_type == "openai":
        headers = {
            "Content-Type": "application/json",
        }
        # 只有当api_key非空时才添加Authorization头
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": output_length,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": True,
            "stream_options": {"include_usage": True}  # 请求返回usage信息
        }
    else:  # ollama
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            "options": {
                "num_predict": output_length,
                "temperature": temperature,
                "top_p": top_p,
            },
            "stream": True
        }
    
    start_time = time.perf_counter()
    first_chunk_time = None  # 第一个chunk到达时间（prefill结束）
    first_token_time = None  # 第一个有内容token的时间（用于ITL）
    output_content = ""
    reasoning_content = ""
    actual_output_tokens = None
    actual_prompt_tokens = None
    reasoning_chunk_count = 0
    content_chunk_count = 0
    server_prefill_time_ms = None
    server_decode_time_ms = None
    usage_info = None

    # ITL (Inter-Token Latency) tracking
    token_timestamps = []  # 记录每个token的时间戳
    last_token_time = None  # 上一个token的时间戳
    
    try:
        print(f"[Request] 发送请求到 {api_url}", flush=True)
        print(f"[Request] Payload大小预估: {len(json.dumps(payload))} 字节", flush=True)
        print(f"[Request] Headers: {headers}", flush=True)

        # 配置httpx以支持大payload和长时间请求
        # httpx 0.13.x 使用不同的Timeout API
        # 使用简单的超时值，httpx会自动应用到各个阶段
        timeout_seconds = timeout / 1000.0  # 转换为秒

        print(f"[Request] 创建httpx客户端 (超时: {timeout_seconds}秒)...", flush=True)
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            verify=False
        ) as client:
            print(f"[Request] 开始发送POST请求...", flush=True)
            async with client.stream("POST", api_url, headers=headers, json=payload) as response:
                print(f"[Response] 收到响应，状态码: {response.status_code}", flush=True)
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = f"HTTP {response.status_code}: {error_text.decode()}"
                    print(f"[Error] {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg
                    }
                
                print(f"[Response] 开始接收流式数据...")
                
                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # Log raw streamed line for debugging
                    # print(f"[RawLine] {line.strip()}")

                    # 处理OpenAI SSE格式
                    if line.startswith("data: "):
                        if "[DONE]" in line:
                            print(f"[Stream] 收到 [DONE]，共处理 {chunk_count} 个chunk")
                            break
                        json_line = line.replace("data: ", "")
                    else:
                        # Ollama直接返回JSON，不需要处理SSE格式
                        json_line = line.strip()

                    try:
                        data = json.loads(json_line)
                        chunk_count += 1

                        # 检查ollama的done字段
                        if data.get("done") is True:
                            print(f"[Stream] Ollama返回done=true，共处理 {chunk_count} 个chunk")
                            # 继续处理这个chunk（包含最终的timing信息），不要break

                        # 记录第一个chunk到达时间（prefill结束时间）
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                            ttft_ms = (first_chunk_time - start_time) * 1000
                            print(f"[FirstChunk] 接收到首个chunk（prefill完成），耗时: {ttft_ms:.2f}ms")

                        # 提取usage信息（OpenAI格式）
                        if data.get("usage"):
                            usage = data["usage"]
                            completion = usage.get("completion_tokens", 0)
                            reasoning = usage.get("reasoning_tokens", 0)
                            prompt = usage.get("prompt_tokens", 0)

                            print(f"[Usage] 找到usage字段！prompt_tokens: {prompt}, completion_tokens: {completion}, reasoning_tokens: {reasoning}")
                            print(f"[Debug] 完整usage字段: {json.dumps(usage, indent=2)}")

                            actual_output_tokens = completion + reasoning
                            actual_prompt_tokens = prompt
                            usage_info = usage.copy()

                            # 提取服务器timing (多种可能的字段名)
                            # llama.cpp/Ollama格式：纳秒
                            if usage.get("prompt_eval_duration"):
                                server_prefill_time_ms = usage["prompt_eval_duration"] / 1_000_000
                                print(f"[Usage] 找到 prompt_eval_duration: {server_prefill_time_ms:.2f}ms")
                            elif usage.get("prompt_eval_time"):
                                server_prefill_time_ms = usage["prompt_eval_time"]
                                print(f"[Usage] 找到 prompt_eval_time: {server_prefill_time_ms:.2f}ms")
                            elif usage.get("prompt_time"):
                                server_prefill_time_ms = usage["prompt_time"] * 1000  # 秒转毫秒
                                print(f"[Usage] 找到 prompt_time: {server_prefill_time_ms:.2f}ms")

                            if usage.get("eval_duration"):
                                server_decode_time_ms = usage["eval_duration"] / 1_000_000
                                print(f"[Usage] 找到 eval_duration: {server_decode_time_ms:.2f}ms")
                            elif usage.get("eval_time"):
                                server_decode_time_ms = usage["eval_time"]
                                print(f"[Usage] 找到 eval_time: {server_decode_time_ms:.2f}ms")
                            elif usage.get("completion_time"):
                                server_decode_time_ms = usage["completion_time"] * 1000  # 秒转毫秒
                                print(f"[Usage] 找到 completion_time: {server_decode_time_ms:.2f}ms")

                            if data.get("timings"):
                                usage_info["timings"] = data["timings"]
                                print(f"[Debug] 完整timings字段: {json.dumps(data['timings'], indent=2)}")

                        # 提取usage信息（Ollama格式 - 直接在顶层）
                        if data.get("prompt_eval_count") is not None:
                            prompt = data.get("prompt_eval_count", 0)
                            completion = data.get("eval_count", 0)

                            print(f"[Usage-Ollama] 找到ollama格式！prompt_eval_count: {prompt}, eval_count: {completion}")

                            actual_output_tokens = completion
                            actual_prompt_tokens = prompt

                            # 构造usage_info
                            usage_info = {
                                "prompt_tokens": prompt,
                                "completion_tokens": completion,
                                "total_tokens": prompt + completion
                            }

                            # 提取ollama的timing信息（纳秒）
                            if data.get("prompt_eval_duration"):
                                server_prefill_time_ms = data["prompt_eval_duration"] / 1_000_000
                                print(f"[Usage-Ollama] 找到 prompt_eval_duration: {server_prefill_time_ms:.2f}ms")
                                usage_info["prompt_eval_duration_ns"] = data["prompt_eval_duration"]

                            if data.get("eval_duration"):
                                server_decode_time_ms = data["eval_duration"] / 1_000_000
                                print(f"[Usage-Ollama] 找到 eval_duration: {server_decode_time_ms:.2f}ms")
                                usage_info["eval_duration_ns"] = data["eval_duration"]

                            if data.get("total_duration"):
                                usage_info["total_duration_ns"] = data["total_duration"]
                                print(f"[Usage-Ollama] total_duration: {data['total_duration'] / 1_000_000:.2f}ms")

                            if data.get("load_duration"):
                                usage_info["load_duration_ns"] = data["load_duration"]
                                print(f"[Usage-Ollama] load_duration: {data['load_duration'] / 1_000_000:.2f}ms")
                        
                        # 提取timings（顶层，某些API实现可能放在这里）
                        if data.get("timings") and not (server_prefill_time_ms and server_decode_time_ms):
                            timings = data["timings"]
                            print(f"[Timings] 检查顶层timings字段...")
                            
                            if not server_prefill_time_ms:
                                if timings.get("prompt_eval_duration"):
                                    server_prefill_time_ms = timings["prompt_eval_duration"] / 1_000_000
                                    print(f"[Timings] 找到 prompt_eval_duration: {server_prefill_time_ms:.2f}ms")
                                elif timings.get("prompt_ms"):
                                    server_prefill_time_ms = timings["prompt_ms"]
                                    print(f"[Timings] 找到 prompt_ms: {server_prefill_time_ms:.2f}ms")
                            
                            if not server_decode_time_ms:
                                if timings.get("eval_duration"):
                                    server_decode_time_ms = timings["eval_duration"] / 1_000_000
                                    print(f"[Timings] 找到 eval_duration: {server_decode_time_ms:.2f}ms")
                                elif timings.get("predicted_ms"):
                                    server_decode_time_ms = timings["predicted_ms"]
                                    print(f"[Timings] 找到 predicted_ms: {server_decode_time_ms:.2f}ms")
                        
                        # 提取内容
                        content_text = None
                        is_reasoning = False
                        # Handle Ollama streaming response field
                        if data.get("response") is not None:
                            content_text = data["response"]
                            content_chunk_count += 1
                        
                        if data.get("choices") and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            
                            if choice.get("delta", {}).get("reasoning_content"):
                                content_text = choice["delta"]["reasoning_content"]
                                is_reasoning = True
                                reasoning_chunk_count += 1
                            elif choice.get("delta", {}).get("content"):
                                content_text = choice["delta"]["content"]
                                content_chunk_count += 1
                            elif choice.get("message", {}).get("content"):
                                content_text = choice["message"]["content"]
                                content_chunk_count += 1
                            elif choice.get("text"):
                                content_text = choice["text"]
                                content_chunk_count += 1
                        elif data.get("message"):
                            message = data["message"]
                            if message.get("thinking"):
                                content_text = message["thinking"]
                                is_reasoning = True
                                reasoning_chunk_count += 1
                            elif message.get("content"):
                                content_text = message["content"]
                                content_chunk_count += 1
                        
                        if content_text:
                                current_token_time = time.perf_counter()

                                if first_token_time is None:
                                    first_token_time = current_token_time
                                    token_latency_ms = (first_token_time - start_time) * 1000
                                    print(f"[FirstToken] 接收到首个内容token，耗时: {token_latency_ms:.2f}ms")

                                # 记录token时间戳和ITL
                                token_timestamps.append({
                                    "timestamp": current_token_time,
                                    "time_from_start_ms": (current_token_time - start_time) * 1000,
                                    "is_reasoning": is_reasoning
                                })

                                # 计算ITL（如果不是第一个token）
                                if last_token_time is not None:
                                    itl = (current_token_time - last_token_time) * 1000  # ms
                                    token_timestamps[-1]["itl_ms"] = itl
                                else:
                                    token_timestamps[-1]["itl_ms"] = None  # 第一个token没有ITL

                                last_token_time = current_token_time

                                if is_reasoning:
                                    reasoning_content += content_text
                                else:
                                    output_content += content_text

                        # 如果ollama标记done=true，处理完这个chunk后退出
                        if data.get("done") is True:
                            print(f"[Stream] Ollama done=true，退出循环")
                            break

                    except json.JSONDecodeError:
                        continue
        
        end_time = time.perf_counter()

        print(f"[Response] 接收完成 - 内容长度: {len(output_content)}, Reasoning: {len(reasoning_content)}")

        # 计算指标 - 使用第一个chunk时间作为prefill结束标志
        total_time_ms = (end_time - start_time) * 1000

        if first_chunk_time is not None:
            # 有chunk数据 - 可以计算准确的prefill和decode时间
            ttft_ms = (first_chunk_time - start_time) * 1000
            decode_time_ms = total_time_ms - ttft_ms  # decode时间 = 总时间 - prefill时间

            # 确保decode时间至少为1ms（避免除零），但保留真实时间用于显示
            real_decode_time_ms = decode_time_ms
            decode_time_ms = max(decode_time_ms, 1)

            if first_token_time is not None:
                # 有实际内容token
                token_first_ms = (first_token_time - start_time) * 1000
                print(f"[Timing] TTFT(首个chunk): {ttft_ms:.2f}ms, 首个内容token: {token_first_ms:.2f}ms, Total: {total_time_ms:.2f}ms")
            else:
                # 没有内容token，但有chunk（服务器完成了处理）
                print(f"[Timing] TTFT(首个chunk): {ttft_ms:.2f}ms, Total: {total_time_ms:.2f}ms, Decode: {real_decode_time_ms:.2f}ms")
                if chunk_count > 0 and actual_output_tokens and actual_output_tokens > 0:
                    if real_decode_time_ms < 10:
                        print(f"[Warning] 收到 {chunk_count} 个chunk，usage显示 {actual_output_tokens} tokens，但decode时间极短 (可能非流式)")
                    else:
                        print(f"[Warning] 收到 {chunk_count} 个chunk，usage显示 {actual_output_tokens} tokens，但无流式内容")
        else:
            # 完全没有收到chunk - 异常情况
            ttft_ms = total_time_ms  # fallback
            decode_time_ms = 1
            print(f"[Timing] 异常：未收到任何chunk, Total: {total_time_ms:.2f}ms")

        print(f"[Timing] Server Prefill: {server_prefill_time_ms}ms, Server Decode: {server_decode_time_ms}ms")
        
        # Token统计
        token_source = ''
        if actual_prompt_tokens is None:
            # Fallback: 使用prompt_length作为估算（因为我们控制了prompt生成）
            actual_prompt_tokens = prompt_length
            print(f"[Token] Prompt估算: {actual_prompt_tokens} (使用设定长度)")
        else:
            print(f"[Token] Prompt来自usage: {actual_prompt_tokens}")

        if actual_output_tokens is None:
            # 使用精确的token估算函数
            reasoning_tokens = estimate_token_count(reasoning_content)
            completion_tokens = estimate_token_count(output_content)
            actual_output_tokens = reasoning_tokens + completion_tokens
            token_source = 'Local Estimation'
            print(f"[Token] Output估算: {actual_output_tokens} (reasoning: {reasoning_tokens}, completion: {completion_tokens})")
        else:
            token_source = 'API'
            print(f"[Token] Output来自usage: {actual_output_tokens}")

        if not token_source:
            token_source = 'Unknown'

        # 计算时间和速度：优先使用服务器返回的timing，否则使用客户端测量

        # 优先使用服务器返回的timing
        if server_prefill_time_ms is not None and server_decode_time_ms is not None:
            prefill_time_ms = server_prefill_time_ms
            output_time_ms = server_decode_time_ms
            time_source = '服务器timing'
            print(f"[TimeSource] 使用服务器timing - Prefill: {prefill_time_ms:.2f}ms, Decode: {output_time_ms:.2f}ms")
            print(f"[TimeSource] 客户端测量（参考）- Prefill(TTFT): {ttft_ms:.2f}ms, Decode: {decode_time_ms:.2f}ms")
        else:
            # 回退到客户端测量的时间
            prefill_time_ms = ttft_ms
            output_time_ms = decode_time_ms
            time_source = '端到端测量（基于chunk时序）'
            print(f"[TimeSource] 使用端到端测量 - Prefill(TTFT): {prefill_time_ms:.2f}ms, Decode: {output_time_ms:.2f}ms")
            if server_prefill_time_ms or server_decode_time_ms:
                print(f"[TimeSource] 部分服务器timing可用 - Prefill: {server_prefill_time_ms}ms, Decode: {server_decode_time_ms}ms")

        # 计算速度 (tokens/second)
        # 使用usage中的token数量，即使没有收到实际内容也计算
        if prefill_time_ms > 0:
            prefill_speed = (actual_prompt_tokens / (prefill_time_ms / 1000))
        else:
            prefill_speed = 0

        if output_time_ms > 0 and actual_output_tokens > 0:
            output_speed = (actual_output_tokens / (output_time_ms / 1000))
        else:
            output_speed = 0

        print(f"[Result] Prefill速度: {prefill_speed:.2f} t/s, Decode速度: {output_speed:.2f} t/s")

        # 添加数据质量警告
        if actual_output_tokens > 0 and len(output_content) == 0 and len(reasoning_content) == 0:
            print(f"[Warning] 速度基于usage ({actual_output_tokens} tokens)和timing计算，但未收到实际流式内容")
            if output_speed > 10000:  # 超过10000 t/s通常不合理
                print(f"[Warning] Decode速度异常高 ({output_speed:.2f} t/s)，可能表示服务器非真正流式生成")

        # 计算ITL统计
        itl_stats = {}
        if len(token_timestamps) > 1:
            itl_values = [t["itl_ms"] for t in token_timestamps if t["itl_ms"] is not None]
            if itl_values:
                itl_stats = {
                    "mean": round(sum(itl_values) / len(itl_values), 2),
                    "min": round(min(itl_values), 2),
                    "max": round(max(itl_values), 2),
                    "std": round((sum((x - sum(itl_values)/len(itl_values))**2 for x in itl_values) / len(itl_values))**0.5, 2) if len(itl_values) > 1 else 0,
                    "count": len(itl_values)
                }
                print(f"[ITL] 平均: {itl_stats['mean']}ms, 最小: {itl_stats['min']}ms, 最大: {itl_stats['max']}ms, 标准差: {itl_stats['std']}ms")

        return {
            "success": True,
            "prompt_length": prompt_length,
            "prompt_tokens": actual_prompt_tokens,
            "output_tokens": actual_output_tokens,
            "ttft_ms": round(ttft_ms, 2),
            "prefill_time_ms": round(prefill_time_ms, 2),
            "prefill_speed": round(prefill_speed, 2),
            "output_time_ms": round(output_time_ms, 2),
            "output_speed": round(output_speed, 2),
            "total_time_ms": round(total_time_ms, 2),
            "prompt_text": prompt_text,  # 添加实际的提示词文本
            "output_content": output_content,
            "reasoning_content": reasoning_content,
            "usage_info": usage_info,
            "server_timing_used": server_prefill_time_ms is not None,
            "has_streaming_content": first_token_time is not None,  # 是否有实际内容token
            "chunk_count": chunk_count,  # chunk数量
            # 添加绝对时间戳用于并发总吞吐计算
            "start_timestamp": start_time,
            "first_chunk_timestamp": first_chunk_time,  # chunk到达时间
            "first_token_timestamp": first_token_time,  # 内容token时间
            "end_timestamp": end_time,
            "token_source": token_source,  # 添加token来源
            # 新增ITL相关数据
            "token_timestamps": token_timestamps,  # 所有token的时间戳数据
            "itl_stats": itl_stats  # ITL统计信息
        }
    
    except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        error_msg = f"请求超时: {type(e).__name__}: {str(e)} (提示词长度: {prompt_length}, 超时设置: {timeout}ms)"
        print(f"[Error] {error_msg}", flush=True)
        import traceback
        print(f"[Error] 详细堆栈:\n{traceback.format_exc()}", flush=True)
        return {
            "success": False,
            "error": error_msg,
            "prompt_length": prompt_length
        }
    except httpx.HTTPError as e:
        error_msg = f"HTTP错误: {type(e).__name__}: {str(e)}"
        print(f"[Error] {error_msg}", flush=True)
        import traceback
        print(f"[Error] 详细堆栈:\n{traceback.format_exc()}", flush=True)
        return {
            "success": False,
            "error": error_msg,
            "prompt_length": prompt_length
        }
    except Exception as e:
        error_msg = f"请求异常: {type(e).__name__}: {str(e)}"
        print(f"[Error] {error_msg}", flush=True)
        import traceback
        print(f"[Error] 详细堆栈:\n{traceback.format_exc()}", flush=True)
        return {
            "success": False,
            "error": error_msg,
            "prompt_length": prompt_length
        }


@app.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """WebSocket端点，用于实时测试进度推送"""
    await websocket.accept()
    print(f"[WebSocket] 客户端已连接")
    
    try:
        config_data = await websocket.receive_json()
        print(f"[WebSocket] 收到原始数据: {config_data}")
        config = TestConfig(**config_data)
        
        print(f"[Config] 接收配置 - API: {config.api_type}, 模型: {config.model_name}, 并发: {config.concurrency}")
        
        # 使用提供的测试点列表，或计算测试点
        if config.test_lengths:
            test_lengths = config.test_lengths
            print(f"[Config] 使用自定义测试点列表: {test_lengths}")
        else:
            test_lengths = list(range(config.min_length, config.max_length + 1, config.step))
            print(f"[Config] 提示词长度: {config.min_length}-{config.max_length} (步长{config.step})")
            print(f"[TestLengths] 计算的测试点列表: {test_lengths}")
        
        await websocket.send_json({
            "type": "info",
            "message": f"开始测试，并发数: {config.concurrency}"
        })
        total_tests = len(test_lengths)
        completed = 0
        
        all_results = []
        
        for length in test_lengths:
            print(f"\n[Test] ===== 测试提示词长度: {length} ({completed+1}/{total_tests}) =====")
            
            await websocket.send_json({
                "type": "progress",
                "current": completed,
                "total": total_tests,
                "testing_length": length
            })
            
            # 创建并发任务（每个任务使用不同的seed避免cache）
            # 根据prompt长度动态计算超时时间
            dynamic_timeout = calculate_dynamic_timeout(length, config.timeout)

            tasks = []
            for i in range(config.concurrency):
                task = execute_single_request(
                    api_url=config.api_url,
                    api_key=config.api_key,
                    api_type=config.api_type,
                    model_name=config.model_name,
                    prompt_length=length,
                    output_length=config.output_length,
                    timeout=dynamic_timeout,  # 使用动态计算的超时
                    temperature=config.temperature,
                    top_p=config.top_p,
                    presence_penalty=config.presence_penalty,
                    frequency_penalty=config.frequency_penalty,
                    seed=i + 1,  # 每个并发请求使用不同seed
                )
                tasks.append(task)
            
            # 并发执行
            print(f"[Test] 启动 {config.concurrency} 个并发请求...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print(f"[Test] 并发请求完成")
            
            # 处理结果
            successful_results = []
            failed_count = 0

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_count += 1
                    print(f"[Error] 请求 #{idx+1} 抛出异常: {type(result).__name__}: {str(result)}", flush=True)
                    import traceback
                    print(f"[Error] 异常堆栈:\n{''.join(traceback.format_exception(type(result), result, result.__traceback__))}", flush=True)
                elif result.get("success"):
                    successful_results.append(result)
                else:
                    failed_count += 1
                    error_msg = result.get("error", "未知错误")
                    print(f"[Error] 请求 #{idx+1} 失败: {error_msg}", flush=True)
            
            if successful_results:
                # 计算聚合统计
                avg_prefill_speed = sum(r["prefill_speed"] for r in successful_results) / len(successful_results)
                avg_output_speed = sum(r["output_speed"] for r in successful_results) / len(successful_results)
                avg_ttft = sum(r["ttft_ms"] for r in successful_results) / len(successful_results)
                
                print(f"[Stats] 成功: {len(successful_results)}/{config.concurrency}")
                print(f"[Stats] 平均 Prefill速度: {avg_prefill_speed:.2f} t/s")
                print(f"[Stats] 平均 Decode速度: {avg_output_speed:.2f} t/s")
                print(f"[Stats] 平均 TTFT: {avg_ttft:.2f} ms")
                
                result_summary = {
                    "type": "result",
                    "prompt_length": length,
                    "concurrency": config.concurrency,
                    "successful": len(successful_results),
                    "failed": failed_count,
                    "avg_prefill_speed": round(avg_prefill_speed, 2),
                    "avg_output_speed": round(avg_output_speed, 2),
                    "avg_ttft_ms": round(avg_ttft, 2),
                    "concurrent_details": successful_results,
                    "status": "成功"
                }
                
                all_results.append(result_summary)
                await websocket.send_json(result_summary)
            else:
                print(f"[Error] 所有请求失败 - 失败数: {failed_count}")
                error_result = {
                    "type": "result",
                    "prompt_length": length,
                    "status": "失败",
                    "error": f"所有 {config.concurrency} 个并发请求都失败了"
                }
                all_results.append(error_result)
                await websocket.send_json(error_result)
            
            completed += 1
            
            # 测试间延迟
            if completed < total_tests:
                print(f"[Test] 等待 1.5 秒后进行下一个测试...")
                await asyncio.sleep(1.5)
        
        print(f"\n[Complete] ===== 所有测试完成 =====")
        print(f"[Complete] 总测试点: {total_tests}, 完成: {completed}")
        
        await websocket.send_json({
            "type": "complete",
            "message": "测试完成",
            "all_results": all_results
        })
    
    except WebSocketDisconnect:
        print("[WebSocket] 客户端断开连接")
    except Exception as e:
        print(f"[Error] WebSocket异常: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": f"测试错误: {str(e)}"
        })


def find_free_port(preferred_port=18000):
    """找到一个未占用的端口，优先使用preferred_port"""
    import socket
    
    # 先尝试使用首选端口
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', preferred_port))
            return preferred_port
    except OSError:
        # 端口被占用，选择随机端口
        pass
    
    # 选择随机未占用端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


if __name__ == "__main__":
    import uvicorn
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    else:
        port = find_free_port()
    
    # 设置全局端口变量
    current_port = port
    
    print("LLM Speed Test Backend Server", flush=True)
    print(f"WebSocket endpoint: ws://localhost:{port}/ws/test", flush=True)
    print(f"Frontend page: http://localhost:{port}/", flush=True)
    print(f"PORT={port}", flush=True)
    
    # 将端口写入配置文件，供bat脚本读取
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        port_file = os.path.join(script_dir, '.backend_port')
        with open(port_file, 'w') as f:
            f.write(str(port))
    except Exception as e:
        pass
    
    uvicorn.run(app, host="0.0.0.0", port=port)
