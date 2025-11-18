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
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
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
    first_token_time = None
    output_content = ""
    reasoning_content = ""
    actual_output_tokens = None
    actual_prompt_tokens = None
    reasoning_chunk_count = 0
    content_chunk_count = 0
    server_prefill_time_ms = None
    server_decode_time_ms = None
    usage_info = None
    
    try:
        print(f"[Request] 发送请求到 {api_url}")
        async with httpx.AsyncClient(timeout=timeout/1000.0) as client:
            async with client.stream("POST", api_url, headers=headers, json=payload) as response:
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
                    
                    if not line.startswith("data: "):
                        continue
                    
                    if "[DONE]" in line:
                        print(f"[Stream] 收到 [DONE]，共处理 {chunk_count} 个chunk")
                        break
                    
                    try:
                        json_line = line.replace("data: ", "")
                        data = json.loads(json_line)
                        chunk_count += 1
                        
                        # 提取usage信息
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
                        if data.get("choices") and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            content_text = None
                            is_reasoning = False
                            
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
                            
                            if content_text:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                    print(f"[FirstToken] 接收到首个token，耗时: {(first_token_time - start_time)*1000:.2f}ms")
                                
                                if is_reasoning:
                                    reasoning_content += content_text
                                else:
                                    output_content += content_text
                    
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.perf_counter()
        
        print(f"[Response] 接收完成 - 内容长度: {len(output_content)}, Reasoning: {len(reasoning_content)}")
        
        # 计算指标
        if first_token_time is None:
            first_token_time = end_time
        
        ttft_ms = (first_token_time - start_time) * 1000
        total_time_ms = (end_time - start_time) * 1000
        
        print(f"[Timing] TTFT: {ttft_ms:.2f}ms, Total: {total_time_ms:.2f}ms")
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
        
        # 计算时间：完全使用端到端测量
        # vLLM的timing字段（prompt_ms/predicted_ms）不可靠，与实际差异很大
        # 原版HTML也是这样处理的：优先服务器timing，但vLLM一般会fallback到端到端
        
        prefill_time_ms = ttft_ms  # TTFT = prefill时间
        output_time_ms = max(total_time_ms - ttft_ms, 1)  # 总时间 - TTFT = decode时间
        
        time_source = '端到端测量'
        print(f"[TimeSource] 使用端到端测量 - Prefill(TTFT): {prefill_time_ms:.2f}ms, Decode: {output_time_ms:.2f}ms")
        
        # 如果有服务器timing，也打印出来作为参考（但不使用）
        if server_prefill_time_ms and server_decode_time_ms:
            print(f"[TimeSource] 服务器timing（仅参考）- Prefill: {server_prefill_time_ms:.2f}ms, Decode: {server_decode_time_ms:.2f}ms")
        
        # 计算速度 (tokens/second)
        prefill_speed = (actual_prompt_tokens / (prefill_time_ms / 1000))
        output_speed = (actual_output_tokens / (output_time_ms / 1000)) if actual_output_tokens > 0 else 0
        
        print(f"[Result] Prefill速度: {prefill_speed:.2f} t/s, Decode速度: {output_speed:.2f} t/s")
        
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
            # 添加绝对时间戳用于并发总吞吐计算
            "start_timestamp": start_time,
            "first_token_timestamp": first_token_time,
            "end_timestamp": end_time,
            "token_source": token_source  # 添加token来源
        }
    
    except Exception as e:
        error_msg = f"请求异常: {str(e)}"
        print(f"[Error] {error_msg}")
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
        print(f"[Config] 提示词长度: {config.min_length}-{config.max_length} (步长{config.step})")
        
        await websocket.send_json({
            "type": "info",
            "message": f"开始测试，并发数: {config.concurrency}"
        })
        
        # 计算测试点
        test_lengths = list(range(config.min_length, config.max_length + 1, config.step))
        print(f"[TestLengths] 测试点列表: {test_lengths}")
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
            tasks = []
            for i in range(config.concurrency):
                task = execute_single_request(
                    api_url=config.api_url,
                    api_key=config.api_key,
                    api_type=config.api_type,
                    model_name=config.model_name,
                    prompt_length=length,
                    output_length=config.output_length,
                    timeout=config.timeout,
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
            
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                elif result.get("success"):
                    successful_results.append(result)
                else:
                    failed_count += 1
            
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
