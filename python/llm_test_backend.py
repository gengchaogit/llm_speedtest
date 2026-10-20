"""
LLM Speed Test Backend
使用Python绕过浏览器并发限制，支持真正的高并发测试
"""
import asyncio
import math
import time
import json
import random
import re
import hashlib
import os
import sys
import platform
import argparse
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable, Awaitable, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import httpx
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from urllib.parse import urlparse, quote

__version__ = "3.5.0"

# ============================================================
# Supabase 配置（云端结果共享）
# ============================================================
SUPABASE_URL = "https://cvezaerrczywfzqcaqmx.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXphZXJyY3p5d2Z6cWNhcW14Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI0Nzc0OTAsImV4cCI6MjA4ODA1MzQ5MH0.-9l2Wcj38m6xh9sz9qMdhdNbnAZ_XoFBSHPhEud9DmQ"
# 注意：只使用公开的 anon key，不在代码中保存 service key
# RLS 策略负责权限控制（见 README 中的建表 SQL）
SUPABASE_TABLE = "speed_results"
PUBLIC_LEADERBOARD_URL = "https://gengchaogit.github.io/llm_speedtest/LLM_Speed_Test_v3_Leaderboard.html"

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

NVIDIA_MODEL_NAME_HINTS = {
    "minimax/minimax-m2.7": "minimaxai/minimax-m2.7",
    "minimax/minimax-m2.5": "minimaxai/minimax-m2.5",
}

app = FastAPI(title="LLM Speed Test Backend", version=__version__)

# 速率限制器
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse
import os
from datetime import datetime

current_port = 18000


def redact_sensitive_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of headers safe for debug logging."""
    sensitive_names = {"authorization", "apikey", "x-api-key"}
    redacted = dict(headers or {})
    for key, value in list(redacted.items()):
        if key.lower() not in sensitive_names:
            continue
        if not isinstance(value, str):
            redacted[key] = "***"
        elif len(value) > 16:
            redacted[key] = f"{value[:8]}...{value[-4:]}"
        else:
            redacted[key] = "***"
    return redacted


def redact_sensitive_text(text: str) -> str:
    """Redact credentials and provider account identifiers in debug/error text."""
    if not text:
        return text
    redacted = re.sub(r'(Bearer\s+)[A-Za-z0-9._~+/=-]+', r'\1<redacted>', text)
    redacted = re.sub(r'sk-[A-Za-z0-9._~+/=-]{12,}', '<redacted_api_key>', redacted)
    redacted = re.sub(r'("user_id"\s*:\s*")[^"]+(")', r'\1<redacted>\2', redacted)
    return redacted


def resolve_model_catalog_endpoint(api_url: str) -> tuple[Optional[str], bool]:
    """Resolve a chat endpoint into its corresponding model-list endpoint."""
    if not api_url:
        return None, False

    normalized = api_url.strip().rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized[:-len("/v1/chat/completions")] + "/v1/models", False
    if normalized.endswith("/v1/models"):
        return normalized, False
    if normalized.endswith("/api/chat"):
        return normalized[:-len("/api/chat")] + "/api/tags", True
    if normalized.endswith("/api/tags"):
        return normalized, True
    return None, False

@app.get("/")
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "LLM_Speed_Test_v3_Python_Backend.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Frontend HTML not found"}

@app.get("/LLM_Speed_Test_v3_Leaderboard.html")
async def get_leaderboard_page():
    base_dir = os.path.dirname(__file__)
    # Prefer the repository-root leaderboard so Python/backend and pure-web share one source.
    root_html_path = os.path.abspath(os.path.join(base_dir, "..", "LLM_Speed_Test_v3_Leaderboard.html"))
    local_html_path = os.path.join(base_dir, "LLM_Speed_Test_v3_Leaderboard.html")

    html_path = root_html_path if os.path.exists(root_html_path) else local_html_path
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Leaderboard HTML not found"}

@app.get("/api/port")
async def get_port():
    """返回当前后端端口号"""
    return {"port": current_port}


class ModelListPayload(BaseModel):
    api_url: str
    api_key: str = ""
    timeout_ms: int = 5000


@app.post("/api/models")
async def get_models(payload: ModelListPayload):
    endpoint, is_ollama = resolve_model_catalog_endpoint(payload.api_url)
    if not endpoint:
        raise HTTPException(
            status_code=400,
            detail="Unsupported API URL. Expected /v1/chat/completions, /v1/models, /api/chat, or /api/tags"
        )

    headers = {"Accept": "application/json"}
    if payload.api_key and payload.api_key.strip():
        headers["Authorization"] = f"Bearer {payload.api_key.strip()}"

    timeout_seconds = max(1.0, min(payload.timeout_ms, 15000) / 1000.0)
    print(f"[ModelSuggest] 拉取模型列表: {endpoint}", flush=True)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, verify=False, follow_redirects=True) as client:
            response = await client.get(endpoint, headers=headers)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch model list from {endpoint}: {exc}") from exc

    if response.status_code != 200:
        error_msg = build_http_error_message(endpoint, "", response.status_code, response.content)
        raise HTTPException(status_code=response.status_code, detail=error_msg)

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON returned by {endpoint}") from exc

    if is_ollama:
        names = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
    else:
        names = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]

    names = sorted({name for name in names if name})
    print(f"[ModelSuggest] 成功拉取 {len(names)} 个模型: {endpoint}", flush=True)
    return {"endpoint": endpoint, "models": names}


# ============================================================
# 云端结果共享 API
# ============================================================

class ResultPoint(BaseModel):
    prompt_length: int
    prefill_speed: float
    output_speed: float
    prefill_time_ms: float
    output_time_ms: float
    avg_ttft_ms: Optional[float] = None
    avg_itl_mean: Optional[float] = None
    avg_itl_std: Optional[float] = None
    concurrency: int = 1
    successful: Optional[int] = None
    boundary_source: Optional[str] = None
    concurrent_details: Optional[List[Dict]] = None
    effective_concurrency: Optional[str] = None
    verification_hash: Optional[str] = None
    loop_detected: Optional[bool] = False
    ttft_stats: Optional[Dict[str, Any]] = None
    itl_stats: Optional[Dict[str, Any]] = None
    telemetry: Optional[Dict[str, Any]] = None
    output_tokens: Optional[int] = None
    e2e_speed: Optional[float] = None
    pure_decode_time_sec: Optional[float] = None
    avg_stream_decode_speed: Optional[float] = None
    actual_duration_sec: Optional[float] = None
    task_type: Optional[str] = None

class UploadPayload(BaseModel):
    user_code: str
    nickname: Optional[str] = None
    model_name: str
    hardware: str
    framework: Optional[str] = None
    quantization: Optional[str] = None
    notes: Optional[str] = None
    run_command: Optional[str] = None
    concurrency: int
    task_type: Optional[str] = None
    avg_prefill_speed: float
    avg_decode_speed: float
    max_prefill_speed: Optional[float] = None
    max_decode_speed: Optional[float] = None
    source: Optional[str] = "python_backend"
    record_tags: Optional[List[str]] = None
    verification_hash: Optional[str] = None
    is_anomaly: Optional[bool] = False
    anomaly_reason: Optional[str] = None
    benchmark_mode: Optional[str] = "1d_sweep"
    matrix_grid: Optional[Dict[str, Any]] = None
    matrix_summary: Optional[Dict[str, Any]] = None
    all_matrices: Optional[Union[Dict[str, Any], List[Any]]] = None
    all_tasks_comparison: Optional[List[Dict[str, Any]]] = None
    sustained_summary: Optional[Dict[str, Any]] = None
    results: Optional[List[ResultPoint]] = None

    @field_validator('user_code')
    @classmethod
    def validate_user_code(cls, v):
        v = v.strip().upper()
        if len(v) < 8 or len(v) > 16 or not v.isalnum():
            raise ValueError('user_code 必须是8-16位字母数字')
        return v

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        v = v.strip()
        if not v or len(v) > 100:
            raise ValueError('model_name 必填且不超过100字符')
        return v

    @field_validator('hardware')
    @classmethod
    def validate_hardware(cls, v):
        v = v.strip()
        if not v or len(v) > 100:
            raise ValueError('hardware 必填且不超过100字符')
        return v

    @field_validator('avg_prefill_speed', 'avg_decode_speed', 'max_prefill_speed', 'max_decode_speed')
    @classmethod
    def validate_speed(cls, v):
        if v is None:
            return v
        if v < 0 or v > 2_000_000:
            raise ValueError('速度值超出合理范围 (0 ~ 2,000,000 t/s)')
        return v

    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        if v is None:
            return "python_backend"
        v = v.strip()
        if not v:
            return "python_backend"
        if len(v) > 50:
            raise ValueError('source must be <= 50 characters')
        return v

    @field_validator('notes')
    @classmethod
    def validate_notes(cls, v):
        if v and len(v) > 200:
            raise ValueError('notes 不超过200字符')
        return v

    @field_validator('run_command')
    @classmethod
    def validate_run_command(cls, v):
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > 2000:
            raise ValueError('run_command 不超过2000字符')
        return v

    @field_validator('nickname')
    @classmethod
    def validate_nickname(cls, v):
        if v and len(v) > 50:
            raise ValueError('nickname 不超过50字符')
        return v


def _get_ip_hash(request: Request) -> str:
    """对客户端 IP 做单向哈希，不存明文"""
    ip = get_remote_address(request) or "unknown"
    return hashlib.sha256(ip.encode()).hexdigest()[:32]


async def _supabase_request(method: str, path: str, extra_headers: dict = None, **kwargs):
    """向 Supabase REST API 发送请求（只使用 anon key）"""
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    if extra_headers:
        headers.update(extra_headers)
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.request(method, url, headers=headers, **kwargs)
    return resp


def compute_verification_hash(
    model_name: str,
    hardware: str,
    framework: Optional[str] = "",
    quantization: Optional[str] = "",
    concurrency: int = 1,
    max_decode_speed: float = 0.0,
    timestamp: Optional[int] = None,
) -> str:
    """
    Computes client verification hash matching PROJECT.md & spec_report.md contract:
    - Base: Platform/Hardware environmental signature
    - Payload: Canonical model, hardware, framework, concurrency, max_decode_speed, timestamp
    - Output: vh_ + SHA-256(fingerprint + payload)[:16]
    """
    if timestamp is None:
        timestamp = int(time.time())
    env_fp = f"{platform.platform()}|{platform.machine()}|{platform.processor()}|{os.cpu_count() or 0}"
    canon_payload = (
        f"model={model_name.strip()}|hw={hardware.strip()}|"
        f"fw={(framework or '').strip()}|qt={(quantization or '').strip()}|"
        f"c={concurrency}|decode={max_decode_speed:.2f}|t={timestamp}"
    )
    raw_hash = hashlib.sha256((env_fp + canon_payload).encode("utf-8")).hexdigest()
    return f"vh_{raw_hash[:16]}"


async def upload_benchmark_to_supabase(
    user_code: str,
    model_name: str,
    hardware: str,
    concurrency: int,
    avg_prefill_speed: float,
    avg_decode_speed: float,
    max_prefill_speed: Optional[float] = None,
    max_decode_speed: Optional[float] = None,
    framework: Optional[str] = None,
    quantization: Optional[str] = None,
    notes: Optional[str] = None,
    nickname: Optional[str] = None,
    run_command: Optional[str] = None,
    record_tags: Optional[List[str]] = None,
    results_json: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    verification_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload benchmark results directly to Supabase speed_results table via REST."""
    ts = int(time.time())
    max_dec = max_decode_speed if max_decode_speed is not None else avg_decode_speed
    max_pref = max_prefill_speed if max_prefill_speed is not None else avg_prefill_speed

    if not verification_hash:
        verification_hash = compute_verification_hash(
            model_name=model_name,
            hardware=hardware,
            framework=framework,
            quantization=quantization,
            concurrency=concurrency,
            max_decode_speed=max_dec,
            timestamp=ts,
        )

    tags = list(record_tags or [])
    if "source:python_v3" not in tags and "source:python_backend" not in tags:
        tags.insert(0, "source:python_v3")
    if "verified" not in tags:
        tags.append("verified")
    tags = list(dict.fromkeys(tags))

    # Ensure verification_hash and metadata are safely embedded inside results_json
    final_results_json = results_json if results_json is not None else []
    if isinstance(final_results_json, dict):
        if verification_hash:
            if "verification" not in final_results_json:
                final_results_json["verification"] = {
                    "hash": verification_hash,
                    "status": "verified",
                    "timestamp": ts,
                }
            elif isinstance(final_results_json.get("verification"), dict):
                final_results_json["verification"].setdefault("hash", verification_hash)
    elif isinstance(final_results_json, list):
        task_tag = next((t.split(":", 1)[1] for t in tags if t.startswith("task:")), "general")
        final_results_json = {
            "benchmark_mode": "1d_sweep",
            "task_type": task_tag,
            "verification": {
                "hash": verification_hash,
                "status": "verified",
                "timestamp": ts,
            },
            "points": final_results_json,
        }

    payload = {
        "user_code": user_code.strip().upper(),
        "nickname": nickname.strip() if nickname else None,
        "model_name": model_name.strip(),
        "hardware": hardware.strip(),
        "framework": framework.strip() if framework else None,
        "quantization": quantization.strip() if quantization else None,
        "notes": notes.strip() if notes else None,
        "run_command": run_command.strip() if run_command else None,
        "concurrency": concurrency,
        "avg_prefill_speed": round(avg_prefill_speed, 2),
        "avg_decode_speed": round(avg_decode_speed, 2),
        "max_prefill_speed": round(max_pref, 2),
        "max_decode_speed": round(max_dec, 2),
        "source": "python_v3",
        "record_tags": tags,
        "results_json": final_results_json,
    }

    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    async with httpx.AsyncClient(timeout=20, verify=False) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code in (200, 201):
            data = resp.json()
            record_id = data[0]["id"] if isinstance(data, list) else data.get("id")
            share_url = f"{PUBLIC_LEADERBOARD_URL}?id={record_id}"
            return {
                "success": True,
                "id": record_id,
                "share_url": share_url,
                "verification_hash": verification_hash,
            }
        else:
            return {
                "success": False,
                "status_code": resp.status_code,
                "error": resp.text,
            }


@app.post("/api/upload")
@limiter.limit("10/hour")
async def upload_result(request: Request, payload: UploadPayload):
    """上传测试结果到云端，返回分享链接"""
    max_prefill_value = payload.max_prefill_speed if payload.max_prefill_speed is not None else payload.avg_prefill_speed
    max_decode_value = payload.max_decode_speed if payload.max_decode_speed is not None else payload.avg_decode_speed
    source_value = payload.source or "python_backend"

    vh = payload.verification_hash or compute_verification_hash(
        model_name=payload.model_name,
        hardware=payload.hardware,
        framework=payload.framework,
        quantization=payload.quantization,
        concurrency=payload.concurrency,
        max_decode_speed=max_decode_value,
    )

    # 构造存储数据
    mode = payload.benchmark_mode or "1d_sweep"
    resolved_task_type = payload.task_type or "general"
    if mode in ("2d_matrix", "sustained_decode"):
        points_data = []
        if payload.results:
            points_data = [
                {
                    "task_type": getattr(r, "task_type", None) or resolved_task_type,
                    "prompt_length": r.prompt_length,
                    "concurrency": r.concurrency,
                    "decode_tps": r.output_speed,
                    "output_speed": r.output_speed,
                    "prefill_speed": r.prefill_speed,
                    "prefill_time_ms": r.prefill_time_ms,
                    "output_time_ms": r.output_time_ms,
                    "avg_ttft_ms": r.avg_ttft_ms,
                    "avg_itl_mean": r.avg_itl_mean,
                    "avg_itl_std": r.avg_itl_std,
                    "output_tokens": getattr(r, "output_tokens", None),
                    "e2e_speed": getattr(r, "e2e_speed", None),
                    "pure_decode_time_sec": getattr(r, "pure_decode_time_sec", None),
                    "avg_stream_decode_speed": getattr(r, "avg_stream_decode_speed", None),
                    "actual_duration_sec": getattr(r, "actual_duration_sec", None),
                    "concurrent_details": getattr(r, "concurrent_details", None),
                    "status": "成功",
                }
                for r in payload.results
            ]
        raw_points = payload.results or []
        all_ttft = [r.avg_ttft_ms for r in raw_points if r.avg_ttft_ms is not None and r.avg_ttft_ms > 0]
        all_itl = [r.avg_itl_mean for r in raw_points if r.avg_itl_mean is not None and r.avg_itl_mean > 0]

        sustained_summary = dict(payload.sustained_summary or {})
        if mode == "sustained_decode":
            p0 = points_data[0] if points_data else {}
            if not sustained_summary and p0:
                sustained_summary = {
                    "steady_decode_tps": p0.get("output_speed") or payload.avg_decode_speed,
                    "avg_decode_speed": p0.get("output_speed") or payload.avg_decode_speed,
                    "prefill_speed": p0.get("prefill_speed") or payload.avg_prefill_speed,
                    "avg_prefill_speed": p0.get("prefill_speed") or payload.avg_prefill_speed,
                    "actual_duration_sec": p0.get("actual_duration_sec") or (p0.get("output_time_ms", 0) / 1000 if p0.get("output_time_ms") else 30.0),
                    "pure_decode_time_s": p0.get("pure_decode_time_sec") or (p0.get("output_time_ms", 0) / 1000 if p0.get("output_time_ms") else 30.0),
                    "pure_decode_time_sec": p0.get("pure_decode_time_sec") or (p0.get("output_time_ms", 0) / 1000 if p0.get("output_time_ms") else 30.0),
                    "completed_streams": len(p0.get("concurrent_details") or []) or p0.get("concurrency") or payload.concurrency,
                    "total_tokens": p0.get("output_tokens"),
                    "avg_stream_decode_tps": p0.get("avg_stream_decode_speed"),
                    "avg_stream_decode_speed": p0.get("avg_stream_decode_speed"),
                    "e2e_throughput_tps": p0.get("e2e_speed"),
                    "e2e_speed": p0.get("e2e_speed"),
                    "concurrency": p0.get("concurrency") or payload.concurrency,
                    "task_type": resolved_task_type,
                    "completed_requests": p0.get("concurrent_details") or [],
                }
            else:
                if "completed_requests" not in sustained_summary and p0.get("concurrent_details"):
                    sustained_summary["completed_requests"] = p0["concurrent_details"]
                if "completed_streams" not in sustained_summary:
                    sustained_summary["completed_streams"] = len(sustained_summary.get("completed_requests") or []) or payload.concurrency

        results_json = {
            "benchmark_mode": mode,
            "task_type": resolved_task_type,
            "matrix": payload.matrix_grid or {},
            "matrix_summary": payload.matrix_summary or {},
            "all_matrices": payload.all_matrices or {},
            "all_tasks_comparison": payload.all_tasks_comparison or [],
            "sustained_summary": sustained_summary,
            "points": points_data,
            "verification": {
                "hash": vh,
                "status": "verified",
                "timestamp": int(time.time()),
            },
            "quantiles": {
                "ttft": {
                    "p50": compute_percentile(all_ttft, 50) if all_ttft else 0.0,
                    "p90": compute_percentile(all_ttft, 90) if all_ttft else 0.0,
                    "p99": compute_percentile(all_ttft, 99) if all_ttft else 0.0,
                    "avg": round(sum(all_ttft) / len(all_ttft), 2) if all_ttft else 0.0,
                },
                "itl": {
                    "p50": compute_percentile(all_itl, 50) if all_itl else 0.0,
                    "p90": compute_percentile(all_itl, 90) if all_itl else 0.0,
                    "p99": compute_percentile(all_itl, 99) if all_itl else 0.0,
                    "avg": round(sum(all_itl) / len(all_itl), 2) if all_itl else 0.0,
                },
            },
            "watchdog": {
                "loop_detected": any(r.loop_detected for r in (payload.results or [])),
            }
        }
    else:
        raw_points = payload.results or []
        all_ttft = [r.avg_ttft_ms for r in raw_points if r.avg_ttft_ms]
        all_itl = [r.avg_itl_mean for r in raw_points if r.avg_itl_mean]

        is_all_tasks = (
            resolved_task_type == "all"
            or bool(payload.all_tasks_comparison)
            or ("all_tasks" in (payload.record_tags or []))
        )
        ordered_tasks = ["general", "code", "structured", "prose", "prefill_16k", "reasoning"]
        cycle = 0
        last_len = -1
        point_tasks = []
        for r in raw_points:
            pt = getattr(r, "task_type", None)
            plen = getattr(r, "prompt_length", None)
            if plen is not None and plen <= last_len and last_len > 0:
                cycle += 1
            if plen is not None:
                last_len = plen
            if pt and pt != "all" and pt in ordered_tasks:
                point_tasks.append(pt)
            elif is_all_tasks:
                point_tasks.append(ordered_tasks[cycle % len(ordered_tasks)])
            else:
                point_tasks.append(pt or resolved_task_type)

        results_json = {
            "benchmark_mode": "1d_sweep",
            "task_type": resolved_task_type,
            "matrix": [],
            "all_tasks_comparison": payload.all_tasks_comparison or [],
            "verification": {
                "hash": vh,
                "status": "verified",
                "timestamp": int(time.time()),
            },
            "quantiles": {
                "ttft": {
                    "p50": compute_percentile(all_ttft, 50),
                    "p90": compute_percentile(all_ttft, 90),
                    "p99": compute_percentile(all_ttft, 99),
                },
                "itl": {
                    "p50": compute_percentile(all_itl, 50),
                    "p90": compute_percentile(all_itl, 90),
                    "p99": compute_percentile(all_itl, 99),
                },
            },
            "watchdog": {
                "loop_detected": any(r.loop_detected for r in raw_points),
            },
            "points": [
                {
                    "task_type": point_tasks[idx] if idx < len(point_tasks) else (getattr(r, "task_type", None) or resolved_task_type),
                    "prompt_length": r.prompt_length,
                    "prefill_speed": r.prefill_speed,
                    "output_speed": r.output_speed,
                    "decode_tps": r.output_speed,
                    "prefill_time_ms": r.prefill_time_ms,
                    "output_time_ms": r.output_time_ms,
                    "avg_ttft_ms": r.avg_ttft_ms,
                    "avg_itl_mean": r.avg_itl_mean,
                    "avg_itl_std": r.avg_itl_std,
                    "concurrency": r.concurrency,
                    "effective_concurrency": r.effective_concurrency or f"{r.concurrency}/{r.concurrency}",
                    "successful": r.successful,
                    "boundary_source": r.boundary_source,
                    "verification_hash": r.verification_hash or vh,
                    "loop_detected": bool(r.loop_detected),
                    "ttft_stats": r.ttft_stats or {"avg": r.avg_ttft_ms},
                    "itl_stats": r.itl_stats or {"avg": r.avg_itl_mean, "std": r.avg_itl_std},
                    "output_tokens": getattr(r, "output_tokens", None),
                    "e2e_speed": getattr(r, "e2e_speed", None),
                    "pure_decode_time_sec": getattr(r, "pure_decode_time_sec", None),
                    "avg_stream_decode_speed": getattr(r, "avg_stream_decode_speed", None),
                    "actual_duration_sec": getattr(r, "actual_duration_sec", None),
                    "concurrent_details": [
                        {
                            "stream_id": d.get("stream_id", d_idx + 1),
                            "prefill_speed": d.get("prefill_speed") or d.get("prefillSpeed"),
                            "output_speed": d.get("output_speed") or d.get("outputSpeed"),
                            "boundary_source": d.get("boundary_source"),
                            "ttft_ms": d.get("ttft_ms") or d.get("ttftMs"),
                            "output_tokens": d.get("output_tokens") or d.get("outputTokens"),
                            "output_time_ms": d.get("output_time_ms") or d.get("outputTimeMs"),
                            "itl_mean": d.get("itl_mean") or d.get("itlMean"),
                            "status": d.get("status"),
                        }
                        for d_idx, d in enumerate(r.concurrent_details or [])
                    ],
                }
                for idx, r in enumerate(raw_points)
            ],
        }

    if payload.is_anomaly and isinstance(results_json, dict):
        results_json["anomaly"] = {
            "is_anomaly": True,
            "reason": payload.anomaly_reason or "Unspecified anomaly",
        }

    record_tags = [tag.strip() for tag in (payload.record_tags or []) if isinstance(tag, str) and tag.strip()]
    if not any(tag.startswith("source:") for tag in record_tags):
        record_tags.insert(0, f"source:{source_value}")
    if "verified" not in record_tags:
        record_tags.append("verified")
    if not any(t.startswith("mode:") for t in record_tags):
        record_tags.append(f"mode:{mode}")
    if payload.is_anomaly:
        record_tags.append("time_anomaly")
        if payload.anomaly_reason:
            record_tags.append(f"anomaly:{payload.anomaly_reason}")
    record_tags = list(dict.fromkeys(record_tags))

    # v3: 速率限制升级，每天每 user_code/ip 100次
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
    ip_hash_val = _get_ip_hash(request)
    
    count_query = f"{SUPABASE_TABLE}?select=id&user_code=eq.{payload.user_code}&created_at=gte.{today_start}"
    count_resp = await _supabase_request("GET", count_query, extra_headers={"Prefer": "count=exact"})
    if count_resp.status_code == 200 or count_resp.status_code == 206:
        content_range = count_resp.headers.get("Content-Range", "")
        if content_range and "/" in content_range:
            count_str = content_range.split("/")[1]
            if count_str.isdigit() and int(count_str) >= 100:
                raise HTTPException(status_code=429, detail="超出每日上传限制 (100次/天)")

    record = {
        "user_code": payload.user_code,
        "nickname": payload.nickname,
        "model_name": payload.model_name,
        "hardware": payload.hardware,
        "framework": payload.framework,
        "quantization": payload.quantization,
        "notes": payload.notes,
        "concurrency": payload.concurrency,
        "avg_prefill_speed": payload.avg_prefill_speed,
        "avg_decode_speed": payload.avg_decode_speed,
        "max_prefill_speed": max_prefill_value,
        "max_decode_speed": max_decode_value,
        "source": source_value,
        "record_tags": record_tags,
        "results_json": results_json,
        "ip_hash": ip_hash_val,
        "status": "public",
    }
    if payload.run_command:
        record["run_command"] = payload.run_command

    resp = await _supabase_request(
        "POST", SUPABASE_TABLE,
        json=record
    )

    if resp.status_code not in (200, 201):
        print(f"[Upload] Supabase error {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=502, detail=f"上传失败: {resp.text}")

    data = resp.json()
    record_id = data[0]["id"] if isinstance(data, list) else data.get("id")
    share_url = f"{PUBLIC_LEADERBOARD_URL}?id={record_id}"
    return {"success": True, "id": record_id, "share_id": record_id, "share_url": share_url, "verification_hash": vh}


@app.get("/api/results")
async def get_results(
    search: Optional[str] = None,
    user_code: Optional[str] = None,
    sort_by: Optional[str] = "created_at",
    limit: int = 100,
    offset: int = 0
):
    """查询排行榜数据"""
    limit = min(limit, 100)
    
    # 允许按时间和速度排序
    if sort_by == "created_at":
        order_col = "created_at"
    elif sort_by in ("max_prefill_speed", "max_decode_speed", "avg_prefill_speed", "avg_decode_speed"):
        order_col = sort_by
    else:
        order_col = "max_decode_speed"
    
    params = f"status=eq.public&order={order_col}.desc&limit={limit}&offset={offset}"
    
    if user_code:
        uc = user_code.strip().upper()
        params += f"&user_code=eq.{uc}"
        
    if search:
        s = search.strip()
        import uuid
        try:
            val = uuid.UUID(s)
            params += f"&id=eq.{str(val)}"
        except ValueError:
            # Not a UUID, so do a fuzzy search on model_name or nickname or exact match on user_code
            safe_s = s.replace(",", "").replace(".", "").replace('"', '')
            params += f"&or=(nickname.ilike.*{safe_s}*,model_name.ilike.*{safe_s}*,user_code.eq.{safe_s.upper()})"

    select_fields = "id,user_code,nickname,model_name,hardware,framework,quantization,notes,run_command,concurrency,avg_prefill_speed,avg_decode_speed,max_prefill_speed,max_decode_speed,source,record_tags,created_at"
    resp = await _supabase_request(
        "GET", f"{SUPABASE_TABLE}?{params}&select={select_fields}",
    )

    if resp.status_code != 200 and "run_command" in resp.text:
        legacy_select_fields = select_fields.replace(",run_command", "")
        resp = await _supabase_request(
            "GET", f"{SUPABASE_TABLE}?{params}&select={legacy_select_fields}",
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"查询失败: {resp.text}")

    return resp.json()


@app.get("/api/results/{record_id}")
async def get_result_detail(record_id: str):
    """按 ID 查询单个记录的完整详情（包含 results_json）"""
    resp = await _supabase_request(
        "GET", f"{SUPABASE_TABLE}?id=eq.{record_id}&select=id,results_json&limit=1",
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"查询详情失败: {resp.text}")
    data = resp.json()
    if not data:
        raise HTTPException(status_code=404, detail="记录不存在")
    return data[0]


# ============================================================
# 用户名系统 API
# ============================================================

USER_PROFILES_TABLE = "user_profiles"

# 用户名规则常量
NICKNAME_MIN_LEN = 2
NICKNAME_MAX_LEN = 20
NICKNAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_\-]{1,19}$')

# 基础违禁词（小写匹配）
BLOCKED_WORDS = {
    "admin", "root", "system", "official", "moderator", "mod",
    "fuck", "shit", "ass", "bitch", "cunt", "dick", "cock",
    "nigger", "faggot", "retard", "whore", "slut",
    "llmtest", "speedtest", "benchmark",  # 保留词
}


def _validate_nickname(nickname: str) -> str | None:
    """验证用户名，返回错误信息字符串，None 表示通过"""
    n = nickname.strip()
    if len(n) < NICKNAME_MIN_LEN:
        return f"用户名至少 {NICKNAME_MIN_LEN} 个字符"
    if len(n) > NICKNAME_MAX_LEN:
        return f"用户名最多 {NICKNAME_MAX_LEN} 个字符"
    if not NICKNAME_PATTERN.match(n):
        return "用户名只能包含字母、数字、下划线、横线，且必须以字母开头"
    if n.isdigit():
        return "用户名不能全为数字"
    lower = n.lower()
    for word in BLOCKED_WORDS:
        if word in lower:
            return f"用户名含有违禁词，请换一个"
    return None


@app.get("/api/user-profile")
async def get_user_profile(user_code: str):
    """根据 user_code 查询用户昵称（用于页面初始化时从云端恢复昵称）"""
    uc = user_code.strip().upper()
    if len(uc) != 8 or not uc.isalnum():
        raise HTTPException(status_code=400, detail="无效的 user_code")

    resp = await _supabase_request(
        "GET",
        f"{USER_PROFILES_TABLE}?user_code=eq.{uc}&select=user_code,nickname,created_at&limit=1",
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="查询失败")

    data = resp.json()
    if not data:
        return {"found": False, "nickname": None}
    return {"found": True, "nickname": data[0]["nickname"], "created_at": data[0]["created_at"]}


@app.get("/api/check-nickname")
async def check_nickname(nickname: str):
    """检查用户名是否可用（格式验证 + 唯一性查询）"""
    error = _validate_nickname(nickname)
    if error:
        return {"available": False, "reason": error}

    resp = await _supabase_request(
        "GET",
        f"{USER_PROFILES_TABLE}?nickname=ilike.{nickname}&select=nickname&limit=1",
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="查询失败")

    if resp.json():
        return {"available": False, "reason": "该用户名已被使用，请换一个"}
    return {"available": True}


def _hash_password(password: str) -> str:
    """使用 PBKDF2-HMAC-SHA256 哈希密码（stdlib，无需额外依赖）
    格式：pbkdf2$<salt_hex>$<hash_hex>
    """
    import os
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 260000)
    return f"pbkdf2${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str, user_code: Optional[str] = None) -> bool:
    """Verify PBKDF2 hashes and legacy browser SHA-256 hashes."""
    import hmac as _hmac
    if not stored:
        return False

    # Standalone browser pages store SHA-256(user_code + ":" + password).
    if re.fullmatch(r"[0-9a-fA-F]{64}", stored or "") and user_code:
        expected = hashlib.sha256(f"{user_code.strip()}:{password}".encode("utf-8")).hexdigest()
        return _hmac.compare_digest(expected.lower(), stored.lower())

    try:
        _, salt_hex, hash_hex = stored.split('$')
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 260000)
        return _hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


class SetNicknamePayload(BaseModel):
    user_code: str
    nickname: str
    password: Optional[str] = None  # 可选，用于后续找回识别码

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if v is None:
            return v
        v = v.strip()
        if len(v) < 6:
            raise ValueError('密码至少6位')
        if len(v) > 72:
            raise ValueError('密码最多72位')
        return v


@app.post("/api/set-nickname")
@limiter.limit("5/hour")
async def set_nickname(request: Request, payload: SetNicknamePayload):
    """为 user_code 设置昵称（只能设置一次，不可修改）"""
    uc = payload.user_code.strip().upper()
    if len(uc) != 8 or not uc.isalnum():
        raise HTTPException(status_code=400, detail="无效的 user_code")

    error = _validate_nickname(payload.nickname)
    if error:
        raise HTTPException(status_code=400, detail=error)

    nickname = payload.nickname.strip()

    # 先检查此 user_code 是否已有昵称
    existing = await _supabase_request(
        "GET",
        f"{USER_PROFILES_TABLE}?user_code=eq.{uc}&select=nickname&limit=1",
    )
    if existing.status_code == 200 and existing.json():
        raise HTTPException(status_code=409, detail="此识别码已设置过用户名，不可更改")

    # 哈希密码（可选）
    password_hash = None
    if payload.password:
        password_hash = _hash_password(payload.password)

    # 插入（UNIQUE 约束会在昵称重复时报错）
    resp = await _supabase_request(
        "POST",
        USER_PROFILES_TABLE,
        json={"user_code": uc, "nickname": nickname, "password_hash": password_hash},
    )

    if resp.status_code in (200, 201):
        return {"success": True, "nickname": nickname}

    # 解析 Supabase 唯一性冲突错误
    err_text = resp.text.lower()
    if "unique" in err_text or "duplicate" in err_text or "23505" in err_text:
        raise HTTPException(status_code=409, detail="该用户名已被使用，请换一个")
    raise HTTPException(status_code=502, detail=f"设置失败: {resp.text}")


class RecoverPayload(BaseModel):
    nickname: str
    password: str


@app.post("/api/recover-user-code")
@limiter.limit("5/hour")
async def recover_user_code(request: Request, payload: RecoverPayload):
    """通过用户名+密码找回识别码。严格速率限制：同 IP 每小时最多 5 次。"""
    # 查询用户名对应记录（含 password_hash）
    nickname_query = quote(payload.nickname.strip(), safe="")
    resp = await _supabase_request(
        "GET",
        f"{USER_PROFILES_TABLE}?nickname=ilike.{nickname_query}&select=user_code,password_hash&limit=1",
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="查询失败")

    data = resp.json()
    # 不论是否找到，都执行哈希比较，防止时序攻击
    stored_hash = data[0]["password_hash"] if data else None
    if not stored_hash:
        # 没有设置密码，无法通过此方式找回
        raise HTTPException(status_code=403, detail="该用户名未设置密码，无法通过此方式找回识别码")

    if not _verify_password(payload.password, stored_hash, data[0].get("user_code") if data else None):
        raise HTTPException(status_code=403, detail="用户名或密码错误")

    return {"user_code": data[0]["user_code"]}



class TestConfig(BaseModel):
    api_url: str
    model_name: str
    api_key: str = ""
    api_type: str = "openai"
    benchmark_mode: str = "1d_sweep"  # "1d_sweep" | "2d_matrix" | "sustained_decode"
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    step: Optional[int] = None
    step_multiplier: Optional[int] = 1
    test_lengths: Optional[list] = None  # 可选的测试点列表，如果提供则忽略min/max/step
    matrix_concurrency_ladder: Optional[List[int]] = None
    matrix_context_ladder: Optional[List[int]] = None
    matrix_max_kv_tokens: int = 32768
    sustained_duration: float = 30.0
    sustained_prompt_length: int = 128
    output_length: int = 128
    concurrency: int = 1
    timeout: int = 120000
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    metrics_url: Optional[str] = None
    enable_watchdog: bool = True
    task_type: str = "general"


def should_skip_matrix_cell(concurrency: int, context: int, max_tokens: int, max_kv_tokens: int) -> bool:
    """Return True if running this cell would exceed the server's KV cache budget."""
    if max_kv_tokens <= 0:
        return False
    tokens_needed = concurrency * (context + max_tokens)
    return tokens_needed > max_kv_tokens


def get_matrix_traversal_order(concurrency_ladder: List[int], context_ladder: List[int]) -> List[Tuple[int, int]]:
    """
    Generate scout traversal order for 2D matrix:
    1. Scout phase: C=1 (or lowest concurrency) across all context lengths
    2. Decode baseline: First context across remaining concurrencies
    3. Progressive grid: Remaining cells
    """
    order = []
    seen = set()

    c_scout = 1 if 1 in concurrency_ladder else min(concurrency_ladder)
    for ctx in context_ladder:
        cell = (c_scout, ctx)
        if cell not in seen:
            order.append(cell)
            seen.add(cell)

    ctx_scout = context_ladder[0]
    for c in concurrency_ladder:
        cell = (c, ctx_scout)
        if cell not in seen:
            order.append(cell)
            seen.add(cell)

    for c in concurrency_ladder:
        for ctx in context_ladder:
            cell = (c, ctx)
            if cell not in seen:
                order.append(cell)
                seen.add(cell)

    return order


def _is_cjk_char(char: str) -> bool:
    return (
        "\u4e00" <= char <= "\u9fff"
        or "\u3400" <= char <= "\u4dbf"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
    )


def compute_percentile(data: List[float], p: float) -> float:
    """Calculate percentile using linear interpolation (matching numpy/standard formula)."""
    if not data:
        return 0.0
    valid = sorted([float(x) for x in data if x is not None and not math.isnan(x)])
    if not valid:
        return 0.0
    k = (len(valid) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return valid[int(k)]
    d0 = valid[f] * (c - k)
    d1 = valid[c] * (k - f)
    return round(d0 + d1, 3)


# ============================================================
# Streaming Repetition Watchdog (Loop Guard)
# Exact periodic tail detector with suffix anchor matching
# ============================================================
LOOP_CELL_TPS = -3.0
LOOP_CHECK_CHARS = 256
LOOP_MAX_PERIOD_CHARS = 32768
LOOP_WINDOW_CHARS = 4 * LOOP_MAX_PERIOD_CHARS  # 131072 (128 KiB)
LOOP_CONFIRMED_CHARS = 4096
LOOP_CONFIRMED_CYCLES = 4
LOOP_SUSPECTED_CHARS = 1024
LOOP_SUSPECTED_CYCLES = 3


def exact_periodic_tail(
    text: str,
    min_cycle_chars: int = 64,
    min_cycles: int = 3,
    min_chars: int = 1024,
) -> Optional[Dict[str, Any]]:
    """
    Find a character-exact periodic suffix with bounded candidate search.
    Follows reference benchmark StreamingLoopDetector algorithm:
    - 64-char anchor matching scanned backwards
    - Fast binary search over repeat length
    - Rejects pure whitespace repetitions
    """
    n = len(text)
    if n < min_chars:
        return None
    anchor_size = min_cycle_chars
    if n < anchor_size:
        return None
    anchor_start = n - anchor_size
    anchor = text[anchor_start:]
    left = max(0, anchor_start - LOOP_MAX_PERIOD_CHARS)
    end = n - 1

    for _ in range(64):
        previous = text.rfind(anchor, left, end)
        if previous < 0:
            break
        period = anchor_start - previous
        end = previous + anchor_size - 1
        minimum = max(min_chars, min_cycles * period)
        if minimum > n or text[n - minimum : n - period] != text[n - minimum + period :]:
            continue
        if not text[-period:].strip():
            continue

        lo, hi = minimum, n
        while lo < hi:
            length = (lo + hi + 1) // 2
            if text[n - length : n - period] == text[n - length + period :]:
                lo = length
            else:
                hi = length - 1

        complete_cycles = lo // period
        return {
            "observed_start_char": n - lo,
            "period_chars": period,
            "repeated_chars": lo,
            "complete_cycles": complete_cycles,
        }
    return None


class StreamingLoopDetector:
    """
    Bounded exact-repetition evidence for one request text channel.
    Reblocks stream fragments into 256-character pieces and scans periodic tail.
    """
    def __init__(
        self,
        channel: str = "content",
        chunk_reblock_chars: int = LOOP_CHECK_CHARS,
        min_cycle_chars: int = 64,
        min_cycles: int = 4,
    ):
        self.channel = channel
        self.chunk_reblock_chars = chunk_reblock_chars
        self.min_cycle_chars = min_cycle_chars
        self.min_cycles = min_cycles
        self.text = ""
        self.pending: List[str] = []
        self.pending_chars = 0
        self.total_chars = 0
        self.reported_level = 0

    def _check(self) -> Optional[Dict[str, Any]]:
        if self.pending:
            self.text = (self.text + "".join(self.pending))[-LOOP_WINDOW_CHARS:]
            self.pending = []
            self.pending_chars = 0
        evidence = exact_periodic_tail(self.text)
        if evidence is None:
            return None
        confirmed = (
            evidence["complete_cycles"] >= LOOP_CONFIRMED_CYCLES
            and evidence["repeated_chars"] >= LOOP_CONFIRMED_CHARS
        )
        level = 2 if confirmed else 1
        if level <= self.reported_level:
            return None
        self.reported_level = level
        offset = self.total_chars - len(self.text)
        start = evidence["observed_start_char"]
        period = evidence["period_chars"]
        cycle = self.text[start : start + period]
        return {
            **evidence,
            "channel": self.channel,
            "status": "confirmed" if confirmed else "suspected",
            "observed_start_char": offset + start,
            "detected_at_char": self.total_chars,
            "coordinate_unit": "unicode_code_points_in_channel",
            "period_sha256": hashlib.sha256(cycle.encode("utf-8")).hexdigest(),
            "cycle_excerpt": cycle[:512],
            "context_excerpt": self.text[max(0, start - 128) : start + min(period, 512)],
            "tail_excerpt": self.text[-1024:],
        }

    def feed(self, fragment: str) -> Optional[Dict[str, Any]]:
        finding = None
        offset = 0
        while offset < len(fragment):
            count = min(LOOP_CHECK_CHARS - self.pending_chars, len(fragment) - offset)
            self.pending.append(fragment[offset : offset + count])
            self.pending_chars += count
            self.total_chars += count
            offset += count
            if self.pending_chars == LOOP_CHECK_CHARS:
                finding = self._check() or finding
                if finding and finding["status"] == "confirmed":
                    break
        return finding

    def finish(self) -> Optional[Dict[str, Any]]:
        return self._check()


# ============================================================
# Prometheus Telemetry Scraper & Server Runtime Monitor
# Supports vLLM and SGLang /metrics endpoints
# ============================================================
PROMETHEUS_METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)"
)


def parse_prometheus_text(text: str) -> List[tuple[str, Dict[str, str], float]]:
    """Parse Prometheus exposition format into (metric_name, labels, float_value) tuples."""
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_METRIC_LINE.match(line)
        if not match:
            continue
        metric_name = match.group(1)
        raw_labels = match.group(2) or ""
        raw_val = match.group(3)
        labels = {}
        if raw_labels.startswith("{") and raw_labels.endswith("}"):
            for k, v in re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', raw_labels):
                labels[k] = v
        # Multi-GPU / Tensor Parallelism deduplication: only take rank 0 or unranked
        if "tp_rank" in labels and labels["tp_rank"] != "0":
            continue
        try:
            samples.append((metric_name, labels, float(raw_val)))
        except ValueError:
            continue
    return samples


def parse_prometheus_metrics(text: str) -> Dict[str, Any]:
    """Parse Prometheus metrics text and return structured metrics dictionary."""
    parsed = parse_prometheus_text(text)
    res = {
        "engine": "unknown",
        "running_requests": 0.0,
        "waiting_requests": 0.0,
        "kv_cache_usage_pct": 0.0,
        "server_generation_tps": 0.0,
    }
    for name, labels, val in parsed:
        if name in ("vllm:num_requests_running", "sglang:num_running_reqs", "sglang:num_requests_running"):
            res["running_requests"] += val
            if "vllm" in name:
                res["engine"] = "vLLM"
            elif "sglang" in name:
                res["engine"] = "SGLang"
        elif name in ("vllm:num_requests_waiting", "sglang:num_queue_reqs", "sglang:num_requests_waiting"):
            res["waiting_requests"] += val
            if "vllm" in name and res["engine"] == "unknown":
                res["engine"] = "vLLM"
            elif "sglang" in name and res["engine"] == "unknown":
                res["engine"] = "SGLang"
        elif name in ("vllm:gpu_cache_usage_factor", "vllm:kv_cache_usage_perc", "sglang:utilization", "sglang:token_usage"):
            if name == "vllm:gpu_cache_usage_factor":
                res["kv_cache_usage_pct"] = max(res["kv_cache_usage_pct"], val * 100.0)
                res["engine"] = "vLLM"
            elif name == "vllm:kv_cache_usage_perc":
                res["kv_cache_usage_pct"] = max(res["kv_cache_usage_pct"], val)
                res["engine"] = "vLLM"
            elif name in ("sglang:utilization", "sglang:token_usage"):
                res["kv_cache_usage_pct"] = max(res["kv_cache_usage_pct"], val * 100.0 if val <= 1.0 else val)
                res["engine"] = "SGLang"
        elif name in ("vllm:avg_generation_throughput_tok_per_s", "vllm:avg_generation_throughput_toks_per_s", "sglang:gen_throughput"):
            res["server_generation_tps"] = max(res["server_generation_tps"], val)
    return res


class PrometheusTelemetryScraper:
    """
    Background Prometheus /metrics scraper for vLLM & SGLang.
    Extracts KV cache usage, queue sizes, scheduler saturation, and throughput.
    """
    def __init__(self, base_or_api_url: Optional[str] = None, explicit_metrics_url: Optional[str] = None):
        if explicit_metrics_url:
            self.metrics_url = explicit_metrics_url
        elif base_or_api_url:
            if base_or_api_url.endswith("/metrics"):
                self.metrics_url = base_or_api_url
            else:
                parsed = urlparse(base_or_api_url)
                base = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else base_or_api_url.rstrip("/")
                self.metrics_url = f"{base}/metrics"
        else:
            self.metrics_url = None

        self.samples: List[Dict[str, Any]] = []
        self._sampling_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.available: Optional[bool] = None
        self.server_engine: Optional[str] = None

    async def start(self, interval: float = 0.5):
        """Async convenience alias for start_sampling."""
        self.start_sampling(interval)

    async def stop(self) -> Dict[str, Any]:
        """Async convenience alias for stop_sampling."""
        return await self.stop_sampling()

    async def scrape_once(self, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """Scrape /metrics once and parse KV cache, queue sizes, and running requests."""
        if not self.metrics_url:
            self.available = False
            return None

        try:
            async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                resp = await client.get(self.metrics_url)
                if resp.status_code != 200:
                    self.available = False
                    return None
                text = resp.text
        except Exception:
            self.available = False
            return None

        parsed = parse_prometheus_text(text)
        if not parsed:
            self.available = False
            return None

        self.available = True
        sample: Dict[str, Any] = {
            "timestamp": time.time(),
            "running_reqs": 0.0,
            "queue_reqs": 0.0,
            "kv_cache_usage_pct": 0.0,
            "server_throughput": 0.0,
            "gpu_cache_factor": None,
        }

        for name, labels, val in parsed:
            # Running requests
            if name in ("vllm:num_requests_running", "sglang:num_running_reqs", "sglang:num_requests_running"):
                sample["running_reqs"] += val
                if "vllm" in name:
                    self.server_engine = "vllm"
                elif "sglang" in name:
                    self.server_engine = "sglang"
            # Queue waiting requests
            elif name in ("vllm:num_requests_waiting", "sglang:num_queue_reqs", "sglang:num_requests_waiting"):
                sample["queue_reqs"] += val
            # KV cache usage
            elif name in ("vllm:gpu_cache_usage_factor", "vllm:kv_cache_usage_perc", "sglang:utilization", "sglang:token_usage"):
                if name == "vllm:gpu_cache_usage_factor":
                    sample["gpu_cache_factor"] = val
                    sample["kv_cache_usage_pct"] = max(sample["kv_cache_usage_pct"], val * 100.0)
                elif name == "vllm:kv_cache_usage_perc":
                    sample["kv_cache_usage_pct"] = max(sample["kv_cache_usage_pct"], val)
                elif name in ("sglang:utilization", "sglang:token_usage"):
                    sample["kv_cache_usage_pct"] = max(sample["kv_cache_usage_pct"], val * 100.0 if val <= 1.0 else val)
            # Server throughput
            elif name in ("vllm:avg_generation_throughput_toks_per_s", "vllm:avg_generation_throughput_tok_per_s", "sglang:gen_throughput"):
                sample["server_throughput"] = max(sample["server_throughput"], val)

        return sample

    async def _sample_loop(self, interval: float):
        while not self._stop_event.is_set():
            sample = await self.scrape_once(timeout=max(1.5, interval))
            if sample:
                self.samples.append(sample)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def start_sampling(self, interval: float = 0.5):
        """Start background sampling task."""
        self.samples.clear()
        self._stop_event.clear()
        self._sampling_task = asyncio.create_task(self._sample_loop(interval))

    async def stop_sampling(self) -> Dict[str, Any]:
        """Stop background sampling and compute summary statistics."""
        self._stop_event.set()
        if self._sampling_task:
            try:
                await self._sampling_task
            except Exception:
                pass
            self._sampling_task = None

        if not self.samples:
            return {
                "available": bool(self.available),
                "engine": self.server_engine or "unknown",
                "sample_count": 0,
                "mean_running": 0.0,
                "max_running": 0.0,
                "mean_queue": 0.0,
                "max_queue": 0.0,
                "mean_kv_cache_pct": 0.0,
                "max_kv_cache_pct": 0.0,
                "server_reported_throughput": 0.0,
            }

        running = [s["running_reqs"] for s in self.samples]
        queue = [s["queue_reqs"] for s in self.samples]
        kv = [s["kv_cache_usage_pct"] for s in self.samples]
        throughput = [s["server_throughput"] for s in self.samples]

        return {
            "available": True,
            "engine": self.server_engine or "detected",
            "sample_count": len(self.samples),
            "mean_running": round(sum(running) / len(running), 2),
            "max_running": round(max(running), 2),
            "mean_queue": round(sum(queue) / len(queue), 2),
            "max_queue": round(max(queue), 2),
            "mean_kv_cache_pct": round(sum(kv) / len(kv), 2),
            "max_kv_cache_pct": round(max(kv), 2),
            "server_reported_throughput": round(max(throughput), 2),
        }


def calculate_effective_concurrency(
    concurrency: int,
    telemetry_summary: Optional[Dict[str, Any]] = None,
    client_inferred_parallelism: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates effective concurrency ratio (X/Y)* detecting server queue saturation.
    Uses Prometheus telemetry when available; gracefully falls back to client timing inference.
    """
    if concurrency <= 1:
        return {
            "label": "1/1",
            "effective_concurrency": 1.0,
            "effective_ratio": 1.0,
            "is_capacity_limited": False,
            "source": "single_request",
        }

    # 1. Prometheus Telemetry Path
    if telemetry_summary and telemetry_summary.get("available") and telemetry_summary.get("sample_count", 0) > 0:
        mean_r = telemetry_summary.get("mean_running", 0.0)
        max_r = telemetry_summary.get("max_running", 0.0)
        mean_q = telemetry_summary.get("mean_queue", 0.0)
        max_q = telemetry_summary.get("max_queue", 0.0)

        is_limited = (
            max_r < concurrency
            or mean_r < concurrency * 0.95
            or mean_q > 0.05
            or max_q > 0
        )
        if is_limited:
            effective_x = max(1, int(round(mean_r)))
            return {
                "label": f"({effective_x}/{concurrency})*",
                "effective_concurrency": float(effective_x),
                "effective_ratio": round(mean_r / concurrency, 3),
                "is_capacity_limited": True,
                "source": "prometheus_telemetry",
                "mean_running": mean_r,
                "mean_queue": mean_q,
            }
        return {
            "label": f"{concurrency}/{concurrency}",
            "effective_concurrency": float(concurrency),
            "effective_ratio": 1.0,
            "is_capacity_limited": False,
            "source": "prometheus_telemetry",
            "mean_running": mean_r,
            "mean_queue": mean_q,
        }

    # 2. Client Timing Inferred Fallback Path
    if client_inferred_parallelism is not None and client_inferred_parallelism > 0:
        if client_inferred_parallelism < concurrency * 0.80:
            effective_x = max(1, int(round(client_inferred_parallelism)))
            return {
                "label": f"({effective_x}/{concurrency})*",
                "effective_concurrency": float(effective_x),
                "effective_ratio": round(client_inferred_parallelism / concurrency, 3),
                "is_capacity_limited": True,
                "source": "client_inferred_fallback",
                "inferred_parallelism": round(client_inferred_parallelism, 2),
            }
        return {
            "label": f"{concurrency}/{concurrency}",
            "effective_concurrency": float(concurrency),
            "effective_ratio": 1.0,
            "is_capacity_limited": False,
            "source": "client_inferred_fallback",
            "inferred_parallelism": round(client_inferred_parallelism, 2),
        }

    # 3. Default nominal
    return {
        "label": f"{concurrency}/{concurrency}",
        "effective_concurrency": float(concurrency),
        "effective_ratio": 1.0,
        "is_capacity_limited": False,
        "source": "nominal",
    }


def _is_ascii_word_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char == "_")


def _estimate_ascii_run_tokens(run: str) -> int:
    """Count a plain ASCII word/number run using the shared lightweight tokenizer."""
    if not run:
        return 0

    # Common English words are usually one token in modern BPE tokenizers. Very long
    # unbroken runs are treated as multiple tokens so random IDs still have weight.
    return max(1, math.ceil(len(run) / 12))


def estimate_token_weight(text: str) -> float:
    """Compatibility wrapper for the shared token estimator."""
    return float(estimate_token_count(text))


def estimate_token_count(text: str) -> int:
    """Estimate tokens with one shared lightweight tokenizer heuristic.

    The benchmark cannot depend on every model-specific tokenizer. This heuristic
    intentionally mirrors the browser implementation: common ASCII word runs count
    as one token, long unbroken ASCII runs are split, CJK characters count one by
    one, whitespace is ignored, and punctuation/symbols count as one token.
    """
    if not text or len(text) == 0:
        return 0

    token_count = 0
    index = 0
    text_length = len(text)

    while index < text_length:
        char = text[index]
        if char.isspace():
            index += 1
            continue

        if _is_cjk_char(char):
            token_count += 1
            index += 1
            continue

        if _is_ascii_word_char(char):
            start = index
            while index < text_length and _is_ascii_word_char(text[index]):
                index += 1
            token_count += _estimate_ascii_run_tokens(text[start:index])
            continue

        token_count += 1
        index += 1

    return max(1, token_count)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _pick_best_token_value(*values: Any) -> Optional[int]:
    parsed = [v for v in (_coerce_int(value) for value in values) if v is not None]
    if not parsed:
        return None
    positives = [v for v in parsed if v > 0]
    return max(positives) if positives else max(parsed)


def extract_usage_token_stats(usage: Dict[str, Any]) -> Dict[str, Optional[int]]:
    completion_details = usage.get("completion_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}

    prompt_tokens = _pick_best_token_value(
        usage.get("prompt_tokens"),
        usage.get("input_tokens"),
        usage.get("prompt_eval_count"),
    )
    completion_tokens = _pick_best_token_value(
        usage.get("completion_tokens"),
        usage.get("output_tokens"),
        usage.get("eval_count"),
    )
    reasoning_tokens_top = _pick_best_token_value(usage.get("reasoning_tokens"))
    reasoning_tokens_nested = _pick_best_token_value(
        completion_details.get("reasoning_tokens"),
        output_details.get("reasoning_tokens"),
    )
    total_tokens = _pick_best_token_value(usage.get("total_tokens"))

    reasoning_tokens = reasoning_tokens_top if reasoning_tokens_top is not None else reasoning_tokens_nested
    output_tokens = None

    if completion_tokens is not None:
        output_tokens = completion_tokens

        # Top-level reasoning tokens are usually separate from completion_tokens.
        if reasoning_tokens_top is not None and reasoning_tokens_top > 0:
            if prompt_tokens is not None and total_tokens is not None:
                if total_tokens == prompt_tokens + completion_tokens + reasoning_tokens_top:
                    output_tokens = completion_tokens + reasoning_tokens_top
                elif total_tokens == prompt_tokens + completion_tokens:
                    output_tokens = completion_tokens
                else:
                    output_tokens = max(completion_tokens, completion_tokens + reasoning_tokens_top)
            else:
                output_tokens = completion_tokens + reasoning_tokens_top
        elif output_tokens <= 0 and reasoning_tokens is not None and reasoning_tokens > 0:
            output_tokens = reasoning_tokens
    elif reasoning_tokens is not None:
        output_tokens = reasoning_tokens
    elif prompt_tokens is not None and total_tokens is not None and total_tokens >= prompt_tokens:
        output_tokens = total_tokens - prompt_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


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


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_BENCHMARK_SUFFIX = (
    "\nBased on the words above, write a short philosophical essay discussing "
    "the meaning of existence, the nature of consciousness, and humanity's "
    "place in the universe. Use clear, coherent sentences."
)

TASK_TYPES = ["general", "code", "structured", "prose", "prefill_16k", "reasoning"]
TASK_TYPE_CHOICES = TASK_TYPES + ["all"]

TASK_SYSTEM_PROMPTS = {
    "general": DEFAULT_SYSTEM_PROMPT,
    "code": (
        "You are an expert systems programmer and computer scientist. "
        "Write clean, idiomatic, high-performance, and syntactically complete code "
        "with thorough edge-case handling, full type annotations, and proper indentation. "
        "Avoid placeholder comments or unfinished stubs."
    ),
    "structured": (
        "You are an automated backend service that strictly outputs valid, parseable JSON conforming to RFC 8259. "
        "Do NOT wrap output in markdown fences. Output ONLY the raw JSON object with complete fields and valid types."
    ),
    "prose": (
        "You are a distinguished essayist, historian, and philosopher. "
        "Compose deep, eloquent, coherent essays with historical nuance, conceptual depth, and lucid argumentation."
    ),
    "prefill_16k": (
        "You are an expert technical documentation analyst. "
        "Read the long technical specification below carefully and provide a rigorous, comprehensive response "
        "to the verification questions at the end."
    ),
    "reasoning": (
        "You are an expert mathematician and logic theorist. "
        "You must think step by step, explicitly detailing every deduction, intermediate lemma, verification step, "
        "and edge case analysis before reaching your final conclusion."
    ),
}

TASK_PROMPT_POOLS = {
    "code": [
        (
            "Implement a high-concurrency LRU Cache with TTL (Time-To-Live) expiration in Python "
            "using a doubly linked list, hash map, and threading.RLock. Provide full get, put, evict, "
            "and cleanup methods along with unit test assertions and comprehensive docstrings."
        ),
        (
            "Write a complete recursive descent Abstract Syntax Tree (AST) parser and expression evaluator "
            "in Python. Support binary operators (+, -, *, /, ^), unary negation, parenthesis grouping, "
            "operator precedence, and variable symbol tables with detailed syntax error reporting."
        ),
        (
            "Implement a Red-Black Tree in Python from scratch with complete node color balance invariants, "
            "insertion rebalancing with left and right rotations, search, and in-order traversal generator "
            "with ASCII tree visualization helper methods."
        ),
        (
            "Implement an asynchronous worker pool with token bucket rate limiting and exponential backoff retry "
            "in Python asyncio. Include graceful shutdown, task cancellation handling, and queue backpressure."
        ),
    ],
    "structured": [
        (
            "Generate a detailed cloud Kubernetes cluster telemetry status report in valid JSON. The object must "
            "contain cluster_id, region, node_count, nodes array (each with node_id, status, cpu_metrics "
            "{usage_pct, load_avg}, memory_metrics {total_gb, used_gb}, pods array with pod_name, status, "
            "restart_count, latency_ms {p50, p90, p99}) and overall health_score."
        ),
        (
            "Output a comprehensive enterprise supply chain purchase order fulfillment record in valid JSON. "
            "Include order_id, customer_info {id, tier, billing_address}, line_items array (sku, description, "
            "quantity, unit_price, tax_pct, fulfillment_status), payment_audit {gateway, transaction_id, "
            "status, fee}, and timeline events with timestamps."
        ),
        (
            "Create an AI inference benchmark experiment record in valid JSON with fields: benchmark_id, "
            "model_metadata {name, architecture, parameter_count_b, quantization}, hardware {gpu_model, "
            "vram_gb, driver_version}, performance_matrix array containing concurrency, prompt_tokens, "
            "output_tokens, ttft_ms, decode_tps, and memory_bandwidth_pct."
        ),
    ],
    "prose": [
        (
            "Examine the intellectual and epistemological shifts brought about by the invention of the "
            "movable type printing press in Renaissance Europe, comparing its historical trajectory to the "
            "societal ramifications of modern generative artificial intelligence."
        ),
        (
            "Discuss the metaphysical and physical implications of thermodynamic entropy, the arrow of time, "
            "and information theory according to Ludwig Boltzmann and Claude Shannon. Analyze whether "
            "biological evolution and computation represent localized entropy reversals."
        ),
        (
            "Analyze Thomas Kuhn's theory of scientific revolutions and paradigm shifts. Explore how anomalous "
            "empirical observations accumulate within 'normal science' to precipitate crises, incommensurability, "
            "and eventual scientific paradigm replacement."
        ),
    ],
    "reasoning": [
        (
            "Solve the following mathematical puzzle with complete step-by-step reasoning: A sequence of positive "
            "integers a_1, a_2, ..., a_n satisfies a_1 = 1, and for all k >= 1, a_{k+1} is the smallest integer "
            "greater than a_k such that the sum of any non-empty subset of {a_1, ..., a_{k+1}} is never equal to a "
            "product of any subset. Determine whether such an infinite sequence exists, prove your assertion, and "
            "derive the asymptotic growth rate of a_n."
        ),
        (
            "Consider an impartial combinatorial game played on a directed acyclic graph where two players alternate "
            "moving a token along edges. A player unable to move loses. Analyze the Sprague-Grundy theorem application, "
            "compute the nim-values (g-values) for each node, and prove the necessary and sufficient winning conditions."
        ),
        (
            "Prove that in any group of 6 people, there are either at least 3 mutual friends or at least 3 mutual strangers "
            "(Ramsey's Theorem R(3,3)=6). Then extend your step-by-step combinatorial proof to find the lower and upper bounds "
            "for R(4,4) using the probabilistic method."
        ),
    ],
}

TASK_PADDING_TOKENS = {
    "code": [
        "class", "def", "self", "return", "import", "async", "await", "try", "except", "int", "str",
        "float", "bool", "dict", "list", "Optional", "Union", "Tuple", "None", "True", "False", "raise",
        "pass", "yield", "for", "in", "while", "if", "elif", "else", "with", "as", "assert", "lambda"
    ],
    "structured": [
        "id", "key", "value", "type", "name", "status", "count", "items", "data", "metadata",
        "config", "timestamp", "metrics", "enabled", "total", "score", "index", "record", "payload",
        "result", "code", "message", "level", "tag", "user", "group", "role", "scope", "target", "schema"
    ],
    "prose": [
        "history", "philosophy", "thought", "knowledge", "society", "culture", "nature", "humanity",
        "universe", "concept", "theory", "meaning", "reality", "mind", "language", "discourse",
        "paradigm", "reason", "virtue", "truth", "dialectic", "context", "episteme", "aesthetic"
    ],
    "reasoning": [
        "assume", "suppose", "lemma", "theorem", "proof", "deduction", "implies", "therefore",
        "contradiction", "invariant", "sequence", "subset", "partition", "induction", "basis",
        "hypothesis", "equation", "inequality", "optimal", "equilibrium", "strategy", "bound"
    ],
}

PREFILL_16K_DOC_SECTIONS = [
    (
        "Chapter 1: Distributed Vector Consensus and Raft State Machine Invariants\n"
        "In a distributed replicated state machine architecture, consensus safety is maintained through strict term monotonicity and election restrictions. "
        "Each replica maintains a persistent log where log entries consist of an integer index, the election term in which the entry was proposed, and the state machine transition command. "
        "A leader is elected only when it receives affirmative votes from a strict majority quorum of servers. A candidate's log must be at least as up-to-date as any other server in the quorum, "
        "defined by lexicographical comparison of the last log term followed by the last log index. If a network partition occurs, the partitioned minority cannot elect a leader or commit entries, "
        "guaranteeing linearizability. Upon partition healing, heartbeat intervals and append entry RPCs enforce the leader's log authority, overwriting uncommitted conflicting entries."
    ),
    (
        "Chapter 2: Log-Structured Merge Tree (LSM) Engine and Leveled Compaction Dynamics\n"
        "High-throughput write workloads utilize Log-Structured Merge (LSM) trees that transform random writes into sequential disk operations. "
        "Incoming writes are initially recorded into an append-only write-ahead log (WAL) and buffered in an active in-memory MemTable implemented as a concurrent SkipList. "
        "When the MemTable reaches its capacity threshold, it becomes immutable and is flushed to persistent storage as a Level 0 (L0) Sorted String Table (SSTable). "
        "Because L0 SSTables contain overlapping key ranges, read operations may inspect multiple files, leading to read amplification. "
        "To bound read latency and reclaim space from tombstone markers and superseded versions, background compaction merges files across hierarchical levels (L1 through LN). "
        "Leveled compaction enforces strictly non-overlapping key ranges within each level above L0, maintaining exponential size multipliers between adjacent tiers."
    ),
    (
        "Chapter 3: Zero-Copy RDMA Transport Fabric and Kernel-Bypass Memory Subsystems\n"
        "Modern hyper-scale data planes bypass the traditional operating system TCP/IP stack using Remote Direct Memory Access (RDMA) over Converged Ethernet (RoCE v2). "
        "Network interface cards (NICs) interact directly with registered user-space memory buffers, completely eliminating kernel context switches and intermediate CPU buffer copies. "
        "Connection state is maintained via Reliable Connected (RC) queue pairs (QPs). Priority Flow Control (PFC) provides hop-by-hop link-level flow control to prevent packet drop "
        "in loss-intolerant fabrics, while Explicit Congestion Notification (ECN) coupled with Data Center Quantized Congestion Notification (DCQCN) throttles transmission rates at the source NIC "
        "prior to switch buffer overflow, achieving microsecond-tier tail latencies at multi-hundred-gigabit line rates."
    ),
    (
        "Chapter 4: Byzantine Fault Detection, Split-Brain Fencing, and Dynamic Reconfiguration\n"
        "Enterprise resilience demands robust fault isolation against asymmetric network partitions, clock drift, and Byzantine corruptions. "
        "Split-brain scenarios are mitigated via epoch-based fencing tokens issued by distributed lease coordinators. Storage targets reject any I/O request bearing an epoch token lower than "
        "the current cluster generation. Membership changes undergo joint consensus transitions wherein configuration changes require dual quorums (both the prior configuration C_old "
        "and the proposed configuration C_new) before committing, ensuring seamless zero-downtime node addition, decommissioning, and storage tier expansion."
    ),
]

def get_system_prompt_for_task(task_type: str = "general") -> str:
    """Return the optimized system prompt for the specified benchmark task type."""
    return TASK_SYSTEM_PROMPTS.get(task_type, DEFAULT_SYSTEM_PROMPT)

ASCII_FILLER_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
PROMPT_TOKEN_WORDS = [
    word for word in (
        "a", "the", "and", "of", "to", "in", "is", "it", "that", "for", "with", "as",
        "on", "by", "from", "this", "be", "are", "or", "not", "we", "you", "they",
        "can", "will", "if", "all", "one", "time", "world", "life", "work", "data",
        "model", "token", "text", "idea", "mind", "story", "light", "space", "future",
        "human", "system", "simple", "clear", "reason", "change", "value", "truth",
    )
    if estimate_token_count(word) == 1
]
SHORT_PROMPT_MAX_LENGTH = 46
BENCHMARK_SUFFIX_TOKEN_LENGTH = SHORT_PROMPT_MAX_LENGTH + 1
TOKEN_SANITY_MIN_TOKENS = 128
TOKEN_SANITY_MAX_RELATIVE_DIFF = 0.8


def get_resolved_prompt_length(length: int) -> int:
    return max(int(length), 1)


def build_ascii_filler(char_count: int, seed: int = 0) -> str:
    offset = max(seed, 0) % len(ASCII_FILLER_CHARS)
    return "".join(
        ASCII_FILLER_CHARS[(offset + i) % len(ASCII_FILLER_CHARS)]
        for i in range(max(char_count, 0))
    )


def build_token_word_sequence(token_count: int, seed: int = 0) -> str:
    """Build text that has exactly token_count tokens under the shared heuristic."""
    target_token_count = max(int(token_count), 0)
    if target_token_count <= 0:
        return ""

    rng = random.Random(time.time_ns() ^ (seed * 1_000_003))
    words = [
        PROMPT_TOKEN_WORDS[rng.randrange(len(PROMPT_TOKEN_WORDS))]
        for _ in range(target_token_count)
    ]
    return " ".join(words)


def build_task_padding_sequence(task_type: str, token_count: int, seed: int = 0) -> str:
    """Build domain-specific filler text matching token_count for the task."""
    target_count = max(int(token_count), 0)
    if target_count <= 0:
        return ""
    words_pool = TASK_PADDING_TOKENS.get(task_type, PROMPT_TOKEN_WORDS)
    rng = random.Random(time.time_ns() ^ (seed * 1_000_003))
    words = [
        words_pool[rng.randrange(len(words_pool))]
        for _ in range(target_count)
    ]
    return " ".join(words)


def generate_prefill_16k_prompt(target_length: int, seed: int = 0) -> str:
    """Generate long technical architecture specification document targeting 16k context."""
    verification_suffix = (
        "\n\n[Architectural Verification Questions based on the document above]:\n"
        "1. Explain how the consensus engine guarantees safety during network partitions when split-brain fencing is triggered.\n"
        "2. Analyze how the leveled compaction strategy balances write amplification versus read amplification in the storage engine.\n"
        "3. Detail the zero-copy RDMA queue-pair lifecycle and congestion notification mechanisms.\n"
        "Provide a thorough, precise architectural analysis citing the sections above."
    )
    suffix_tokens = estimate_token_count(verification_suffix)
    needed_body_tokens = max(target_length - suffix_tokens, 64)

    sections = PREFILL_16K_DOC_SECTIONS
    session_nonce = f"[Benchmark Session ID: {int(time.time() * 1000):x}-{random.randint(0, 100000):x}]\n"
    body_parts = [session_nonce]
    current_tokens = estimate_token_count(session_nonce)
    cycle = 0
    while current_tokens < needed_body_tokens:
        for idx, sec in enumerate(sections):
            header = f"\n\n--- Module Cluster {cycle + 1}.{idx + 1} ---\n" if cycle > 0 else "\n\n"
            part = header + sec
            part_tokens = estimate_token_count(part)
            if current_tokens + part_tokens > needed_body_tokens:
                remaining = needed_body_tokens - current_tokens
                padding = build_token_word_sequence(remaining, seed + cycle * 100 + idx)
                if padding:
                    body_parts.append("\n" + padding)
                current_tokens = needed_body_tokens
                break
            body_parts.append(part)
            current_tokens += part_tokens
        cycle += 1

    return "".join(body_parts).strip() + verification_suffix


def generate_task_prompt(task_type: str, length: int, seed: int = 0) -> str:
    """Generate prompt from the curated task library."""
    if task_type == "prefill_16k":
        target_len = max(length, 1024) if length <= SHORT_PROMPT_MAX_LENGTH else length
        return generate_prefill_16k_prompt(target_len, seed)

    pool = TASK_PROMPT_POOLS.get(task_type, [])
    if not pool:
        return generate_prompt(length, seed, task_type="general")

    base_prompt = pool[abs(seed) % len(pool)]
    base_tokens = estimate_token_count(base_prompt)

    if length <= SHORT_PROMPT_MAX_LENGTH:
        if task_type == "code":
            return f"Write a complete Python implementation: {base_prompt[:60]}..."
        elif task_type == "structured":
            return f"Generate pure JSON matching: {base_prompt[:60]}..."
        elif task_type == "reasoning":
            return f"Solve step by step with proof: {base_prompt[:60]}..."
        else:
            return base_prompt[:100]

    if length <= base_tokens:
        return base_prompt

    prefix_token_count = max(length - base_tokens, 0)
    prefix_text = build_task_padding_sequence(task_type, prefix_token_count, seed)
    if prefix_text:
        return prefix_text + "\n\n" + base_prompt
    return base_prompt


def generate_short_prompt(length: int, seed: int = 0) -> str:
    """Generate a short prompt without the benchmark essay suffix."""
    target_prompt_length = get_resolved_prompt_length(length)
    return build_token_word_sequence(target_prompt_length, seed)


def generate_prompt(length: int, seed: int = 0, task_type: str = "general") -> str:
    """Generate a randomized prompt whose final estimated length matches the target semantics."""
    target_prompt_length = get_resolved_prompt_length(length)
    if task_type == "general" or task_type not in TASK_SYSTEM_PROMPTS:
        if target_prompt_length <= SHORT_PROMPT_MAX_LENGTH:
            return generate_short_prompt(target_prompt_length, seed)

        suffix_text = DEFAULT_BENCHMARK_SUFFIX.strip()
        suffix_tokens = estimate_token_count(suffix_text)
        prefix_token_count = max(target_prompt_length - suffix_tokens, 0)
        prefix_text = build_token_word_sequence(prefix_token_count, seed)

        if prefix_text:
            return prefix_text + "\n" + suffix_text
        return suffix_text

    return generate_task_prompt(task_type, target_prompt_length, seed)


def estimate_prompt_tokens_for_messages(prompt_text: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> int:
    """Estimate total prompt tokens for the system + user messages."""
    total_tokens = 0
    if system_prompt:
        total_tokens += estimate_token_count(system_prompt)
    total_tokens += estimate_token_count(prompt_text)
    return max(1, total_tokens)


def build_prompt_calibration(prompt_text: str, actual_prompt_tokens: Optional[int]) -> Optional[Dict[str, Any]]:
    """Build a one-sample calibration from warmup usage, when the provider returns it."""
    if not _has_positive_number(actual_prompt_tokens):
        return None

    estimated_prompt_tokens = estimate_prompt_tokens_for_messages(prompt_text)
    token_offset = int(actual_prompt_tokens) - estimated_prompt_tokens
    token_ratio = int(actual_prompt_tokens) / max(estimated_prompt_tokens, 1)
    return {
        "source": "warmup_usage",
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "actual_prompt_tokens": int(actual_prompt_tokens),
        "token_offset": token_offset,
        "token_ratio": token_ratio,
    }


def apply_prompt_calibration(estimated_prompt_tokens: int, calibration: Optional[Dict[str, Any]]) -> int:
    """Apply warmup-derived fixed overhead to a local prompt token estimate."""
    if not calibration:
        return max(int(estimated_prompt_tokens or 0), 1)
    token_offset = calibration.get("token_offset")
    if not isinstance(token_offset, (int, float)):
        return max(int(estimated_prompt_tokens or 0), 1)
    return max(int(round(estimated_prompt_tokens + token_offset)), 1)


def get_calibrated_generation_prompt_length(length: int, calibration: Optional[Dict[str, Any]] = None) -> int:
    """Choose a local user-prompt length that should land closer to the requested API prompt tokens."""
    resolved_length = get_resolved_prompt_length(length)
    if not calibration:
        return resolved_length

    token_offset = calibration.get("token_offset")
    if not isinstance(token_offset, (int, float)):
        return resolved_length

    if resolved_length <= SHORT_PROMPT_MAX_LENGTH:
        return resolved_length

    system_tokens = estimate_token_count(DEFAULT_SYSTEM_PROMPT)
    calibrated_user_length = int(round(resolved_length - system_tokens - token_offset))
    return max(BENCHMARK_SUFFIX_TOKEN_LENGTH, calibrated_user_length)


def estimate_generated_prompt_tokens(
    length: int,
    seed: int = 1,
    calibration: Optional[Dict[str, Any]] = None,
) -> int:
    """Estimate actual API prompt tokens after applying optional warmup calibration."""
    generation_length = get_calibrated_generation_prompt_length(length, calibration)
    return apply_prompt_calibration(
        estimate_prompt_tokens_for_messages(generate_prompt(generation_length, seed)),
        calibration,
    )


def _has_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def resolve_token_count_with_sanity_check(
    kind: str,
    api_tokens: Any,
    estimated_tokens: Any,
) -> Dict[str, Any]:
    """Prefer API token stats unless they look like cumulative/concurrent totals."""
    api_value = _coerce_int(api_tokens)
    estimated_value = _coerce_int(estimated_tokens)

    if not _has_positive_number(api_value):
        if _has_positive_number(estimated_value):
            return {"tokens": estimated_value, "source": "Local Estimation", "warning": None}
        return {"tokens": 0, "source": "Unknown", "warning": None}

    if _has_positive_number(estimated_value):
        relative_diff = abs(api_value - estimated_value) / max(estimated_value, 1)
        max_tokens = max(api_value, estimated_value)
        if max_tokens >= TOKEN_SANITY_MIN_TOKENS and relative_diff >= TOKEN_SANITY_MAX_RELATIVE_DIFF:
            return {
                "tokens": estimated_value,
                "source": "API Stats Anomaly",
                "warning": {
                    "type": "api_token_mismatch",
                    "kind": kind,
                    "api_tokens": api_value,
                    "estimated_tokens": estimated_value,
                    "relative_diff": relative_diff,
                },
            }

    return {"tokens": api_value, "source": "API", "warning": None}


def has_llamacpp_style_timings(usage_info: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(usage_info, dict):
        return False
    timings = usage_info.get("timings")
    if not isinstance(timings, dict):
        return False
    return any(
        _has_positive_number(timings.get(key))
        for key in ("prompt_ms", "predicted_ms", "prompt_n", "predicted_n")
    )


def calculate_aggregate_throughput_metrics(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate concurrent throughput with server timings when available."""
    if not results:
        return {
            "prefill_time_ms": 0.0,
            "output_time_ms": 0.0,
            "prefill_speed": 0.0,
            "output_speed": 0.0,
            "client_prefill_time_ms": 0.0,
            "client_output_time_ms": 0.0,
            "client_total_time_ms": 0.0,
            "client_has_phase_boundary": False,
            "prefill_source": "client",
            "output_source": "client",
            "warnings": [],
        }

    total_prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in results)
    total_output_tokens = sum(r.get("output_tokens") or 0 for r in results)
    average_network_latency_ms = get_average_network_latency_ms([
        r.get("network_latency_ms") or 0 for r in results
    ])
    min_start = min(r.get("start_timestamp") or 0 for r in results)
    has_client_phase_boundary = all(
        _has_positive_number(r.get("first_token_timestamp") or r.get("boundary_timestamp"))
        and r.get("boundary_source") != "no_boundary_fallback"
        for r in results
    )
    max_boundary = max(
        r.get("first_token_timestamp")
        or r.get("boundary_timestamp")
        or r.get("end_timestamp")
        or min_start
        for r in results
    )
    client_prefill_time_ms = max(((max_boundary - min_start) * 1000) - average_network_latency_ms, 1)
    min_decode_start = min(
        r.get("first_token_timestamp")
        or r.get("boundary_timestamp")
        or r.get("start_timestamp")
        or 0
        for r in results
    )
    max_end = max(r.get("end_timestamp") or min_decode_start for r in results)
    client_output_time_ms = max((max_end - min_decode_start) * 1000, 1)
    client_total_time_ms = max(((max_end - min_start) * 1000) - average_network_latency_ms, 1)

    use_server_prefill = all(
        _has_positive_number(r.get("server_prefill_time_ms"))
        and r.get("prefill_time_source") in (None, "server")
        for r in results
    )
    if use_server_prefill:
        prefill_time_ms = max(r["server_prefill_time_ms"] for r in results)
        prefill_source = "server"
    else:
        prefill_time_ms = client_prefill_time_ms
        prefill_source = "client_latency_adjusted" if average_network_latency_ms > 0 else "client"

    use_server_output = all(
        _has_positive_number(r.get("server_decode_time_ms"))
        and r.get("output_time_source") in (None, "server")
        for r in results
    )
    if use_server_output:
        output_time_ms = max(r["server_decode_time_ms"] for r in results)
        output_source = "server"
    else:
        output_time_ms = client_output_time_ms
        output_source = "client"

    prefill_speed = total_prompt_tokens / (prefill_time_ms / 1000) if prefill_time_ms > 0 else 0.0
    output_speed = total_output_tokens / (output_time_ms / 1000) if output_time_ms > 0 else 0.0
    warnings: List[Dict[str, Any]] = []

    has_concurrent_llamacpp_timings = (
        len(results) > 1
        and has_client_phase_boundary
        and any(has_llamacpp_style_timings(r.get("usage_info")) for r in results)
    )
    server_prefill_times = [
        r.get("server_prefill_time_ms") for r in results
        if _has_positive_number(r.get("server_prefill_time_ms"))
    ]
    server_output_times = [
        r.get("server_decode_time_ms") for r in results
        if _has_positive_number(r.get("server_decode_time_ms"))
    ]
    server_total_time_ms = sum(server_prefill_times) + sum(server_output_times)
    inferred_parallelism = 0.0
    if has_concurrent_llamacpp_timings and server_total_time_ms > 0 and client_total_time_ms > 0:
        inferred_parallelism = min(
            float(len(results)),
            max(1.0, server_total_time_ms / client_total_time_ms),
        )

    configured_parallelism = max(float(len(results)), 1.0)
    has_limited_server_parallelism = (
        inferred_parallelism > 0
        and inferred_parallelism < configured_parallelism * 0.8
    )
    if has_limited_server_parallelism and use_server_prefill and server_prefill_times:
        prefill_time_ms = client_prefill_time_ms
        prefill_speed = total_prompt_tokens / (client_prefill_time_ms / 1000)
        prefill_source = "client_llamacpp_timing_anomaly"
        warnings.append({
            "type": "api_timing_mismatch",
            "kind": "prefill_timing",
            "reason": "llamacpp_limited_parallel_slots",
            "api_time_ms": round(max(server_prefill_times), 2),
            "client_time_ms": round(client_prefill_time_ms, 2),
            "inferred_parallelism": round(inferred_parallelism, 2),
            "configured_concurrency": len(results),
        })

    if has_limited_server_parallelism and use_server_output and server_output_times:
        output_time_ms = client_output_time_ms
        output_speed = total_output_tokens / (client_output_time_ms / 1000)
        output_source = "client_llamacpp_timing_anomaly"
        warnings.append({
            "type": "api_timing_mismatch",
            "kind": "decode_timing",
            "reason": "llamacpp_limited_parallel_slots",
            "api_time_ms": round(max(server_output_times), 2),
            "client_time_ms": round(client_output_time_ms, 2),
            "inferred_parallelism": round(inferred_parallelism, 2),
            "configured_concurrency": len(results),
        })

    return {
        "prefill_time_ms": prefill_time_ms,
        "output_time_ms": output_time_ms,
        "prefill_speed": prefill_speed,
        "output_speed": output_speed,
        "client_prefill_time_ms": client_prefill_time_ms,
        "client_output_time_ms": client_output_time_ms,
        "client_total_time_ms": client_total_time_ms,
        "client_has_phase_boundary": has_client_phase_boundary,
        "prefill_source": prefill_source,
        "output_source": output_source,
        "warnings": warnings,
    }


def generate_warmup_prompt() -> str:
    """Generate a short randomized warmup prompt using the benchmark tokenizer path."""
    seed = random.randint(1, 1_000_000)
    return generate_prompt(96, seed=seed)


async def send_warmup_request(
    api_url: str,
    api_key: str,
    api_type: str,
    model_name: str,
    timeout: int,
    temperature: float,
    top_p: float,
) -> Optional[Dict[str, Any]]:
    """Send a short warmup request and wait for it to finish before benchmarking."""
    warmup_prompt = generate_warmup_prompt()
    warmup_timeout = max(timeout, 15000)
    timeout_seconds = warmup_timeout / 1000.0

    if api_type == "openai":
        headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": warmup_prompt},
            ],
            "max_tokens": 8,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
    else:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": warmup_prompt},
            ],
            "options": {
                "num_predict": 8,
                "temperature": temperature,
                "top_p": top_p,
            },
            "stream": False,
        }

    print(
        f"[Warmup] Sending {api_type} warmup request before benchmark "
        f"(timeout={warmup_timeout}ms)...",
        flush=True,
    )

    calibration = None
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(api_url, headers=headers, json=payload)
            response_bytes = response.content
            if response.status_code >= 400:
                raise RuntimeError(
                    build_http_error_message(api_url, model_name, response.status_code, response_bytes)
                )
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {}
            actual_prompt_tokens = None
            if isinstance(data, dict):
                if isinstance(data.get("usage"), dict):
                    actual_prompt_tokens = extract_usage_token_stats(data["usage"]).get("prompt_tokens")
                elif data.get("prompt_eval_count") is not None:
                    actual_prompt_tokens = _coerce_int(data.get("prompt_eval_count"))
            calibration = build_prompt_calibration(warmup_prompt, actual_prompt_tokens)
        print("[Warmup] Warmup request completed.", flush=True)
        if calibration:
            print(
                f"[Warmup] Prompt calibration: estimated={calibration['estimated_prompt_tokens']}, "
                f"actual={calibration['actual_prompt_tokens']}, "
                f"offset={calibration['token_offset']}, ratio={calibration['token_ratio']:.3f}",
                flush=True,
            )
    except Exception as exc:
        print(f"[Warmup] Warmup request failed: {exc}", flush=True)

    await asyncio.sleep(0.8)
    return calibration


def get_average_network_latency_ms(latency_samples: list[float]) -> float:
    valid_samples = [sample for sample in latency_samples if isinstance(sample, (int, float)) and sample > 0]
    if not valid_samples:
        return 0.0
    if len(valid_samples) >= 3:
        sorted_samples = sorted(valid_samples)
        valid_samples = sorted_samples[1:-1]
    return sum(valid_samples) / len(valid_samples)


async def measure_network_latency_sample(
    api_url: str,
    api_key: str,
    api_type: str,
    model_name: str,
    timeout: int,
    temperature: float,
    top_p: float,
) -> Optional[Dict[str, Any]]:
    probe_timeout = min(max((timeout or 0) / 1000.0, 3.0), 8.0)
    endpoint, _ = resolve_model_catalog_endpoint(api_url)
    if not endpoint:
        print("[Latency] No non-inference latency endpoint could be derived; skipping latency sample.", flush=True)
        return None

    headers: Dict[str, str] = {}
    if api_type == "openai" and api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=probe_timeout, verify=False) as client:
            start = time.perf_counter()
            response = await client.get(endpoint, headers=headers)
            _ = response.content
            latency_ms = (time.perf_counter() - start) * 1000
            return {"latency_ms": latency_ms, "probe_type": f"model_catalog {response.status_code}"}
    except Exception as exc:
        print(f"[Latency] model catalog latency probe failed: {exc}", flush=True)
        return None


async def collect_network_latency_sample(
    latency_samples: list[float],
    api_url: str,
    api_key: str,
    api_type: str,
    model_name: str,
    timeout: int,
    temperature: float,
    top_p: float,
) -> float:
    sample = await measure_network_latency_sample(
        api_url=api_url,
        api_key=api_key,
        api_type=api_type,
        model_name=model_name,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
    )
    if sample and isinstance(sample.get("latency_ms"), (int, float)) and sample["latency_ms"] > 0:
        latency_samples.append(float(sample["latency_ms"]))
        if len(latency_samples) > 20:
            del latency_samples[0]
        print(
            f"[Latency] {sample['probe_type']}: {sample['latency_ms']:.2f}ms, "
            f"avg={get_average_network_latency_ms(latency_samples):.2f}ms, "
            f"samples={len(latency_samples)}",
            flush=True,
        )

    return get_average_network_latency_ms(latency_samples)


def build_http_error_message(api_url: str, model_name: str, status_code: int, error_bytes: bytes) -> str:
    """Build a more actionable HTTP error message for common provider quirks."""
    error_text = redact_sensitive_text(error_bytes.decode(errors="replace"))
    error_msg = f"HTTP {status_code}: {error_text}"

    host = urlparse(api_url).netloc.lower()
    if status_code == 404 and "integrate.api.nvidia.com" in host:
        hint_parts = [
            "NVIDIA 的 /v1/chat/completions 入口通常是对的，404 更像是模型 ID 不存在、命名不匹配，或当前目录下没有路由到该模型。"
        ]

        suggested_model = NVIDIA_MODEL_NAME_HINTS.get(model_name)
        if suggested_model:
            hint_parts.append(f"当前传入的是 '{model_name}'，可优先改试官方命名 '{suggested_model}'。")
        elif model_name.startswith("minimax/"):
            hint_parts.append("NVIDIA 文档里的 Minimax 模型前缀通常是 'minimaxai/'，不是 'minimax/'。")

        error_msg = f"{error_msg} | 提示: {' '.join(hint_parts)}"

    return error_msg


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
    network_latency_ms: float = 0.0,
    network_latency_sample_count: int = 0,
    prompt_calibration: Optional[Dict[str, Any]] = None,
    client: Optional[httpx.AsyncClient] = None,
    enable_watchdog: bool = True,
    task_type: str = "general",
) -> Dict[str, Any]:
    """执行单个测试请求（支持共享连接池、流式死循环守卫与多场景题库）"""
    
    resolved_prompt_length = get_resolved_prompt_length(prompt_length)
    generation_prompt_length = get_calibrated_generation_prompt_length(prompt_length, prompt_calibration)
    print(
        f"[Request] 开始请求 - 任务: {task_type}, Prompt长度: {prompt_length}, "
        f"目标长度: {resolved_prompt_length}, 输出长度: {output_length}, Seed: {seed}"
    )

    system_prompt = get_system_prompt_for_task(task_type)
    prompt_text = generate_prompt(generation_prompt_length, seed, task_type=task_type)
    local_estimated_prompt_tokens = estimate_prompt_tokens_for_messages(prompt_text, system_prompt=system_prompt)
    estimated_prompt_tokens = apply_prompt_calibration(local_estimated_prompt_tokens, prompt_calibration)
    
    if api_type == "openai":
        headers = {
            "Content-Type": "application/json",
        }
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": output_length,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
    else:  # ollama
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
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
    first_chunk_time = None
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

    # Watchdogs
    content_watchdog = StreamingLoopDetector(channel="content") if enable_watchdog else None
    reasoning_watchdog = StreamingLoopDetector(channel="reasoning") if enable_watchdog else None
    loop_detected = False
    loop_finding: Optional[Dict[str, Any]] = None

    # ITL (Inter-Token Latency) tracking
    token_timestamps = []
    last_token_time = None
    
    timeout_seconds = timeout / 1000.0
    timeout_obj = httpx.Timeout(timeout_seconds, connect=15.0)

    should_close_client = False
    http_client = client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=timeout_obj, verify=False, http2=True)
        should_close_client = True

    try:
        try:
            async with http_client.stream("POST", api_url, headers=headers, json=payload, timeout=timeout_obj) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = build_http_error_message(api_url, model_name, response.status_code, error_text)
                    print(f"[Error] {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg
                    }
                
                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # 处理OpenAI SSE格式
                    if line.startswith("data: "):
                        if "[DONE]" in line:
                            break
                        json_line = line.replace("data: ", "")
                    else:
                        json_line = line.strip()

                    try:
                        data = json.loads(json_line)
                        chunk_count += 1

                        if data.get("done") is True:
                            # Ollama final timing chunk
                            pass

                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                            ttft_ms = (first_chunk_time - start_time) * 1000

                        # 提取usage信息（OpenAI格式）
                        if data.get("usage"):
                            usage = data["usage"]
                            usage_stats = extract_usage_token_stats(usage)
                            actual_output_tokens = usage_stats["output_tokens"]
                            if _has_positive_number(usage_stats["prompt_tokens"]):
                                actual_prompt_tokens = usage_stats["prompt_tokens"]
                            usage_info = usage.copy()

                            if usage.get("prompt_eval_duration"):
                                server_prefill_time_ms = usage["prompt_eval_duration"] / 1_000_000
                            elif usage.get("prompt_eval_time"):
                                server_prefill_time_ms = usage["prompt_eval_time"]
                            elif usage.get("prompt_time"):
                                server_prefill_time_ms = usage["prompt_time"] * 1000

                            if usage.get("eval_duration"):
                                server_decode_time_ms = usage["eval_duration"] / 1_000_000
                            elif usage.get("eval_time"):
                                server_decode_time_ms = usage["eval_time"]
                            elif usage.get("completion_time"):
                                server_decode_time_ms = usage["completion_time"] * 1000

                            if data.get("timings"):
                                usage_info["timings"] = data["timings"]

                        # 提取usage信息（Ollama格式）
                        if data.get("prompt_eval_count") is not None:
                            prompt = data.get("prompt_eval_count", 0)
                            completion = data.get("eval_count", 0)
                            actual_output_tokens = completion
                            if _has_positive_number(prompt):
                                actual_prompt_tokens = prompt

                            usage_info = {
                                "prompt_tokens": prompt,
                                "completion_tokens": completion,
                                "total_tokens": prompt + completion
                            }
                            if data.get("prompt_eval_duration"):
                                server_prefill_time_ms = data["prompt_eval_duration"] / 1_000_000
                                usage_info["prompt_eval_duration_ns"] = data["prompt_eval_duration"]
                            if data.get("eval_duration"):
                                server_decode_time_ms = data["eval_duration"] / 1_000_000
                                usage_info["eval_duration_ns"] = data["eval_duration"]
                            if data.get("total_duration"):
                                usage_info["total_duration_ns"] = data["total_duration"]
                            if data.get("load_duration"):
                                usage_info["load_duration_ns"] = data["load_duration"]
                        
                        # 提取timings
                        if data.get("timings") and not (server_prefill_time_ms and server_decode_time_ms):
                            timings = data["timings"]
                            if not server_prefill_time_ms:
                                if timings.get("prompt_eval_duration"):
                                    server_prefill_time_ms = timings["prompt_eval_duration"] / 1_000_000
                                elif timings.get("prompt_ms"):
                                    server_prefill_time_ms = timings["prompt_ms"]
                            if not server_decode_time_ms:
                                if timings.get("eval_duration"):
                                    server_decode_time_ms = timings["eval_duration"] / 1_000_000
                                elif timings.get("predicted_ms"):
                                    server_decode_time_ms = timings["predicted_ms"]

                        # 提取文本内容
                        content_text = None
                        is_reasoning = False
                        if data.get("response") is not None:
                            content_text = data["response"]
                            content_chunk_count += 1
                        
                        if data.get("choices") and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if choice.get("delta", {}).get("reasoning_content"):
                                content_text = choice["delta"]["reasoning_content"]
                                is_reasoning = True
                                reasoning_chunk_count += 1
                            elif choice.get("delta", {}).get("reasoning"):
                                content_text = choice["delta"]["reasoning"]
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

                            token_timestamps.append({
                                "timestamp": current_token_time,
                                "time_from_start_ms": (current_token_time - start_time) * 1000,
                                "is_reasoning": is_reasoning
                            })

                            if last_token_time is not None:
                                itl = (current_token_time - last_token_time) * 1000
                                token_timestamps[-1]["itl_ms"] = itl
                            else:
                                token_timestamps[-1]["itl_ms"] = None

                            last_token_time = current_token_time

                            # Watchdog checks
                            if is_reasoning:
                                reasoning_content += content_text
                                if reasoning_watchdog:
                                    wf = reasoning_watchdog.feed(content_text)
                                    if wf and wf.get("status") == "confirmed":
                                        loop_detected = True
                                        loop_finding = wf
                                        print(f"[Watchdog] 思考流死循环熔断触发! Period={wf['period_chars']}, Cycles={wf['complete_cycles']}. 立即中止流。")
                                        break
                            else:
                                output_content += content_text
                                if content_watchdog:
                                    wf = content_watchdog.feed(content_text)
                                    if wf and wf.get("status") == "confirmed":
                                        loop_detected = True
                                        loop_finding = wf
                                        print(f"[Watchdog] 内容流死循环熔断触发! Period={wf['period_chars']}, Cycles={wf['complete_cycles']}. 立即中止流。")
                                        break


                        if loop_detected:
                            break

                        if data.get("done") is True:
                            break

                    except json.JSONDecodeError:
                        continue
        finally:
            if should_close_client:
                await http_client.aclose()
        
        end_time = time.perf_counter()

        # Final loop watchdog flush check
        if not loop_detected:
            if content_watchdog:
                wf = content_watchdog.finish()
                if wf and wf.get("status") == "confirmed":
                    loop_detected = True
                    loop_finding = wf
                    print(f"[Watchdog] 流结束时发现死循环: {wf}")
            if reasoning_watchdog and not loop_detected:
                wf = reasoning_watchdog.finish()
                if wf and wf.get("status") == "confirmed":
                    loop_detected = True
                    loop_finding = wf
                    print(f"[Watchdog] 流结束时发现思考流死循环: {wf}")

        print(f"[Response] 接收完成 - 内容长度: {len(output_content)}, Reasoning: {len(reasoning_content)}, Loop: {loop_detected}")

        total_time_ms = (end_time - start_time) * 1000

        boundary_source = "first_content_token"
        if first_token_time is not None:
            boundary_time = first_token_time
            ttft_ms = (boundary_time - start_time) * 1000
        elif first_chunk_time is not None:
            boundary_source = "first_chunk_fallback"
            boundary_time = first_chunk_time
            ttft_ms = (boundary_time - start_time) * 1000
        else:
            boundary_source = "no_boundary_fallback"
            boundary_time = end_time
            ttft_ms = total_time_ms

        decode_time_ms = max(total_time_ms - ttft_ms, 1)

        token_warnings: List[Dict[str, Any]] = []
        api_prompt_tokens = actual_prompt_tokens
        api_output_tokens = actual_output_tokens
        local_output_tokens = (
            estimate_token_count(reasoning_content) + estimate_token_count(output_content)
            if (reasoning_content or output_content)
            else None
        )

        prompt_resolution = resolve_token_count_with_sanity_check(
            "prompt",
            api_prompt_tokens,
            estimated_prompt_tokens,
        )
        actual_prompt_tokens = prompt_resolution["tokens"] or estimated_prompt_tokens
        if prompt_resolution["warning"]:
            token_warnings.append(prompt_resolution["warning"])

        output_resolution = resolve_token_count_with_sanity_check(
            "decode",
            api_output_tokens,
            local_output_tokens,
        )
        actual_output_tokens = output_resolution["tokens"]
        if output_resolution["warning"]:
            token_warnings.append(output_resolution["warning"])

        if token_warnings:
            token_source = "API Stats Anomaly"
        elif output_resolution["source"] == "API" or prompt_resolution["source"] == "API":
            token_source = "API"
        elif output_resolution["source"] == "Local Estimation" or prompt_resolution["source"] == "Local Estimation":
            token_source = "Local Estimation"
        else:
            token_source = "Unknown"

        # 优先使用服务器返回的timing
        if server_prefill_time_ms is not None and server_decode_time_ms is not None:
            prefill_time_ms = server_prefill_time_ms
            output_time_ms = server_decode_time_ms
            time_source = '服务器timing'
        else:
            prefill_time_ms = ttft_ms
            output_time_ms = decode_time_ms
            time_source = '端到端测量（基于chunk时序）'

        has_server_prefill_timing = _has_positive_number(server_prefill_time_ms)
        has_server_decode_timing = _has_positive_number(server_decode_time_ms)
        latency_adjustment_ms = network_latency_ms if _has_positive_number(network_latency_ms) else 0.0
        client_prefill_time_ms = max(ttft_ms - latency_adjustment_ms, 1)
        client_decode_time_ms = decode_time_ms
        if has_server_prefill_timing or has_server_decode_timing:
            prefill_time_ms = server_prefill_time_ms if has_server_prefill_timing else client_prefill_time_ms
            output_time_ms = server_decode_time_ms if has_server_decode_timing else client_decode_time_ms
            prefill_time_source = "server" if has_server_prefill_timing else ("client_latency_adjusted" if latency_adjustment_ms > 0 else "client")
            output_time_source = "server" if has_server_decode_timing else "client"
        else:
            prefill_time_ms = client_prefill_time_ms
            output_time_ms = client_decode_time_ms
            prefill_time_source = "client_latency_adjusted" if latency_adjustment_ms > 0 else "client"
            output_time_source = "client"

        # 计算速度 (tokens/second)
        if prefill_time_ms > 0:
            prefill_speed = (actual_prompt_tokens / (prefill_time_ms / 1000))
        else:
            prefill_speed = 0

        if output_time_ms > 0 and actual_output_tokens > 0:
            output_speed = (actual_output_tokens / (output_time_ms / 1000))
        else:
            output_speed = 0

        # 如果熔断器捕捉到死循环，设置哨兵速率 -3.0
        if loop_detected:
            output_speed = LOOP_CELL_TPS

        # 计算ITL分位数及统计
        itl_stats = {}
        if len(token_timestamps) > 1:
            itl_values = [t["itl_ms"] for t in token_timestamps if t["itl_ms"] is not None]
            if itl_values:
                itl_stats = {
                    "mean": round(sum(itl_values) / len(itl_values), 2),
                    "p50": compute_percentile(itl_values, 50),
                    "p90": compute_percentile(itl_values, 90),
                    "p99": compute_percentile(itl_values, 99),
                    "min": round(min(itl_values), 2),
                    "max": round(max(itl_values), 2),
                    "std": round((sum((x - sum(itl_values)/len(itl_values))**2 for x in itl_values) / len(itl_values))**0.5, 2) if len(itl_values) > 1 else 0,
                    "count": len(itl_values)
                }

        return {
            "success": True,
            "prompt_length": resolved_prompt_length,
            "requested_prompt_length": prompt_length,
            "generated_prompt_length": generation_prompt_length,
            "prompt_tokens": actual_prompt_tokens,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "local_estimated_prompt_tokens": local_estimated_prompt_tokens,
            "local_estimated_output_tokens": local_output_tokens,
            "prompt_calibration": prompt_calibration,
            "output_tokens": actual_output_tokens,
            "ttft_ms": round(ttft_ms, 2),
            "prefill_time_ms": round(prefill_time_ms, 2),
            "prefill_speed": round(prefill_speed, 2),
            "output_time_ms": round(output_time_ms, 2),
            "output_speed": round(output_speed, 2),
            "total_time_ms": round(total_time_ms, 2),
            "prompt_text": prompt_text,
            "output_content": output_content,
            "reasoning_content": reasoning_content,
            "usage_info": usage_info,
            "server_timing_used": server_prefill_time_ms is not None or server_decode_time_ms is not None,
            "server_prefill_time_ms": round(server_prefill_time_ms, 2) if server_prefill_time_ms is not None else None,
            "server_decode_time_ms": round(server_decode_time_ms, 2) if server_decode_time_ms is not None else None,
            "prefill_time_source": prefill_time_source,
            "output_time_source": output_time_source,
            "network_latency_ms": round(latency_adjustment_ms, 2),
            "network_latency_sample_count": network_latency_sample_count,
            "client_prefill_time_ms": round(client_prefill_time_ms, 2),
            "client_decode_time_ms": round(client_decode_time_ms, 2),
            "client_total_time_ms": round(max(total_time_ms - latency_adjustment_ms, 1), 2),
            "client_has_phase_boundary": boundary_source != "no_boundary_fallback",
            "has_streaming_content": first_token_time is not None,
            "chunk_count": chunk_count,
            "start_timestamp": start_time,
            "first_chunk_timestamp": first_chunk_time,
            "first_token_timestamp": first_token_time,
            "boundary_timestamp": boundary_time,
            "boundary_source": boundary_source,
            "end_timestamp": end_time,
            "token_source": token_source,
            "token_warnings": token_warnings,
            "token_timestamps": token_timestamps,
            "itl_stats": itl_stats,
            "loop_detected": loop_detected,
            "loop_diagnostics": loop_finding,
            "cjk_drift_detected": False,
        }
    
    except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        error_msg = f"请求超时: {type(e).__name__}: {str(e)} (提示词长度: {prompt_length}, 超时设置: {timeout}ms)"
        print(f"[Error] {error_msg}", flush=True)
        import traceback
        print(f"[Error] 详细堆栈:\n{traceback.format_exc()}", flush=True)
        return {
            "success": False,
            "error": error_msg,
            "prompt_length": resolved_prompt_length,
            "requested_prompt_length": prompt_length,
            "generated_prompt_length": generation_prompt_length,
        }
    except httpx.HTTPError as e:
        error_msg = f"HTTP错误: {type(e).__name__}: {str(e)}"
        print(f"[Error] {error_msg}", flush=True)
        import traceback
        print(f"[Error] 详细堆栈:\n{traceback.format_exc()}", flush=True)
        return {
            "success": False,
            "error": error_msg,
            "prompt_length": resolved_prompt_length,
            "requested_prompt_length": prompt_length,
            "generated_prompt_length": generation_prompt_length,
        }
    except Exception as e:
        error_msg = f"请求异常: {type(e).__name__}: {str(e)}"
        print(f"[Error] {error_msg}", flush=True)
        import traceback
        print(f"[Error] 详细堆栈:\n{traceback.format_exc()}", flush=True)
        return {
            "success": False,
            "error": error_msg,
            "prompt_length": resolved_prompt_length,
            "requested_prompt_length": prompt_length,
            "generated_prompt_length": generation_prompt_length,
        }


async def run_2d_matrix_benchmark(
    api_url: str,
    api_key: str,
    api_type: str,
    model_name: str,
    concurrency_ladder: List[int],
    context_ladder: List[int],
    max_tokens: int = 128,
    max_kv_tokens: int = 32768,
    timeout: int = 120000,
    temperature: float = 0.7,
    top_p: float = 0.9,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    metrics_url: Optional[str] = None,
    hardware: Optional[str] = None,
    framework: Optional[str] = None,
    quantization: Optional[str] = None,
    enable_watchdog: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    task_type: str = "general",
) -> Dict[str, Any]:
    """Execute 2D Concurrency x Context Matrix Sweep with scout traversal, capacity skipping and cell error isolation."""
    hardware = hardware or detect_hardware()
    framework = framework or "vLLM"
    quantization = quantization or "FP16"

    # Normalize ladders
    concurrency_ladder = [int(c) for c in concurrency_ladder if int(c) > 0]
    context_ladder = [int(ctx) for ctx in context_ladder if int(ctx) > 0]
    if not concurrency_ladder:
        concurrency_ladder = [1, 2, 4, 8]
    if not context_ladder:
        context_ladder = [512, 1024, 2048]

    cell_order = get_matrix_traversal_order(concurrency_ladder, context_ladder)
    total_cells = len(cell_order)

    matrix_grid: Dict[str, Dict[str, Any]] = {str(c): {} for c in concurrency_ladder}
    all_cell_results: List[Dict[str, Any]] = []
    all_successful_requests: List[Dict[str, Any]] = []

    completed_cells = 0
    skipped_cells = 0
    errored_cells = 0
    peak_decode_tps = 0.0
    peak_point: Optional[Dict[str, Any]] = None
    saturation_onset: str = "None"

    print(f"\n[Matrix] Starting 2D Matrix Sweep ({len(concurrency_ladder)}x{len(context_ladder)} = {total_cells} cells)...")

    for idx, (conc, ctx) in enumerate(cell_order):
        tokens_needed = conc * (ctx + max_tokens)
        if should_skip_matrix_cell(conc, ctx, max_tokens, max_kv_tokens):
            skipped_cells += 1
            print(f"[Matrix] Cell ({conc}x, {ctx}ctx): SKIP_KV_LIMIT ({tokens_needed} > {max_kv_tokens})")
            cell_result = {
                "concurrency": conc,
                "context": ctx,
                "prompt_length": ctx,
                "requested_prompt_length": ctx,
                "decode_tps": -1.0,
                "avg_decode_speed": -1.0,
                "avg_prefill_speed": 0.0,
                "max_decode_speed": 0.0,
                "max_prefill_speed": 0.0,
                "ttft_ms": 0.0,
                "avg_ttft_ms": 0.0,
                "status": "SKIP_KV_LIMIT",
                "skipped": True,
                "skip_reason": f"KV limit exceeded: {tokens_needed} > {max_kv_tokens}",
                "tokens_needed": tokens_needed,
                "max_kv_tokens": max_kv_tokens,
                "success_count": 0,
                "failed_count": 0,
                "effective_concurrency": f"{conc}/{conc}",
            }
            matrix_grid[str(conc)][str(ctx)] = cell_result
            all_cell_results.append(cell_result)

            if progress_callback:
                try:
                    await progress_callback({
                        "type": "progress",
                        "benchmark_mode": "2d_matrix",
                        "cell_current": idx + 1,
                        "cell_total": total_cells,
                        "concurrency": conc,
                        "context": ctx,
                        "tokens_needed": tokens_needed,
                        "status": "SKIP_KV_LIMIT",
                    })
                except Exception as cb_err:
                    print(f"[Matrix] Progress callback disconnected: {cb_err}")
                    break
            if result_callback:
                msg = dict(cell_result)
                msg["type"] = "result"
                msg["benchmark_mode"] = "2d_matrix"
                try:
                    await result_callback(msg)
                except Exception as cb_err:
                    print(f"[Matrix] Result callback disconnected: {cb_err}")
                    break
            continue

        print(f"[Matrix] Cell ({conc}x, {ctx}ctx): Running {conc} streams ({idx + 1}/{total_cells})...")
        if progress_callback:
            try:
                await progress_callback({
                    "type": "progress",
                    "benchmark_mode": "2d_matrix",
                    "cell_current": idx + 1,
                    "cell_total": total_cells,
                    "concurrency": conc,
                    "context": ctx,
                    "tokens_needed": tokens_needed,
                    "status": "testing",
                })
            except Exception as cb_err:
                print(f"[Matrix] Progress callback disconnected: {cb_err}")
                break

        try:
            client_limits = httpx.Limits(
                max_connections=max(256, conc * 2),
                max_keepalive_connections=max(128, conc),
            )
            async with httpx.AsyncClient(limits=client_limits, http2=True, timeout=timeout / 1000.0, verify=False) as http_client:
                seeds = [random.randint(1000, 999999) for _ in range(conc)]
                tasks = [
                    execute_single_request(
                        api_url=api_url,
                        api_key=api_key,
                        api_type=api_type,
                        model_name=model_name,
                        prompt_length=ctx,
                        output_length=max_tokens,
                        timeout=timeout,
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        seed=s,
                        client=http_client,
                        enable_watchdog=enable_watchdog,
                        task_type=task_type,
                    )
                    for s in seeds
                ]
                req_results = await asyncio.gather(*tasks, return_exceptions=True)

            succ = [r for r in req_results if isinstance(r, dict) and r.get("success")]
            failed_count = conc - len(succ)

            if not succ:
                errored_cells += 1
                cell_result = {
                    "concurrency": conc,
                    "context": ctx,
                    "prompt_length": ctx,
                    "requested_prompt_length": ctx,
                    "decode_tps": 0.0,
                    "avg_decode_speed": 0.0,
                    "avg_prefill_speed": 0.0,
                    "max_decode_speed": 0.0,
                    "max_prefill_speed": 0.0,
                    "ttft_ms": 0.0,
                    "avg_ttft_ms": 0.0,
                    "status": "ERROR",
                    "error": f"All {conc} requests failed",
                    "skipped": False,
                    "tokens_needed": tokens_needed,
                    "failed_count": failed_count,
                    "success_count": 0,
                    "effective_concurrency": f"(0/{conc})*",
                }
            else:
                completed_cells += 1
                all_successful_requests.extend(succ)

                avg_prefill = sum(r["prefill_speed"] for r in succ) / len(succ)
                avg_decode = sum(r["output_speed"] for r in succ) / len(succ)
                max_prefill = max(r["prefill_speed"] for r in succ)
                max_decode = max(r["output_speed"] for r in succ)
                avg_ttft = sum(r["ttft_ms"] for r in succ) / len(succ)

                aggregate_metrics = calculate_aggregate_throughput_metrics(succ)
                eff_conc = calculate_effective_concurrency(
                    concurrency=conc,
                    telemetry_summary={},
                    client_inferred_parallelism=aggregate_metrics.get("client_inferred_parallelism"),
                )

                all_ttft = [r["ttft_ms"] for r in succ if r.get("ttft_ms")]
                ttft_stats = {
                    "avg": round(sum(all_ttft) / len(all_ttft), 2) if all_ttft else 0.0,
                    "p50": compute_percentile(all_ttft, 50),
                    "p90": compute_percentile(all_ttft, 90),
                    "p99": compute_percentile(all_ttft, 99),
                }

                all_itl = [
                    t["itl_ms"]
                    for r in succ
                    for t in r.get("token_timestamps", [])
                    if t.get("itl_ms") is not None
                ]
                itl_stats = {
                    "avg": round(sum(all_itl) / len(all_itl), 2) if all_itl else 0.0,
                    "p50": compute_percentile(all_itl, 50),
                    "p90": compute_percentile(all_itl, 90),
                    "p99": compute_percentile(all_itl, 99),
                }

                loop_detected = any(r.get("loop_detected") for r in succ)
                status_str = "成功" if not loop_detected else "ERROR: loop"

                if avg_decode > peak_decode_tps:
                    peak_decode_tps = round(avg_decode, 2)
                    peak_point = {"concurrency": conc, "context": ctx, "decode_tps": peak_decode_tps}

                if saturation_onset == "None" and eff_conc.get("is_saturated"):
                    saturation_onset = eff_conc.get("label", f"{conc}x")

                cell_vh = compute_verification_hash(
                    model_name=model_name,
                    hardware=hardware,
                    framework=framework,
                    quantization=quantization,
                    concurrency=conc,
                    max_decode_speed=max_decode,
                )

                cell_result = {
                    "concurrency": conc,
                    "context": ctx,
                    "prompt_length": ctx,
                    "requested_prompt_length": ctx,
                    "decode_tps": round(avg_decode, 2),
                    "avg_decode_speed": round(avg_decode, 2),
                    "avg_prefill_speed": round(avg_prefill, 2),
                    "max_decode_speed": round(max_decode, 2),
                    "max_prefill_speed": round(max_prefill, 2),
                    "ttft_ms": round(avg_ttft, 2),
                    "avg_ttft_ms": round(avg_ttft, 2),
                    "ttft_stats": ttft_stats,
                    "itl_stats": itl_stats,
                    "effective_concurrency": eff_conc.get("label", f"{conc}/{conc}"),
                    "effective_concurrency_detail": eff_conc,
                    "loop_detected": loop_detected,
                    "skipped": False,
                    "status": status_str,
                    "tokens_needed": tokens_needed,
                    "success_count": len(succ),
                    "failed_count": failed_count,
                    "verification_hash": cell_vh,
                    "concurrent_details": [
                        {
                            "concurrency": r.get("concurrency", 1),
                            "output_speed": r.get("output_speed"),
                            "prefill_speed": r.get("prefill_speed"),
                            "ttft_ms": r.get("ttft_ms"),
                            "avg_ttft_ms": r.get("ttft_ms"),
                            "output_tokens": r.get("output_tokens"),
                            "total_time_ms": r.get("total_time_ms"),
                            "loop_detected": r.get("loop_detected", False),
                            "status": r.get("status", "成功"),
                        }
                        for r in succ
                    ],
                }

        except Exception as e:
            errored_cells += 1
            print(f"[Matrix] Cell ({conc}x, {ctx}ctx) exception: {e}")
            cell_result = {
                "concurrency": conc,
                "context": ctx,
                "prompt_length": ctx,
                "requested_prompt_length": ctx,
                "decode_tps": 0.0,
                "avg_decode_speed": 0.0,
                "avg_prefill_speed": 0.0,
                "max_decode_speed": 0.0,
                "max_prefill_speed": 0.0,
                "ttft_ms": 0.0,
                "avg_ttft_ms": 0.0,
                "status": "ERROR",
                "error": str(e),
                "skipped": False,
                "tokens_needed": tokens_needed,
                "failed_count": conc,
                "success_count": 0,
                "effective_concurrency": f"(0/{conc})*",
            }

        matrix_grid[str(conc)][str(ctx)] = cell_result
        all_cell_results.append(cell_result)

        if result_callback:
            msg = dict(cell_result)
            msg["type"] = "result"
            msg["benchmark_mode"] = "2d_matrix"
            try:
                await result_callback(msg)
            except Exception as cb_err:
                print(f"[Matrix] Result callback disconnected: {cb_err}")
                break

        # Cooldown between cells
        await asyncio.sleep(1.0)

    peak_concurrency = max(concurrency_ladder) if concurrency_ladder else 1
    valid_decodes = [c["avg_decode_speed"] for c in all_cell_results if not c.get("skipped") and c.get("avg_decode_speed", 0) > 0]
    valid_prefills = [c["avg_prefill_speed"] for c in all_cell_results if not c.get("skipped") and c.get("avg_prefill_speed", 0) > 0]

    overall_avg_decode = sum(valid_decodes) / len(valid_decodes) if valid_decodes else 0.0
    overall_avg_prefill = sum(valid_prefills) / len(valid_prefills) if valid_prefills else 0.0
    overall_max_decode = max((c.get("max_decode_speed", 0.0) for c in all_cell_results if not c.get("skipped")), default=0.0)
    overall_max_prefill = max((c.get("max_prefill_speed", 0.0) for c in all_cell_results if not c.get("skipped")), default=0.0)

    matrix_summary = {
        "task_type": task_type,
        "peak_decode_tps": peak_decode_tps,
        "peak_point": peak_point or {},
        "saturation_onset": saturation_onset,
        "total_cells": total_cells,
        "completed_cells": completed_cells,
        "skipped_cells": skipped_cells,
        "errored_cells": errored_cells,
        "overall_avg_decode": round(overall_avg_decode, 2),
        "overall_avg_prefill": round(overall_avg_prefill, 2),
        "overall_max_decode": round(overall_max_decode, 2),
        "overall_max_prefill": round(overall_max_prefill, 2),
        "peak_concurrency": peak_concurrency,
    }

    points = [
        {
            "task_type": task_type,
            "concurrency": c.get("concurrency"),
            "context": c.get("context"),
            "prompt_length": c.get("context"),
            "decode_tps": c.get("decode_tps", 0.0),
            "output_speed": c.get("decode_tps", 0.0),
            "prefill_speed": c.get("avg_prefill_speed", 0.0),
            "avg_ttft_ms": c.get("avg_ttft_ms", 0.0),
            "ttft_p50_ms": c.get("ttft_stats", {}).get("p50", 0.0),
            "itl_p50_ms": c.get("itl_stats", {}).get("p50", 0.0),
            "effective_concurrency": c.get("effective_concurrency", ""),
            "status": c.get("status", ""),
            "skipped": c.get("skipped", False),
        }
        for c in all_cell_results
    ]

    all_ttft_values = [r["ttft_ms"] for r in all_successful_requests if r.get("ttft_ms")]
    all_itl_values = [
        t["itl_ms"]
        for r in all_successful_requests
        for t in r.get("token_timestamps", [])
        if t.get("itl_ms") is not None
    ]

    quantiles = {
        "ttft": {
            "avg": round(sum(all_ttft_values) / len(all_ttft_values), 2) if all_ttft_values else 0.0,
            "p50": compute_percentile(all_ttft_values, 50),
            "p90": compute_percentile(all_ttft_values, 90),
            "p99": compute_percentile(all_ttft_values, 99),
        },
        "itl": {
            "avg": round(sum(all_itl_values) / len(all_itl_values), 2) if all_itl_values else 0.0,
            "p50": compute_percentile(all_itl_values, 50),
            "p90": compute_percentile(all_itl_values, 90),
            "p99": compute_percentile(all_itl_values, 99),
        }
    }

    overall_vh = compute_verification_hash(
        model_name=model_name,
        hardware=hardware,
        framework=framework,
        quantization=quantization,
        concurrency=peak_concurrency,
        max_decode_speed=overall_max_decode,
    )

    return {
        "benchmark_mode": "2d_matrix",
        "model_name": model_name,
        "hardware": hardware,
        "framework": framework,
        "quantization": quantization,
        "concurrency_ladder": concurrency_ladder,
        "context_ladder": context_ladder,
        "max_kv_tokens": max_kv_tokens,
        "peak_concurrency": peak_concurrency,
        "matrix_grid": matrix_grid,
        "matrix_summary": matrix_summary,
        "all_cell_results": all_cell_results,
        "points": points,
        "quantiles": quantiles,
        "overall_avg_decode": round(overall_avg_decode, 2),
        "overall_avg_prefill": round(overall_avg_prefill, 2),
        "overall_max_decode": round(overall_max_decode, 2),
        "overall_max_prefill": round(overall_max_prefill, 2),
        "verification_hash": overall_vh,
    }


async def run_sustained_decode_benchmark(
    api_url: str,
    api_key: str,
    api_type: str,
    model_name: str,
    concurrency: int = 1,
    prompt_length: int = 128,
    output_length: int = 128,
    duration_seconds: float = 30.0,
    timeout: int = 120000,
    temperature: float = 0.7,
    top_p: float = 0.9,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    metrics_url: Optional[str] = None,
    hardware: Optional[str] = None,
    framework: Optional[str] = None,
    quantization: Optional[str] = None,
    enable_watchdog: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    task_type: str = "general",
) -> Dict[str, Any]:
    """Execute Sustained Steady-State Decode benchmark with worker pool replenishment and window enforcement."""
    hardware = hardware or detect_hardware()
    framework = framework or "vLLM"
    quantization = quantization or "FP16"
    concurrency = max(1, int(concurrency))
    duration_seconds = max(1.0, float(duration_seconds))

    window_start = time.monotonic()
    window_deadline = window_start + duration_seconds
    total_tokens_generated = 0
    completed_requests: List[Dict[str, Any]] = []
    active_stream_count = 0
    lock = asyncio.Lock()
    stop_event = asyncio.Event()

    print(f"\n[Sustained] Starting Sustained Decode: Concurrency={concurrency}, Duration={duration_seconds}s, Prompt={prompt_length} tokens...")

    client_limits = httpx.Limits(
        max_connections=max(256, concurrency * 2),
        max_keepalive_connections=max(128, concurrency),
    )

    async with httpx.AsyncClient(limits=client_limits, http2=True, timeout=timeout / 1000.0, verify=False) as http_client:
        async def worker(worker_id: int):
            nonlocal total_tokens_generated, active_stream_count
            req_index = 0
            while not stop_event.is_set() and time.monotonic() < window_deadline:
                time_left = window_deadline - time.monotonic()
                if time_left <= 0.2:
                    break
                req_timeout = min(timeout, int(time_left * 1000) + 2000)

                async with lock:
                    active_stream_count += 1
                try:
                    seed = worker_id * 10000 + req_index * 100 + random.randint(1, 99)
                    req_index += 1
                    res = await execute_single_request(
                        api_url=api_url,
                        api_key=api_key,
                        api_type=api_type,
                        model_name=model_name,
                        prompt_length=prompt_length,
                        output_length=output_length,
                        timeout=req_timeout,
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        seed=seed,
                        client=http_client,
                        enable_watchdog=enable_watchdog,
                        task_type=task_type,
                    )
                    if res.get("success") and (res.get("output_tokens") or 0) > 0:
                        async with lock:
                            total_tokens_generated += res["output_tokens"]
                            completed_requests.append(res)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[SustainedWorker {worker_id}] Request error: {e}")
                finally:
                    async with lock:
                        active_stream_count -= 1

                def compute_live_progress() -> Dict[str, Any]:
                    elapsed = time.monotonic() - window_start
                    total_decode_s = sum(max(r.get("output_time_ms", 0), 1) / 1000.0 for r in completed_requests)
                    if total_decode_s > 0 and len(completed_requests) > 0:
                        cur_stream_speed = total_tokens_generated / total_decode_s
                        active_conc = min(float(concurrency), max(1.0, float(len(completed_requests))))
                        live_tps = cur_stream_speed * active_conc
                    else:
                        live_tps = 0.0
                    return {
                        "elapsed_sec": round(elapsed, 2),
                        "duration_sec": round(duration_seconds, 2),
                        "total_tokens": total_tokens_generated,
                        "completed_streams": len(completed_requests),
                        "live_tps": round(live_tps, 2),
                        "e2e_tps": round(total_tokens_generated / max(elapsed, 0.001), 2),
                        "active_workers": active_stream_count,
                    }

                if progress_callback and not stop_event.is_set():
                    await progress_callback(compute_live_progress())

        async def ticker():
            while not stop_event.is_set() and time.monotonic() < window_deadline:
                await asyncio.sleep(0.5)
                if progress_callback and not stop_event.is_set():
                    elapsed = time.monotonic() - window_start
                    total_decode_s = sum(max(r.get("output_time_ms", 0), 1) / 1000.0 for r in completed_requests)
                    if total_decode_s > 0 and len(completed_requests) > 0:
                        cur_stream_speed = total_tokens_generated / total_decode_s
                        active_conc = min(float(concurrency), max(1.0, float(len(completed_requests))))
                        live_tps = cur_stream_speed * active_conc
                    else:
                        live_tps = 0.0
                    await progress_callback({
                        "elapsed_sec": round(elapsed, 2),
                        "duration_sec": round(duration_seconds, 2),
                        "total_tokens": total_tokens_generated,
                        "completed_streams": len(completed_requests),
                        "live_tps": round(live_tps, 2),
                        "e2e_tps": round(total_tokens_generated / max(elapsed, 0.001), 2),
                        "active_workers": active_stream_count,
                    })

        ticker_task = asyncio.create_task(ticker())
        workers = [asyncio.create_task(worker(w + 1)) for w in range(concurrency)]

        sleep_duration = max(0.0, window_deadline - time.monotonic())
        await asyncio.sleep(sleep_duration)
        stop_event.set()
        ticker_task.cancel()

        done, pending = await asyncio.wait(workers, timeout=2.0)
        for p in pending:
            p.cancel()

    actual_duration_sec = time.monotonic() - window_start
    stream_durations = [r["output_time_ms"] / 1000.0 for r in completed_requests if r.get("output_time_ms")]
    sum_stream_durations = sum(stream_durations)
    avg_stream_decode_speed = (total_tokens_generated / sum_stream_durations) if sum_stream_durations > 0 else 0.0

    all_ttft = [r["ttft_ms"] for r in completed_requests if r.get("ttft_ms")]
    sum_ttft_s = sum(all_ttft) / 1000.0
    total_prompt_tokens = sum(r.get("prompt_tokens") or prompt_length for r in completed_requests)
    avg_prefill = round(total_prompt_tokens / sum_ttft_s, 2) if sum_ttft_s > 0 else 0.0

    first_tokens = [
        r.get("first_token_timestamp") or r.get("boundary_timestamp") or r.get("start_timestamp")
        for r in completed_requests
        if r.get("start_timestamp") is not None
    ]
    end_tokens = [r.get("end_timestamp") for r in completed_requests if r.get("end_timestamp") is not None]
    if first_tokens and end_tokens and len(completed_requests) > 1:
        active_decode_span_s = max(max(end_tokens) - min(first_tokens), 0.001)
    else:
        active_decode_span_s = sum_stream_durations if concurrency == 1 else actual_duration_sec

    inferred_parallelism = min(float(concurrency), max(1.0, sum_stream_durations / active_decode_span_s)) if active_decode_span_s > 0 else float(concurrency)
    is_saturated = concurrency > 1 and inferred_parallelism < (concurrency * 0.8)
    p_effective = inferred_parallelism if is_saturated else float(concurrency)
    eff_conc_label = f"({round(inferred_parallelism)}/{concurrency})*" if is_saturated else f"{concurrency}/{concurrency}"

    # Pure steady-state decode throughput (deducting TTFT / prefill duration)
    steady_decode_tps = avg_stream_decode_speed * p_effective
    e2e_throughput_tps = total_tokens_generated / max(actual_duration_sec, 0.001)

    all_itl = [
        t["itl_ms"]
        for r in completed_requests
        for t in r.get("token_timestamps", [])
        if t.get("itl_ms") is not None
    ]

    ttft_stats = {
        "avg": round(sum(all_ttft) / len(all_ttft), 2) if all_ttft else 0.0,
        "p50": compute_percentile(all_ttft, 50),
        "p90": compute_percentile(all_ttft, 90),
        "p99": compute_percentile(all_ttft, 99),
        "std": round((sum((x - (sum(all_ttft)/len(all_ttft)))**2 for x in all_ttft) / len(all_ttft))**0.5, 2) if len(all_ttft) > 1 else 0.0
    }

    itl_stats = {
        "avg": round(sum(all_itl) / len(all_itl), 2) if all_itl else 0.0,
        "p50": compute_percentile(all_itl, 50),
        "p90": compute_percentile(all_itl, 90),
        "p99": compute_percentile(all_itl, 99),
        "std": round((sum((x - (sum(all_itl)/len(all_itl)))**2 for x in all_itl) / len(all_itl))**0.5, 2) if len(all_itl) > 1 else 0.0
    }

    loop_detected = any(r.get("loop_detected") for r in completed_requests)

    overall_vh = compute_verification_hash(
        model_name=model_name,
        hardware=hardware,
        framework=framework,
        quantization=quantization,
        concurrency=concurrency,
        max_decode_speed=steady_decode_tps,
    )

    sustained_summary = {
        "benchmark_mode": "sustained_decode",
        "task_type": task_type,
        "concurrency": concurrency,
        "prompt_length": prompt_length,
        "target_duration_sec": duration_seconds,
        "actual_duration_sec": round(actual_duration_sec, 2),
        "pure_decode_time_s": round(sum_stream_durations, 2),
        "total_tokens": total_tokens_generated,
        "completed_streams": len(completed_requests),
        "avg_decode_speed": round(steady_decode_tps, 2),
        "steady_decode_tps": round(steady_decode_tps, 2),
        "pure_decode_tps": round(steady_decode_tps, 2),
        "e2e_throughput_tps": round(e2e_throughput_tps, 2),
        "avg_stream_decode_tps": round(avg_stream_decode_speed, 2),
        "max_decode_speed": round(steady_decode_tps, 2),
        "avg_prefill_speed": avg_prefill,
        "max_prefill_speed": avg_prefill,
        "avg_ttft_ms": ttft_stats["avg"],
        "ttft_stats": ttft_stats,
        "itl_stats": itl_stats,
        "effective_concurrency": eff_conc_label,
        "inferred_parallelism": round(inferred_parallelism, 2),
        "pure_decode_time_sec": round(sum_stream_durations, 2),
        "avg_stream_decode_speed": round(avg_stream_decode_speed, 2),
        "e2e_speed": round(e2e_throughput_tps, 2),
        "is_saturated": is_saturated,
        "loop_detected": loop_detected,
        "completed_requests": completed_requests,
        "verification_hash": overall_vh,
    }

    points = [
        {
            "concurrency": concurrency,
            "prompt_length": prompt_length,
            "task_type": task_type,
            "decode_tps": round(steady_decode_tps, 2),
            "output_speed": round(steady_decode_tps, 2),
            "prefill_speed": avg_prefill,
            "avg_ttft_ms": ttft_stats["avg"],
            "ttft_p50_ms": ttft_stats["p50"],
            "itl_p50_ms": itl_stats["p50"],
            "effective_concurrency": eff_conc_label,
            "output_tokens": total_tokens_generated,
            "actual_duration_sec": round(actual_duration_sec, 2),
            "pure_decode_time_sec": round(sum_stream_durations, 2),
            "avg_stream_decode_speed": round(avg_stream_decode_speed, 2),
            "e2e_speed": round(e2e_throughput_tps, 2),
            "concurrent_details": completed_requests,
            "status": "成功" if not loop_detected else "ERROR: loop",
            "skipped": False,
        }
    ]

    quantiles = {
        "ttft": ttft_stats,
        "itl": itl_stats,
    }

    return {
        **sustained_summary,
        "model_name": model_name,
        "hardware": hardware,
        "framework": framework,
        "quantization": quantization,
        "sustained_summary": sustained_summary,
        "points": points,
        "quantiles": quantiles,
        "completed_requests": completed_requests,
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
        
        mode = getattr(config, "benchmark_mode", "1d_sweep") or "1d_sweep"
        task_type = getattr(config, "task_type", "general") or "general"
        print(f"[Config] 接收配置 - 模式: {mode}, 任务: {task_type}, API: {config.api_type}, 模型: {config.model_name}, 并发: {config.concurrency}")

        if mode == "2d_matrix":
            concurrency_ladder = config.matrix_concurrency_ladder or [1, 2, 4, 8]
            context_ladder = config.matrix_context_ladder or [512, 1024, 2048]
            max_kv_tokens = config.matrix_max_kv_tokens if config.matrix_max_kv_tokens is not None else 32768

            await websocket.send_json({
                "type": "info",
                "message": f"开始 2D 矩阵测试 ({task_type})，并发阶梯: {concurrency_ladder}, 上下文阶梯: {context_ladder}",
                "benchmark_mode": "2d_matrix",
                "task_type": task_type,
            })

            async def ws_matrix_progress(msg: Dict[str, Any]):
                await websocket.send_json(msg)

            async def ws_matrix_result(msg: Dict[str, Any]):
                await websocket.send_json(msg)

            matrix_res = await run_2d_matrix_benchmark(
                api_url=config.api_url,
                api_key=config.api_key,
                api_type=config.api_type,
                model_name=config.model_name,
                concurrency_ladder=concurrency_ladder,
                context_ladder=context_ladder,
                max_tokens=config.output_length,
                max_kv_tokens=max_kv_tokens,
                timeout=config.timeout,
                temperature=config.temperature,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                metrics_url=config.metrics_url,
                enable_watchdog=config.enable_watchdog,
                progress_callback=ws_matrix_progress,
                result_callback=ws_matrix_result,
                task_type=task_type,
            )

            await websocket.send_json({
                "type": "complete",
                "message": "2D 矩阵测试完成",
                "benchmark_mode": "2d_matrix",
                "task_type": task_type,
                "all_results": matrix_res.get("all_cell_results", []),
                "matrix_grid": matrix_res.get("matrix_grid", {}),
                "matrix_summary": matrix_res.get("matrix_summary", {}),
                "points": matrix_res.get("points", []),
            })
            await asyncio.sleep(0.05)
            try:
                await websocket.close(code=1000)
            except Exception:
                pass
            return

        elif mode == "sustained_decode":
            concurrency = config.concurrency or 1
            duration = config.sustained_duration if config.sustained_duration is not None else 30.0
            prompt_length = config.sustained_prompt_length or 128

            await websocket.send_json({
                "type": "info",
                "message": f"开始稳态持续解码测试 ({task_type})，并发流: {concurrency}, 时长: {duration}s, 提示词: {prompt_length} tokens",
                "benchmark_mode": "sustained_decode",
                "task_type": task_type,
            })

            async def ws_sustained_progress(data: Dict[str, Any]):
                msg = dict(data)
                msg["type"] = "progress"
                msg["benchmark_mode"] = "sustained_decode"
                msg["task_type"] = task_type
                await websocket.send_json(msg)

            sustained_res = await run_sustained_decode_benchmark(
                api_url=config.api_url,
                api_key=config.api_key,
                api_type=config.api_type,
                model_name=config.model_name,
                concurrency=concurrency,
                prompt_length=prompt_length,
                output_length=config.output_length,
                duration_seconds=duration,
                timeout=config.timeout,
                temperature=config.temperature,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                metrics_url=config.metrics_url,
                enable_watchdog=config.enable_watchdog,
                progress_callback=ws_sustained_progress,
                task_type=task_type,
            )

            result_payload = dict(sustained_res)
            result_payload["type"] = "result"
            result_payload["benchmark_mode"] = "sustained_decode"
            await websocket.send_json(result_payload)

            await websocket.send_json({
                "type": "complete",
                "message": "稳态持续解码测试完成",
                "benchmark_mode": "sustained_decode",
                "all_results": [sustained_res],
                "sustained_summary": sustained_res,
                "points": sustained_res.get("points", []),
            })
            await asyncio.sleep(0.05)
            try:
                await websocket.close(code=1000)
            except Exception:
                pass
            return

        # 1D Sweep Mode (Default)
        min_len = config.min_length if config.min_length is not None else 100
        max_len = config.max_length if config.max_length is not None else 2000
        step_val = config.step if config.step is not None else 500
        # 使用提供的测试点列表，或计算测试点
        if config.test_lengths:
            test_lengths = config.test_lengths
            print(f"[Config] 使用自定义测试点列表: {test_lengths}")
        else:
            test_lengths = list(range(min_len, max_len + 1, step_val))
            print(f"[Config] 提示词长度: {min_len}-{max_len} (步长{step_val})")
            print(f"[TestLengths] 计算的测试点列表: {test_lengths}")
        
        await websocket.send_json({
            "type": "info",
            "message": f"开始测试，并发数: {config.concurrency}",
            "benchmark_mode": "1d_sweep",
            "task_type": task_type,
        })
        total_tests = len(test_lengths)
        completed = 0
        
        all_results = []
        latency_samples: list[float] = []
        prompt_calibration = None

        if total_tests > 0:
            prompt_calibration = await send_warmup_request(
                api_url=config.api_url,
                api_key=config.api_key,
                api_type=config.api_type,
                model_name=config.model_name,
                timeout=config.timeout,
                temperature=config.temperature,
                top_p=config.top_p,
            )
        
            await collect_network_latency_sample(
                latency_samples,
                config.api_url,
                config.api_key,
                config.api_type,
                config.model_name,
                config.timeout,
                config.temperature,
                config.top_p,
            )
            await collect_network_latency_sample(
                latency_samples,
                config.api_url,
                config.api_key,
                config.api_type,
                config.model_name,
                config.timeout,
                config.temperature,
                config.top_p,
            )

        for length in test_lengths:
            print(f"\n[Test] ===== 测试提示词长度: {length} ({completed+1}/{total_tests}) =====")
            
            network_latency_ms = await collect_network_latency_sample(
                latency_samples,
                config.api_url,
                config.api_key,
                config.api_type,
                config.model_name,
                config.timeout,
                config.temperature,
                config.top_p,
            )

            await websocket.send_json({
                "type": "progress",
                "current": completed,
                "total": total_tests,
                "testing_length": length,
                "task_type": task_type,
                "benchmark_mode": "1d_sweep",
                "network_latency_ms": round(network_latency_ms, 2),
                "network_latency_sample_count": len(latency_samples),
            })

            # 创建并发任务（每个任务使用不同的seed避免cache）
            # 根据prompt长度动态计算超时时间
            estimated_prompt_tokens_for_timeout = estimate_generated_prompt_tokens(length, calibration=prompt_calibration)
            print(
                f"[PromptEstimate] Requested={length}, estimated_actual_prompt_tokens="
                f"{estimated_prompt_tokens_for_timeout}",
                flush=True,
            )
            dynamic_timeout = calculate_dynamic_timeout(estimated_prompt_tokens_for_timeout, config.timeout)

            # 启动 Prometheus Telemetry 采样
            metrics_url = getattr(config, "metrics_url", None)
            scraper = PrometheusTelemetryScraper(config.api_url, metrics_url)
            scraper.start_sampling(interval=0.5)

            # 使用共享连接池运行高并发请求（突破连接限制，支持50+并发）
            limits = httpx.Limits(
                max_connections=max(250, config.concurrency * 2),
                max_keepalive_connections=max(100, config.concurrency),
            )
            timeout_obj = httpx.Timeout(dynamic_timeout / 1000.0, connect=15.0)

            async with httpx.AsyncClient(
                limits=limits,
                timeout=timeout_obj,
                verify=False,
                http2=True,
            ) as shared_client:
                tasks = []
                for i in range(config.concurrency):
                    task = execute_single_request(
                        api_url=config.api_url,
                        api_key=config.api_key,
                        api_type=config.api_type,
                        model_name=config.model_name,
                        prompt_length=length,
                        output_length=config.output_length,
                        timeout=dynamic_timeout,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        presence_penalty=config.presence_penalty,
                        frequency_penalty=config.frequency_penalty,
                        network_latency_ms=network_latency_ms,
                        network_latency_sample_count=len(latency_samples),
                        prompt_calibration=prompt_calibration,
                        seed=i + 1,
                        client=shared_client,
                        enable_watchdog=getattr(config, "enable_watchdog", True),
                        task_type=task_type,
                    )
                    tasks.append(task)
                
                # 并发执行
                print(f"[Test] 启动 {config.concurrency} 个并发请求 (连接池模式)...")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                print(f"[Test] 并发请求完成")

            telemetry_summary = await scraper.stop_sampling()
            
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
                max_prefill_speed = max(r["prefill_speed"] for r in successful_results)
                max_output_speed = max(r["output_speed"] for r in successful_results)
                avg_ttft = sum(r["ttft_ms"] for r in successful_results) / len(successful_results)
                avg_prompt_tokens = sum(r["prompt_tokens"] for r in successful_results) / len(successful_results)
                aggregate_metrics = calculate_aggregate_throughput_metrics(successful_results)
                boundary_fallback_count = sum(1 for r in successful_results if r.get("boundary_source") == "first_chunk_fallback")
                aggregate_warnings = aggregate_metrics.get("warnings") or []
                token_warning_count = (
                    sum(len(r.get("token_warnings") or []) for r in successful_results)
                    + len(aggregate_warnings)
                )

                # 计算分位数
                all_ttft = [r["ttft_ms"] for r in successful_results if r.get("ttft_ms")]
                ttft_p50 = compute_percentile(all_ttft, 50)
                ttft_p90 = compute_percentile(all_ttft, 90)
                ttft_p99 = compute_percentile(all_ttft, 99)
                ttft_std = (
                    (sum((x - avg_ttft) ** 2 for x in all_ttft) / len(all_ttft)) ** 0.5
                    if len(all_ttft) > 1 else 0.0
                )

                all_itl = [
                    t["itl_ms"]
                    for r in successful_results
                    for t in r.get("token_timestamps", [])
                    if t.get("itl_ms") is not None
                ]
                itl_avg = sum(all_itl) / len(all_itl) if all_itl else 0.0
                itl_p50 = compute_percentile(all_itl, 50)
                itl_p90 = compute_percentile(all_itl, 90)
                itl_p99 = compute_percentile(all_itl, 99)
                itl_std = (
                    (sum((x - itl_avg) ** 2 for x in all_itl) / len(all_itl)) ** 0.5
                    if len(all_itl) > 1 else 0.0
                )

                ttft_stats = {
                    "avg": round(avg_ttft, 2),
                    "p50": round(ttft_p50, 2),
                    "p90": round(ttft_p90, 2),
                    "p99": round(ttft_p99, 2),
                    "std": round(ttft_std, 2),
                }
                itl_stats = {
                    "avg": round(itl_avg, 2),
                    "p50": round(itl_p50, 2),
                    "p90": round(itl_p90, 2),
                    "p99": round(itl_p99, 2),
                    "std": round(itl_std, 2),
                }

                client_inferred_parallelism = aggregate_metrics.get("client_inferred_parallelism")
                eff_conc = calculate_effective_concurrency(
                    concurrency=config.concurrency,
                    telemetry_summary=telemetry_summary,
                    client_inferred_parallelism=client_inferred_parallelism,
                )

                cell_vh = compute_verification_hash(
                    model_name=config.model_name,
                    hardware="Server/Local",
                    framework=telemetry_summary.get("engine") or "api",
                    concurrency=config.concurrency,
                    max_decode_speed=max_output_speed,
                )
                loop_detected = any(r.get("loop_detected") for r in successful_results)

                record_tags = ["source:python_backend", "verified"]
                if boundary_fallback_count > 0:
                    record_tags.append("boundary_source:first_chunk_fallback")
                if token_warning_count > 0:
                    record_tags.append("token_source:api_stats_anomaly")
                if aggregate_metrics["prefill_source"] == "server":
                    record_tags.append("prefill_source:server")
                elif aggregate_metrics["prefill_source"] == "client_latency_adjusted":
                    record_tags.append("prefill_source:client_latency_adjusted")
                elif aggregate_metrics["prefill_source"] == "client_llamacpp_timing_anomaly":
                    record_tags.append("prefill_source:client_llamacpp_timing_anomaly")
                if aggregate_metrics["output_source"] == "server":
                    record_tags.append("decode_source:server")
                elif aggregate_metrics["output_source"] == "client_llamacpp_timing_anomaly":
                    record_tags.append("decode_source:client_llamacpp_timing_anomaly")
                if eff_conc.get("is_capacity_limited"):
                    record_tags.append(f"effective_concurrency:{eff_conc['label']}")
                if loop_detected:
                    record_tags.append("status:loop_detected")
                if telemetry_summary.get("available") and telemetry_summary.get("engine"):
                    record_tags.append(f"engine:{telemetry_summary['engine']}")
                record_tags = list(dict.fromkeys(record_tags))
                
                print(f"[Stats] 成功: {len(successful_results)}/{config.concurrency}")
                print(f"[Stats] 平均 Prefill速度: {avg_prefill_speed:.2f} t/s, Decode速度: {avg_output_speed:.2f} t/s")
                print(f"[Stats] 有效并发率: {eff_conc['label']}, TTFT P50: {ttft_p50:.2f}ms, ITL P50: {itl_p50:.2f}ms")
                if telemetry_summary.get("available"):
                    print(f"[Stats] Telemetry - KV Cache: {telemetry_summary['mean_kv_cache_pct']:.1f}%, Queue: {telemetry_summary['mean_queue']:.1f}")

                result_summary = {
                    "type": "result",
                    "prompt_length": get_resolved_prompt_length(length),
                    "requested_prompt_length": length,
                    "task_type": task_type,
                    "benchmark_mode": "1d_sweep",
                    "network_latency_ms": round(network_latency_ms, 2),
                    "network_latency_sample_count": len(latency_samples),
                    "avg_prompt_tokens": round(avg_prompt_tokens, 2),
                    "concurrency": config.concurrency,
                    "effective_concurrency": eff_conc["label"],
                    "effective_concurrency_detail": eff_conc,
                    "telemetry": telemetry_summary,
                    "loop_detected": loop_detected,
                    "verification_hash": cell_vh,
                    "ttft_stats": ttft_stats,
                    "itl_stats": itl_stats,
                    "successful": len(successful_results),
                    "failed": failed_count,
                    "avg_prefill_speed": round(avg_prefill_speed, 2),
                    "avg_output_speed": round(avg_output_speed, 2),
                    "max_prefill_speed": round(max_prefill_speed, 2),
                    "max_output_speed": round(max_output_speed, 2),
                    "aggregate_prefill_time_ms": round(aggregate_metrics["prefill_time_ms"], 2),
                    "aggregate_output_time_ms": round(aggregate_metrics["output_time_ms"], 2),
                    "aggregate_prefill_speed": round(aggregate_metrics["prefill_speed"], 2),
                    "aggregate_output_speed": round(aggregate_metrics["output_speed"], 2),
                    "client_prefill_time_ms": round(aggregate_metrics.get("client_prefill_time_ms") or 0, 2),
                    "client_output_time_ms": round(aggregate_metrics.get("client_output_time_ms") or 0, 2),
                    "client_total_time_ms": round(aggregate_metrics.get("client_total_time_ms") or 0, 2),
                    "client_has_phase_boundary": bool(aggregate_metrics.get("client_has_phase_boundary")),
                    "aggregate_prefill_source": aggregate_metrics["prefill_source"],
                    "aggregate_output_source": aggregate_metrics["output_source"],
                    "avg_ttft_ms": round(avg_ttft, 2),
                    "source": "python_backend",
                    "record_tags": record_tags,
                    "boundary_fallback_count": boundary_fallback_count,
                    "token_warning_count": token_warning_count,
                    "aggregate_warnings": aggregate_warnings,
                    "concurrent_details": successful_results,
                    "status": "成功" if not loop_detected else "ERROR: loop"
                }
                
                all_results.append(result_summary)
                await websocket.send_json(result_summary)
            else:
                print(f"[Error] 所有请求失败 - 失败数: {failed_count}")
                error_result = {
                    "type": "result",
                    "prompt_length": get_resolved_prompt_length(length),
                    "requested_prompt_length": length,
                    "task_type": task_type,
                    "benchmark_mode": "1d_sweep",
                    "network_latency_ms": round(network_latency_ms, 2),
                    "network_latency_sample_count": len(latency_samples),
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
            "benchmark_mode": "1d_sweep",
            "task_type": task_type,
            "all_results": all_results
        })
        await asyncio.sleep(0.05)
        try:
            await websocket.close(code=1000)
        except Exception:
            pass
    
    except WebSocketDisconnect:
        print("[WebSocket] 客户端断开连接")
    except Exception as e:
        print(f"[Error] WebSocket异常: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"测试错误: {str(e)}"
            })
        except Exception:
            pass


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


def detect_hardware() -> str:
    """Auto-detect hardware platform (GPU or CPU) for telemetry and benchmark reports."""
    # 1. Try nvidia-smi
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
            if lines:
                return lines[0]
    except Exception:
        pass

    # 2. Try rocm-smi
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for line in proc.stdout.strip().splitlines():
                if "Card series:" in line or "Product Name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass

    # 3. macOS Darwin
    if platform.system() == "Darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return f"Apple Silicon ({proc.stdout.strip()})"
        except Exception:
            pass
        return f"Apple {platform.machine()}"

    # 4. Linux /proc/cpuinfo or processor
    try:
        cpu_name = platform.processor()
        if not cpu_name or cpu_name in ("x86_64", "aarch64"):
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":", 1)[1].strip()
                            break
        if cpu_name:
            return f"{cpu_name} ({platform.machine()})"
    except Exception:
        pass

    return f"{platform.system()} {platform.machine()}"


def build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser supporting both Server and Benchmark CLI modes."""
    parser = argparse.ArgumentParser(
        description="LLM Speed Test — High-Concurrency AsyncIO Benchmark Suite & Telemetry Backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    # Server mode flags
    parser.add_argument("--server", "--serve", action="store_true", help="Launch FastAPI / WebSocket server mode")
    parser.add_argument("--port", type=int, default=None, help="Port to bind server (default: 18000 or next free)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind server")

    # Benchmark CLI flags
    parser.add_argument("--benchmark", "--cli", action="store_true", help="Run in CLI benchmark mode")
    parser.add_argument(
        "--benchmark-mode", "--mode",
        type=str,
        choices=["1d_sweep", "2d_matrix", "sustained_decode"],
        default="1d_sweep",
        help="Benchmark execution mode (1d_sweep, 2d_matrix, sustained_decode)"
    )
    parser.add_argument(
        "--task-type", "--task",
        type=str,
        choices=TASK_TYPE_CHOICES,
        default="general",
        help="Benchmark task workload type (general, code, structured, prose, prefill_16k, reasoning, all)"
    )
    parser.add_argument(
        "--matrix-concurrency",
        type=str,
        default="1,2,4,8",
        help="Comma-separated concurrency levels for 2D matrix sweep (e.g. '1,2,4,8')"
    )
    parser.add_argument(
        "--matrix-context",
        type=str,
        default="512,1024,2048",
        help="Comma-separated context token lengths for 2D matrix sweep (e.g. '512,1024,2048')"
    )
    parser.add_argument(
        "--matrix-max-kv-tokens", "--max-kv-tokens",
        type=int,
        default=32768,
        help="Maximum token budget for KV cache capacity skipping (0 for unlimited)"
    )
    parser.add_argument(
        "--sustained-duration", "--duration",
        type=float,
        default=30.0,
        help="Target measurement duration in seconds for sustained decode benchmark"
    )
    parser.add_argument(
        "--sustained-prompt-length", "--sustained-context",
        type=int,
        default=128,
        help="Prompt token length for sustained decode requests"
    )
    parser.add_argument("--model", type=str, default=None, help="Target LLM model name")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1", help="Target API URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API authentication key")
    parser.add_argument("--api-type", type=str, choices=["openai", "ollama"], default="openai", help="API protocol type")
    parser.add_argument("--concurrency", "-c", type=str, default="1", help="Concurrency level or comma-separated levels (e.g. '1,4,16,64' or '50')")
    parser.add_argument("--contexts", "--prompt-lengths", "-l", type=str, default="100,500,2000", help="Comma-separated prompt token lengths")
    parser.add_argument("--max-tokens", "-m", type=int, default=128, help="Max output tokens to generate per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top_p")
    parser.add_argument("--metrics-url", type=str, default=None, help="Prometheus /metrics endpoint (e.g. http://localhost:8000/metrics)")
    parser.add_argument("--hardware", type=str, default=None, help="Hardware description (auto-detected if omitted)")
    parser.add_argument("--framework", type=str, default="vLLM", help="Serving framework (vLLM, SGLang, TGI, Ollama, etc.)")
    parser.add_argument("--quantization", type=str, default="FP16", help="Model quantization (FP16, BF16, AWQ, FP8, etc.)")
    parser.add_argument("--upload", action="store_true", help="Upload benchmark results directly to Supabase cloud leaderboard")
    parser.add_argument("--user-code", type=str, default=None, help="User code (8-16 alphanumeric chars) for cloud upload")
    parser.add_argument("--nickname", type=str, default=None, help="Nickname for cloud upload")
    parser.add_argument("--notes", type=str, default=None, help="Notes or run configuration comments")
    parser.add_argument("--run-command", type=str, default=None, help="Command used to launch server (e.g. vllm serve ...)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to write JSON benchmark report")
    parser.add_argument("--no-watchdog", action="store_true", help="Disable repetition loop watchdog")
    parser.add_argument("--timeout", type=int, default=120000, help="Per-request timeout in milliseconds")

    return parser


async def run_cli_benchmark(args: argparse.Namespace) -> int:
    """Run full benchmark directly from CLI, print Rich-style tabular output, save JSON and upload."""
    model_name = (args.model or "").strip()
    if not model_name:
        print("❌ Error: --model is required for CLI benchmark mode.", file=sys.stderr)
        return 1

    api_url = args.api_url.strip()
    if args.api_type == "openai" and not api_url.endswith("/chat/completions"):
        api_url = f"{api_url.rstrip('/')}/chat/completions"

    api_key = args.api_key
    api_type = args.api_type
    try:
        concurrency_list = [int(c.strip()) for c in str(args.concurrency).split(",") if c.strip()]
    except ValueError:
        print(f"❌ Error: Invalid --concurrency values '{args.concurrency}'", file=sys.stderr)
        return 1

    try:
        context_list = [int(ctx.strip()) for ctx in str(args.contexts).split(",") if ctx.strip()]
    except ValueError:
        print(f"❌ Error: Invalid --contexts values '{args.contexts}'", file=sys.stderr)
        return 1

    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    metrics_url = args.metrics_url
    hardware = args.hardware.strip() if args.hardware else detect_hardware()
    framework = args.framework.strip() if args.framework else "vLLM"
    quantization = args.quantization.strip() if args.quantization else "FP16"
    timeout = args.timeout
    enable_watchdog = not args.no_watchdog

    # 1. Probe for metrics endpoint if not provided
    if not metrics_url:
        try:
            parsed = urlparse(api_url)
            candidate_metrics = f"{parsed.scheme}://{parsed.netloc}/metrics"
            async with httpx.AsyncClient(timeout=2.0) as probe_client:
                resp = await probe_client.get(candidate_metrics)
                if resp.status_code == 200 and ("vllm:" in resp.text or "sglang:" in resp.text or "prompt_tokens" in resp.text):
                    metrics_url = candidate_metrics
                    print(f"[Telemetry] Auto-detected Prometheus metrics endpoint: {metrics_url}")
        except Exception:
            pass

    benchmark_mode = getattr(args, "benchmark_mode", "1d_sweep") or "1d_sweep"
    task_type = getattr(args, "task_type", "general") or "general"

    if benchmark_mode == "2d_matrix":
        matrix_conc_raw = getattr(args, "matrix_concurrency", None) or args.concurrency
        try:
            concurrency_ladder = [int(c.strip()) for c in str(matrix_conc_raw).split(",") if c.strip()]
        except ValueError:
            print(f"❌ Error: Invalid --matrix-concurrency values '{matrix_conc_raw}'", file=sys.stderr)
            return 1

        matrix_ctx_raw = getattr(args, "matrix_context", None) or args.contexts
        try:
            context_ladder = [int(ctx.strip()) for ctx in str(matrix_ctx_raw).split(",") if ctx.strip()]
        except ValueError:
            print(f"❌ Error: Invalid --matrix-context values '{matrix_ctx_raw}'", file=sys.stderr)
            return 1

        max_kv_tokens = getattr(args, "matrix_max_kv_tokens", 32768)

        print("\n" + "=" * 84)
        print("🚀 LLM 2D Matrix Inference Benchmark (Concurrency × Context Engine)")
        print("=" * 84)
        print(f"Target API     : {api_url}")
        print(f"Model          : {model_name}")
        print(f"Hardware       : {hardware}")
        print(f"Framework      : {framework} ({quantization})")
        print(f"Task Workload  : {task_type}")
        print(f"Concurrencies  : {concurrency_ladder}")
        print(f"Context Lengths: {context_ladder}")
        print(f"Max KV Tokens  : {max_kv_tokens}")
        print(f"Output Tokens  : {max_tokens}")
        print(f"Watchdogs      : {'Loop: ON' if enable_watchdog else 'Loop: OFF'}")
        print("=" * 84 + "\n")

        matrix_res = await run_2d_matrix_benchmark(
            api_url=api_url,
            api_key=api_key,
            api_type=api_type,
            model_name=model_name,
            concurrency_ladder=concurrency_ladder,
            context_ladder=context_ladder,
            max_tokens=max_tokens,
            max_kv_tokens=max_kv_tokens,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            metrics_url=metrics_url,
            hardware=hardware,
            framework=framework,
            quantization=quantization,
            enable_watchdog=enable_watchdog,
            task_type=task_type,
        )

        print("\n" + "=" * 90)
        print("📊 2D Concurrency × Context Matrix Results")
        print("=" * 90)
        print(f"{'Concurrency':>12} | {'Context':>8} | {'Tokens':>8} | {'TTFT P50':>10} | {'Decode TPS':>11} | {'Eff. Conc':>10} | {'Status'}")
        print("-" * 90)
        for cell in matrix_res["all_cell_results"]:
            conc = cell["concurrency"]
            ctx = cell["context"]
            needed = cell.get("tokens_needed", conc * (ctx + max_tokens))
            ttft = f"{cell.get('ttft_stats', {}).get('p50', cell.get('ttft_ms', 0.0)):.1f}ms" if not cell.get("skipped") else "N/A"
            tps = f"{cell.get('decode_tps', 0.0):.1f}" if not cell.get("skipped") else "SKIP"
            eff = cell.get("effective_concurrency", f"{conc}/{conc}")
            st = cell.get("status", "")
            print(f"{conc:>12} | {ctx:>8} | {needed:>8} | {ttft:>10} | {tps:>11} | {eff:>10} | {st}")
        print("=" * 90)
        print(f"Peak Concurrency    : {matrix_res['peak_concurrency']}")
        print(f"Peak Decode TPS     : {matrix_res['matrix_summary'].get('peak_decode_tps', 0.0)} tokens/s")
        print(f"Peak Point          : {matrix_res['matrix_summary'].get('peak_point', {})}")
        print(f"Saturation Onset    : {matrix_res['matrix_summary'].get('saturation_onset', 'None')}")
        print(f"Verification Hash   : {matrix_res['verification_hash']}")
        print("=" * 90)

        report_data = {
            "benchmark_mode": "2d_matrix",
            "task_type": task_type,
            "model_name": model_name,
            "hardware": hardware,
            "framework": framework,
            "quantization": quantization,
            "notes": args.notes,
            "run_command": args.run_command,
            "concurrency": matrix_res["peak_concurrency"],
            "avg_prefill_speed": matrix_res["overall_avg_prefill"],
            "avg_decode_speed": matrix_res["overall_avg_decode"],
            "max_prefill_speed": matrix_res["overall_max_prefill"],
            "max_decode_speed": matrix_res["overall_max_decode"],
            "verification_hash": matrix_res["verification_hash"],
            "source": "python_v3",
            "timestamp": int(time.time()),
            "matrix_concurrency_ladder": concurrency_ladder,
            "matrix_context_ladder": context_ladder,
            "matrix_max_kv_tokens": max_kv_tokens,
            "matrix_summary": matrix_res["matrix_summary"],
            "matrix": matrix_res["matrix_grid"],
            "points": matrix_res["points"],
            "quantiles": matrix_res["quantiles"],
            "results": matrix_res["all_cell_results"],
        }

        output_path = args.output
        if not output_path:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)[:30]
            output_path = f"benchmark_matrix_{safe_model}_{ts_str}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n📁 Local 2D matrix benchmark report saved to: {os.path.abspath(output_path)}")

        if args.upload:
            print("\n☁️ Uploading benchmark to Supabase Leaderboard...")
            user_code = args.user_code
            if not user_code or len(user_code.strip()) < 8:
                user_code = "CLI" + hashlib.sha256(f"{hardware}{model_name}{time.time()}".encode()).hexdigest()[:5].upper()
                print(f"ℹ️ Auto-generated User Code: {user_code}")
            else:
                user_code = user_code.strip().upper()

            upload_res = await upload_benchmark_to_supabase(
                user_code=user_code,
                model_name=model_name,
                hardware=hardware,
                concurrency=matrix_res["peak_concurrency"],
                avg_prefill_speed=matrix_res["overall_avg_prefill"],
                avg_decode_speed=matrix_res["overall_avg_decode"],
                max_prefill_speed=matrix_res["overall_max_prefill"],
                max_decode_speed=matrix_res["overall_max_decode"],
                framework=framework,
                quantization=quantization,
                notes=args.notes,
                nickname=args.nickname,
                run_command=args.run_command,
                record_tags=["source:python_v3", "verified", "mode:2d_matrix", f"task:{task_type}"],
                results_json={
                    "benchmark_mode": "2d_matrix",
                    "task_type": task_type,
                    "matrix": matrix_res["matrix_grid"],
                    "matrix_summary": matrix_res["matrix_summary"],
                    "points": matrix_res["points"],
                    "quantiles": matrix_res["quantiles"],
                    "watchdog": {
                        "loop_detected": any(c.get("loop_detected") for c in matrix_res["all_cell_results"]),
                    },
                    "verification": {
                        "hash": matrix_res["verification_hash"],
                        "status": "verified",
                        "timestamp": int(time.time()),
                    }
                },
                verification_hash=matrix_res["verification_hash"],
            )
            if upload_res.get("success"):
                share_id = upload_res.get("id")
                print("✅ Cloud upload successful!")
                print(f"  Record ID          : {share_id}")
                print(f"  Verification Hash  : {upload_res.get('verification_hash')}")
                print(f"  Leaderboard Share  : {PUBLIC_LEADERBOARD_URL}?id={share_id}")
            else:
                print(f"⚠️ Cloud upload failed: {upload_res.get('error')}")

        return 0

    if task_type == "all":
        sustained_concurrency = concurrency_list[0] if concurrency_list else 1
        sustained_duration = getattr(args, "sustained_duration", 30.0) or 30.0
        sustained_prompt_len = getattr(args, "sustained_prompt_length", 128) or 128

        print("\n" + "=" * 90)
        print("⚡ LLM All Tasks Sequence Benchmark (全场景自动化顺序测试)")
        print("=" * 90)
        print(f"Target API     : {api_url}")
        print(f"Model          : {model_name}")
        print(f"Hardware       : {hardware}")
        print(f"Framework      : {framework} ({quantization})")
        print(f"Workloads      : 6 Tasks ({', '.join(TASK_TYPES)})")
        print(f"Concurrency    : {sustained_concurrency}")
        print(f"Duration/Task  : {sustained_duration}s")
        print(f"Prompt Length  : {sustained_prompt_len} tokens")
        print(f"Output Tokens  : {max_tokens}")
        print(f"Watchdogs      : {'Loop: ON' if enable_watchdog else 'Loop: OFF'}")
        print("=" * 90 + "\n")

        all_task_results = []
        for idx, t_type in enumerate(TASK_TYPES, start=1):
            print(f"\n--- [Task {idx}/6: {t_type.upper()}] Starting sustained benchmark ({sustained_duration}s)... ---")
            t_res = await run_sustained_decode_benchmark(
                api_url=api_url,
                api_key=api_key,
                api_type=api_type,
                model_name=model_name,
                concurrency=sustained_concurrency,
                prompt_length=sustained_prompt_len,
                output_length=max_tokens,
                duration_seconds=sustained_duration,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                metrics_url=metrics_url,
                hardware=hardware,
                framework=framework,
                quantization=quantization,
                enable_watchdog=enable_watchdog,
                task_type=t_type,
            )
            all_task_results.append(t_res)
            print(f"  ✔️ [{t_type}] Completed: {t_res.get('steady_decode_tps', 0):.2f} t/s, TTFT P50: {t_res.get('ttft_stats', {}).get('p50', 0):.1f}ms")

        baseline = next((r for r in all_task_results if r.get("task_type") == "general"), all_task_results[0])
        baseline_tps = baseline.get("steady_decode_tps", 0.0)

        print("\n" + "=" * 105)
        print(f"🏆 All Workloads Performance Comparison (全场景工作负载性能对比: {model_name})")
        print("=" * 105)
        print(f"{'Task Workload':<15} | {'Pure TPS':>10} | {'vs Base':>9} | {'E2E TPS':>9} | {'Stream TPS':>10} | {'TTFT P50':>9} | {'ITL P50':>9} | {'Eff Conc':>8} | {'Tokens':>7}")
        print("-" * 15 + "-+-" + "-" * 10 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 10 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 8 + "-+-" + "-" * 7)

        for r in all_task_results:
            t = r.get("task_type", "general")
            tps = r.get("steady_decode_tps", 0.0)
            if baseline_tps > 0 and t != "general":
                delta = ((tps - baseline_tps) / baseline_tps) * 100
                delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            else:
                delta_str = "0.0%"
            e2e = r.get("e2e_throughput_tps", 0.0)
            stream_tps = r.get("avg_stream_decode_tps", 0.0)
            ttft_p50 = f"{r.get('ttft_stats', {}).get('p50', 0):.1f}ms"
            itl_p50 = f"{r.get('itl_stats', {}).get('p50', 0):.1f}ms"
            eff = r.get("effective_concurrency", f"{sustained_concurrency}/{sustained_concurrency}")
            toks = r.get("total_tokens", 0)
            print(f"{t:<15} | {tps:>10.2f} | {delta_str:>9} | {e2e:>9.2f} | {stream_tps:>10.2f} | {ttft_p50:>9} | {itl_p50:>9} | {eff:>8} | {toks:>7}")
        print("=" * 105)

        max_tps_item = max(all_task_results, key=lambda x: x.get("steady_decode_tps", 0))
        min_tps_item = min(all_task_results, key=lambda x: x.get("steady_decode_tps", 0))
        max_ttft_item = max(all_task_results, key=lambda x: x.get("ttft_stats", {}).get("p50", 0))
        span_ratio = f"{max_tps_item.get('steady_decode_tps', 0) / min_tps_item.get('steady_decode_tps', 1):.2f}x" if min_tps_item.get("steady_decode_tps", 0) > 0 else "N/A"

        print(f"🚀 Peak Pure Decode Task : {max_tps_item.get('task_type')} ({max_tps_item.get('steady_decode_tps', 0):.2f} t/s)")
        print(f"🐢 Heavy Load / Min TPS   : {min_tps_item.get('task_type')} ({min_tps_item.get('steady_decode_tps', 0):.2f} t/s)")
        print(f"⏱️ Peak TTFT Latency Task : {max_ttft_item.get('task_type')} ({max_ttft_item.get('ttft_stats', {}).get('p50', 0):.1f} ms)")
        print(f"📊 Workload Span Ratio   : {span_ratio}")
        print("=" * 105)

        avg_decode_all = round(sum(r.get("steady_decode_tps", 0) for r in all_task_results) / len(all_task_results), 2)
        max_decode_all = round(max(r.get("steady_decode_tps", 0) for r in all_task_results), 2)
        avg_prefill_all = round(sum(r.get("avg_prefill_speed", 0) for r in all_task_results) / len(all_task_results), 2)
        max_prefill_all = round(max(r.get("max_prefill_speed", 0) for r in all_task_results), 2)
        all_points = [pt for r in all_task_results for pt in r.get("points", [])]

        combined_vh = compute_verification_hash(
            model_name=model_name,
            hardware=hardware,
            framework=framework,
            quantization=quantization,
            concurrency=sustained_concurrency,
            max_decode_speed=max_decode_all,
        )

        report_data = {
            "benchmark_mode": "sustained_decode",
            "task_type": "all",
            "model_name": model_name,
            "hardware": hardware,
            "framework": framework,
            "quantization": quantization,
            "notes": args.notes,
            "run_command": args.run_command,
            "concurrency": sustained_concurrency,
            "avg_prefill_speed": avg_prefill_all,
            "avg_decode_speed": avg_decode_all,
            "max_prefill_speed": max_prefill_all,
            "max_decode_speed": max_decode_all,
            "verification_hash": combined_vh,
            "source": "python_v3",
            "timestamp": int(time.time()),
            "sustained_duration": sustained_duration,
            "sustained_prompt_length": sustained_prompt_len,
            "all_tasks_results": all_task_results,
            "points": all_points,
            "quantiles": {
                "ttft": max_ttft_item.get("ttft_stats", {}),
                "itl": max_tps_item.get("itl_stats", {}),
            },
            "results": all_points,
        }

        output_path = args.output
        if not output_path:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)[:30]
            output_path = f"benchmark_all_tasks_{safe_model}_{ts_str}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n📁 Local all tasks benchmark report saved to: {os.path.abspath(output_path)}")

        if args.upload:
            print("\n☁️ Uploading benchmark to Supabase Leaderboard...")
            user_code = args.user_code
            if not user_code or len(user_code.strip()) < 8:
                user_code = "CLI" + hashlib.sha256(f"{hardware}{model_name}{time.time()}".encode()).hexdigest()[:5].upper()
                print(f"ℹ️ Auto-generated User Code: {user_code}")
            else:
                user_code = user_code.strip().upper()

            upload_res = await upload_benchmark_to_supabase(
                user_code=user_code,
                model_name=model_name,
                hardware=hardware,
                concurrency=sustained_concurrency,
                avg_prefill_speed=avg_prefill_all,
                avg_decode_speed=avg_decode_all,
                max_prefill_speed=max_prefill_all,
                max_decode_speed=max_decode_all,
                framework=framework,
                quantization=quantization,
                notes=args.notes,
                nickname=args.nickname,
                run_command=args.run_command,
                record_tags=["source:python_v3", "verified", "mode:sustained_decode", "task:all"],
                results_json={
                    "benchmark_mode": "sustained_decode",
                    "task_type": "all",
                    "all_tasks_results": all_task_results,
                    "points": all_points,
                    "quantiles": report_data["quantiles"],
                    "watchdog": {
                        "loop_detected": any(r.get("loop_detected", False) for r in all_task_results),
                    },
                    "verification": {
                        "hash": combined_vh,
                        "status": "verified",
                        "timestamp": int(time.time()),
                    }
                },
                verification_hash=combined_vh,
            )
            if upload_res.get("success"):
                share_id = upload_res.get("id")
                print("✅ Cloud upload successful!")
                print(f"  Record ID          : {share_id}")
                print(f"  Verification Hash  : {upload_res.get('verification_hash')}")
                print(f"  Leaderboard Share  : {PUBLIC_LEADERBOARD_URL}?id={share_id}")
            else:
                print(f"⚠️ Cloud upload failed: {upload_res.get('error')}")

        return 0

    if benchmark_mode == "sustained_decode":
        sustained_concurrency = concurrency_list[0] if concurrency_list else 1
        sustained_duration = getattr(args, "sustained_duration", 30.0) or 30.0
        sustained_prompt_len = getattr(args, "sustained_prompt_length", 128) or 128

        print("\n" + "=" * 84)
        print("⏱️ LLM Sustained Steady-State Decode Benchmark")
        print("=" * 84)
        print(f"Target API     : {api_url}")
        print(f"Model          : {model_name}")
        print(f"Hardware       : {hardware}")
        print(f"Framework      : {framework} ({quantization})")
        print(f"Task Workload  : {task_type}")
        print(f"Concurrency    : {sustained_concurrency}")
        print(f"Target Duration: {sustained_duration}s")
        print(f"Prompt Length  : {sustained_prompt_len} tokens")
        print(f"Output Tokens  : {max_tokens}")
        print(f"Watchdogs      : {'Loop: ON' if enable_watchdog else 'Loop: OFF'}")
        print("=" * 84 + "\n")

        sustained_res = await run_sustained_decode_benchmark(
            api_url=api_url,
            api_key=api_key,
            api_type=api_type,
            model_name=model_name,
            concurrency=sustained_concurrency,
            prompt_length=sustained_prompt_len,
            output_length=max_tokens,
            duration_seconds=sustained_duration,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            metrics_url=metrics_url,
            hardware=hardware,
            framework=framework,
            quantization=quantization,
            enable_watchdog=enable_watchdog,
            task_type=task_type,
        )

        actual_dur = sustained_res.get('actual_duration_sec', sustained_duration)
        pure_dec_s = sustained_res.get('pure_decode_time_s')
        e2e_tps = sustained_res.get('e2e_throughput_tps')
        if e2e_tps is None and actual_dur and actual_dur > 0:
            e2e_tps = sustained_res.get('total_tokens', 0) / actual_dur
        avg_stream_tps = sustained_res.get('avg_stream_decode_tps')
        if avg_stream_tps is None and sustained_concurrency > 0:
            avg_stream_tps = sustained_res.get('steady_decode_tps', 0) / sustained_concurrency

        print("\n" + "=" * 84)
        print("📊 Sustained Steady-State Decode Summary")
        print("=" * 84)
        print(f"Target Duration      : {sustained_duration}s")
        if pure_dec_s is not None:
            print(f"Actual Duration      : {actual_dur:.1f}s (Pure Decode Worker Time: {pure_dec_s:.2f}s)")
        else:
            print(f"Actual Duration      : {actual_dur:.1f}s")
        print(f"Total Tokens         : {sustained_res['total_tokens']}")
        print(f"Completed Streams    : {sustained_res['completed_streams']}")
        print(f"Steady Decode TPS    : {sustained_res['steady_decode_tps']:.2f} tokens/s (Pure Decode, Prefill/TTFT Deducted)")
        print(f"End-to-End TPS       : {e2e_tps:.2f} tokens/s (Wall-Clock, Incl. Prefill/Queueing)")
        print(f"Single-Stream Decode : {avg_stream_tps:.2f} tokens/s (Avg Per-Stream Decode Rate)")
        print(f"Avg TTFT             : {sustained_res['avg_ttft_ms']:.2f}ms (P50: {sustained_res['ttft_stats']['p50']:.1f}ms, P90: {sustained_res['ttft_stats']['p90']:.1f}ms, P99: {sustained_res['ttft_stats']['p99']:.1f}ms)")
        print(f"ITL P50              : {sustained_res['itl_stats']['p50']:.2f}ms (P90: {sustained_res['itl_stats']['p90']:.1f}ms)")
        print(f"Effective Concurrency: {sustained_res['effective_concurrency']}")
        print(f"Verification Hash    : {sustained_res['verification_hash']}")
        print("=" * 84)

        report_data = {
            "benchmark_mode": "sustained_decode",
            "task_type": task_type,
            "model_name": model_name,
            "hardware": hardware,
            "framework": framework,
            "quantization": quantization,
            "notes": args.notes,
            "run_command": args.run_command,
            "concurrency": sustained_concurrency,
            "avg_prefill_speed": sustained_res["avg_prefill_speed"],
            "avg_decode_speed": sustained_res["steady_decode_tps"],
            "max_prefill_speed": sustained_res["max_prefill_speed"],
            "max_decode_speed": sustained_res["steady_decode_tps"],
            "verification_hash": sustained_res["verification_hash"],
            "source": "python_v3",
            "timestamp": int(time.time()),
            "sustained_duration": sustained_duration,
            "sustained_prompt_length": sustained_prompt_len,
            "sustained_summary": sustained_res["sustained_summary"],
            "points": sustained_res["points"],
            "quantiles": sustained_res["quantiles"],
            "results": sustained_res["points"],
        }

        output_path = args.output
        if not output_path:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)[:30]
            output_path = f"benchmark_sustained_{safe_model}_{ts_str}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n📁 Local sustained benchmark report saved to: {os.path.abspath(output_path)}")

        if args.upload:
            print("\n☁️ Uploading benchmark to Supabase Leaderboard...")
            user_code = args.user_code
            if not user_code or len(user_code.strip()) < 8:
                user_code = "CLI" + hashlib.sha256(f"{hardware}{model_name}{time.time()}".encode()).hexdigest()[:5].upper()
                print(f"ℹ️ Auto-generated User Code: {user_code}")
            else:
                user_code = user_code.strip().upper()

            upload_res = await upload_benchmark_to_supabase(
                user_code=user_code,
                model_name=model_name,
                hardware=hardware,
                concurrency=sustained_concurrency,
                avg_prefill_speed=sustained_res["avg_prefill_speed"],
                avg_decode_speed=sustained_res["steady_decode_tps"],
                max_prefill_speed=sustained_res["max_prefill_speed"],
                max_decode_speed=sustained_res["steady_decode_tps"],
                framework=framework,
                quantization=quantization,
                notes=args.notes,
                nickname=args.nickname,
                run_command=args.run_command,
                record_tags=["source:python_v3", "verified", "mode:sustained_decode", f"task:{task_type}"],
                results_json={
                    "benchmark_mode": "sustained_decode",
                    "sustained_summary": sustained_res["sustained_summary"],
                    "points": sustained_res["points"],
                    "completed_requests": sustained_res.get("completed_requests", []),
                    "quantiles": sustained_res["quantiles"],
                    "watchdog": {
                        "loop_detected": sustained_res.get("loop_detected", False),
                    },
                    "verification": {
                        "hash": sustained_res["verification_hash"],
                        "status": "verified",
                        "timestamp": int(time.time()),
                    }
                },
                verification_hash=sustained_res["verification_hash"],
            )
            if upload_res.get("success"):
                share_id = upload_res.get("id")
                print("✅ Cloud upload successful!")
                print(f"  Record ID          : {share_id}")
                print(f"  Verification Hash  : {upload_res.get('verification_hash')}")
                print(f"  Leaderboard Share  : {PUBLIC_LEADERBOARD_URL}?id={share_id}")
            else:
                print(f"⚠️ Cloud upload failed: {upload_res.get('error')}")

        return 0

    # 2. Warmup & network latency baseline (1D Sweep Mode)
    print("\n[Init] Probing target API baseline & warming up...")
    prompt_calibration = None
    try:
        prompt_calibration = await send_warmup_request(
            api_url=api_url,
            api_key=api_key,
            api_type=api_type,
            model_name=model_name,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
        )
    except Exception as e:
        print(f"[Init] Warmup request note: {e}")

    latency_samples: List[float] = []
    for _ in range(2):
        await collect_network_latency_sample(
            latency_samples,
            api_url=api_url,
            api_key=api_key,
            api_type=api_type,
            model_name=model_name,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
        )
    network_latency_ms = get_average_network_latency_ms(latency_samples)

    # 3. Display header
    print("\n" + "=" * 84)
    print("🚀 LLM High-Concurrency Inference Benchmark (AsyncIO HTTP/2 Engine)")
    print("=" * 84)
    print(f"Target API     : {api_url}")
    print(f"Model          : {model_name}")
    print(f"Hardware       : {hardware}")
    print(f"Framework      : {framework} ({quantization})")
    print(f"Concurrencies  : {concurrency_list}")
    print(f"Context Lengths: {context_list}")
    print(f"Output Tokens  : {max_tokens}")
    print(f"Telemetry URL  : {metrics_url or 'Disabled / Not found'}")
    print(f"Watchdogs      : {'Loop Watchdog: ON' if enable_watchdog else 'Loop: OFF'}")
    print("=" * 84 + "\n")

    all_cell_results: List[Dict[str, Any]] = []
    all_succ_results: List[Dict[str, Any]] = []

    for concurrency in concurrency_list:
        print(f"\n====================== Concurrency Level: {concurrency} ======================")
        client_limits = httpx.Limits(
            max_connections=max(256, concurrency * 2),
            max_keepalive_connections=max(128, concurrency),
        )
        async with httpx.AsyncClient(limits=client_limits, http2=True, timeout=timeout / 1000.0, verify=False) as http_client:
            for length in context_list:
                resolved_length = get_resolved_prompt_length(length)
                print(f"[Run] Testing Concurrency={concurrency}, Context={resolved_length} tokens...")

                network_latency_ms = await collect_network_latency_sample(
                    latency_samples,
                    api_url=api_url,
                    api_key=api_key,
                    api_type=api_type,
                    model_name=model_name,
                    timeout=timeout,
                    temperature=temperature,
                    top_p=top_p,
                )

                scraper = PrometheusTelemetryScraper(metrics_url)
                await scraper.start()

                seeds = [random.randint(1000, 999999) for _ in range(concurrency)]
                tasks = [
                    execute_single_request(
                        api_url=api_url,
                        api_key=api_key,
                        api_type=api_type,
                        model_name=model_name,
                        prompt_length=length,
                        output_length=max_tokens,
                        timeout=timeout,
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        seed=s,
                        network_latency_ms=network_latency_ms,
                        network_latency_sample_count=len(latency_samples),
                        prompt_calibration=prompt_calibration,
                        client=http_client,
                        enable_watchdog=enable_watchdog,
                        task_type=task_type,
                    )
                    for s in seeds
                ]

                req_results = await asyncio.gather(*tasks, return_exceptions=True)
                telemetry_summary = await scraper.stop()

                successful_results: List[Dict[str, Any]] = []
                failed_count = 0
                for r in req_results:
                    if isinstance(r, dict) and r.get("success"):
                        successful_results.append(r)
                    else:
                        failed_count += 1
                        if isinstance(r, dict) and r.get("error"):
                            print(f"  [Failure] {r.get('error')[:120]}")
                        elif isinstance(r, Exception):
                            print(f"  [Exception] {type(r).__name__}: {r}")

                if not successful_results:
                    print(f"  ❌ All {concurrency} requests failed for Context={resolved_length}!")
                    continue

                all_succ_results.extend(successful_results)

                avg_prefill = sum(r["prefill_speed"] for r in successful_results) / len(successful_results)
                avg_decode = sum(r["output_speed"] for r in successful_results) / len(successful_results)
                max_prefill = max(r["prefill_speed"] for r in successful_results)
                max_decode = max(r["output_speed"] for r in successful_results)
                avg_ttft = sum(r["ttft_ms"] for r in successful_results) / len(successful_results)

                aggregate_metrics = calculate_aggregate_throughput_metrics(successful_results)
                client_inferred_parallelism = aggregate_metrics.get("client_inferred_parallelism")
                eff_conc = calculate_effective_concurrency(
                    concurrency=concurrency,
                    telemetry_summary=telemetry_summary,
                    client_inferred_parallelism=client_inferred_parallelism,
                )

                all_ttft = [r["ttft_ms"] for r in successful_results if r.get("ttft_ms")]
                ttft_p50 = compute_percentile(all_ttft, 50)
                ttft_p90 = compute_percentile(all_ttft, 90)
                ttft_p99 = compute_percentile(all_ttft, 99)

                all_itl = [
                    t["itl_ms"]
                    for r in successful_results
                    for t in r.get("token_timestamps", [])
                    if t.get("itl_ms") is not None
                ]
                itl_p50 = compute_percentile(all_itl, 50)
                itl_p90 = compute_percentile(all_itl, 90)
                itl_p99 = compute_percentile(all_itl, 99)

                loop_detected = any(r.get("loop_detected") for r in successful_results)

                kv_cache_str = (
                    f"{telemetry_summary['kv_cache_usage_pct']:.1f}%"
                    if telemetry_summary.get("kv_cache_usage_pct") is not None
                    else "N/A"
                )

                cell_vh = compute_verification_hash(
                    model_name=model_name,
                    hardware=hardware,
                    framework=framework,
                    quantization=quantization,
                    concurrency=concurrency,
                    max_decode_speed=max_decode,
                )

                cell_data = {
                    "task_type": task_type,
                    "prompt_length": resolved_length,
                    "requested_prompt_length": length,
                    "concurrency": concurrency,
                    "avg_ttft_ms": round(avg_ttft, 2),
                    "ttft_p50_ms": round(ttft_p50, 2),
                    "ttft_p90_ms": round(ttft_p90, 2),
                    "ttft_p99_ms": round(ttft_p99, 2),
                    "avg_prefill_speed": round(avg_prefill, 2),
                    "avg_decode_speed": round(avg_decode, 2),
                    "max_prefill_speed": round(max_prefill, 2),
                    "max_decode_speed": round(max_decode, 2),
                    "output_speed": round(avg_decode, 2),
                    "decode_tps": round(avg_decode, 2),
                    "prefill_speed": round(avg_prefill, 2),
                    "itl_p50_ms": round(itl_p50, 2),
                    "itl_p90_ms": round(itl_p90, 2),
                    "itl_p99_ms": round(itl_p99, 2),
                    "effective_concurrency": eff_conc["effective_concurrency"],
                    "effective_concurrency_label": eff_conc["label"],
                    "verification_hash": cell_vh,
                    "telemetry": telemetry_summary,
                    "loop_detected": loop_detected,
                    "success_count": len(successful_results),
                    "failed_count": failed_count,
                    "status": "成功" if not loop_detected else "ERROR: loop",
                }
                all_cell_results.append(cell_data)

                status_str = "✅ PASS" if not loop_detected else "⚠️ LOOP DETECTED"
                print(
                    f"  -> C={concurrency:3d} | L={resolved_length:5d} | "
                    f"TTFT P50={ttft_p50:6.1f}ms | Decode TPS={avg_decode:6.1f} | "
                    f"ITL P50={itl_p50:5.1f}ms | KV Cache={kv_cache_str:>6} | "
                    f"Eff. Conc={eff_conc['label']:>7} | {status_str}"
                )
                await asyncio.sleep(1.0)

    if not all_succ_results:
        print("\n❌ Benchmark finished with 0 successful requests.", file=sys.stderr)
        return 1

    overall_avg_prefill = sum(r["prefill_speed"] for r in all_succ_results) / len(all_succ_results)
    overall_avg_decode = sum(r["output_speed"] for r in all_succ_results) / len(all_succ_results)
    overall_max_prefill = max(r["prefill_speed"] for r in all_succ_results)
    overall_max_decode = max(r["output_speed"] for r in all_succ_results)
    max_concurrency = max(concurrency_list)

    overall_vh = compute_verification_hash(
        model_name=model_name,
        hardware=hardware,
        framework=framework,
        quantization=quantization,
        concurrency=max_concurrency,
        max_decode_speed=overall_max_decode,
    )

    print("\n" + "=" * 84)
    print("📊 Benchmark Summary")
    print("=" * 84)
    print(f"Model Name           : {model_name}")
    print(f"Hardware             : {hardware}")
    print(f"Framework            : {framework} ({quantization})")
    print(f"Peak Concurrency     : {max_concurrency}")
    print(f"Avg Prefill Speed    : {overall_avg_prefill:.2f} tokens/s")
    print(f"Avg Decode Speed     : {overall_avg_decode:.2f} tokens/s")
    print(f"Max Decode Speed     : {overall_max_decode:.2f} tokens/s")
    print(f"Verification Hash    : {overall_vh}")
    print("=" * 84)

    # Save JSON report
    report_data = {
        "benchmark_mode": "1d_sweep",
        "task_type": task_type,
        "model_name": model_name,
        "hardware": hardware,
        "framework": framework,
        "quantization": quantization,
        "notes": args.notes,
        "run_command": args.run_command,
        "concurrency": max_concurrency,
        "avg_prefill_speed": round(overall_avg_prefill, 2),
        "avg_decode_speed": round(overall_avg_decode, 2),
        "max_prefill_speed": round(overall_max_prefill, 2),
        "max_decode_speed": round(overall_max_decode, 2),
        "verification_hash": overall_vh,
        "source": "python_v3",
        "timestamp": int(time.time()),
        "results": all_cell_results,
    }

    output_path = args.output
    if not output_path:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)[:30]
        output_path = f"benchmark_{safe_model}_{ts_str}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Local benchmark report saved to: {os.path.abspath(output_path)}")

    # Cloud upload if requested
    if args.upload:
        print("\n☁️ Uploading benchmark to Supabase Leaderboard...")
        user_code = args.user_code
        if not user_code or len(user_code.strip()) < 8:
            user_code = "CLI" + hashlib.sha256(f"{hardware}{model_name}{time.time()}".encode()).hexdigest()[:5].upper()
            print(f"ℹ️ Auto-generated User Code: {user_code}")
        else:
            user_code = user_code.strip().upper()

        upload_res = await upload_benchmark_to_supabase(
            user_code=user_code,
            model_name=model_name,
            hardware=hardware,
            concurrency=max_concurrency,
            avg_prefill_speed=overall_avg_prefill,
            avg_decode_speed=overall_avg_decode,
            max_prefill_speed=overall_max_prefill,
            max_decode_speed=overall_max_decode,
            framework=framework,
            quantization=quantization,
            notes=args.notes,
            nickname=args.nickname,
            run_command=args.run_command,
            record_tags=["source:python_v3", "verified", "mode:1d_sweep", f"task:{task_type}"],
            results_json=all_cell_results,
            verification_hash=overall_vh,
        )

        if upload_res.get("success"):
            share_id = upload_res.get("id")
            print("✅ Cloud upload successful!")
            print(f"  Record ID          : {share_id}")
            print(f"  Verification Hash  : {upload_res.get('verification_hash')}")
            print(f"  Leaderboard Share  : {PUBLIC_LEADERBOARD_URL}?id={share_id}")
        else:
            print(f"⚠️ Cloud upload failed: {upload_res.get('error')}")
            print("  (Your local JSON report is safely saved above)")

    return 0


def run_server(port: int, host: str = "0.0.0.0"):
    """Launch FastAPI / WebSocket server."""
    global current_port
    current_port = port

    print("LLM Speed Test Backend Server", flush=True)
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print(f"WebSocket endpoint: ws://{display_host}:{port}/ws/test", flush=True)
    print(f"Frontend page: http://{display_host}:{port}/", flush=True)
    print(f"PORT={port}", flush=True)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        port_file = os.path.join(script_dir, '.backend_port')
        with open(port_file, 'w') as f:
            f.write(str(port))
    except Exception:
        pass

    import uvicorn
    sockets = []
    if host in ("0.0.0.0", "::"):
        try:
            import socket
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('::', port))
            sock.listen(128)
            sockets.append(sock)
            print(f"[Server] IPv4/IPv6 双栈监听已启用 (支持 localhost, 127.0.0.1 及 [::1], 端口 {port})", flush=True)
        except Exception as e:
            print(f"[Server] 双栈监听绑定提示: {e}，回退至标准 host={host}", flush=True)
            sockets = []

    config = uvicorn.Config(
        app,
        host=host if not sockets else None,
        port=port if not sockets else None,
        ws_ping_interval=30.0,
        ws_ping_timeout=30.0,
        ws_max_size=33554432,
    )
    server = uvicorn.Server(config)
    if sockets:
        server.run(sockets=sockets)
    else:
        server.run()


def main():
    """Main CLI entrypoint supporting Server and Benchmark modes."""
    # Fast path: single numeric argument (e.g. `python llm_test_backend.py 18000`)
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
        run_server(port)
        return

    parser = build_cli_parser()

    # If help requested
    if "-h" in sys.argv or "--help" in sys.argv:
        parser.print_help()
        return

    # Check whether benchmark mode is specified
    benchmark_flags = {"--benchmark", "--cli", "--model", "-m"}
    is_benchmark = any(arg in benchmark_flags for arg in sys.argv)

    if not is_benchmark or "--server" in sys.argv or "--serve" in sys.argv:
        args, _ = parser.parse_known_args()
        port = args.port or find_free_port()
        run_server(port, host=args.host)
        return

    args = parser.parse_args()
    if not args.model:
        parser.error("--model is required for CLI benchmark mode")
    sys.exit(asyncio.run(run_cli_benchmark(args)))


if __name__ == "__main__":
    main()

