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
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import httpx
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from urllib.parse import urlparse, quote

# ============================================================
# Supabase 配置（云端结果共享）
# ============================================================
SUPABASE_URL = "https://cvezaerrczywfzqcaqmx.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXphZXJyY3p5d2Z6cWNhcW14Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI0Nzc0OTAsImV4cCI6MjA4ODA1MzQ5MH0.-9l2Wcj38m6xh9sz9qMdhdNbnAZ_XoFBSHPhEud9DmQ"
# 注意：只使用公开的 anon key，不在代码中保存 service key
# RLS 策略负责权限控制（见 README 中的建表 SQL）
SUPABASE_TABLE = "speed_results"

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

app = FastAPI(title="LLM Speed Test Backend")

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
    avg_prefill_speed: float
    avg_decode_speed: float
    max_prefill_speed: Optional[float] = None
    max_decode_speed: Optional[float] = None
    source: Optional[str] = "python_backend"
    record_tags: Optional[List[str]] = None
    results: List[ResultPoint]

    @field_validator('user_code')
    @classmethod
    def validate_user_code(cls, v):
        v = v.strip().upper()
        if len(v) != 8 or not v.isalnum():
            raise ValueError('user_code 必须是8位字母数字')
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


@app.post("/api/upload")
@limiter.limit("10/hour")
async def upload_result(request: Request, payload: UploadPayload):
    """上传测试结果到云端，返回分享链接"""
    # 构造存储数据
    results_json = [
        {
            "prompt_length": r.prompt_length,
            "prefill_speed": r.prefill_speed,
            "output_speed": r.output_speed,
            "prefill_time_ms": r.prefill_time_ms,
            "output_time_ms": r.output_time_ms,
            "avg_ttft_ms": r.avg_ttft_ms,
            "avg_itl_mean": r.avg_itl_mean,
            "avg_itl_std": r.avg_itl_std,
            "concurrency": r.concurrency,
            "successful": r.successful,
            "boundary_source": r.boundary_source,
            "concurrent_details": [
                {
                    "prefill_speed": d.get("prefill_speed"),
                    "output_speed": d.get("output_speed"),
                    "boundary_source": d.get("boundary_source")
                }
                for d in (r.concurrent_details or [])
            ]
        }
        for r in payload.results
    ]

    max_prefill_value = payload.max_prefill_speed if payload.max_prefill_speed is not None else payload.avg_prefill_speed
    max_decode_value = payload.max_decode_speed if payload.max_decode_speed is not None else payload.avg_decode_speed
    source_value = payload.source or "python_backend"
    record_tags = [tag.strip() for tag in (payload.record_tags or []) if isinstance(tag, str) and tag.strip()]
    if not any(tag.startswith("source:") for tag in record_tags):
        record_tags.insert(0, f"source:{source_value}")
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
    share_url = f"{SUPABASE_URL.replace('supabase.co', 'supabase.co')}/leaderboard/{record_id}"
    # 实际分享链接指向排行榜，由用户复制 record_id 分享
    return {"success": True, "id": record_id, "share_id": record_id}


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

    select_fields = "id,user_code,nickname,model_name,hardware,framework,quantization,notes,run_command,concurrency,avg_prefill_speed,avg_decode_speed,max_prefill_speed,max_decode_speed,source,record_tags,created_at,results_json"
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


def _is_cjk_char(char: str) -> bool:
    return (
        "\u4e00" <= char <= "\u9fff"
        or "\u3400" <= char <= "\u4dbf"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
    )


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


def generate_short_prompt(length: int, seed: int = 0) -> str:
    """Generate a short prompt without the benchmark essay suffix."""
    target_prompt_length = get_resolved_prompt_length(length)
    return build_token_word_sequence(target_prompt_length, seed)


def generate_prompt(length: int, seed: int = 0) -> str:
    """Generate a randomized prompt whose final estimated length matches the target semantics."""
    target_prompt_length = get_resolved_prompt_length(length)
    if target_prompt_length <= SHORT_PROMPT_MAX_LENGTH:
        return generate_short_prompt(target_prompt_length, seed)

    suffix_text = DEFAULT_BENCHMARK_SUFFIX.strip()
    suffix_tokens = estimate_token_count(suffix_text)
    prefix_token_count = max(target_prompt_length - suffix_tokens, 0)
    prefix_text = build_token_word_sequence(prefix_token_count, seed)

    if prefix_text:
        return prefix_text + "\n" + suffix_text
    return suffix_text


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


def calculate_aggregate_throughput_metrics(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate concurrent throughput with server timings when available."""
    if not results:
        return {
            "prefill_time_ms": 0.0,
            "output_time_ms": 0.0,
            "prefill_speed": 0.0,
            "output_speed": 0.0,
            "prefill_source": "client",
            "output_source": "client",
        }

    total_prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in results)
    total_output_tokens = sum(r.get("output_tokens") or 0 for r in results)

    use_server_prefill = all(_has_positive_number(r.get("server_prefill_time_ms")) for r in results)
    if use_server_prefill:
        prefill_time_ms = max(r["server_prefill_time_ms"] for r in results)
        prefill_source = "server"
    else:
        min_start = min(r.get("start_timestamp") or 0 for r in results)
        max_boundary = max(
            r.get("first_token_timestamp")
            or r.get("boundary_timestamp")
            or r.get("end_timestamp")
            or min_start
            for r in results
        )
        average_network_latency_ms = get_average_network_latency_ms([
            r.get("network_latency_ms") or 0 for r in results
        ])
        prefill_time_ms = max(((max_boundary - min_start) * 1000) - average_network_latency_ms, 1)
        prefill_source = "client_latency_adjusted" if average_network_latency_ms > 0 else "client"

    use_server_output = all(_has_positive_number(r.get("server_decode_time_ms")) for r in results)
    if use_server_output:
        output_time_ms = max(r["server_decode_time_ms"] for r in results)
        output_source = "server"
    else:
        min_decode_start = min(
            r.get("first_token_timestamp")
            or r.get("boundary_timestamp")
            or r.get("start_timestamp")
            or 0
            for r in results
        )
        max_end = max(r.get("end_timestamp") or min_decode_start for r in results)
        output_time_ms = max((max_end - min_decode_start) * 1000, 1)
        output_source = "client"

    prefill_speed = total_prompt_tokens / (prefill_time_ms / 1000) if prefill_time_ms > 0 else 0.0
    output_speed = total_output_tokens / (output_time_ms / 1000) if output_time_ms > 0 else 0.0

    return {
        "prefill_time_ms": prefill_time_ms,
        "output_time_ms": output_time_ms,
        "prefill_speed": prefill_speed,
        "output_speed": output_speed,
        "prefill_source": prefill_source,
        "output_source": output_source,
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
) -> Dict[str, Any]:
    """执行单个测试请求"""
    
    resolved_prompt_length = get_resolved_prompt_length(prompt_length)
    generation_prompt_length = get_calibrated_generation_prompt_length(prompt_length, prompt_calibration)
    print(
        f"[Request] 开始请求 - Prompt长度: {prompt_length}, "
        f"目标长度: {resolved_prompt_length}, 输出长度: {output_length}, Seed: {seed}"
    )

    # 使用随机单词生成prompt，避免cache
    prompt_text = generate_prompt(generation_prompt_length, seed)
    local_estimated_prompt_tokens = estimate_prompt_tokens_for_messages(prompt_text)
    estimated_prompt_tokens = apply_prompt_calibration(local_estimated_prompt_tokens, prompt_calibration)
    
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
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
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
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
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
        print(f"[Request] Headers: {redact_sensitive_headers(headers)}", flush=True)

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
                    error_msg = build_http_error_message(api_url, model_name, response.status_code, error_text)
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
                            usage_stats = extract_usage_token_stats(usage)
                            completion = usage_stats["completion_tokens"] or 0
                            reasoning = usage_stats["reasoning_tokens"] or 0
                            prompt = usage_stats["prompt_tokens"] or 0

                            print(f"[Usage] 找到usage字段！prompt_tokens: {prompt}, completion_tokens: {completion}, reasoning_tokens: {reasoning}")
                            print(f"[Debug] 完整usage字段: {json.dumps(usage, indent=2)}")

                            actual_output_tokens = usage_stats["output_tokens"]
                            if _has_positive_number(usage_stats["prompt_tokens"]):
                                actual_prompt_tokens = usage_stats["prompt_tokens"]
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
                            if _has_positive_number(prompt):
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
                            # 思考模型格式：delta.reasoning（思考过程）
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

        # Use first content token as default prefill/decode boundary.
        boundary_source = "first_content_token"
        if first_token_time is not None:
            boundary_time = first_token_time
            ttft_ms = (boundary_time - start_time) * 1000
            print(f"[Timing] Boundary=first_content_token, TTFT: {ttft_ms:.2f}ms, Total: {total_time_ms:.2f}ms")
        elif first_chunk_time is not None:
            boundary_source = "first_chunk_fallback"
            boundary_time = first_chunk_time
            ttft_ms = (boundary_time - start_time) * 1000
            print(f"[Timing] Boundary fallback to first_chunk, TTFT: {ttft_ms:.2f}ms, Total: {total_time_ms:.2f}ms")
        else:
            boundary_source = "no_boundary_fallback"
            boundary_time = end_time
            ttft_ms = total_time_ms
            print(f"[Timing] Warning: no token/chunk timestamp, fallback to end_time. Total: {total_time_ms:.2f}ms")

        decode_time_ms = max(total_time_ms - ttft_ms, 1)

        print(f"[Timing] Server Prefill: {server_prefill_time_ms}ms, Server Decode: {server_decode_time_ms}ms")
        
        # Token sanity check: llama.cpp-like servers may return cumulative totals per concurrent response.
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
            print(
                "[TokenWarning] Prompt API stats look abnormal; "
                f"api={prompt_resolution['warning']['api_tokens']}, "
                f"local={prompt_resolution['warning']['estimated_tokens']}, "
                f"diff={prompt_resolution['warning']['relative_diff'] * 100:.1f}%. "
                "Using local estimate.",
                flush=True,
            )
        elif prompt_resolution["source"] == "API":
            print(f"[Token] Prompt from API usage: {actual_prompt_tokens}")
        else:
            print(f"[Token] Prompt local estimate: {actual_prompt_tokens}")

        output_resolution = resolve_token_count_with_sanity_check(
            "decode",
            api_output_tokens,
            local_output_tokens,
        )
        actual_output_tokens = output_resolution["tokens"]
        if output_resolution["warning"]:
            token_warnings.append(output_resolution["warning"])
            print(
                "[TokenWarning] Decode API stats look abnormal; "
                f"api={output_resolution['warning']['api_tokens']}, "
                f"local={output_resolution['warning']['estimated_tokens']}, "
                f"diff={output_resolution['warning']['relative_diff'] * 100:.1f}%. "
                "Using local estimate.",
                flush=True,
            )
        elif output_resolution["source"] == "API":
            print(f"[Token] Output from API usage: {actual_output_tokens}")
        else:
            print(f"[Token] Output local estimate: {actual_output_tokens}")

        if token_warnings:
            token_source = "API Stats Anomaly"
        elif output_resolution["source"] == "API" or prompt_resolution["source"] == "API":
            token_source = "API"
        elif output_resolution["source"] == "Local Estimation" or prompt_resolution["source"] == "Local Estimation":
            token_source = "Local Estimation"
        else:
            token_source = "Unknown"

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
            print(f"[TimeSource] client timing fallback - Prefill(TTFT): {prefill_time_ms:.2f}ms, Decode: {output_time_ms:.2f}ms, Boundary: {boundary_source}")
            if server_prefill_time_ms or server_decode_time_ms:
                print(f"[TimeSource] 部分服务器timing可用 - Prefill: {server_prefill_time_ms}ms, Decode: {server_decode_time_ms}ms")

        has_server_prefill_timing = _has_positive_number(server_prefill_time_ms)
        has_server_decode_timing = _has_positive_number(server_decode_time_ms)
        latency_adjustment_ms = network_latency_ms if _has_positive_number(network_latency_ms) else 0.0
        if has_server_prefill_timing or has_server_decode_timing:
            prefill_time_ms = server_prefill_time_ms if has_server_prefill_timing else max(ttft_ms - latency_adjustment_ms, 1)
            output_time_ms = server_decode_time_ms if has_server_decode_timing else decode_time_ms
            prefill_time_source = "server" if has_server_prefill_timing else ("client_latency_adjusted" if latency_adjustment_ms > 0 else "client")
            output_time_source = "server" if has_server_decode_timing else "client"
            print(
                f"[TimeSource] Adjusted per-phase timing - Prefill={prefill_time_source} "
                f"({prefill_time_ms:.2f}ms), Decode={output_time_source} ({output_time_ms:.2f}ms)"
            )
        else:
            prefill_time_ms = max(ttft_ms - latency_adjustment_ms, 1)
            prefill_time_source = "client_latency_adjusted" if latency_adjustment_ms > 0 else "client"
            output_time_source = "client"

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
            "prompt_length": resolved_prompt_length,
            "requested_prompt_length": prompt_length,
            "generated_prompt_length": generation_prompt_length,
            "prompt_tokens": actual_prompt_tokens,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "local_estimated_prompt_tokens": local_estimated_prompt_tokens,
            "prompt_calibration": prompt_calibration,
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
            "server_timing_used": server_prefill_time_ms is not None or server_decode_time_ms is not None,
            "server_prefill_time_ms": round(server_prefill_time_ms, 2) if server_prefill_time_ms is not None else None,
            "server_decode_time_ms": round(server_decode_time_ms, 2) if server_decode_time_ms is not None else None,
            "prefill_time_source": prefill_time_source,
            "output_time_source": output_time_source,
            "network_latency_ms": round(latency_adjustment_ms, 2),
            "network_latency_sample_count": network_latency_sample_count,
            "has_streaming_content": first_token_time is not None,  # 是否有实际内容token
            "chunk_count": chunk_count,  # chunk数量
            # 添加绝对时间戳用于并发总吞吐计算
            "start_timestamp": start_time,
            "first_chunk_timestamp": first_chunk_time,  # chunk到达时间
            "first_token_timestamp": first_token_time,  # 内容token时间
            "boundary_timestamp": boundary_time,
            "boundary_source": boundary_source,
            "end_timestamp": end_time,
            "token_source": token_source,  # 添加token来源
            "token_warnings": token_warnings,
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
                    network_latency_ms=network_latency_ms,
                    network_latency_sample_count=len(latency_samples),
                    prompt_calibration=prompt_calibration,
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
                max_prefill_speed = max(r["prefill_speed"] for r in successful_results)
                max_output_speed = max(r["output_speed"] for r in successful_results)
                avg_ttft = sum(r["ttft_ms"] for r in successful_results) / len(successful_results)
                avg_prompt_tokens = sum(r["prompt_tokens"] for r in successful_results) / len(successful_results)
                aggregate_metrics = calculate_aggregate_throughput_metrics(successful_results)
                boundary_fallback_count = sum(1 for r in successful_results if r.get("boundary_source") == "first_chunk_fallback")
                token_warning_count = sum(len(r.get("token_warnings") or []) for r in successful_results)
                record_tags = ["source:python_backend"]
                if boundary_fallback_count > 0:
                    record_tags.append("boundary_source:first_chunk_fallback")
                if token_warning_count > 0:
                    record_tags.append("token_source:api_stats_anomaly")
                if aggregate_metrics["prefill_source"] == "server":
                    record_tags.append("prefill_source:server")
                elif aggregate_metrics["prefill_source"] == "client_latency_adjusted":
                    record_tags.append("prefill_source:client_latency_adjusted")
                if aggregate_metrics["output_source"] == "server":
                    record_tags.append("decode_source:server")
                
                print(f"[Stats] 成功: {len(successful_results)}/{config.concurrency}")
                print(f"[Stats] 平均 Prefill速度: {avg_prefill_speed:.2f} t/s")
                print(f"[Stats] 平均 Decode速度: {avg_output_speed:.2f} t/s")
                print(f"[Stats] Max Prefill speed: {max_prefill_speed:.2f} t/s")
                print(f"[Stats] Max Decode speed: {max_output_speed:.2f} t/s")
                print(f"[Stats] 平均 TTFT: {avg_ttft:.2f} ms")
                
                print(
                    f"[Stats] Aggregate Prefill: {aggregate_metrics['prefill_speed']:.2f} t/s "
                    f"({aggregate_metrics['prefill_time_ms']:.2f}ms, source={aggregate_metrics['prefill_source']})"
                )
                print(
                    f"[Stats] Aggregate Decode: {aggregate_metrics['output_speed']:.2f} t/s "
                    f"({aggregate_metrics['output_time_ms']:.2f}ms, source={aggregate_metrics['output_source']})"
                )

                result_summary = {
                    "type": "result",
                    "prompt_length": get_resolved_prompt_length(length),
                    "requested_prompt_length": length,
                    "network_latency_ms": round(network_latency_ms, 2),
                    "network_latency_sample_count": len(latency_samples),
                    "avg_prompt_tokens": round(avg_prompt_tokens, 2),
                    "concurrency": config.concurrency,
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
                    "aggregate_prefill_source": aggregate_metrics["prefill_source"],
                    "aggregate_output_source": aggregate_metrics["output_source"],
                    "avg_ttft_ms": round(avg_ttft, 2),
                    "source": "python_backend",
                    "record_tags": record_tags,
                    "boundary_fallback_count": boundary_fallback_count,
                    "token_warning_count": token_warning_count,
                    "concurrent_details": successful_results,
                    "status": "成功"
                }
                
                all_results.append(result_summary)
                await websocket.send_json(result_summary)
            else:
                print(f"[Error] 所有请求失败 - 失败数: {failed_count}")
                error_result = {
                    "type": "result",
                    "prompt_length": get_resolved_prompt_length(length),
                    "requested_prompt_length": length,
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
