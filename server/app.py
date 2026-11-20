"""YULUO-snow 限时 API —— llama.cpp(Q8_0) 引擎版

关键前提（踩过的坑，别再改回去）：
QLoRA 的增量是在「nf4 4-bit 量化后的底座」上学出来的。直接把 adapter 合并进
原始 fp16 底座再转 GGUF，行为会退化为基座（键名变 product_service、完全没有
evidence）。正确做法是合并进 **反量化后的 4-bit 权重**（见 merge_into_dequant.py），
产物 merged4deq → GGUF Q8_0，既保住训练行为，又能用 llama.cpp 跑得快。

模型只负责「读懂并引用」；字符偏移重锚定与算钱交给确定性代码
（spanfix / validator / calculator），校验不通过就如实返回错误。
"""
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 默认指向仓库根目录（本文件在 server/ 下），可用 APP 环境变量覆盖
APP = Path(os.environ.get("APP", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(APP / "src"))

from fastapi import FastAPI, Request                       # noqa: E402
from fastapi.middleware.cors import CORSMiddleware         # noqa: E402
from fastapi.responses import JSONResponse                 # noqa: E402
from pydantic import BaseModel                             # noqa: E402

# 训练同源提示词，不复制、不追加（实测追加约束会把模型推出分布：2/5 -> 0/5）
from generate_synthetic_data import SYSTEM_PROMPT           # noqa: E402

# ---------------------------------------------------------------- 配置
NL = chr(10)
DEADLINE = datetime(2026, 11, 30, 18, 0, 0, tzinfo=timezone(timedelta(hours=8)))
LLAMA_URL = os.environ.get("YULUO_BACKEND",
                           "http://127.0.0.1:8081/v1/chat/completions")
SCHEMA = APP / "schema" / "commitment_schema_v1.json"
SCHEMA_OBJ = json.loads(SCHEMA.read_text(encoding="utf-8")) if SCHEMA.exists() else None
MAX_INPUT = 4000
N_PREDICT = int(os.environ.get("YULUO_N_PREDICT", "800"))
RATE_LIMIT, RATE_WINDOW = 20, 600

API_KEY = os.environ.get("YULUO_API_KEY", "").strip()
ALLOWED_ORIGINS = ["https://commitment-receipt-27t03pzl6zt.qoder.zone"]
_extra = os.environ.get("YULUO_EXTRA_ORIGINS", "").strip()
if _extra:
    ALLOWED_ORIGINS += [o.strip() for o in _extra.split(",") if o.strip()]

_hits = defaultdict(deque)
app = FastAPI(title="YULUO-snow limited inference API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["content-type", "x-api-key"], max_age=600)


class ExtractIn(BaseModel):
    text: str
    source_id: str = "web_input"


# ---------------------------------------------------------------- 闸门
def deadline_state():
    left = (DEADLINE - datetime.now(timezone.utc)).total_seconds()
    return left > 0, int(max(left, 0))


def rate_limited(ip):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True, int(RATE_WINDOW - (now - q[0]))
    q.append(now)
    return False, 0


# ---------------------------------------------------------------- 解析
def first_json_object(raw):
    """取第一个「括号配平且可解析」的对象。模型偶尔输出 {...}}, {...} 序列，
    贪婪正则会把两个对象连起来，让合格的第一个也变成非法 JSON。"""
    if not raw:
        return None
    try:
        v = json.loads(raw.strip())
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == chr(92):
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        v = json.loads(raw[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except Exception:
                        pass
    return None


def backend_alive():
    base = LLAMA_URL.split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------- 归一
PLACEHOLDERS = ("未知商品", "未知服务", "未知产品", "未知内容", "未知", "未提及",
                "未说明", "未给出", "未命名", "unknown", "Unknown", "UNKNOWN",
                "N/A", "n/a", "无", "-")


def normalize_placeholders(obj):
    """把模型用于表示「原文没给」的占位词，归一为 schema 的正式表达。

    这不是掩饰幻觉：模型用「未知商品」表达"文本里没有名称"，但 schema 里
    「有名称」必须给证据，把占位词当 known 就会配上伪造 span。归一到
    name='' / status='unknown' / evidence=[] 才是正确语义，且不需要证据。
    仅对 product_or_service.name 生效，范围最小化。
    """
    ps = obj.get("product_or_service")
    if isinstance(ps, dict):
        name = str(ps.get("name") or "").strip()
        if name in PLACEHOLDERS or name.startswith("未知"):
            ps["name"] = ""
            ps["status"] = "unknown"
            ps["evidence"] = []
    return obj


# ---------------------------------------------------------------- 路由
@app.get("/v1/health")
def health():
    ok, left = deadline_state()
    return {
        "ok": True,
        "model": "YULUO-snow v1.0",
        "engine": "llama.cpp/Q8_0 (merged-from-4bit)",
        "backend_ready": backend_alive(),
        "open": ok,
        "seconds_left": left,
        "hours_left": round(left / 3600, 1),
        "window": DEADLINE.strftime("%Y-%m-%d %H:%M (UTC+8)"),
        "limits": {"max_input_chars": MAX_INPUT, "n_predict": N_PREDICT,
                   "rate": f"{RATE_LIMIT}/{RATE_WINDOW}s per IP"},
        "auth": "x-api-key" if API_KEY else "none",
    }


@app.get("/v1/selfcheck")
def selfcheck():
    ok, left = deadline_state()
    return {"deadline_utc8": DEADLINE.isoformat(), "open": ok,
            "schema_exists": SCHEMA.exists(), "backend": LLAMA_URL,
            "backend_ready": backend_alive(), "cors": ALLOWED_ORIGINS}


@app.post("/v1/extract")
def extract(body: ExtractIn, request: Request):
    if API_KEY and request.headers.get("x-api-key", "") != API_KEY:
        return JSONResponse(status_code=401,
                           content={"ok": False, "reason": "bad_api_key"})
    ok, _ = deadline_state()
    if not ok:
        return JSONResponse(status_code=410, content={
            "ok": False, "reason": "expired",
            "message": "评审期已于 2026-11-30 18:00 (UTC+8) 结束"})

    ip = request.client.host if request.client else "-"
    blocked, retry = rate_limited(ip)
    if blocked:
        return JSONResponse(status_code=429, content={
            "ok": False, "reason": "rate_limited", "retry_after_s": retry})

    text = (body.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"ok": False, "reason": "empty"})
    if len(text) > MAX_INPUT:
        return JSONResponse(status_code=413, content={
            "ok": False, "reason": "too_long", "limit": MAX_INPUT, "got": len(text)})
    if not backend_alive():
        return JSONResponse(status_code=503, content={
            "ok": False, "reason": "backend_down",
            "message": "推理后端未就绪，请稍后重试"})

    sid = (body.source_id or "web_input")[:64]
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "source_id: " + sid + NL + "待抽取文本：" + NL + text},
        ],
        "max_tokens": N_PREDICT,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        # 受语法约束的解码：把 schema 交给 llama.cpp，在解码阶段就排除
        # 「自创字段 / 缺 required / 枚举越界」，结构错误归零。
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "commitment", "schema": SCHEMA_OBJ,
                            "strict": True},
        },
    }).encode()

    t0 = time.time()
    try:
        req = urllib.request.Request(LLAMA_URL, data=payload,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=420) as r:
            raw = json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as exc:
        return JSONResponse(status_code=503, content={
            "ok": False, "reason": "engine_error", "message": str(exc)[:200]})
    elapsed = round(time.time() - t0, 2)

    from spanfix import reanchor
    from validator import validate_extraction
    from calculator import compute_commitment_summary

    parsed = first_json_object(raw)
    if parsed is None:
        return {"ok": False, "engine": "llama.cpp/Q8_0", "elapsed_s": elapsed,
                "stage": "parse", "raw": raw[:400]}

    parsed = normalize_placeholders(parsed)
    anchored, stats = reanchor(parsed, text)
    valid, errs = validate_extraction(anchored, text, sid, SCHEMA)
    totals = None
    if valid:
        totals = compute_commitment_summary(
            anchored, order_date_iso=datetime.now().date().isoformat())

    return {"ok": valid, "engine": "llama.cpp/Q8_0", "elapsed_s": elapsed,
            "errors": errs, "span_reanchor": stats,
            "extracted": anchored, "totals": totals}
