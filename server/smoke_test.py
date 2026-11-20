#!/usr/bin/env python3
"""End-to-end smoke test for the YULUO-snow limited API.

Runs against a live server on the given base URL and checks:
  health contract, CORS headers, deadline state, input validation,
  rate limiting, and (in replay mode) a known-case hit.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
ORIGIN = "https://commitment-receipt-27t03pzl6zt.qoder.zone"


def call(method: str, path: str, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("content-type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(req, timeout=180) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", "replace")


def main() -> None:
    fail = 0

    print("1) GET /v1/health")
    st, hd, body = call("GET", "/v1/health")
    print("   status:", st)
    print("   body:", body[:240])
    if st != 200:
        fail += 1
    else:
        d = json.loads(body)
        for k in ("model", "engine", "open", "seconds_left", "window"):
            if k not in d:
                print(f"   [FAIL] 缺少字段 {k}")
                fail += 1
        print(f"   模型={d.get('model')} 引擎={d.get('engine')} 开放={d.get('open')} "
              f"剩余={d.get('seconds_left')}s")

    print("\n2) CORS 预检（评委站点来源）")
    st, hd, body = call("OPTIONS", "/v1/extract", headers={
        "Origin": ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    allow = hd.get("access-control-allow-origin") or hd.get("Access-Control-Allow-Origin")
    print("   status:", st, "| ACAO:", allow)
    if allow != ORIGIN:
        print("   [FAIL] CORS 未放行站点来源")
        fail += 1

    print("\n3) 空输入")
    st, hd, body = call("POST", "/v1/extract", {"text": "   "})
    print("   status:", st, "| body:", body[:120])
    if st != 400:
        print("   [FAIL] 空输入应 400")
        fail += 1

    print("\n4) 超长输入")
    st, hd, body = call("POST", "/v1/extract", {"text": "x" * 5000})
    print("   status:", st, "| body:", body[:160])
    if st != 413:
        print("   [FAIL] 超长应 413")
        fail += 1

    print("\n5) POST /v1/extract（真实条款）")
    st, hd, body = call("POST", "/v1/extract", {
        "text": "首月仅 0.1 元，之后每 30 天自动续费 25 元，已支付金额不退。",
        "source_id": "smoke_001",
    })
    print("   status:", st)
    print("   body:", body[:400])
    if st not in (200, 404):
        print("   [FAIL] 期望 200（真实推理）或 404（replay 未命中）")
        fail += 1

    print("\n6) 限流（上限 20 次/10 分钟）")
    codes = []
    for _ in range(24):
        st, _, _ = call("POST", "/v1/extract", {"text": "第 30 天扣款 25 元"})
        codes.append(st)
    print("   statuses:", codes)
    if 429 not in codes:
        print("   [WARN] 未触发 429，请确认 RATE_LIMIT 生效（replay 未命中也计入）")

    print("\n7) selfcheck")
    st, hd, body = call("GET", "/v1/selfcheck")
    print("   status:", st, "|", body[:260])

    print("\n结果:", "ALL PASS" if fail == 0 else f"{fail} 项失败")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
