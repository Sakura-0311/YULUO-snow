# 推理 API 契约

线上实例：`https://yuluo.sakura1225.cloud/yuluo/v1`
评审期：至 **2026-11-30 18:00 (UTC+8)**，之后所有抽取请求返回 410。

## GET /v1/health

```json
{
  "ok": true,
  "model": "YULUO-snow v1.0",
  "engine": "llama.cpp/Q8_0 (merged-from-4bit)",
  "backend_ready": true,
  "open": true,
  "seconds_left": 888062,
  "hours_left": 246.7,
  "window": "2026-11-30 18:00 (UTC+8)",
  "limits": {"max_input_chars": 4000, "n_predict": 800, "rate": "20/600s per IP"},
  "auth": "none"
}
```

## POST /v1/extract

请求：

```json
{"text": "会员首月 9.9 元，次月起每月 29 元自动续费，可随时取消，已付费用不予退还。",
 "source_id": "web_input"}
```

成功（HTTP 200，**注意 `ok` 可能为 false**，表示请求跑通了但严格校验未通过）：

```json
{
  "ok": true,
  "engine": "llama.cpp/Q8_0",
  "elapsed_s": 27.4,
  "errors": [],
  "span_reanchor": {"spans": 3, "fixed": 3, "already_ok": 0, "unresolved": 0},
  "extracted": { "...": "归一后的结构化承诺单" },
  "totals": { "...": "ok=true 时由 calculator.py 确定性算出" }
}
```

`ok:false` 时 `errors` 给出具体原因，调用方**必须展示**而不是当成网络失败：

```
product_or_service.evidence[0]: span mismatch: text[8:14]='元，之后每 ' != quote='未知服务'
jsonschema error: Additional properties are not allowed ('renewal_add_on' was unexpected)
```

错误码：

| 码 | reason | 说明 |
|---|---|---|
| 400 | `empty` | text 为空 |
| 410 | `expired` | 评审期已结束 |
| 413 | `too_long` | 超 4000 字符，带 `limit` / `got` |
| 429 | `rate_limited` | 超 20 次/10 分钟，带 `retry_after_s`；`busy` 表示单并发已占用 |
| 503 | `backend_down` / `loading` / `engine_error` | 后端未就绪或推理异常 |

## 部署要点

服务只绑 `127.0.0.1:8000`，由已有 Caddy 在 443 上反代 `/yuluo/*`：

```caddyfile
# 必须放在兜底 handle 之前，且不能经过 forward_auth，
# 否则外部访问会先被登录页拦住。
handle_path /yuluo/* {
    reverse_proxy 127.0.0.1:8000
}
```

`handle_path` 会剥掉 `/yuluo` 前缀，所以后端路由是 `/v1/extract` 而非 `/yuluo/v1/extract`。

## 引擎选择

| 方案 | 通过率 | 单例耗时 | 适用 |
|---|---|---|---|
| transformers + bnb 4-bit + adapter | 2/5 | 62–80s | 想完全复刻训练配置 |
| **GGUF Q8_0（合并进反量化 4-bit）+ 语法约束** | **11/12** | **20–36s** | 推荐，线上用这套 |
| GGUF Q4_K_M（合并进原始 fp16） | 0/5 | — | 不可用，见 docs/FINDINGS.md |
