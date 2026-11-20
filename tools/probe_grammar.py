"""探测 llama-server 是否支持 response_format=json_schema（受语法约束的解码）。

若支持，把我们的 commitment_schema 交给它，可以在解码阶段就排除
「自创字段 / 缺 required / 枚举越界」这些结构错误。
"""
from pathlib import Path
import json
import sys
import urllib.request

sys.path.insert(0, str(ROOT / "src"))
from generate_synthetic_data import SYSTEM_PROMPT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # 仓库根目录

URL = os.environ.get("YULUO_BACKEND", "http://127.0.0.1:8081/v1/chat/completions")
SCHEMA = json.loads((ROOT / "schema" / "commitment_schema_v1.json").read_text(encoding="utf-8"))
TEXT = "首月 9.9 元，次月起每月 29 元自动续费，可随时取消，已付费用不予退还。"
NL = chr(10)


def call(with_schema: bool):
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "source_id: g" + NL + "待抽取文本：" + NL + TEXT},
        ],
        "max_tokens": 900,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    if with_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "commitment", "schema": SCHEMA,
                            "strict": True},
        }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"]["content"], None
    except Exception as exc:
        detail = ""
        if hasattr(exc, "read"):
            try:
                detail = exc.read().decode()[:300]
            except Exception:
                pass
        return None, f"{type(exc).__name__}: {exc} {detail}"


print("schema 顶层 required:", SCHEMA.get("required"))
print("schema 大小:", len(json.dumps(SCHEMA)), "字节")
print()

print("=== A. 不加 grammar ===")
out, err = call(False)
if err:
    print("  失败:", err)
else:
    print("  len:", len(out), "| evidence:", "evidence" in out)
    try:
        obj = json.loads(out)
        extra = [k for k in obj if k not in SCHEMA.get("properties", {})]
        print("  顶级多余键:", extra)
    except Exception as e:
        print("  JSON 解析失败:", e)

print()
print("=== B. 加 response_format=json_schema ===")
out2, err2 = call(True)
if err2:
    print("  不支持或失败:", err2)
else:
    print("  len:", len(out2), "| evidence:", "evidence" in out2)
    ok = True
    try:
        obj2 = json.loads(out2)
        extra = [k for k in obj2 if k not in SCHEMA.get("properties", {})]
        print("  顶级多余键:", extra)
        missing = [k for k in SCHEMA.get("required", []) if k not in obj2]
        print("  缺 required:", missing)
        ok = not extra and not missing
    except Exception as e:
        print("  JSON 解析失败:", e)
        ok = False
    print("  结构是否完全符合 schema:", ok)
    print("  前 200 字:", out2[:200].replace(NL, " "))
