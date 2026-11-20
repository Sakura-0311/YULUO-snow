#!/usr/bin/env python3
"""YULUO-snow 最小可运行示例。

用法：
    python 使用示例.py                     # 跑内置的 3 条演示条款
    python 使用示例.py --text-file a.txt   # 从文件读一段条款
    python 使用示例.py --no-4bit           # 没有 GPU 时用 fp32 加载（慢）

流程：模型生成 → 确定性 Span 重锚定 → 严格校验 → 确定性算钱。
注意：reanchor 不是可选项。模型给的字符偏移不可靠，必须用原文重新定位。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = HERE / "model"
SCHEMA = HERE / "schema" / "commitment_schema_v1.json"

DEMO = [
    ("demo_trial", "0 元免费试用 7 天，试用结束后自动转为月付 25 元，"
                   "已支付的费用不予退还。"),
    ("demo_firstmonth", "首月特价 0.1 元，次月起每 30 天 25 元自动续费，"
                        "随时可取消，取消后权益保留至当前周期结束。"),
    ("demo_conflict", "横幅写着「每月仅 19.9 元」，但下方细则说明"
                      "首月 19.9 元、之后每月 29.9 元。"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", type=Path, default=None)
    ap.add_argument("--no-4bit", action="store_true", help="不用 4-bit（CPU 调试用）")
    ap.add_argument("--max-new-tokens", type=int, default=704)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    from calculator import compute_commitment_summary
    from generate_synthetic_data import SYSTEM_PROMPT
    from infer import extract_json_block
    from spanfix import reanchor
    from validator import validate_extraction

    print(f"[1/3] 加载底座 {MODEL_ID} + YULUO-snow adapter ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.no_4bit:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
        dev = "cpu"
    else:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True),
            device_map={"": 0}, trust_remote_code=True)
        dev = "cuda:0"

    model = PeftModel.from_pretrained(base, str(ADAPTER)).eval()
    eos_ids = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end > 0 and im_end not in eos_ids:
        eos_ids.append(im_end)

    if args.text_file:
        cases = [(args.text_file.stem, args.text_file.read_text(encoding="utf-8"))]
    else:
        cases = DEMO

    print(f"[2/3] 推理 {len(cases)} 条条款 ...")
    results = []
    for sid, body in cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"source_id: {sid}\n待抽取文本：\n{body}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=eos_ids)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True)

        parsed = extract_json_block(raw)
        if parsed is None:
            results.append({"source_id": sid, "valid": False,
                            "error": "JSON 解析失败", "raw": raw[:400]})
            continue

        anchored, stats = reanchor(parsed, body)
        ok, errs = validate_extraction(anchored, body, sid, SCHEMA)
        calc = compute_commitment_summary(anchored, order_date_iso="2026-11-19") if ok else None
        results.append({
            "source_id": sid,
            "valid": ok,
            "validation_errors": errs,
            "span_reanchor": stats,
            "extracted": anchored,
            "total_cost": calc,
        })

    print("[3/3] 结果\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    out_path = HERE / "示例输出.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n已写入 {out_path.name}")


if __name__ == "__main__":
    main()
