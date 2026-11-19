#!/usr/bin/env python3
"""
Inference pipeline for "承诺单：下单前看清账".
Loads base model (+ optional LoRA adapter), extracts structured JSON,
runs strict schema + span validation, and passes extracted fields to
the deterministic calculator (calculator.py).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from calculator import compute_commitment_summary
from generate_synthetic_data import SYSTEM_PROMPT
from validator import validate_extraction


def extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw_text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def run_hf_inference(
    base_model: str,
    adapter_dir: Optional[str],
    source_id: str,
    input_text: str,
    max_new_tokens: int = 1024,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"source_id: {source_id}\n待抽取文本：\n{input_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction + validation + deterministic calculation.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--source-id", default="cli_input_001")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--order-date", default="2026-11-18")
    parser.add_argument("--schema", type=Path, default=Path("schema/commitment_schema_v1.json"))
    args = parser.parse_args()

    text = args.input_file.read_text(encoding="utf-8")
    raw_output = run_hf_inference(args.base_model, args.adapter_dir, args.source_id, text)
    parsed = extract_json_block(raw_output)
    if parsed is None:
        print(json.dumps({"valid": False, "error": "JSON parse failed", "raw": raw_output}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    ok, errs = validate_extraction(parsed, text, args.source_id, args.schema)
    calc = compute_commitment_summary(parsed, order_date_iso=args.order_date) if ok else None
    print(json.dumps({
        "valid": ok,
        "validation_errors": errs,
        "extracted": parsed,
        "deterministic_calculation": calc,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
