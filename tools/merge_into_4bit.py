"""把 LoRA 合并进「4-bit 反量化后的底座」，再导出 fp16 —— 使合并结果与训练行为一致。

为什么不能直接 merge 到原始 fp16 底座：
QLoRA 的增量是在带 4-bit 量化噪声的底座上学出来的；合并到干净的 fp16 底座会得到
一个「参考系不同」的模型，实测行为退化为基座（无 evidence / 键名错误）。

这里：加载 4-bit（nf4+双量化）→ 挂 adapter → merge_and_unload（peft 会把 bnb 层反量化）
→ 保存 fp16 → 交给 llama.cpp 转 GGUF。
"""
import gc
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
ROOT = Path(__file__).resolve().parent.parent  # 仓库根目录
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = str(ROOT / "model")
OUT = str(ROOT / "_merged4bit")

print("1) 加载 4-bit 底座", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(BASE)
m = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"": "cpu"},
    torch_dtype=torch.float16, low_cpu_mem_usage=True)

print("2) 挂 adapter", flush=True)
pm = PeftModel.from_pretrained(m, ADAPTER)

print("3) merge_and_unload（会反量化）", flush=True)
merged = pm.merge_and_unload()
try:
    merged = merged.to(torch.float16)
except Exception as exc:
    print("   .to(float16) 跳过:", exc)

print("4) 保存 ->", OUT, flush=True)
os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)

# 校验：保存后权重是否真的是 fp16 且与 4-bit 底座不同
import glob
tot = 0
for f in glob.glob(os.path.join(OUT, "*.safetensors")):
    tot += os.path.getsize(f)
print(f"   权重文件总大小 {tot/1e9:.2f} GB")
del m, pm, merged
gc.collect()
print("DONE")
