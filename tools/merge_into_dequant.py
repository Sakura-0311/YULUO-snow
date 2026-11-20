"""方案 B：把 LoRA 合并进「4-bit 反量化后的权重」，导出可在 llama.cpp 跑的全量 fp16 模型。

为什么这样做：
训练时底座是 nf4 双量化，LoRA 增量学的是「相对量化后底座」的偏移。
直接合并进原始 fp16 底座会换参考系（实测行为退化为基座）。
这里取 dequant(4bit) + scale * B@A，得到的才是训练时那个模型。

内存策略（机器只有 7.9G / 可用 ~6G，有 4G swap）：
  4-bit 底座 ≈ 1.1G + 全量 fp16 目标模型 ≈ 3.1G ≈ 4.2G，逐层合并后立即落盘，最后删 4-bit。
"""
import gc
import os
from pathlib import Path
import shutil

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
ROOT = Path(__file__).resolve().parent.parent  # 仓库根目录
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import bitsandbytes as bnbmo

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = str(ROOT / "model")
OUT = str(ROOT / "_merged4deq")

torch.set_num_threads(3)

print("1) 4-bit 底座 + adapter（与训练同配置）", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
m4 = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"": "cpu"},
    torch_dtype=torch.float16, low_cpu_mem_usage=True)
pm = PeftModel.from_pretrained(m4, ADAPTER)

print("2) 全量 fp16 骨架（将被逐层覆盖为 dequant+delta）", flush=True)
plain = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True)

print("3) 逐层合并", flush=True)
done, skipped, rel = 0, [], []
for name, mod in pm.named_modules():
    if not isinstance(mod, LoraLayer):
        continue
    if "default" not in getattr(mod, "lora_A", {}):
        skipped.append(name + " (无 default adapter)")
        continue
    plain_name = name.replace("base_model.model.", "", 1)
    try:
        target = plain.get_submodule(plain_name)
    except AttributeError:
        skipped.append(plain_name)
        continue

    # bnb 的 4-bit 权重是打包过的，直接 .dequantize() 会得到半尺寸张量
    # （把 2 个 4-bit 值当成 1 个 uint8 读）。必须显式带上 quant_state 解包。
    wq = mod.weight
    try:
        W = bnbmo.functional.dequantize_4bit(wq.data, wq.quant_state).float()
    except Exception:
        W = wq.dequantize().float()
    if W.shape[0] * 2 == wq.data.shape[0] if W.dim() == 2 else False:
        raise RuntimeError("解包失败，仍是半尺寸: " + str(W.shape))
    A = mod.lora_A["default"].weight.float()
    B = mod.lora_B["default"].weight.float()
    scale = float(mod.scaling["default"])
    delta = (B @ A) * scale
    merged = W + delta

    target.weight.data = merged.to(torch.float16)
    dn, wn = delta.norm().item(), W.norm().item()
    rel.append(dn / wn if wn else 0.0)
    done += 1

print(f"   合并模块 {done} 个，跳过 {len(skipped)}", flush=True)
if skipped[:3]:
    print("   跳过示例:", skipped[:3])
if rel:
    rel.sort()
    print(f"   相对变化 ||Δ||/||W||: min={rel[0]:.2e} "
          f"median={rel[len(rel)//2]:.2e} max={rel[-1]:.2e}", flush=True)

del pm, m4
gc.collect()

print("4) 保存 ->", OUT, flush=True)
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
plain.save_pretrained(OUT, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained(OUT)

size = sum(os.path.getsize(os.path.join(OUT, f))
           for f in os.listdir(OUT) if f.endswith(".safetensors"))
print(f"   权重 {size/1e9:.2f} GB", flush=True)
print("DONE")
