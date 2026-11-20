"""验证 LoRA 是否真的合并进了权重。

不靠属性链（PeftModel 包装后 .model.layers/.base_model.model.model.layers 各不相同），
直接遍历 named_parameters() 按名字前缀匹配 —— 对 base / peft / merged 三种形态都稳。
"""
from pathlib import Path
import hashlib
import os
import sys

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent  # 仓库根目录

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = str(ROOT / "model")
TARGETS = ("layers.0.self_attn.q_proj.weight", "layers.10.mlp.up_proj.weight")


def sig(model, tag):
    """返回 {目标张量名: (sha, mean|w|)}"""
    got = {}
    for name, param in model.named_parameters():
        for t in TARGETS:
            if name.endswith(t):
                w = param.detach().float().cpu()
                got[t] = (hashlib.sha256(w.numpy().tobytes()).hexdigest()[:16],
                          float(w.abs().mean()))
    if not got:
        print(f"[{tag}] 没匹配到目标张量，示例名字：",
              [n for n, _ in list(model.named_parameters())[:3]], file=sys.stderr)
    return got


print("loading base ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True)
r_base = sig(base, "base")

print("attaching adapter ...", flush=True)
pm = PeftModel.from_pretrained(base, ADAPTER)
r_peft = sig(pm, "peft")

print("merging ...", flush=True)
merged = pm.merge_and_unload()
r_merged = sig(merged, "merged")

print("\n" + "=" * 62)
verdict_ok = True
for t in TARGETS:
    if t not in r_base or t not in r_merged:
        print(f"{t}: 缺少数据，无法判断")
        verdict_ok = False
        continue
    hb, mb = r_base[t]
    hp, mp = r_peft.get(t, ("-", float("nan")))
    hm, mm = r_merged[t]
    print(f"{t}")
    print(f"   base    sha={hb}  mean|w|={mb:.6f}")
    print(f"   peft    sha={hp}  mean|w|={mp:.6f}")
    print(f"   merged  sha={hm}  mean|w|={mm:.6f}")
    changed = (hb != hm)
    print(f"   结论: {'LoRA 已合并（权重已改变）' if changed else 'LoRA 未生效（权重未变）'}")
    verdict_ok = verdict_ok and changed
print("=" * 62)
print("总体:", "MERGE OK" if verdict_ok else "MERGE FAILED")
