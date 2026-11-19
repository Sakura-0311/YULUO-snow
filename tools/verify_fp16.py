#!/usr/bin/env python3
"""Characterize an F32 -> F16 safetensors conversion honestly.

The naive metric max(|a-b|/max(|a|,eps)) is useless for LoRA weights: lora_B starts
at zero and many trained entries stay tiny, so anything that underflows to 0 in
fp16 reports a 100% "error" while being numerically irrelevant. This script
reports:
  * max absolute error and its location
  * max relative error restricted to elements with |a| above a threshold
  * how many non-zero elements flushed to exactly 0 (underflow count)
  * inf / nan violations
  * per-tensor worst case, so a single bad tensor cannot hide in the average
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_st(path: Path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    header.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for name, info in header.items():
        s, e = info["data_offsets"]
        out[name] = (info["dtype"], info["shape"], raw[base + s:base + e])
    return out


def main() -> None:
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    A, B = read_st(a_path), read_st(b_path)
    print(f"源 {a_path.name}: {len(A)} 张量")
    print(f"新 {b_path.name}: {len(B)} 张量\n")
    assert set(A) == set(B)

    worst_abs = (0.0, "")
    worst_rel = (0.0, "")
    total_elems = 0
    flushed = 0
    nonzero_src = 0
    inf_nan = 0
    REL_FLOOR = 1e-4          # 只对量级够大的元素算相对误差
    rows = []

    for name in A:
        d1, sh1, b1 = A[name]
        d2, sh2, b2 = B[name]
        assert sh1 == sh2, f"{name} shape 变化"
        a = np.frombuffer(b1, dtype=np.float32)
        b = np.frombuffer(b2, dtype=np.float16).astype(np.float32)
        assert a.size == b.size

        total_elems += a.size
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if diff.size:
            i = int(np.argmax(diff))
            if diff[i] > worst_abs[0]:
                worst_abs = (float(diff[i]), name)

        big = np.abs(a) >= REL_FLOOR
        if big.any() and diff.size:
            rel = diff[big] / np.abs(a[big]).astype(np.float64)
            j = int(np.argmax(rel))
            idx = np.flatnonzero(big)[j]
            if rel[j] > worst_rel[0]:
                worst_rel = (float(rel[j]), f"{name}[{idx}] a={a[idx]:.6g} b={b[idx]:.6g}")

        nz = a != 0
        nonzero_src += int(nz.sum())
        flushed += int((nz & (b == 0)).sum())
        inf_nan += int((~np.isfinite(b)).sum())

        if diff.size:
            rows.append((float(diff.max()), name, a.size))

    rows.sort(reverse=True)
    print(f"总元素            {total_elems:,}")
    print(f"源非零元素        {nonzero_src:,}")
    print(f"下溢为 0 的元素    {flushed:,}  ({flushed/max(nonzero_src,1)*100:.3f}% of 非零)")
    print(f"inf/nan 违规      {inf_nan}")
    print(f"\n最大绝对误差      {worst_abs[0]:.3e}   @ {worst_abs[1]}")
    print(f"最大相对误差(≥{REL_FLOOR:g}) {worst_rel[0]:.3e}   @ {worst_rel[1]}")
    print("\n绝对误差最大的 5 个张量:")
    for v, n, c in rows[:5]:
        print(f"  {v:.3e}  {n}  ({c} 元素)")

    ok = (inf_nan == 0 and worst_rel[0] < 2e-3 and flushed / max(nonzero_src, 1) < 0.02)
    print("\n判定:", "PASS" if ok else "REVIEW")
    print("  依据: 无 inf/nan；量级≥1e-4 的元素相对误差 < 2e-3（fp16 理论 ~4.9e-4）；"
          "下溢比例 < 2%")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
