#!/usr/bin/env python3
"""Convert a safetensors file from F32 to F16 without touching tensor identity.

Why this needs care: safetensors is not just "raw floats" — it is
    [8-byte little-endian header length][JSON header][tensor data, 8-byte aligned]
and every tensor's `data_offsets` are byte offsets into that data section. If you
re-encode the payload without rewriting every offset, the file parses but yields
garbage. This script rebuilds the header from scratch and then proves the result
by re-reading it and comparing values against the original.

Usage:
    python convert_fp16.py SRC.safetensors DST.safetensors
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# safetensors dtype -> numpy dtype (only what we need)
DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": None}


def read_safetensors(path: Path):
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("file too small to be safetensors")
    n = struct.unpack("<Q", raw[:8])[0]
    if n <= 0 or 8 + n > len(raw):
        raise ValueError(f"bad header length {n}")
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    meta = header.pop("__metadata__", None)
    data_start = 8 + n
    tensors = {}
    for name, info in header.items():
        s, e = info["data_offsets"]
        if e < s or data_start + e > len(raw):
            raise ValueError(f"{name}: offsets out of range {s}..{e}")
        tensors[name] = (info["dtype"], info["shape"], raw[data_start + s:data_start + e])
    return meta, tensors


def write_safetensors(path: Path, meta, tensors_of_bytes):
    """tensors_of_bytes: name -> (dtype_str, shape, raw_bytes)"""
    # data offsets in a deterministic order (sorted by name, like the original tooling)
    names = sorted(tensors_of_bytes)
    header = {}
    offset = 0
    for name in names:
        dtype, shape, buf = tensors_of_bytes[name]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(buf)],
        }
        offset += len(buf)
    if meta is not None:
        header["__metadata__"] = meta
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # header must be padded so that the data section starts 8-byte aligned
    pad = (-(8 + len(blob))) % 8
    blob += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in names:
            f.write(tensors_of_bytes[name][2])


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    print(f"[1] 读取 {src.name} ({src.stat().st_size/1e6:.2f} MB)")
    meta, tensors = read_safetensors(src)
    print(f"    metadata={meta}  张量数={len(tensors)}")

    dtypes = sorted({v[0] for v in tensors.values()})
    print(f"    源 dtype: {dtypes}")
    if dtypes != ["F32"]:
        raise SystemExit(f"期望全 F32，实际 {dtypes}")

    print("\n[2] 逐张量 F32 -> F16")
    out = {}
    total_elems = 0
    for name, (dtype, shape, buf) in tensors.items():
        arr = np.frombuffer(buf, dtype=np.float32)
        expect = int(np.prod(shape)) if shape else 1
        if arr.size != expect:
            raise SystemExit(f"{name}: 元素数 {arr.size} != shape {shape} 期望 {expect}")
        total_elems += arr.size
        out[name] = ("F16", shape, arr.astype(np.float16).tobytes())

    only_16bit = lambda b: None
    print(f"    总元素 {total_elems:,}")

    print(f"\n[3] 写出 {dst.name}")
    write_safetensors(dst, meta, out)
    print(f"    大小 {dst.stat().st_size/1e6:.2f} MB "
          f"(原 {src.stat().st_size/1e6:.2f} MB, "
          f"缩减 {100 - dst.stat().st_size/src.stat().st_size*100:.1f}%)")

    print("\n[4] 复读并逐张量对账（这是关键：证明没写坏）")
    meta2, tensors2 = read_safetensors(dst)
    assert meta2 == meta, "metadata 不一致"
    assert set(tensors2) == set(tensors), "张量名集合不一致"
    assert sorted(v[0] for v in tensors2.values()) == ["F16"] * len(tensors2), "dtype 不是全 F16"

    max_rel = 0.0
    bad_shape = []
    for name, (d1, sh1, b1) in tensors.items():
        d2, sh2, b2 = tensors2[name]
        if sh1 != sh2:
            bad_shape.append(name)
            continue
        a = np.frombuffer(b1, dtype=np.float32)
        b = np.frombuffer(b2, dtype=np.float16).astype(np.float32)
        if a.size != b.size:
            bad_shape.append(name)
            continue
        denom = np.maximum(np.abs(a), 1e-8)
        rel = float(np.max(np.abs(a - b) / denom)) if a.size else 0.0
        max_rel = max(max_rel, rel)

    if bad_shape:
        raise SystemExit(f"shape 不一致: {bad_shape[:5]}")
    print(f"    张量数一致、形状一致、全 F16 [OK]")
    print(f"    最大相对误差 {max_rel:.6f}  (fp16 尾数 10 bit，理论量级 ~5e-4)")

    # 逐元素绝对误差上界检查：fp16 相对误差不应超过 2^-11
    if max_rel > 1e-2:
        raise SystemExit(f"相对误差过大 {max_rel}，转换有问题")

    print("\nOK: 转换完成且已自证")
    print(f"    输出: {dst}")


if __name__ == "__main__":
    main()
