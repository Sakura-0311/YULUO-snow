# YULUO-snow

**消费条款结构化抽取模型**：把电商 / 会员页里的促销与小字条款，转成机器可校验的 JSON —— 金额统一成分、时间统一成天、每个结论都带原文逐字引用和字符位置；原文没写的项一律标 `unknown`，不替商家做任何推断。

> 项目「承诺单：下单前看清账」（2026 Qoder 高校创造力大赛 · LifeHack）。它不是聊天模型，不回答「这个划算吗」，只做一件事：**读懂条款 → 吐出可核对的字段 + 原文证据**。

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![Base](https://img.shields.io/badge/base-Qwen2.5--1.5B--Instruct-blue)
![Adapter](https://img.shields.io/badge/LoRA-r16%20F32%2073.91MB-green)

---

## 效果（100 条留出样本，Kaggle T4 x2 实测）

同一批样本、同一套指标，底座零样本 vs YULUO-snow（经 `spanfix` 重锚定后）：

| 指标 | 底座零样本 | **YULUO-snow** | Δ |
| --- | ---: | ---: | ---: |
| JSON 可解析率 | 0.8800 | **0.9600** | +0.0800 |
| 字段精确匹配 | 0.3431 | **0.8654** | **+0.5223** |
| 未知 / 歧义识别 F1 | 0.0000 | **0.9170** | +0.9170 |
| 未知项脑补幻觉率 | 1.0000 | **0.0000** | **−1.0000** |
| 证据覆盖率 | 0.0000 | **0.7422** | +0.7422 |
| 原文证据 Span F1（重锚定后） | 0.0000 | **0.8429** | +0.8429 |
| 结构严格合法率（全或无） | 0.0000 | 0.0000 | — |

严格合法率为什么是 0：该指标全或无——一条样本里任意一处缺陷整条判 0。实测 336 个证据 span 中 141 个偏移本来就精确、192 个由 `spanfix.reanchor` 确定性救回、仅 3 个不可恢复（span 可解决率 99.1%），但 100 条里每条都至少有一处缺陷（4 条 JSON 解析失败 + 3 个不可恢复 span + 部分字段漏证据）。详细分析见 `reports/ablation_report_verified.json` 与 `README-模型说明书.md` §7。

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

需要 `torch`、`transformers>=4.45`、`peft>=0.12`、`bitsandbytes>=0.43`、`accelerate`、`jsonschema`。4-bit 加载需要 CUDA；纯 CPU 可跑（慢，且需把 `load_in_4bit` 关掉）。

### 加载（底座 + adapter）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # 底座需另行获取，或用完整版里的 base_model/

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
    device_map={"": 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(base, "model").eval()   # model/ 即本仓库的 adapter
```

完整可运行版本见 `使用示例.py`。

### 推理流程（生成 → 重锚定 → 校验 → 算钱）

```python
import sys, json
sys.path.insert(0, "src")
from infer import extract_json_block
from spanfix import reanchor          # 必须用：修正模型输出不准的字符偏移
from validator import validate_extraction
from calculator import compute_commitment_summary
from pathlib import Path

pred = extract_json_block(raw_output)                 # 1) 模型输出
pred, stats = reanchor(pred, source_text)             # 2) 确定性重锚定（关键一步）
ok, errs = validate_extraction(pred, source_text, "demo_001",
                               Path("schema/commitment_schema_v1.json"))  # 3) 严格校验
calc = compute_commitment_summary(pred, order_date_iso="2026-11-19") if ok else None  # 4) 算钱交给代码
```

> **必读**：模型擅长「引用哪一句」，不擅长「这句在第几个字符」。**永远不要直接信任模型给的 `start_char` / `end_char`**，一律用 `spanfix.reanchor()` 拿 quote 在原文里重新定位。不做这一步，Span F1 会从 0.8429 掉到 0.3550。

关键约定：金额一律整数「分」（19.9 元 → `1990`）；`status` 三态 `stated` / `unknown` / `contradicted`；`unknown` 时 `evidence` 必须为空，且不得外推为「否」；总额计算一律走 `src/calculator.py`，不要让模型自己算钱。

## 目录说明

```
.
├── README.md                     ← 本文件
├── README-模型说明书.md           ← 完整模型说明书（规格 / 指标 / 局限）
├── LICENSE                       ← CC BY-NC-SA 4.0 全文
├── NOTICE                        ← 许可适用范围（项目 vs 底座）
├── 使用示例.py                    ← 可直接运行的最小示例
├── requirements.txt
├── model/                        ← LoRA adapter（F32 完整版，73.91 MB）+ tokenizer
├── src/                          ← 推理与后处理：infer / validator / spanfix / calculator / evaluate / 数据生成器
├── schema/                       ← 输出 JSON Schema（commitment_schema_v1.json）
├── ontology/                     ← 消费承诺本体（consumer_commitment_ontology_v1.yaml）
├── benchmark/                    ← 15 例人工边界基准
├── reports/                      ← 消融报告（含已核验版）
├── tests/                        ← spanfix 测试（5/5）与规则基线测试（10/15）
└── tools/                        ← F32→F16 转换与逐张量对账工具
```

技术规格速览：LoRA r=16 / alpha=32 / dropout=0.05，可训练参数 18,464,768（1.18%）；4-bit NF4 QLoRA，2 epochs，126 步，最终 loss 0.0020；训练数据 1000 条确定性合成样本（seed=42，零真实用户数据），留出 100 条按模板族分组切分。

## 完整版下载（含 Qwen 基座，离线可用）

本仓库只含 adapter 与代码。**完整版**把底座 `Qwen/Qwen2.5-1.5B-Instruct` 一并打包（`base_model/` 目录），下载后无需再访问 Hugging Face。请到本仓库 **Releases** 下载：

- `YULUO-snow-complete-with-qwen-base.zip.part-aa`
- `YULUO-snow-complete-with-qwen-base.zip.part-ab`
- `SHA256SUMS.txt`（每片与合并后 zip 的 sha256）

合并与校验：

```bash
cat YULUO-snow-complete-with-qwen-base.zip.part-* > YULUO-snow-complete-with-qwen-base.zip
sha256sum -c SHA256SUMS.txt
unzip YULUO-snow-complete-with-qwen-base.zip
```

> 资产名与分片数以 Release 页面实际发布为准（当前为占位说明，发布时按实际分片调整）。

## 许可

- **本项目**（adapter 权重、代码、文档、合成数据说明）：**CC BY-NC-SA 4.0** —— 署名、**禁止商业用途**、衍生作品相同方式共享。全文见 `LICENSE`。
- **底座模型** `Qwen/Qwen2.5-1.5B-Instruct`：遵循其原始 **Apache-2.0** 许可，**不受上述非商业条款约束**；完整版 `base_model/` 内附带其许可文件。范围划分见 `NOTICE`。
- 训练数据 100% 由确定性生成器合成，不含任何真实用户订单、账号或账单。

## 已知局限

只在合成数据上训练，真实页面存在分布差异；`max_len=2048`，超长条款需分段；不做法律判断；`unknown` 是特性不是缺陷——宁可少说，不说错。详见 `README-模型说明书.md` §8。
