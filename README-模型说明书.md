# YULUO-snow · 模型说明书

> 版本 **v1.0** ｜ 2026-11-19 ｜ 项目「承诺单：下单前看清账」（2026 Qoder 高校创造力大赛 · LifeHack）

---

## 1. 一句话说明

**YULUO-snow 是一个消费条款结构化抽取模型**：把电商/会员页里的促销与小字条款，转成机器可校验的 JSON —— 金额统一成分、时间统一成天、每个结论都带原文逐字引用和字符位置；原文没写的项一律标 `unknown`，不替商家做任何推断。

它不是聊天模型，**不要指望它回答"这个划算吗"**。它只做一件事：**读懂条款 → 吐出可核对的字段 + 原文证据**。

---

## 2. 它解决什么问题

| 用户痛点 | YULUO-snow 的输出 |
| --- | --- |
| "首月 0.1 元"之后到底多少钱 | `initial_charge` + `recurring_charge` 分字段列出，含周期 |
| 到期会不会自动续费 | `renewal_interval` + 自动续费标志 |
| 30 / 90 / 365 天真实支出 | 模型给结构，`calculator.py` 用整数「分」确定性累加 |
| 条款原文在哪一句 | 每个字段带 `quote` + `[start_char, end_char)` |
| 页面没提退款怎么办 | 输出 `status="unknown"`，**不猜"不退"** |
| 两处价格互相打架 | 标 `contradicted`，阻断总额计算 |

---

## 3. 技术规格

| 项 | 值 |
| --- | --- |
| 模型名 | **YULUO-snow** |
| 类型 | LoRA Adapter（PEFT），非全量模型 |
| 底座 | `Qwen/Qwen2.5-1.5B-Instruct`（需另行获取） |
| 量化 | 4-bit NF4 + double quant（加载时） |
| LoRA 配置 | r=16, alpha=32, dropout=0.05 |
| 可训练参数 | 18,464,768 / 1,562,179,072 = **1.18%** |
| 训练 | 2 epochs, lr=2e-4, max_len=2048, 126 optimizer steps |
| 最终训练 loss | **0.0020** |
| 训练数据 | 1000 条**确定性合成**样本（生成器 seed=42），**零真实用户数据** |
| 留出评测 | 100 条（按模板族分组切分，与训练无模板重叠） |
| 训练/评测硬件 | Kaggle GPU T4 x2（免费额度） |
| 输出格式 | 严格 JSON，`schema_version 1.0.0` |
| 权重精度 | **F32（本包为完整版）**，73.91 MB |

**Adapter 权重文件**：`model/adapter_model.safetensors` — **73.91 MB**（F32 完整版）

### 3.1 关于精度与体积

| 精度 | 权重文件 | 说明 |
| --- | --- | --- |
| **F32（本包 = 完整版）** | **73.91 MB** | peft 默认存盘精度，原始训练产物，零精度损失 |
| F16（精简版，另行发布） | 36.98 MB | 由 F32 转换；指 adapter 权重存盘精度——训练同配置（4-bit 底座）下计算本就按 fp16，无实质损失；**不代表底座可改用 fp16 加载（会退化，见 §8 第 7 条）** |

转换不是简单改 dtype —— safetensors 头部记录每个张量的 `data_offsets`，必须整体重建。转换后用 `tools/verify_fp16.py` 逐张量对账，实测：

- inf / nan：**0**
- 下溢归零：**171 / 18,464,768 = 0.001%**
- 最大绝对误差：**1.52e-05**
- 量级 ≥1e-4 的元素最大相对误差：**4.880e-04 = 2⁻¹¹**（正好是 fp16 10 位尾数的一个 ULP，即理论下限）

本包即 F32 完整版，权重就是原始训练产物，无需另行回取其他版本；上表对账数据为 F16 精简版转换时的逐张量核验结果。

---

## 4. 目录结构

```
YULUO-snow/
├── README-模型说明书.md          ← 本文件
├── 使用示例.py                   ← 可直接运行的最小示例
├── requirements.txt
├── model/                        ← LoRA adapter + tokenizer
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
├── src/                          ← 推理与后处理代码
│   ├── infer.py                  ← 单条推理入口
│   ├── validator.py              ← JSON Schema + 原文 Span 严格校验
│   ├── spanfix.py                ← 确定性 Span 重锚定（关键）
│   ├── calculator.py             ← 30/90/365 天确定性算术
│   ├── evaluate.py               ← 严格指标
│   └── generate_synthetic_data.py← 训练数据生成器（可复现）
├── schema/commitment_schema_v1.json
├── ontology/consumer_commitment_ontology_v1.yaml
├── benchmark/benchmark_cases.json ← 15 例人工边界基准
├── reports/
│   ├── ablation_report.json
│   └── ablation_report_verified.json
├── tests/
│   ├── test_spanfix.py
│   └── test_rule_baseline.py
└── tools/
    ├── convert_fp16.py           ← F32→F16 转换（重建 safetensors 偏移表）
    └── verify_fp16.py            ← 转换后逐张量对账
```

---

## 5. 快速开始

### 5.1 安装

```bash
pip install -r requirements.txt
```

需要：`torch`、`transformers>=4.45`、`peft>=0.12`、`bitsandbytes>=0.43`、`accelerate`、`jsonschema`。
4-bit 加载需要 CUDA；**纯 CPU 不要把 `load_in_4bit` 关掉改用 fp32/fp16 底座——能加载但行为退化、输出不可信（见 §8 第 7 条）**。

### 5.2 最小调用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch, json, sys
sys.path.insert(0, "src")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
    device_map={"": 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(base, "model").eval()
```

完整可运行版本见 `使用示例.py`。

### 5.3 推理（推荐流程：生成 → 重锚定 → 校验 → 算钱）

```python
from infer import extract_json_block
from spanfix import reanchor          # 必须用：修正模型输出不准的字符偏移
from validator import validate_extraction
from calculator import compute_commitment_summary
from pathlib import Path

text = """首月仅 0.1 元，之后每 30 天自动续费 25 元，已支付金额不退。"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},          # 见 src/generate_synthetic_data.py
    {"role": "user", "content": f"source_id: demo_001\n待抽取文本：\n{text}"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to("cuda:0")
out = model.generate(**ids, max_new_tokens=704, do_sample=False,
                     pad_token_id=tok.pad_token_id, eos_token_id=[151645])
raw = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

pred = extract_json_block(raw)                  # 1) 模型输出
pred, stats = reanchor(pred, text)              # 2) 确定性重锚定（关键一步）
ok, errs = validate_extraction(pred, text, "demo_001",
                               Path("schema/commitment_schema_v1.json"))  # 3) 严格校验
calc = compute_commitment_summary(pred, order_date_iso="2026-11-19") if ok else None
print(json.dumps({"valid": ok, "errors": errs, "extracted": pred,
                  "total": calc}, ensure_ascii=False, indent=2))
```

> **必读**：模型擅长「引用哪一句」，不擅长「这句在第几个字符」。**永远不要直接信任模型给的 `start_char`/`end_char`**，一律用 `spanfix.reanchor()` 用 quote 在原文里重新定位。不做这一步，严格合法率会从接近满值掉到 0。

---

## 6. 输入 / 输出

### 输入
- 一段**原始中文条款文本**（几十到几千字），不做任何预处理
- 一个 `source_id` 字符串（用于回填证据的归属）

### 输出（JSON 摘要）

```json
{
  "schema_version": "1.0.0",
  "source_id": "demo_001",
  "product_or_service": { "name": "...", "category": "...", "status": "stated",
                          "evidence": [{"quote": "...", "start_char": 0, "end_char": 6}] },
  "initial_charge":     { "amount_fen": 10, "status": "stated", "evidence": [...] },
  "recurring_charge":   { "amount_fen": 2500, "status": "stated", "evidence": [...] },
  "renewal_interval":   { "interval_days": 30, "auto_renew": true, "status": "stated", ... },
  "refund_status":      { "status": "unknown", "evidence": [] },
  "uncertainties_and_conflicts": [ ... ]
}
```

关键约定：
- **金额一律整数「分」**：19.9 元 → `1990`
- **`status` 三态**：`stated`（原文写明）/ `unknown`（原文没提）/ `contradicted`（前后矛盾）
- **`status="unknown"` 时 `evidence` 必须为空**，且不得外推为「否」
- 每个非 unknown 结论必须带 ≥1 个 `evidence`，且 `原文[start_char:end_char] == quote`

---

## 7. 实测效果（100 条留出样本，Kaggle T4 x2）

同一批样本、同一套指标，底座零样本 vs YULUO-snow：

| 指标 | 底座零样本 | **YULUO-snow** | Δ |
| --- | ---: | ---: | ---: |
| JSON 可解析率 | 0.8800 | **0.9600** | +0.0800 |
| 字段精确匹配 | 0.3431 | **0.8654** | **+0.5223** |
| 未知/歧义识别 F1 | 0.0000 | **0.9170** | **+0.9170** |
| 未知项脑补幻觉率 | 1.0000 | **0.0000** | **−1.0000** |
| 证据覆盖率 | 0.0000 | **0.7422** | +0.7422 |
| 原文证据 Span F1（重锚定后） | 0.0000 | **0.8429** | +0.8429 |
| 原文证据 Span F1（模型原始偏移） | 0.0000 | 0.3550 | +0.3550 |
| 结构严格合法率（全或无） | 0.0000 | 0.0000 | +0.0000 |

### 为什么「结构严格合法率」是 0 —— 请务必理解这条

该指标是**全或无**的：一条样本里只要有任意一个字段缺证据、或任意一个 span 偏移不精确，整条判 0。

实测细节：
- 底座零样本在 100 条里**一个 evidence span 都不产出** → Span F1 与合法率自然为 0
- YULUO-snow 产出 **336 个 span**：**141 个（42%）偏移本来就精确**，**192 个**由 `spanfix.reanchor` 确定性救回，仅 **3 个**不可恢复 → **span 可解决率 99.1%**
- 但 100 条里每条都至少有一处缺陷（4 条 JSON 解析失败 + 3 个不可恢复 span + 部分字段漏证据），所以全或无指标仍为 0

**结论**：模型确实学会了「读懂并引用条款」，短板只在「数第几个字符」——这一项我们用确定性代码兜住了。

- 复现命令：`python tests/test_spanfix.py`（含「偏移 +7 复现失败模式 → 重锚定恢复 1.0」的完整证明）
- 规则基线对照：`python tests/test_rule_baseline.py`（15 例人工基准，10/15）

---

## 8. 已知局限（请如实告知下游）

1. **只在合成数据上训练**：真实电商页面措辞更散、表格/图片更多，分布差异明显。上线前必须用真实页面语料做增量评估。
2. **长文本会截断**：训练与推理 `max_len=2048`，且实测样本最长 1259 token。超过这个长度的条款页需要分段后合并。
3. **不做算术**：模型只输出结构化金额与周期；总额一律交给 `calculator.py`。**不要让模型自己算钱**。
4. **不做法律判断**：它不评价条款是否合法、是否霸王条款。
5. **`unknown` 是特性不是缺陷**：宁可少说，不说错。下游不要把 `unknown` 当成"没有此条款"。
6. **1.5B 小模型的边界**：复杂嵌套条件（如"满 3 件且非特价商品方可用券"）识别率会明显下降，建议配合规则。
7. **adapter 与 4-bit 底座耦合（2026-11-20 部署实测）**：LoRA 增量是在带量化噪声的 4-bit 底座上学出来的（合并后权重相对变化仅约 2.7e-3）。实测：底座改 fp16 加载 + adapter 输出退化；`merge_and_unload` 合并到 fp16 再转 GGUF q4 会退化成基座行为。transformers 4.46 与 5.x 均复现，排除版本因素。**只能按训练同配置（4-bit NF4 + double quant + compute fp16）加载使用，目前没有可用的合并版 / GGUF 部署路径。**注意 §3.1 说的「fp16 无实质损失」只指 adapter 权重自身的存盘精度（F16 vs F32），不是说底座可以用 fp16 加载。
8. **提示词高度敏感**：必须使用与训练完全一致的 `SYSTEM_PROMPT`，不要追加约束或改写。部署抽测（5 例）中追加约束提示后严格通过率从 2/5 掉到 0/5（span 产出从 19 个掉到 2 个、字段开始缺 `status`）。

---

## 9. 许可与数据声明

- **本项目以 CC BY-NC-SA 4.0（署名—非商业性使用—相同方式共享）发布**：本仓库的 adapter 权重、代码、文档与合成数据说明均适用该许可，**禁止商业用途**；使用时须署名，衍生作品须以相同许可发布。全文见仓库根目录 `LICENSE`，适用范围说明见 `NOTICE`。
- 训练数据 100% 由确定性生成器合成（`src/generate_synthetic_data.py`，seed=42），**不含任何真实用户订单、账号或账单**
- 本 adapter 仅可配合 `Qwen/Qwen2.5-1.5B-Instruct` 底座使用；**底座遵循其原始 Apache-2.0 许可，不受上述非商业条款约束**（见 `NOTICE`）
- 用于「承诺单：下单前看清账」项目及其衍生场景

---

## 10. 版本记录

| 版本 | 日期 | 变化 |
| --- | --- | --- |
| v1.0 | 2026-11-19 | 首个可用版本：Qwen2.5-1.5B 4-bit QLoRA，126 步，loss 0.0020；引入 `spanfix` 确定性重锚定；完成 100 条留出消融 |
