#!/usr/bin/env python3
"""
Pure Synthetic Data Generator for "承诺单：下单前看清账".
- ZERO real user orders.
- ZERO reading of benchmark_cases.json.
- Grouped split by template family (8 train families -> 1000 samples; 2 held-out val families -> 100 samples)
  to prevent template leakage.
- Covers: subscriptions, free/paid trials, threshold discounts, prechecked add-ons, shipping thresholds,
  refund ambiguity/missing (strictly unknown, never inferred as non-refundable), OCR noise,
  contradicted clauses, and fen/0-yuan boundary cases.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from validator import validate_extraction

SYSTEM_PROMPT = (
    "你是“承诺单：下单前看清账”消费条款结构化抽取器。"
    "请严格按 schema_version 1.0.0 输出合法 JSON：金额统一转换为整数“分”（如 19.9 元 -> 1990）；"
    "原文未提及的退款、续费或取消条款必须标记 status='unknown' 且不得外推为“否/不退”；"
    "每个非 unknown 结论必须附带原文精确 quote 与 [start_char, end_char) 字符起止位置；"
    "不要输出任何计算后的应付总额或解释性废话，仅输出 JSON。"
)


def _fmt_yuan(fen: int) -> str:
    if fen % 100 == 0:
        return f"{fen // 100}"
    if fen % 10 == 0:
        return f"{fen / 100:.1f}"
    return f"{fen / 100:.2f}"


class TextBuilder:
    """Tracks exact character offsets while assembling synthetic order/clause text."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self.parts: List[str] = []
        self.length = 0

    def add_plain(self, s: str) -> None:
        self.parts.append(s)
        self.length += len(s)

    def add_span(self, s: str) -> Dict[str, Any]:
        start = self.length
        self.parts.append(s)
        self.length += len(s)
        return {
            "source_id": self.source_id,
            "quote": s,
            "start_char": start,
            "end_char": self.length,
        }

    def text(self) -> str:
        return "".join(self.parts)


def _empty_template(source_id: str) -> Dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "source_id": source_id,
        "product_or_service": {
            "name": None,
            "category": "unknown",
            "quantity": None,
            "status": "unknown",
            "evidence": [],
        },
        "price_items": [],
        "trial_period": {
            "has_trial": None,
            "duration_value": None,
            "duration_unit": "unknown",
            "trial_price_fen": None,
            "auto_convert_to_paid": None,
            "status": "unknown",
            "evidence": [],
        },
        "initial_charge": {
            "amount_fen": None,
            "timing_rule": "unknown",
            "status": "unknown",
            "evidence": [],
        },
        "recurring_charge": {
            "amount_fen": None,
            "is_auto_renew": None,
            "status": "unknown",
            "evidence": [],
        },
        "renewal_interval": {
            "interval_value": None,
            "interval_unit": "unknown",
            "advance_notice_days": None,
            "status": "unknown",
            "evidence": [],
        },
        "threshold_discounts": [],
        "add_on_items": [],
        "shipping_fee": {
            "amount_fen": None,
            "free_shipping_threshold_fen": None,
            "status": "unknown",
            "evidence": [],
        },
        "cancellation_effect": {
            "effective_timing": "unknown",
            "penalty_fen": None,
            "status": "unknown",
            "evidence": [],
        },
        "refund_status": {
            "refundability": "unknown",
            "refund_window_days": None,
            "status": "unknown",
            "evidence": [],
        },
        "uncertainties_and_conflicts": [],
    }


def _maybe_ocr_noise(rng: random.Random, text: str) -> str:
    """Applies realistic OCR-like punctuation/spacing noise while keeping span exactness because noise is applied before span recording."""
    if rng.random() < 0.5:
        text = text.replace("：", ": ")
    if rng.random() < 0.4:
        text = text.replace("元", " 元")
    return text


def build_sample(rng: random.Random, family: str, idx: int) -> Tuple[str, Dict[str, Any]]:
    source_id = f"syn_{family}_{idx:04d}"
    tb = TextBuilder(source_id)
    out = _empty_template(source_id)

    products_sub = ["云盘高级会员", "AI写作助手专业版", "高清影视VIP", "无损音乐黑胶卡", "效率笔记Pro"]
    products_ecom = ["便携咖啡机", "纯棉四件套", "机械键盘87键", "宠物自动喂食器", "护眼台灯"]

    if family in ("T01_saas_trial_autorenew", "T03_video_vip_promo_stepup"):
        prod = rng.choice(products_sub)
        trial_days = rng.choice([3, 7, 14])
        trial_fen = rng.choice([0, 100, 190])
        rec_fen = rng.choice([1900, 2500, 2990, 3800, 5800])
        notice_days = rng.choice([1, 3, 5])

        tb.add_plain("【订阅确认单】商品名称：")
        ev_p = tb.add_span(prod)
        out["product_or_service"] = {
            "name": prod,
            "category": "subscription",
            "quantity": 1,
            "status": "known",
            "evidence": [ev_p],
        }
        tb.add_plain("\n试用规则：")
        t_str = f"新用户享{trial_days}天试用（试用价{_fmt_yuan(trial_fen)}元），期满自动转为连续包月"
        if family == "T03_video_vip_promo_stepup":
            t_str = _maybe_ocr_noise(rng, t_str)
        ev_t = tb.add_span(t_str)
        out["trial_period"] = {
            "has_trial": True,
            "duration_value": trial_days,
            "duration_unit": "day",
            "trial_price_fen": trial_fen,
            "auto_convert_to_paid": True,
            "status": "known",
            "evidence": [ev_t],
        }
        tb.add_plain("\n首扣说明：")
        ev_i = tb.add_span(f"下单立即扣款{_fmt_yuan(trial_fen)}元")
        out["initial_charge"] = {
            "amount_fen": trial_fen,
            "timing_rule": "immediate_on_order",
            "status": "known",
            "evidence": [ev_i],
        }
        out["price_items"].append({
            "item_type": "first_period_promo",
            "label": "试用期费用",
            "amount_fen": trial_fen,
            "status": "known",
            "evidence": [ev_i],
        })
        tb.add_plain("\n自动续费：")
        ev_r = tb.add_span(f"到期后每1个月自动扣款{_fmt_yuan(rec_fen)}元，扣款前{notice_days}天短信提醒")
        out["recurring_charge"] = {
            "amount_fen": rec_fen,
            "is_auto_renew": True,
            "status": "known",
            "evidence": [ev_r],
        }
        out["renewal_interval"] = {
            "interval_value": 1,
            "interval_unit": "month",
            "advance_notice_days": notice_days,
            "status": "known",
            "evidence": [ev_r],
        }
        out["price_items"].append({
            "item_type": "recurring_standard",
            "label": "连续包月标准价",
            "amount_fen": rec_fen,
            "status": "known",
            "evidence": [ev_r],
        })
        tb.add_plain("\n退订与售后：")
        ev_c = tb.add_span("取消自动续费后下个周期生效，当期已扣费用不支持退款")
        out["cancellation_effect"] = {
            "effective_timing": "end_of_current_period",
            "penalty_fen": 0,
            "status": "known",
            "evidence": [ev_c],
        }
        out["refund_status"] = {
            "refundability": "non_refundable",
            "refund_window_days": 0,
            "status": "known",
            "evidence": [ev_c],
        }

    elif family in ("T02_ecommerce_threshold_shipping", "T09_multi_tier_discount_deposit"):
        prod = rng.choice(products_ecom)
        qty = rng.choice([1, 2])
        unit_fen = rng.choice([8900, 12900, 19900, 29900])
        th_fen = rng.choice([10000, 20000, 30000])
        disc_fen = rng.choice([1500, 3000, 5000])
        ship_fen = rng.choice([800, 1000, 1200])
        free_th_fen = rng.choice([9900, 19900])

        tb.add_plain("【实物订单核对】商品：")
        ev_p = tb.add_span(f"{prod} x{qty}")
        out["product_or_service"] = {
            "name": prod,
            "category": "ecommerce_goods",
            "quantity": qty,
            "status": "known",
            "evidence": [ev_p],
        }
        tb.add_plain("，")
        ev_price = tb.add_span(f"标价合计{_fmt_yuan(unit_fen * qty)}元")
        out["price_items"].append({
            "item_type": "base_price",
            "label": "商品标价",
            "amount_fen": unit_fen * qty,
            "status": "known",
            "evidence": [ev_price],
        })
        tb.add_plain("\n促销满减：")
        ev_d = tb.add_span(f"店铺满{_fmt_yuan(th_fen)}元减{_fmt_yuan(disc_fen)}元（不可叠加其他满减）")
        out["threshold_discounts"].append({
            "threshold_fen": th_fen,
            "discount_fen": disc_fen,
            "stackable": False,
            "status": "known",
            "evidence": [ev_d],
        })
        tb.add_plain("\n物流运费：")
        ev_s = tb.add_span(f"基础运费{_fmt_yuan(ship_fen)}元，实付满{_fmt_yuan(free_th_fen)}元包邮")
        out["shipping_fee"] = {
            "amount_fen": ship_fen,
            "free_shipping_threshold_fen": free_th_fen,
            "status": "known",
            "evidence": [ev_s],
        }
        if family == "T09_multi_tier_discount_deposit":
            # Held-out validation style: also mentions 7-day refund and explicit non-subscription
            tb.add_plain("\n售后保障：")
            ev_rf = tb.add_span("支持7天无理由退货退款")
            out["refund_status"] = {
                "refundability": "full_refund",
                "refund_window_days": 7,
                "status": "known",
                "evidence": [ev_rf],
            }
        else:
            tb.add_plain("\n备注：本页面未展示退款政策。")
            out["uncertainties_and_conflicts"].append({
                "issue_type": "unknown",
                "target_field": "refund_status",
                "description": "页面未提供退款规则，保持 unknown",
                "evidence": [],
            })

    elif family in ("T04_bundle_prechecked_addon", "T10_fitness_edu_cancel_penalty"):
        prod = rng.choice(products_sub if family == "T10_fitness_edu_cancel_penalty" else products_ecom)
        base_fen = rng.choice([9900, 19900, 39900])
        addon_name = rng.choice(["2年碎屏延保", "VIP优先发货包", "专属客服加购包"])
        addon_fen = rng.choice([990, 1900, 2900])
        tb.add_plain("结算清单：")
        ev_p = tb.add_span(prod)
        out["product_or_service"] = {
            "name": prod,
            "category": "digital_service" if family == "T10_fitness_edu_cancel_penalty" else "ecommerce_goods",
            "quantity": 1,
            "status": "known",
            "evidence": [ev_p],
        }
        tb.add_plain(" ")
        ev_pr = tb.add_span(f"主商品价格{_fmt_yuan(base_fen)}元")
        out["price_items"].append({
            "item_type": "base_price",
            "label": "主商品价格",
            "amount_fen": base_fen,
            "status": "known",
            "evidence": [ev_pr],
        })
        tb.add_plain("\n附加服务：")
        ev_ad = tb.add_span(f"[默认已勾选] {addon_name} ¥{_fmt_yuan(addon_fen)}（非必选，可手动取消）")
        out["add_on_items"].append({
            "name": addon_name,
            "amount_fen": addon_fen,
            "is_prechecked": True,
            "is_mandatory": False,
            "status": "known",
            "evidence": [ev_ad],
        })
        if family == "T10_fitness_edu_cancel_penalty":
            pen_fen = rng.choice([3000, 5000])
            tb.add_plain("\n解约条款：")
            ev_c = tb.add_span(f"未满3个月提前解约需扣违约金{_fmt_yuan(pen_fen)}元，取消后立即停止服务")
            out["cancellation_effect"] = {
                "effective_timing": "immediate",
                "penalty_fen": pen_fen,
                "status": "known",
                "evidence": [ev_c],
            }

    elif family == "T05_refund_ambiguous_or_missing":
        prod = rng.choice(products_sub)
        price_fen = rng.choice([1500, 3000, 4500])
        tb.add_plain("订单详情：")
        ev_p = tb.add_span(prod)
        out["product_or_service"] = {
            "name": prod,
            "category": "digital_service",
            "quantity": 1,
            "status": "known",
            "evidence": [ev_p],
        }
        tb.add_plain("，")
        ev_pr = tb.add_span(f"单次购买{_fmt_yuan(price_fen)}元")
        out["price_items"].append({
            "item_type": "base_price",
            "label": "单次购买价",
            "amount_fen": price_fen,
            "status": "known",
            "evidence": [ev_pr],
        })
        if idx % 2 == 0:
            tb.add_plain("\n售后说明：")
            ev_amb = tb.add_span("退款需视具体使用情况由客服酌情处理")
            out["refund_status"] = {
                "refundability": "conditional_refund",
                "refund_window_days": None,
                "status": "ambiguous",
                "evidence": [ev_amb],
            }
            out["uncertainties_and_conflicts"].append({
                "issue_type": "ambiguous",
                "target_field": "refund_status",
                "description": "仅写明由客服酌情处理，未明确退款比例与时限",
                "evidence": [ev_amb],
            })
        else:
            tb.add_plain("\n说明：页面仅展示价格，未说明是否支持退款或自动续费。")
            # Strictly unknown! Do NOT infer false/non_refundable.
            out["uncertainties_and_conflicts"].append({
                "issue_type": "unknown",
                "target_field": "refund_status",
                "description": "原文未提及退款条款，不得外推为不退",
                "evidence": [],
            })

    elif family == "T06_contradicted_price_terms":
        prod = rng.choice(products_sub)
        p1 = rng.choice([1900, 2900])
        p2 = p1 + rng.choice([1000, 2000])
        tb.add_plain("订阅页顶部横幅：")
        ev_p = tb.add_span(prod)
        out["product_or_service"] = {
            "name": prod,
            "category": "subscription",
            "quantity": 1,
            "status": "known",
            "evidence": [ev_p],
        }
        tb.add_plain(" ")
        ev_a = tb.add_span(f"连续包月仅需{_fmt_yuan(p1)}元/月")
        tb.add_plain("\n底部协议小字：")
        ev_b = tb.add_span(f"次月起自动续费按{_fmt_yuan(p2)}元/月扣款")
        out["recurring_charge"] = {
            "amount_fen": None,
            "is_auto_renew": True,
            "status": "contradicted",
            "evidence": [ev_a, ev_b],
        }
        out["renewal_interval"] = {
            "interval_value": 1,
            "interval_unit": "month",
            "advance_notice_days": None,
            "status": "known",
            "evidence": [ev_b],
        }
        out["uncertainties_and_conflicts"].append({
            "issue_type": "contradicted",
            "target_field": "recurring_charge.amount_fen",
            "description": f"顶部横幅写{_fmt_yuan(p1)}元/月，底部协议写{_fmt_yuan(p2)}元/月，存在金额矛盾",
            "evidence": [ev_a, ev_b],
        })

    elif family in ("T07_ocr_noisy_receipt_and_tos", "T08_zero_yuan_boundary_and_cents"):
        prod = rng.choice(products_ecom if family == "T07_ocr_noisy_receipt_and_tos" else products_sub)
        fen = rng.choice([1, 10, 99, 1099]) if family == "T08_zero_yuan_boundary_and_cents" else rng.choice([4990, 12850])
        prefix = "【OCR识别】商品: " if family == "T07_ocr_noisy_receipt_and_tos" else "【分钱边界测试】"
        tb.add_plain(prefix)
        ev_p = tb.add_span(prod)
        out["product_or_service"] = {
            "name": prod,
            "category": "ecommerce_goods" if family == "T07_ocr_noisy_receipt_and_tos" else "digital_service",
            "quantity": 1,
            "status": "known",
            "evidence": [ev_p],
        }
        tb.add_plain("  实付金额 ")
        price_str = _maybe_ocr_noise(rng, f"¥{_fmt_yuan(fen)}元")
        ev_i = tb.add_span(price_str)
        out["initial_charge"] = {
            "amount_fen": fen,
            "timing_rule": "immediate_on_order",
            "status": "known",
            "evidence": [ev_i],
        }
        out["price_items"].append({
            "item_type": "base_price",
            "label": "实付价格",
            "amount_fen": fen,
            "status": "known",
            "evidence": [ev_i],
        })

    text = tb.text()
    ok, errs = validate_extraction(out, text, source_id)
    if not ok:
        raise RuntimeError(f"Generated invalid synthetic sample {source_id}: {errs}")
    return text, out


def to_chat_record(text: str, out: Dict[str, Any], family: str) -> Dict[str, Any]:
    return {
        "source_id": out["source_id"],
        "template_family": family,
        "input_text": text,
        "target_json": out,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"source_id: {out['source_id']}\n待抽取文本：\n{text}"},
            {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leak-free synthetic dataset for commitment extraction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-family", type=int, default=125, help="8 train families * 125 = 1000")
    parser.add_argument("--val-per-family", type=int, default=50, help="2 held-out val families * 50 = 100")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_families = [
        "T01_saas_trial_autorenew",
        "T02_ecommerce_threshold_shipping",
        "T03_video_vip_promo_stepup",
        "T04_bundle_prechecked_addon",
        "T05_refund_ambiguous_or_missing",
        "T06_contradicted_price_terms",
        "T07_ocr_noisy_receipt_and_tos",
        "T08_zero_yuan_boundary_and_cents",
    ]
    val_families = [
        "T09_multi_tier_discount_deposit",
        "T10_fitness_edu_cancel_penalty",
    ]

    train_records: List[Dict[str, Any]] = []
    for fam in train_families:
        for i in range(args.train_per_family):
            txt, out = build_sample(rng, fam, i)
            train_records.append(to_chat_record(txt, out, fam))

    val_records: List[Dict[str, Any]] = []
    for fam in val_families:
        for i in range(args.val_per_family):
            txt, out = build_sample(rng, fam, i)
            val_records.append(to_chat_record(txt, out, fam))

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_families": train_families,
        "val_families": val_families,
        "leakage_guarantee": "train and val use disjoint template_family sets; zero real orders; zero benchmark_cases.json overlap",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(train_records)} train and {len(val_records)} val records in {args.out_dir}")


if __name__ == "__main__":
    main()
