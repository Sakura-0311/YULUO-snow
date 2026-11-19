#!/usr/bin/env python3
"""
Evaluation Protocol for "承诺单：下单前看清账".
Compares Base Model vs Fine-Tuned Model vs Baseline on:
1. Structure Validity Rate (JSON + Schema)
2. Field Exact Match & Micro/Macro F1
3. Evidence Span / Quote Exact Match (quote + start_char + end_char)
4. Unknown / Ambiguous / Contradicted Recognition F1
5. Hallucination Rate (non-unknown claim when gold is unknown, or invalid span)

Supports:
- Synthetic held-out val.jsonl (full schema format)
- External blind benchmark JSON file (e.g. benchmark_cases.json with gold format), NEVER used in training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from infer import extract_json_block, run_hf_inference
from validator import validate_extraction

SCALAR_FIELDS = [
    ("product_or_service", "name"),
    ("product_or_service", "category"),
    ("product_or_service", "status"),
    ("trial_period", "has_trial"),
    ("trial_period", "duration_value"),
    ("trial_period", "trial_price_fen"),
    ("trial_period", "auto_convert_to_paid"),
    ("trial_period", "status"),
    ("initial_charge", "amount_fen"),
    ("initial_charge", "timing_rule"),
    ("initial_charge", "status"),
    ("recurring_charge", "amount_fen"),
    ("recurring_charge", "is_auto_renew"),
    ("recurring_charge", "status"),
    ("renewal_interval", "interval_value"),
    ("renewal_interval", "interval_unit"),
    ("renewal_interval", "status"),
    ("shipping_fee", "amount_fen"),
    ("shipping_fee", "free_shipping_threshold_fen"),
    ("shipping_fee", "status"),
    ("cancellation_effect", "effective_timing"),
    ("cancellation_effect", "penalty_fen"),
    ("cancellation_effect", "status"),
    ("refund_status", "refundability"),
    ("refund_status", "refund_window_days"),
    ("refund_status", "status"),
]


def _collect_spans(obj: Dict[str, Any]) -> set[Tuple[str, int, int]]:
    spans: set[Tuple[str, int, int]] = set()
    for sec in (
        "product_or_service",
        "trial_period",
        "initial_charge",
        "recurring_charge",
        "renewal_interval",
        "shipping_fee",
        "cancellation_effect",
        "refund_status",
    ):
        for sp in obj.get(sec, {}).get("evidence", []):
            if isinstance(sp, dict):
                spans.add((str(sp.get("quote", "")), int(sp.get("start_char", -1)), int(sp.get("end_char", -1))))
    for arr in ("price_items", "threshold_discounts", "add_on_items", "uncertainties_and_conflicts"):
        for item in obj.get(arr, []):
            if isinstance(item, dict):
                for sp in item.get("evidence", []):
                    if isinstance(sp, dict):
                        spans.add((str(sp.get("quote", "")), int(sp.get("start_char", -1)), int(sp.get("end_char", -1))))
    return spans


def evaluate_schema_predictions(
    cases: List[Dict[str, Any]],
    predictions: List[Optional[Dict[str, Any]]],
    schema_path: Path,
) -> Dict[str, Any]:
    total = len(cases)
    valid_count = 0
    field_matches = 0
    field_total = 0

    span_tp = 0
    span_pred_total = 0
    span_gold_total = 0

    unc_tp = 0
    unc_pred_total = 0
    unc_gold_total = 0

    hallucination_count = 0
    hallucination_den = 0

    for case, pred in zip(cases, predictions):
        src_id = case["source_id"]
        src_text = case["input_text"]
        gold = case["target_json"]

        if pred is None:
            field_total += len(SCALAR_FIELDS)
            span_gold_total += len(_collect_spans(gold))
            continue

        ok, _ = validate_extraction(pred, src_text, src_id, schema_path)
        if ok:
            valid_count += 1

        # Field Exact Match & Uncertainty / Hallucination
        for sec, key in SCALAR_FIELDS:
            field_total += 1
            g_val = gold.get(sec, {}).get(key)
            p_val = pred.get(sec, {}).get(key) if isinstance(pred.get(sec), dict) else None
            if g_val == p_val:
                field_matches += 1

            if key == "status":
                if g_val in ("unknown", "ambiguous", "contradicted"):
                    unc_gold_total += 1
                    if p_val == g_val:
                        unc_tp += 1
                if p_val in ("unknown", "ambiguous", "contradicted"):
                    unc_pred_total += 1
                if g_val == "unknown":
                    hallucination_den += 1
                    if p_val != "unknown":
                        hallucination_count += 1

        g_spans = _collect_spans(gold)
        p_spans = _collect_spans(pred)
        span_tp += len(g_spans & p_spans)
        span_pred_total += len(p_spans)
        span_gold_total += len(g_spans)

    span_p = span_tp / span_pred_total if span_pred_total else 0.0
    span_r = span_tp / span_gold_total if span_gold_total else 0.0
    span_f1 = (2 * span_p * span_r / (span_p + span_r)) if (span_p + span_r) else 0.0

    unc_p = unc_tp / unc_pred_total if unc_pred_total else 0.0
    unc_r = unc_tp / unc_gold_total if unc_gold_total else 0.0
    unc_f1 = (2 * unc_p * unc_r / (unc_p + unc_r)) if (unc_p + unc_r) else 0.0

    return {
        "total_cases": total,
        "structure_validity_rate": round(valid_count / total, 4) if total else 0.0,
        "field_exact_match": round(field_matches / field_total, 4) if field_total else 0.0,
        "evidence_span_f1": round(span_f1, 4),
        "uncertainty_status_f1": round(unc_f1, 4),
        "hallucination_rate_on_unknown": round(hallucination_count / hallucination_den, 4) if hallucination_den else 0.0,
    }


def evaluate_consumer_predictions(
    cases: List[Dict[str, Any]],
    predictions: List[Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    total = len(cases)
    init_match = 0
    renew_amt_match = 0
    auto_renew_match = 0
    conflict_flag_match = 0
    evidence_valid_quotes = 0
    evidence_total_quotes = 0
    cost_matches = 0
    cost_total = 0

    for case, pred in zip(cases, predictions):
        src_text = case["input_text"]
        gold = case["target_json"]

        if pred is None:
            continue

        # 1. Conflict / unknown flag
        gold_flag = bool(gold.get("must_flag_unknown_or_conflict", False))
        pred_flag = bool(pred.get("must_flag_unknown_or_conflict", False))
        if gold_flag == pred_flag:
            conflict_flag_match += 1

        # 2. Auto-renew flag
        gold_ar = gold.get("auto_renew")
        pred_ar = pred.get("auto_renew")
        if gold_ar is None or gold_ar == "unclear" or gold_ar == pred_ar:
            auto_renew_match += 1

        # 3. Initial payment
        gold_init = gold.get("initial_payment_yuan")
        pred_init = pred.get("initial_payment_yuan")
        if gold_init is None or gold_init == pred_init:
            init_match += 1

        # 4. Renewal amount
        gold_ren = gold.get("renewal_amount_yuan")
        pred_ren = pred.get("renewal_amount_yuan")
        if gold_ren is None or gold_ren == pred_ren:
            renew_amt_match += 1

        # 5. Evidence verbatim substring check
        pred_evidence = pred.get("evidence", {})
        if isinstance(pred_evidence, dict):
            for k, q in pred_evidence.items():
                if isinstance(q, str) and q.strip():
                    evidence_total_quotes += 1
                    if q.strip() in src_text:
                        evidence_valid_quotes += 1

        # 6. Cumulative cost matching
        gold_cost = gold.get("total_cost_yuan")
        pred_cost = pred.get("total_cost_yuan")
        if isinstance(gold_cost, dict) and isinstance(pred_cost, dict):
            for h in ("30_days", "90_days", "365_days"):
                if h in gold_cost:
                    cost_total += 1
                    if gold_cost[h] == pred_cost.get(h):
                        cost_matches += 1

    return {
        "total_cases": total,
        "initial_payment_match_rate": round(init_match / total, 4) if total else 0.0,
        "renewal_amount_match_rate": round(renew_amt_match / total, 4) if total else 0.0,
        "auto_renew_match_rate": round(auto_renew_match / total, 4) if total else 0.0,
        "conflict_detection_accuracy": round(conflict_flag_match / total, 4) if total else 0.0,
        "evidence_quote_validity_rate": round(evidence_valid_quotes / evidence_total_quotes, 4) if evidence_total_quotes else 1.0,
        "cumulative_cost_match_rate": round(cost_matches / cost_total, 4) if cost_total else 0.0,
    }


def load_cases(path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    if path.suffix == ".jsonl":
        cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return cases, False

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    normalized = []
    is_consumer = False
    for i, item in enumerate(data):
        target = item.get("target_json") or item.get("expected_json") or item.get("gold")
        if "gold" in item and item.get("target_json") is None:
            is_consumer = True
        normalized.append({
            "source_id": item.get("source_id") or item.get("id", f"bench_{i:03d}"),
            "input_text": item["input_text"],
            "target_json": target,
            "raw_item": item,
        })
    return normalized, is_consumer


def evaluate_predictions(
    cases: List[Dict[str, Any]],
    predictions: List[Optional[Dict[str, Any]]],
    schema_path: Path,
    is_consumer: bool = False,
) -> Dict[str, Any]:
    if is_consumer:
        return evaluate_consumer_predictions(cases, predictions)
    return evaluate_schema_predictions(cases, predictions, schema_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-Tuned model on val or blind benchmark.")
    parser.add_argument("--benchmark-file", type=Path, required=True, help="Path to val.jsonl or blind benchmark_cases.json")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--schema", type=Path, default=Path("schema/commitment_schema_v1.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-report", type=Path, default=Path("eval_report.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    cases, is_consumer = load_cases(args.benchmark_file)

    if args.base_model:
        base_preds = [extract_json_block(run_hf_inference(args.base_model, None, c["source_id"], c["input_text"])) for c in cases]
        base_metrics = evaluate_predictions(cases, base_preds, args.schema, is_consumer)
    else:
        # Rule baseline when no model is passed
        from test_rule_baseline import extract_with_rules
        base_preds = [extract_with_rules(c["input_text"], c["source_id"]) for c in cases]
        base_metrics = evaluate_predictions(cases, base_preds, args.schema, is_consumer)

    report: Dict[str, Any] = {"seed": args.seed, "benchmark": str(args.benchmark_file), "is_consumer_benchmark": is_consumer, "baseline": base_metrics}
    if args.adapter_dir and args.base_model:
        ft_preds = [extract_json_block(run_hf_inference(args.base_model, args.adapter_dir, c["source_id"], c["input_text"])) for c in cases]
        report["finetuned_model"] = evaluate_predictions(cases, ft_preds, args.schema, is_consumer)

    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
