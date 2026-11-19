#!/usr/bin/env python3
"""Unit test for spanfix.reanchor: perturb gold offsets and prove recovery.

Runs offline, no model needed:
  * gold as-is            -> already_ok, metrics unchanged (1.0)
  * gold with offsets +7  -> fixed, strict validity restored to 1.0
  * gold with quote re-spaced -> fixed via whitespace-insensitive path
  * unrecoverable quote   -> unresolved, case stays invalid (no silent pass)
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from evaluate import evaluate_schema_predictions  # noqa: E402
from spanfix import reanchor  # noqa: E402

SCHEMA = HERE.parent / "schema" / "commitment_schema_v1.json"
_full = HERE.parent / "data" / "val.jsonl"
VAL = _full if _full.exists() else (HERE / "val_sample.jsonl")
SEC = ("product_or_service", "trial_period", "initial_charge", "recurring_charge",
       "renewal_interval", "shipping_fee", "cancellation_effect", "refund_status")


def all_spans(obj):
    for s in SEC:
        n = obj.get(s)
        if isinstance(n, dict):
            for sp in n.get("evidence", []) or []:
                yield sp
    for a in ("price_items", "threshold_discounts", "add_on_items",
              "uncertainties_and_conflicts"):
        for it in obj.get(a, []) or []:
            if isinstance(it, dict):
                for sp in it.get("evidence", []) or []:
                    yield sp


def main() -> None:
    cases = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines() if l.strip()][:40]
    gold = [c["target_json"] for c in cases]

    base = evaluate_schema_predictions(cases, gold, SCHEMA)
    print("1) gold untouched          validity=%.4f span_f1=%.4f"
          % (base["structure_validity_rate"], base["evidence_span_f1"]))
    assert base["structure_validity_rate"] == 1.0, "gold must be strictly valid"

    # perturb: shift every offset by +7 (model-like off-by-N error)
    pert = []
    for c in cases:
        o = copy.deepcopy(c["target_json"])
        for sp in all_spans(o):
            if isinstance(sp.get("start_char"), int):
                sp["start_char"] += 7
            if isinstance(sp.get("end_char"), int):
                sp["end_char"] += 7
        pert.append(o)
    m_pert = evaluate_schema_predictions(cases, pert, SCHEMA)
    print("2) offsets shifted by +7    validity=%.4f span_f1=%.4f  <- the failure mode"
          % (m_pert["structure_validity_rate"], m_pert["evidence_span_f1"]))
    assert m_pert["structure_validity_rate"] == 0.0, "shifted offsets must break strict validity"

    fixed, stats = [], {"spans": 0, "fixed": 0, "already_ok": 0, "unresolved": 0}
    pert_before = json.dumps(pert, ensure_ascii=False, sort_keys=True)
    for c, o in zip(cases, copy.deepcopy(pert)):
        o2, st = reanchor(o, c["input_text"])
        for k in stats:
            stats[k] += st[k]
        fixed.append(o2)
    assert json.dumps(pert, ensure_ascii=False, sort_keys=True) == pert_before, \
        "reanchor must not mutate its input (the raw baseline would be corrupted)"
    m_fix = evaluate_schema_predictions(cases, fixed, SCHEMA)
    print("   reanchor stats:", stats)
    print("3) after reanchor           validity=%.4f span_f1=%.4f"
          % (m_fix["structure_validity_rate"], m_fix["evidence_span_f1"]))
    assert m_fix["structure_validity_rate"] == 1.0, "reanchor must restore strict validity"
    assert m_fix["evidence_span_f1"] == 1.0, "reanchor must restore span F1"

    # re-spaced quote: insert spaces inside the quote, offsets garbage
    resp = []
    for c in cases:
        o = copy.deepcopy(c["target_json"])
        for sp in all_spans(o):
            q = str(sp.get("quote", ""))
            if len(q) > 6:
                sp["quote"] = q[:3] + "  " + q[3:]   # same chars, extra whitespace
                sp["start_char"], sp["end_char"] = 0, 0
        resp.append(o)
    fixed2, stats2 = [], {"spans": 0, "fixed": 0, "already_ok": 0, "unresolved": 0}
    for c, o in zip(cases, copy.deepcopy(resp)):
        o2, st = reanchor(o, c["input_text"])
        for k in stats2:
            stats2[k] += st[k]
        fixed2.append(o2)
    print("4) re-spaced quotes         reanchor stats:", stats2)
    m2 = evaluate_schema_predictions(cases, fixed2, SCHEMA)
    print("   after reanchor           validity=%.4f span_f1=%.4f"
          % (m2["structure_validity_rate"], m2["evidence_span_f1"]))
    assert m2["structure_validity_rate"] == 1.0, "whitespace path must recover"

    # unrecoverable: quote that does not exist in the text must stay unresolved
    bad = copy.deepcopy(cases[0]["target_json"])
    first = next(iter(all_spans(bad)))
    first["quote"] = "这段文字在原文里绝对不存在XYZ"
    first["start_char"], first["end_char"] = 0, 1
    o2, st = reanchor(bad, cases[0]["input_text"])
    print("5) bogus quote              stats:", st)
    assert st["unresolved"] >= 1, "unresolvable quote must be reported, not silently fixed"

    print("\nALL SPANFIX TESTS PASSED")


if __name__ == "__main__":
    main()
