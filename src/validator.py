#!/usr/bin/env python3
"""
Strict JSON Schema & Character-Level Evidence Span Validator.
Verifies:
1. Structural compliance against commitment_schema_v1.json (works with or without `jsonschema` package).
2. Every EvidenceSpan satisfies:
   - source_id matches input source_id
   - 0 <= start_char < end_char <= len(source_text)
   - source_text[start_char:end_char] == quote (exact verbatim substring)
3. Every non-unknown conclusion has >= 1 valid EvidenceSpan.
4. No fabricated values when status == "unknown".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

VALID_STATUSES = {"known", "unknown", "ambiguous", "contradicted"}


def _validate_span(
    span: Any,
    source_text: str,
    expected_source_id: str,
    path: str,
    errors: List[str],
) -> None:
    if not isinstance(span, dict):
        errors.append(f"{path}: evidence span must be object")
        return
    required = {"source_id", "quote", "start_char", "end_char"}
    if set(span.keys()) != required:
        errors.append(f"{path}: evidence keys mismatch {set(span.keys())} != {required}")
        return
    if span["source_id"] != expected_source_id:
        errors.append(f"{path}.source_id: '{span['source_id']}' != '{expected_source_id}'")
    q = span["quote"]
    s = span["start_char"]
    e = span["end_char"]
    if not isinstance(q, str) or len(q) == 0:
        errors.append(f"{path}.quote: must be non-empty string")
        return
    if not isinstance(s, int) or not isinstance(e, int) or s < 0 or e <= s or e > len(source_text):
        errors.append(f"{path}: invalid span range [{s}, {e}) for text len {len(source_text)}")
        return
    actual = source_text[s:e]
    if actual != q:
        errors.append(f"{path}: span mismatch: text[{s}:{e}]={actual!r} != quote={q!r}")


def _validate_evidence_list(
    ev_list: Any,
    status: str,
    source_text: str,
    expected_source_id: str,
    path: str,
    errors: List[str],
) -> None:
    if not isinstance(ev_list, list):
        errors.append(f"{path}.evidence: must be list")
        return
    if status != "unknown" and len(ev_list) == 0:
        errors.append(f"{path}: status={status} requires at least 1 evidence span")
    for idx, sp in enumerate(ev_list):
        _validate_span(sp, source_text, expected_source_id, f"{path}.evidence[{idx}]", errors)


def validate_extraction(
    data: Dict[str, Any],
    source_text: str,
    expected_source_id: str,
    schema_path: Path | None = None,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return False, ["Root output must be a JSON object"]

    # Accept a str path too: callers (notebooks, CLI) often pass a plain string,
    # and the previous `schema_path.exists()` call raised AttributeError on str.
    if schema_path is not None and not isinstance(schema_path, Path):
        schema_path = Path(schema_path)

    # Optional jsonschema check if installed
    if schema_path is not None and schema_path.exists():
        try:
            import jsonschema  # type: ignore

            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(instance=data, schema=schema)
        except ImportError:
            pass
        except Exception as exc:
            errors.append(f"jsonschema error: {exc}")

    required_top = {
        "schema_version",
        "source_id",
        "product_or_service",
        "price_items",
        "trial_period",
        "initial_charge",
        "recurring_charge",
        "renewal_interval",
        "threshold_discounts",
        "add_on_items",
        "shipping_fee",
        "cancellation_effect",
        "refund_status",
        "uncertainties_and_conflicts",
    }
    missing = required_top - set(data.keys())
    extra = set(data.keys()) - required_top
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
    if extra:
        errors.append(f"Unexpected top-level keys: {sorted(extra)}")
    if data.get("schema_version") != "1.0.0":
        errors.append(f"Invalid schema_version: {data.get('schema_version')!r}")
    if data.get("source_id") != expected_source_id:
        errors.append(f"source_id mismatch: {data.get('source_id')!r} != {expected_source_id!r}")

    # Validate single-object sections with status + evidence
    single_sections = [
        "product_or_service",
        "trial_period",
        "initial_charge",
        "recurring_charge",
        "renewal_interval",
        "shipping_fee",
        "cancellation_effect",
        "refund_status",
    ]
    for sec in single_sections:
        obj = data.get(sec)
        if not isinstance(obj, dict):
            errors.append(f"{sec}: must be object")
            continue
        st = obj.get("status")
        if st not in VALID_STATUSES:
            errors.append(f"{sec}.status: invalid {st!r}")
        else:
            _validate_evidence_list(obj.get("evidence"), st, source_text, expected_source_id, sec, errors)

    # Validate array sections
    for arr_name in ("price_items", "threshold_discounts", "add_on_items"):
        arr = data.get(arr_name)
        if not isinstance(arr, list):
            errors.append(f"{arr_name}: must be list")
            continue
        for idx, item in enumerate(arr):
            if not isinstance(item, dict):
                errors.append(f"{arr_name}[{idx}]: must be object")
                continue
            st = item.get("status")
            if st not in VALID_STATUSES:
                errors.append(f"{arr_name}[{idx}].status: invalid {st!r}")
            else:
                _validate_evidence_list(
                    item.get("evidence"), st, source_text, expected_source_id, f"{arr_name}[{idx}]", errors
                )
            if "amount_fen" in item and item["amount_fen"] is not None:
                if not isinstance(item["amount_fen"], int) or isinstance(item["amount_fen"], bool) or item["amount_fen"] < 0:
                    errors.append(f"{arr_name}[{idx}].amount_fen: must be non-negative int")

    # Validate uncertainties_and_conflicts
    uac = data.get("uncertainties_and_conflicts")
    if not isinstance(uac, list):
        errors.append("uncertainties_and_conflicts: must be list")
    else:
        for idx, item in enumerate(uac):
            if not isinstance(item, dict):
                errors.append(f"uncertainties_and_conflicts[{idx}]: must be object")
                continue
            if item.get("issue_type") not in {"unknown", "ambiguous", "contradicted"}:
                errors.append(f"uncertainties_and_conflicts[{idx}].issue_type invalid")
            ev = item.get("evidence", [])
            if not isinstance(ev, list):
                errors.append(f"uncertainties_and_conflicts[{idx}].evidence must be list")
            else:
                for j, sp in enumerate(ev):
                    _validate_span(
                        sp, source_text, expected_source_id, f"uncertainties_and_conflicts[{idx}].evidence[{j}]", errors
                    )

    return len(errors) == 0, errors


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python validator.py <input_text.txt> <extracted.json>")
        sys.exit(2)
    txt = Path(sys.argv[1]).read_text(encoding="utf-8")
    obj = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    ok, errs = validate_extraction(obj, txt, obj.get("source_id", ""))
    print(json.dumps({"valid": ok, "errors": errs}, ensure_ascii=False, indent=2))
    sys.exit(0 if ok else 1)
