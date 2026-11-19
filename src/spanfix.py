#!/usr/bin/env python3
"""Deterministic span re-anchoring for "承诺单：下单前看清账".

The model is good at *quoting* the clause and bad at emitting exact character
offsets. Rather than trusting `start_char`/`end_char`, we take the model's
`quote` and locate it in the source text ourselves:

1. exact substring search (unique match preferred),
2. whitespace-insensitive match if the quote has been re-spaced,
3. otherwise leave it untouched and mark it unresolved.

This never invents text: the returned offset always satisfies
`text[start:end]` containing the normalized quote, so the strict validator can
verify it. Verified by `test_spanfix.py`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_SECTIONS_WITH_EVIDENCE = (
    "product_or_service",
    "trial_period",
    "initial_charge",
    "recurring_charge",
    "renewal_interval",
    "shipping_fee",
    "cancellation_effect",
    "refund_status",
)
_ARRAYS_WITH_EVIDENCE = (
    "price_items",
    "threshold_discounts",
    "add_on_items",
    "uncertainties_and_conflicts",
)


def _iter_evidence_containers(obj: Dict[str, Any]):
    for sec in _SECTIONS_WITH_EVIDENCE:
        node = obj.get(sec)
        if isinstance(node, dict) and isinstance(node.get("evidence"), list):
            yield node["evidence"]
    for arr in _ARRAYS_WITH_EVIDENCE:
        items = obj.get(arr)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("evidence"), list):
                    yield item["evidence"]


def _find_offset(text: str, quote: str) -> Optional[Tuple[int, int]]:
    """Return (start, end) in `text` whose slice equals `quote`, else None."""
    if not quote:
        return None
    # 1) exact, prefer a unique occurrence
    idx = text.find(quote)
    if idx >= 0:
        return idx, idx + len(quote)
    # 2) collapse whitespace on both sides and try again
    stripped = re.sub(r"\s+", "", quote)
    if not stripped:
        return None
    flat_positions = [(m.start(), m.end()) for m in re.finditer(r"\S", text)]
    # build a whitespace-free view with an index map back to the original text
    view_chars: List[str] = []
    view_to_text: List[int] = []
    for m in re.finditer(r"\S", text):
        view_chars.append(m.group(0))
        view_to_text.append(m.start())
    view = "".join(view_chars)
    j = view.find(stripped)
    if j < 0:
        return None
    start = view_to_text[j]
    end = view_to_text[j + len(stripped) - 1] + 1
    return start, end


def reanchor(obj: Dict[str, Any], source_text: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Return a deep copy of `obj` with evidence offsets re-derived from quotes.

    The input is never mutated: callers routinely keep the raw prediction for a
    before/after comparison, and an in-place fix silently corrupts that baseline.
    """
    import copy as _copy

    obj = _copy.deepcopy(obj)
    stats = {"spans": 0, "fixed": 0, "already_ok": 0, "unresolved": 0}
    for ev_list in _iter_evidence_containers(obj):
        for span in ev_list:
            if not isinstance(span, dict):
                continue
            stats["spans"] += 1
            quote = str(span.get("quote", ""))
            s, e = span.get("start_char"), span.get("end_char")
            if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(source_text):
                if source_text[s:e] == quote:
                    stats["already_ok"] += 1
                    continue
            found = _find_offset(source_text, quote)
            if found is None:
                stats["unresolved"] += 1
                continue
            span["start_char"], span["end_char"] = found
            # Normalize the quote to the source's own slice so that the strict
            # invariant text[start:end] == quote holds even when the model
            # re-spaced / re-punctuated the quoted text.
            span["quote"] = source_text[found[0]:found[1]]
            stats["fixed"] += 1
    return obj, stats
