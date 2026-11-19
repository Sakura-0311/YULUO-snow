#!/usr/bin/env python3
"""
Deterministic calculator for "承诺单：下单前看清账".
All arithmetic (fen totals, threshold discounts, shipping thresholds,
trial/renewal date windows, 12-month projections) is computed here—NEVER by the LLM.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional


def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    # Clamp day to month length
    days_in_month = [
        31,
        29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ][month - 1]
    return date(year, month, min(d.day, days_in_month))


def advance_date(start_date: date, value: int, unit: str) -> Optional[date]:
    if value is None or value <= 0:
        return None
    if unit == "day":
        return start_date + timedelta(days=value)
    if unit == "week":
        return start_date + timedelta(weeks=value)
    if unit == "month":
        return _add_months(start_date, value)
    if unit == "quarter":
        return _add_months(start_date, value * 3)
    if unit == "year":
        return _add_months(start_date, value * 12)
    return None


def compute_commitment_summary(
    extracted: Dict[str, Any],
    order_date_iso: Optional[str] = None,
    include_prechecked_addons: bool = True,
) -> Dict[str, Any]:
    """Deterministically compute payable amounts and date windows from extracted JSON."""
    warnings: List[str] = []

    for conflict in extracted.get("uncertainties_and_conflicts", []):
        warnings.append(
            f"[{conflict.get('issue_type', 'unknown').upper()}] "
            f"{conflict.get('target_field', '')}: {conflict.get('description', '')}"
        )

    # 1. Base & initial price items
    merchandise_fen = 0
    has_known_price = False
    for item in extracted.get("price_items", []):
        st = item.get("status")
        amt = item.get("amount_fen")
        itype = item.get("item_type")
        if st == "known" and isinstance(amt, int) and itype in ("base_price", "first_period_promo", "deposit", "service_fee"):
            merchandise_fen += amt
            has_known_price = True
        elif st in ("ambiguous", "contradicted"):
            warnings.append(f"价格项 '{item.get('label', itype)}' 状态为 {st}，无法确定计入。")

    # 2. Add-ons
    addon_fen = 0
    for addon in extracted.get("add_on_items", []):
        st = addon.get("status")
        amt = addon.get("amount_fen")
        selected = bool(addon.get("is_mandatory")) or (include_prechecked_addons and bool(addon.get("is_prechecked")))
        if selected and st == "known" and isinstance(amt, int):
            addon_fen += amt
        elif selected and st in ("ambiguous", "contradicted", "unknown"):
            warnings.append(f"加购项 '{addon.get('name', '未知')}' 金额状态为 {st}。")

    subtotal_before_discount_fen = merchandise_fen + addon_fen if has_known_price else None

    # 3. Threshold discounts (deterministic rule evaluation)
    discount_applied_fen = 0
    if subtotal_before_discount_fen is not None:
        discounts = extracted.get("threshold_discounts", [])
        stackable_sum = 0
        non_stackable_best = 0
        for d in discounts:
            if d.get("status") != "known":
                if d.get("status") in ("ambiguous", "contradicted"):
                    warnings.append(f"满减规则存在 {d.get('status')}，未自动抵扣。")
                continue
            th = d.get("threshold_fen")
            disc = d.get("discount_fen")
            if isinstance(th, int) and isinstance(disc, int) and subtotal_before_discount_fen >= th:
                if d.get("stackable") is True:
                    stackable_sum += disc
                else:
                    non_stackable_best = max(non_stackable_best, disc)
        discount_applied_fen = min(subtotal_before_discount_fen, stackable_sum + non_stackable_best)

    after_discount_fen = (
        max(0, subtotal_before_discount_fen - discount_applied_fen)
        if subtotal_before_discount_fen is not None
        else None
    )

    # 4. Shipping fee
    shipping_applied_fen: Optional[int] = None
    ship = extracted.get("shipping_fee", {})
    ship_st = ship.get("status")
    if ship_st == "known":
        base_ship = ship.get("amount_fen")
        free_th = ship.get("free_shipping_threshold_fen")
        if after_discount_fen is not None and isinstance(free_th, int) and after_discount_fen >= free_th:
            shipping_applied_fen = 0
        elif isinstance(base_ship, int):
            shipping_applied_fen = base_ship
    elif ship_st in ("ambiguous", "contradicted"):
        warnings.append(f"运费条款状态为 {ship_st}。")

    # 5. Initial charge / due now
    init_chg = extracted.get("initial_charge", {})
    due_now_fen: Optional[int] = None
    if init_chg.get("status") == "known" and isinstance(init_chg.get("amount_fen"), int):
        due_now_fen = init_chg["amount_fen"]
    elif after_discount_fen is not None:
        due_now_fen = after_discount_fen + (shipping_applied_fen or 0)

    # 6. Date windows & 12-month projection
    trial = extracted.get("trial_period", {})
    rec = extracted.get("recurring_charge", {})
    interval = extracted.get("renewal_interval", {})

    trial_end_date_iso: Optional[str] = None
    first_recurring_date_iso: Optional[str] = None
    if order_date_iso:
        base_dt = date.fromisoformat(order_date_iso)
        if trial.get("status") == "known" and trial.get("has_trial") is True:
            t_val = trial.get("duration_value")
            t_unit = trial.get("duration_unit")
            if isinstance(t_val, int) and t_unit in ("day", "week", "month"):
                end_dt = advance_date(base_dt, t_val, t_unit)
                if end_dt:
                    trial_end_date_iso = end_dt.isoformat()
                    if trial.get("auto_convert_to_paid") is True:
                        first_recurring_date_iso = end_dt.isoformat()
        elif rec.get("status") == "known" and rec.get("is_auto_renew") is True:
            i_val = interval.get("interval_value")
            i_unit = interval.get("interval_unit")
            if isinstance(i_val, int) and i_unit in ("day", "week", "month", "quarter", "year"):
                next_dt = advance_date(base_dt, i_val, i_unit)
                if next_dt:
                    first_recurring_date_iso = next_dt.isoformat()

    estimated_12m_total_fen: Optional[int] = None
    if (
        due_now_fen is not None
        and rec.get("status") == "known"
        and rec.get("is_auto_renew") is True
        and isinstance(rec.get("amount_fen"), int)
        and interval.get("status") == "known"
        and isinstance(interval.get("interval_value"), int)
    ):
        unit = interval.get("interval_unit")
        val = interval.get("interval_value")
        cycles_per_year = {
            "month": 12 / val,
            "quarter": 4 / val,
            "year": 1 / val,
            "week": 52 / val,
            "day": 365 / val,
        }.get(unit)
        if cycles_per_year is not None and cycles_per_year >= 1:
            # First period covered by due_now_fen if no free trial, else all cycles after trial
            has_trial = trial.get("status") == "known" and trial.get("has_trial") is True
            remaining_cycles = int(cycles_per_year) if has_trial else max(0, int(cycles_per_year) - 1)
            estimated_12m_total_fen = due_now_fen + remaining_cycles * rec["amount_fen"]

    return {
        "merchandise_subtotal_fen": merchandise_fen if has_known_price else None,
        "addon_subtotal_fen": addon_fen,
        "discount_applied_fen": discount_applied_fen,
        "shipping_applied_fen": shipping_applied_fen,
        "due_now_fen": due_now_fen,
        "trial_end_date": trial_end_date_iso,
        "first_recurring_date": first_recurring_date_iso,
        "estimated_12m_total_fen": estimated_12m_total_fen,
        "warnings": warnings,
    }
