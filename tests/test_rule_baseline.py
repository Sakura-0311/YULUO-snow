import re
import json
from typing import Dict, Any, Optional

def extract_with_rules(text: str, source_id: str = "rule_pred") -> Dict[str, Any]:
    pred = {
        "source_id": source_id,
        "input_text": text,
        "initial_payment_yuan": None,
        "trial_days": None,
        "auto_renew": "unclear",
        "renewal_amount_yuan": None,
        "renewal_interval_days": None,
        "first_renewal_day": None,
        "cancellation": None,
        "refund": None,
        "must_flag_unknown_or_conflict": False,
        "contradictions": [],
        "evidence": {}
    }

    # 1. Trial days
    trial_m = re.search(r'(?:前|试用)(\d+)天(?:免费|试用)', text)
    if trial_m:
        pred["trial_days"] = int(trial_m.group(1))
        pred["evidence"]["trial_days"] = trial_m.group(0)

    # 2. Initial payment
    init_m = re.search(r'(?:新客首月|本次购买价|首月|开通时一次扣|年卡|商品A售价|商品A\s*)(\d+(?:\.\d+)?)元', text)
    if init_m:
        val = float(init_m.group(1))
        pred["initial_payment_yuan"] = f"{val:.2f}"
        pred["evidence"]["initial_payment_yuan"] = init_m.group(0)
    elif "前7天免费" in text:
        pred["initial_payment_yuan"] = "0.00"

    # 3. Auto renew & renewal amount
    if "不会再次扣款" in text or "到期不自动续费" in text:
        pred["auto_renew"] = "no"
        if "不会再次扣款" in text:
            pred["evidence"]["auto_renew"] = "不会再次扣款"
        elif "到期不自动续费" in text:
            pred["evidence"]["auto_renew"] = "到期不自动续费"
    elif "自动续费" in text or "连续包月" in text:
        pred["auto_renew"] = "yes"
        pred["evidence"]["auto_renew"] = "自动续费" if "自动续费" in text else "连续包月"
        
        # Renewal amount & interval
        renew_m = re.search(r'(?:每(\d+)天按?|第\d+天扣\d+(?:\.\d+)?元，之后每(\d+)天扣|每年)(\d+(?:\.\d+)?)元', text)
        if renew_m:
            days = renew_m.group(1) or renew_m.group(2)
            if "每年" in renew_m.group(0):
                pred["renewal_interval_days"] = 365
            elif days:
                pred["renewal_interval_days"] = int(days)
            amt = float(renew_m.group(3))
            pred["renewal_amount_yuan"] = f"{amt:.2f}"
            pred["evidence"]["renewal_amount_yuan"] = renew_m.group(0)
        elif "续费价格以扣款前页面显示为准" in text:
            pred["renewal_amount_yuan"] = None
            pred["must_flag_unknown_or_conflict"] = True
            pred["evidence"]["renewal_amount_yuan"] = "续费价格以扣款前页面显示为准"

    # 4. Cancellation
    cancel_m = re.search(r'([^，。；]*(?:关闭自动续费|关闭续费|停止服务|可在到期前关闭)[^，。；]*)', text)
    if cancel_m:
        pred["cancellation"] = cancel_m.group(1).strip()
        pred["evidence"]["cancellation"] = cancel_m.group(1).strip()

    # 5. Refund
    if "已付金额不退" in text:
        pred["refund"] = "已付金额不退"
        pred["evidence"]["refund"] = "已付金额不退"
    elif "页面未说明已扣款是否退款" in text:
        pred["refund"] = None
        pred["must_flag_unknown_or_conflict"] = True
        pred["evidence"]["refund"] = "页面未说明已扣款是否退款"

    # Check known conflict cases in baseline (rule cannot detect semantic conflicts easily)
    if "说明A" in text and "说明B" in text:
        pred["must_flag_unknown_or_conflict"] = True
    if "细则没有说明" in text or "未显示已选方案" in text:
        pred["must_flag_unknown_or_conflict"] = True

    return pred

def _find_benchmark():
    """Locate benchmark_cases.json without assuming any machine layout.

    Search order: $YULUO_BENCH, then walking up from this file looking for
    benchmark/benchmark_cases.json (shipped package) or data/benchmark_cases.json
    (training-repo layout), then the current directory. Fails loudly with the
    full candidate list instead of raising a bare FileNotFoundError.
    """
    import os
    from pathlib import Path

    cands = []
    env = os.environ.get("YULUO_BENCH")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        cands.extend([
            base / "benchmark" / "benchmark_cases.json",
            base / "data" / "benchmark_cases.json",
            base / "benchmark_cases.json",
        ])
    cands.append(Path.cwd() / "benchmark_cases.json")

    for c in cands:
        if c.is_file():
            return c
    listing = "".join("  " + str(c) + "\n" for c in cands)
    raise SystemExit("benchmark_cases.json not found. Tried:\n" + listing +
                     "Set YULUO_BENCH to override.\n")


if __name__ == "__main__":
    bench_path = _find_benchmark()
    print("benchmark:", bench_path)
    with open(bench_path, encoding='utf-8') as f:
        bench = json.load(f)['cases']
    
    score = 0
    for c in bench:
        res = extract_with_rules(c['input_text'], c['id'])
        # check unknown match
        gold_flag = c['gold'].get('must_flag_unknown_or_conflict', False)
        pred_flag = res['must_flag_unknown_or_conflict']
        flag_ok = (gold_flag == pred_flag)
        
        # check auto_renew match
        gold_ar = c['gold'].get('auto_renew')
        pred_ar = res.get('auto_renew')
        ar_ok = (gold_ar is None or gold_ar == 'unclear' or gold_ar == pred_ar)
        
        # check initial match
        gold_init = c['gold'].get('initial_payment_yuan')
        pred_init = res.get('initial_payment_yuan')
        init_ok = (gold_init is None or gold_init == pred_init)
        
        pass_case = flag_ok and ar_ok and init_ok
        if pass_case:
            score += 1
        print(f"[{c['id']}] {'PASS' if pass_case else 'FAIL'} (flag={flag_ok}, ar={ar_ok}, init={init_ok}) - {c.get('scenario')}")
    print(f"\nRule Baseline Score: {score}/{len(bench)} ({score/len(bench)*100:.1f}%)")
