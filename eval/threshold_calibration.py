"""
F1-optimal threshold calibration for dual-mode gating.

Reads existing results from results/dual_mode_ablation.json (does NOT re-run pipeline),
sweeps thresholds, finds optimal operating points, recomputes D1-D9 under 4 configurations,
and saves analysis to results/threshold_analysis/.
"""

import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import cohens_kappa, precision_recall_f1

console = Console()

RESULTS_PATH = Path(__file__).parent.parent / "results" / "dual_mode_ablation.json"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "threshold_analysis"

THRESHOLDS = [round(t * 0.05, 2) for t in range(21)]
METRICS = ["faithfulness", "answer_relevance", "context_relevance", "completeness"]
SCORE_KEYS = {
    "faithfulness": "sync_faithfulness",
    "answer_relevance": "sync_answer_relevance",
    "context_relevance": "async_context_relevance",
    "completeness": "async_completeness",
}
COST_FN = 10
COST_FP = 1

PRODUCTION_THRESHOLDS = {
    "faithfulness": 0.30,
    "answer_relevance": 0.15,
    "context_relevance": 0.30,
    "completeness": 0.40,
}
EVAL_CURRENT_THRESHOLDS = {
    "faithfulness": 0.40,
    "answer_relevance": 0.30,
    "context_relevance": 0.50,
    "completeness": 0.50,
}


def sweep_metric(records: list[dict], metric: str) -> list[dict]:
    score_key = SCORE_KEYS[metric]
    is_positive = [r["gt_label"] == "UNGROUNDED" for r in records]
    results = []

    for threshold in THRESHOLDS:
        tp = fp = fn = tn = 0
        for i, r in enumerate(records):
            score = r[score_key]
            flagged = score is not None and score < threshold
            pos = is_positive[i]
            if pos and flagged:
                tp += 1
            elif not pos and flagged:
                fp += 1
            elif pos and not flagged:
                fn += 1
            else:
                tn += 1

        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden_j = sensitivity + specificity - 1.0
        cost = fn * COST_FN + fp * COST_FP

        results.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "youden_j": round(youden_j, 4),
            "cost": cost,
        })

    return results


def find_optimal(sweep_results: list[dict]) -> dict:
    f1_best = max(sweep_results, key=lambda x: x["f1"])
    j_best = max(sweep_results, key=lambda x: x["youden_j"])
    cost_best = min(sweep_results, key=lambda x: x["cost"])

    return {
        "f1_optimal": {
            "threshold": f1_best["threshold"],
            "f1": f1_best["f1"],
            "precision": f1_best["precision"],
            "recall": f1_best["recall"],
        },
        "youden_optimal": {
            "threshold": j_best["threshold"],
            "j": j_best["youden_j"],
        },
        "asymmetric_optimal": {
            "threshold": cost_best["threshold"],
            "cost": cost_best["cost"],
        },
    }


def apply_config(records: list[dict], thresholds: dict, use_badge_sync: bool = False) -> list[dict]:
    updated = deepcopy(records)
    for r in updated:
        faith = r["sync_faithfulness"]
        rel = r["sync_answer_relevance"]
        ctx = r["async_context_relevance"]
        comp = r["async_completeness"]

        if use_badge_sync:
            r["sync_flags"] = r["sync_badge"] in ("red", "gray")
        else:
            r["sync_flags"] = (
                (faith is not None and faith < thresholds["faithfulness"])
                or (rel is not None and rel < thresholds["answer_relevance"])
            )
        r["async_flags"] = (
            (ctx is not None and ctx < thresholds["context_relevance"])
            or (comp is not None and comp < thresholds["completeness"])
        )
        r["dual_flags"] = r["sync_flags"] or r["async_flags"]
    return updated


def detection_recall(flagged_key: str, subset: list[dict]) -> float:
    if not subset:
        return 0.0
    return sum(1 for r in subset if r[flagged_key]) / len(subset)


def compute_d1_d9(records: list[dict]) -> dict:
    valid = [r for r in records if r["gt_label"] != "ERROR"]
    ungrounded = [r for r in valid if r["gt_label"] == "UNGROUNDED"]

    sync_recall = detection_recall("sync_flags", ungrounded)
    async_recall = detection_recall("async_flags", ungrounded)
    dual_recall = detection_recall("dual_flags", ungrounded)

    sync_flagged = [r for r in valid if r["sync_flags"]]
    async_flagged = [r for r in valid if r["async_flags"]]
    dual_flagged = [r for r in valid if r["dual_flags"]]

    sync_precision = (sum(1 for r in sync_flagged if r["gt_label"] == "UNGROUNDED") / len(sync_flagged)) if sync_flagged else 0.0
    async_precision = (sum(1 for r in async_flagged if r["gt_label"] == "UNGROUNDED") / len(async_flagged)) if async_flagged else 0.0
    dual_precision = (sum(1 for r in dual_flagged if r["gt_label"] == "UNGROUNDED") / len(dual_flagged)) if dual_flagged else 0.0

    sync_latencies = [r["sync_latency_ms"] for r in valid if r["sync_latency_ms"] > 0]
    sync_p95 = sorted(sync_latencies)[int(len(sync_latencies) * 0.95)] if sync_latencies else 0

    cell_a = sum(1 for r in valid if not r["sync_flags"] and not r["async_flags"])
    cell_b = sum(1 for r in valid if not r["sync_flags"] and r["async_flags"])
    cell_c = sum(1 for r in valid if r["sync_flags"] and not r["async_flags"])
    cell_d = sum(1 for r in valid if r["sync_flags"] and r["async_flags"])

    sync_labels = [1 if r["sync_flags"] else 0 for r in valid]
    async_labels = [1 if r["async_flags"] else 0 for r in valid]
    kappa = float(cohens_kappa(sync_labels, async_labels))

    only_sync = sum(1 for r in ungrounded if r["sync_flags"] and not r["async_flags"])
    only_async = sum(1 for r in ungrounded if r["async_flags"] and not r["sync_flags"])
    both_caught = sum(1 for r in ungrounded if r["sync_flags"] and r["async_flags"])
    total_detected = only_sync + only_async + both_caught

    sync_unique_pct = only_sync / total_detected if total_detected else 0
    async_unique_pct = only_async / total_detected if total_detected else 0

    tier_data = defaultdict(list)
    for r in valid:
        tier_data[r["difficulty_tier"]].append(r)

    # D1: sync recall >= 50% AND p95 <= 3000ms
    d1 = sync_recall >= 0.50 and sync_p95 <= 3000
    # D2: async recall >= 40%
    d2 = async_recall >= 0.40
    # D3: Cell B > 0
    d3 = cell_b > 0
    # D4: Cell C > 0
    d4 = cell_c > 0
    # D5: kappa in [0.3, 0.8]
    d5 = 0.3 <= kappa <= 0.8
    # D6: both unique >= 10%
    d6 = sync_unique_pct >= 0.10 and async_unique_pct >= 0.10
    # D7: dual - sync gap >= 5pp
    dual_sync_gap = (dual_recall - sync_recall) * 100
    d7 = dual_sync_gap >= 5.0
    # D8: dual >= async
    d8 = dual_recall >= async_recall
    # D9: dual > sync AND dual >= async on every tier
    d9 = True
    tier_details = {}
    for tier in ["easy", "medium", "hard"]:
        tier_ung = [r for r in tier_data.get(tier, []) if r["gt_label"] == "UNGROUNDED"]
        if tier_ung:
            t_sr = detection_recall("sync_flags", tier_ung)
            t_ar = detection_recall("async_flags", tier_ung)
            t_dr = detection_recall("dual_flags", tier_ung)
            tier_ok = t_dr > t_sr and t_dr >= t_ar
            tier_details[tier] = {"sync": t_sr, "async": t_ar, "dual": t_dr, "pass": tier_ok}
            if not tier_ok:
                d9 = False
        else:
            d9 = False
            tier_details[tier] = {"sync": 0, "async": 0, "dual": 0, "pass": False}

    verdicts = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6, "D7": d7, "D8": d8, "D9": d9}

    return {
        "verdicts": verdicts,
        "pass_count": sum(verdicts.values()),
        "metrics": {
            "sync_recall": round(sync_recall, 4),
            "async_recall": round(async_recall, 4),
            "dual_recall": round(dual_recall, 4),
            "sync_precision": round(sync_precision, 4),
            "async_precision": round(async_precision, 4),
            "dual_precision": round(dual_precision, 4),
            "sync_p95_ms": round(sync_p95, 1),
            "kappa": round(kappa, 4),
            "cell_a": cell_a, "cell_b": cell_b, "cell_c": cell_c, "cell_d": cell_d,
            "only_sync": only_sync, "only_async": only_async, "both_caught": both_caught,
            "sync_unique_pct": round(sync_unique_pct, 4),
            "async_unique_pct": round(async_unique_pct, 4),
            "dual_sync_gap_pp": round(dual_sync_gap, 1),
            "sync_flags": sum(1 for r in valid if r["sync_flags"]),
            "async_flags": sum(1 for r in valid if r["async_flags"]),
            "dual_flags": sum(1 for r in valid if r["dual_flags"]),
        },
        "tier_details": tier_details,
    }


def sensitivity_analysis(records: list[dict], f1_thresholds: dict) -> dict:
    deltas = [-0.10, -0.05, 0.05, 0.10]
    baseline = compute_d1_d9(apply_config(records, f1_thresholds))
    results = {}

    for metric in METRICS:
        results[metric] = []
        for delta in deltas:
            varied = dict(f1_thresholds)
            new_val = round(varied[metric] + delta, 2)
            new_val = max(0.0, min(1.0, new_val))
            varied[metric] = new_val
            varied_result = compute_d1_d9(apply_config(records, varied))

            flipped = []
            for goal in [f"D{i}" for i in range(1, 10)]:
                if varied_result["verdicts"][goal] != baseline["verdicts"][goal]:
                    direction = "PASS→FAIL" if baseline["verdicts"][goal] else "FAIL→PASS"
                    flipped.append({"goal": goal, "direction": direction})

            results[metric].append({
                "delta": delta,
                "threshold": new_val,
                "pass_count": varied_result["pass_count"],
                "flipped_goals": flipped,
            })

    goal_robustness = {}
    for goal in [f"D{i}" for i in range(1, 10)]:
        flips = set()
        for metric in METRICS:
            for entry in results[metric]:
                for flip in entry["flipped_goals"]:
                    if flip["goal"] == goal:
                        flips.add(f"{metric} {'+' if entry['delta'] > 0 else ''}{entry['delta']}")
        goal_robustness[goal] = {
            "robust": len(flips) == 0,
            "flips_at": sorted(flips) if flips else [],
        }

    return {"per_metric": results, "goal_robustness": goal_robustness}


def print_tables(sweep_data: dict, optimal: dict, config_results: dict, sensitivity: dict):
    # Table 1: Per-metric threshold sweep (top 5 by F1)
    for metric in METRICS:
        sweep = sweep_data[metric]
        top5 = sorted(sweep, key=lambda x: x["f1"], reverse=True)[:5]
        best_t = optimal[metric]["f1_optimal"]["threshold"]

        t = Table(title=f"{metric.upper()} THRESHOLD SWEEP (top 5 by F1)")
        for col in ["Threshold", "TP", "FP", "FN", "TN", "Prec", "Rec", "F1"]:
            t.add_column(col, justify="right")

        for row in top5:
            star = " *" if row["threshold"] == best_t else ""
            t.add_row(
                f"{row['threshold']:.2f}",
                str(row["tp"]), str(row["fp"]), str(row["fn"]), str(row["tn"]),
                f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                f"{row['f1']:.3f}{star}",
            )
        console.print(t)
        console.print()

    # Table 2: Optimal thresholds summary
    t = Table(title="OPTIMAL THRESHOLDS")
    t.add_column("Metric")
    t.add_column("F1-opt", justify="right")
    t.add_column("Youden-J", justify="right")
    t.add_column("Asymmetric", justify="right")
    t.add_column("Production", justify="right")

    for metric in METRICS:
        o = optimal[metric]
        t.add_row(
            metric,
            f"{o['f1_optimal']['threshold']:.2f}",
            f"{o['youden_optimal']['threshold']:.2f}",
            f"{o['asymmetric_optimal']['threshold']:.2f}",
            f"{PRODUCTION_THRESHOLDS[metric]:.2f}",
        )
    console.print(t)
    console.print()

    # Table 3: D1-D9 verdicts across configurations
    config_names = ["f1_optimal", "asymmetric_cost", "production", "eval_current"]
    display_names = {"f1_optimal": "F1-opt", "asymmetric_cost": "Asym-cost", "production": "Production", "eval_current": "Eval-cur"}
    goal_descriptions = {
        "D1": "Sync recall>=50%, p95<=3s",
        "D2": "Async recall>=40%",
        "D3": "Cell B > 0",
        "D4": "Cell C > 0",
        "D5": "kappa in [0.3, 0.8]",
        "D6": "Both unique>=10%",
        "D7": "Dual-sync gap>=5pp",
        "D8": "Dual >= async",
        "D9": "Dual wins all tiers",
    }

    t = Table(title="D1-D9 VERDICTS ACROSS THRESHOLD CONFIGURATIONS")
    t.add_column("Goal")
    t.add_column("Description")
    for name in config_names:
        t.add_column(display_names[name], justify="center")

    for goal in [f"D{i}" for i in range(1, 10)]:
        row = [goal, goal_descriptions[goal]]
        for name in config_names:
            v = config_results[name]["verdicts"][goal]
            row.append("[green]PASS[/green]" if v else "[red]FAIL[/red]")
        t.add_row(*row)

    totals = ["", "TOTAL"]
    for name in config_names:
        totals.append(f"[bold]{config_results[name]['pass_count']}/9[/bold]")
    t.add_row(*totals, style="bold")
    console.print(t)
    console.print()

    # Table 4: Key metrics comparison
    t = Table(title="KEY METRICS ACROSS CONFIGURATIONS")
    t.add_column("Metric")
    for name in config_names:
        t.add_column(display_names[name], justify="right")

    metric_rows = [
        ("Sync flags", "sync_flags", "{}", False),
        ("Async flags", "async_flags", "{}", False),
        ("Dual flags", "dual_flags", "{}", False),
        ("Sync recall", "sync_recall", "{:.1%}", True),
        ("Async recall", "async_recall", "{:.1%}", True),
        ("Dual recall", "dual_recall", "{:.1%}", True),
        ("Cohen's kappa", "kappa", "{:.3f}", True),
        ("Cell B", "cell_b", "{}", False),
        ("Cell C", "cell_c", "{}", False),
        ("Sync unique %", "sync_unique_pct", "{:.1%}", True),
        ("Async unique %", "async_unique_pct", "{:.1%}", True),
    ]

    for label, key, fmt, _ in metric_rows:
        row = [label]
        for name in config_names:
            val = config_results[name]["metrics"][key]
            row.append(fmt.format(val))
        t.add_row(*row)
    console.print(t)
    console.print()

    # Table 5: Sensitivity analysis
    t = Table(title="SENSITIVITY: WHICH GOALS FLIP WITH +/-0.10 THRESHOLD VARIATION?")
    t.add_column("Goal")
    t.add_column("Robust?", justify="center")
    t.add_column("Flips at")

    for goal in [f"D{i}" for i in range(1, 10)]:
        rob = sensitivity["goal_robustness"][goal]
        if rob["robust"]:
            status = "[green]ROBUST[/green]"
            detail = "passes at all tested thresholds"
        else:
            status = "[red]SENSITIVE[/red]"
            detail = ", ".join(rob["flips_at"])
        t.add_row(goal, status, detail)
    console.print(t)


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    records = data["records"]
    console.print(f"[bold]Loaded {len(records)} records from {RESULTS_PATH.name}[/bold]\n")

    # Step 1-2: Sweep each metric
    sweep_data = {}
    optimal = {}
    for metric in METRICS:
        sweep_data[metric] = sweep_metric(records, metric)
        optimal[metric] = find_optimal(sweep_data[metric])

    # Step 3: Build 4 threshold configs
    f1_thresholds = {m: optimal[m]["f1_optimal"]["threshold"] for m in METRICS}
    ac_thresholds = {m: optimal[m]["asymmetric_optimal"]["threshold"] for m in METRICS}

    configs = {
        "f1_optimal": f1_thresholds,
        "asymmetric_cost": ac_thresholds,
        "production": dict(PRODUCTION_THRESHOLDS),
        "eval_current": dict(EVAL_CURRENT_THRESHOLDS),
    }

    # Step 4: Recompute D1-D9 for each config
    config_results = {}
    for name, thresholds in configs.items():
        updated = apply_config(records, thresholds, use_badge_sync=(name == "eval_current"))
        result = compute_d1_d9(updated)
        result["thresholds"] = thresholds
        config_results[name] = result

    # Step 5: Sensitivity analysis
    sens = sensitivity_analysis(records, f1_thresholds)

    # Step 6: Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep_output = {}
    for metric in METRICS:
        sweep_output[metric] = {
            "sweep": sweep_data[metric],
            **optimal[metric],
        }
    with open(OUTPUT_DIR / "threshold_sweep.json", "w") as f:
        json.dump(sweep_output, f, indent=2)

    with open(OUTPUT_DIR / "config_comparison.json", "w") as f:
        json.dump(config_results, f, indent=2)

    with open(OUTPUT_DIR / "sensitivity.json", "w") as f:
        json.dump(sens, f, indent=2)

    console.print(f"[green]Saved results to {OUTPUT_DIR}/[/green]\n")

    # Step 7: Print tables
    print_tables(sweep_data, optimal, config_results, sens)


if __name__ == "__main__":
    main()
