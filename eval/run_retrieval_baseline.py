import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EVAL_DATASET_PATH, RESULTS_DIR, TOP_K
from src.flat_rag import get_collection, retrieve
from src.metrics import bootstrap_ci, hit_rate_at_k, mrr

logging.basicConfig(level=logging.WARNING)
console = Console()

GOALS = {
    "R1": {"target": 0.60, "fail": 0.50, "label": "Hit Rate @5"},
    "R2": {"target": 10, "fail": 5, "label": "Difficulty gap (pp)"},
    "R3": {"target": 0.10, "fail": 0.05, "label": "Complete miss rate"},
}


def run():
    with open(EVAL_DATASET_PATH) as f:
        records = json.load(f)

    collection = get_collection()
    results = []
    retrieved_ids_all = []
    gt_ids_all = []

    for r in tqdm(records, desc="Retrieval baseline"):
        t0 = time.perf_counter()
        retrieved = retrieve(r["observation"], k=TOP_K, collection=collection)
        latency_ms = (time.perf_counter() - t0) * 1000

        gt_ids = set(r.get("ground_truth_chunk_ids", [r["ground_truth_chunk_id"]]))
        ret_ids = [c.chunk_id for c in retrieved]
        hit = any(rid in gt_ids for rid in ret_ids)

        rank = None
        for i, rid in enumerate(ret_ids):
            if rid in gt_ids:
                rank = i + 1
                break

        results.append({
            "id": r["id"],
            "hit": hit,
            "rank": rank,
            "difficulty_tier": r["difficulty_tier"],
            "origin": r["origin"],
            "observation_type": r["observation_type"],
            "manual_source": r["manual_source"],
            "cosine_scores": [c.cosine_score for c in retrieved],
            "latency_ms": latency_ms,
            "retrieved_ids": ret_ids,
        })
        retrieved_ids_all.append(ret_ids)
        gt_ids_all.append(gt_ids)

    # Compute metrics
    overall_hr = hit_rate_at_k(retrieved_ids_all, gt_ids_all, k=TOP_K)
    overall_mrr = mrr(retrieved_ids_all, gt_ids_all)
    miss_rate = sum(1 for r in results if not r["hit"]) / len(results)

    # Stratify by difficulty
    tier_results = defaultdict(list)
    for r in results:
        tier_results[r["difficulty_tier"]].append(r)

    tier_metrics = {}
    for tier, recs in sorted(tier_results.items()):
        tier_hr = sum(1 for r in recs if r["hit"]) / len(recs)
        tier_metrics[tier] = {"hit_rate": tier_hr, "count": len(recs)}

    # Stratify by observation type
    type_results = defaultdict(list)
    for r in results:
        type_results[r["observation_type"]].append(r)

    type_metrics = {}
    for otype, recs in sorted(type_results.items()):
        type_hr = sum(1 for r in recs if r["hit"]) / len(recs)
        type_metrics[otype] = {"hit_rate": type_hr, "count": len(recs)}

    # Bootstrap CI
    hit_values = [1.0 if r["hit"] else 0.0 for r in results]
    mean_hr, ci_low, ci_high = bootstrap_ci(hit_values)

    # Print results
    console.print("\n[bold]═══ RETRIEVAL BASELINE RESULTS ═══[/bold]\n")

    t = Table(title="Overall Metrics")
    t.add_column("Metric"); t.add_column("Value")
    t.add_row("Hit Rate @5", f"{overall_hr:.1%}")
    t.add_row("MRR", f"{overall_mrr:.3f}")
    t.add_row("Miss Rate", f"{miss_rate:.1%}")
    t.add_row("95% CI", f"[{ci_low:.1%}, {ci_high:.1%}]")
    t.add_row("N", str(len(results)))
    console.print(t)

    t2 = Table(title="By Difficulty Tier")
    t2.add_column("Tier"); t2.add_column("N"); t2.add_column("Hit Rate @5")
    for tier in ["easy", "medium", "hard"]:
        if tier in tier_metrics:
            m = tier_metrics[tier]
            t2.add_row(tier, str(m["count"]), f"{m['hit_rate']:.1%}")
    console.print(t2)

    t3 = Table(title="By Observation Type")
    t3.add_column("Type"); t3.add_column("N"); t3.add_column("Hit Rate @5")
    for otype, m in sorted(type_metrics.items()):
        t3.add_row(otype, str(m["count"]), f"{m['hit_rate']:.1%}")
    console.print(t3)

    # Verdicts
    easy_hr = tier_metrics.get("easy", {}).get("hit_rate", 0)
    hard_hr = tier_metrics.get("hard", {}).get("hit_rate", 0)
    gap_pp = (easy_hr - hard_hr) * 100

    console.print("\n[bold]═══ RESEARCH GOAL VERDICTS ═══[/bold]\n")

    r1_pass = overall_hr >= GOALS["R1"]["target"]
    r1_status = "[green]PASS[/green]" if r1_pass else ("[red]FAIL[/red]" if overall_hr < GOALS["R1"]["fail"] else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"  R1  {r1_status}  Hit Rate @5 = {overall_hr:.1%} (target >= 60%)")
    console.print(f'      -> "The retriever finds the correct manual section in its')
    console.print(f'         top 5 for {overall_hr:.0%} of cases."')

    r2_pass = gap_pp >= GOALS["R2"]["target"]
    r2_status = "[green]PASS[/green]" if r2_pass else ("[red]FAIL[/red]" if gap_pp < GOALS["R2"]["fail"] else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"\n  R2  {r2_status}  Gap = {gap_pp:.1f}pp (easy {easy_hr:.1%} - hard {hard_hr:.1%}, target >= 10pp)")
    console.print(f'      -> "Easy cases hit at {easy_hr:.0%}, hard cases at {hard_hr:.0%} — the')
    console.print(f'         difficulty tiers genuinely separate retrieval difficulty."')

    r3_pass = miss_rate >= GOALS["R3"]["target"]
    r3_status = "[green]PASS[/green]" if r3_pass else ("[red]FAIL[/red]" if miss_rate < GOALS["R3"]["fail"] else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"\n  R3  {r3_status}  Miss rate = {miss_rate:.1%} (target >= 10%)")
    console.print(f'      -> "{miss_rate:.0%} of cases return zero relevant chunks — these are the')
    console.print(f'         cases where the sync gate MUST block."')

    passed = sum([r1_pass, r2_pass, r3_pass])
    console.print(f"\n  OVERALL: {passed}/3 PASS\n")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "overall_hit_rate": overall_hr,
        "overall_mrr": overall_mrr,
        "miss_rate": miss_rate,
        "ci_95": [ci_low, ci_high],
        "tier_metrics": tier_metrics,
        "type_metrics": type_metrics,
        "verdicts": {
            "R1": {"pass": r1_pass, "value": overall_hr},
            "R2": {"pass": r2_pass, "value": gap_pp},
            "R3": {"pass": r3_pass, "value": miss_rate},
        },
        "records": results,
    }
    with open(RESULTS_DIR / "retrieval_baseline.json", "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"[dim]Results saved to {RESULTS_DIR / 'retrieval_baseline.json'}[/dim]")

    return output


if __name__ == "__main__":
    run()
