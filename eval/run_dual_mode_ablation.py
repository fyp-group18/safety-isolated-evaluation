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

from src.async_eval import evaluate_async
from src.config import EVAL_DATASET_PATH, RESULTS_DIR, TOP_K
from src.flat_rag import get_collection, retrieve
from src.generator import generate_repair_plan
from src.metrics import auroc, bootstrap_ci, cohens_kappa
from src.sync_gate import evaluate_sync

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

CHECKPOINT_PATH = RESULTS_DIR / "dual_mode_checkpoint.jsonl"


def _load_checkpoint() -> dict[str, dict]:
    done = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
    return done


def _append_checkpoint(record: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def run():
    with open(EVAL_DATASET_PATH) as f:
        records = json.load(f)

    collection = get_collection()
    checkpoint = _load_checkpoint()
    results = list(checkpoint.values())
    done_ids = set(checkpoint.keys())

    remaining = [r for r in records if r["id"] not in done_ids]
    console.print(f"[bold]Dual-mode ablation: {len(remaining)} remaining of {len(records)}[/bold]")

    for r in tqdm(remaining, desc="Dual-mode ablation"):
        try:
            # 1. Retrieve
            retrieved = retrieve(r["observation"], k=TOP_K, collection=collection)
            gt_ids = set(r.get("ground_truth_chunk_ids", [r["ground_truth_chunk_id"]]))
            ret_ids = [c.chunk_id for c in retrieved]
            hit = any(rid in gt_ids for rid in ret_ids)

            # 2. Build context
            context = "\n\n---\n\n".join(
                f"[CHUNK-ID: {c.chunk_id}]\n{c.text}" for c in retrieved
            )

            # 3. Generate repair plan
            plan = generate_repair_plan(r["observation"], context)
            if not plan:
                plan = {"repair_steps": [], "root_cause": "generation_failed", "confidence": 0.0}

            plan_text = "\n".join(
                s.get("text", str(s)) if isinstance(s, dict) else str(s)
                for s in plan.get("repair_steps", [])
            )

            # 4. Sync gate
            sync_result = evaluate_sync(
                question=r["observation"],
                repair_plan=plan,
                context=context,
                chunk_ids=ret_ids,
            )

            # 5. Async eval
            async_result = evaluate_async(
                question=r["observation"],
                plan_text=plan_text,
                context=context,
            )

            # 6. Auto-label ground truth
            fault_match = r["ground_truth_fault_code"].lower() in (plan.get("root_cause", "") or "").lower()
            if not hit and not fault_match:
                gt_label = "UNGROUNDED"
            elif hit and fault_match:
                gt_label = "GROUNDED"
            else:
                gt_label = "AMBIGUOUS"

            # 7. Detection labels
            sync_flags = not sync_result.passed
            async_flags = (
                (async_result.context_relevance is not None and async_result.context_relevance < 0.5) or
                (async_result.completeness is not None and async_result.completeness < 0.5)
            )

            record = {
                "id": r["id"],
                "difficulty_tier": r["difficulty_tier"],
                "observation_type": r["observation_type"],
                "hit": hit,
                "gt_label": gt_label,
                "sync_passed": sync_result.passed,
                "sync_badge": sync_result.badge,
                "sync_faithfulness": sync_result.faithfulness_score,
                "sync_answer_relevance": sync_result.answer_relevance_score,
                "sync_latency_ms": sync_result.latency_ms,
                "sync_flags": sync_flags,
                "async_context_relevance": async_result.context_relevance,
                "async_completeness": async_result.completeness,
                "async_safety_coverage": async_result.safety_coverage,
                "async_latency_ms": async_result.latency_ms,
                "async_flags": async_flags,
                "dual_flags": sync_flags or async_flags,
                "plan_steps": len(plan.get("repair_steps", [])),
            }

            results.append(record)
            _append_checkpoint(record)

        except Exception as e:
            logger.error(f"Case {r['id']} failed: {e}")
            record = {
                "id": r["id"],
                "difficulty_tier": r["difficulty_tier"],
                "observation_type": r["observation_type"],
                "error": str(e),
                "hit": False,
                "gt_label": "ERROR",
                "sync_passed": None, "sync_badge": "gray",
                "sync_faithfulness": None, "sync_answer_relevance": None,
                "sync_latency_ms": 0, "sync_flags": False,
                "async_context_relevance": None, "async_completeness": None,
                "async_safety_coverage": None, "async_latency_ms": 0,
                "async_flags": False, "dual_flags": False, "plan_steps": 0,
            }
            results.append(record)
            _append_checkpoint(record)

    # Compute metrics
    valid = [r for r in results if r["gt_label"] != "ERROR"]
    grounded = [r for r in valid if r["gt_label"] == "GROUNDED"]
    ungrounded = [r for r in valid if r["gt_label"] == "UNGROUNDED"]
    non_ambiguous = [r for r in valid if r["gt_label"] != "AMBIGUOUS"]

    console.print(f"\n[bold]═══ DUAL-MODE ABLATION RESULTS ═══[/bold]\n")

    # Label distribution
    t = Table(title="Ground Truth Label Distribution")
    t.add_column("Label"); t.add_column("Count"); t.add_column("Pct")
    for label in ["GROUNDED", "UNGROUNDED", "AMBIGUOUS", "ERROR"]:
        count = sum(1 for r in results if r["gt_label"] == label)
        t.add_row(label, str(count), f"{count/len(results):.1%}")
    console.print(t)

    # Sync gate metrics
    sync_scores = [r["sync_faithfulness"] for r in valid if r["sync_faithfulness"] is not None]
    sync_latencies = [r["sync_latency_ms"] for r in valid]

    t2 = Table(title="Sync Gate Metrics")
    t2.add_column("Metric"); t2.add_column("Value")
    if sync_scores:
        t2.add_row("Mean Faithfulness", f"{sum(sync_scores)/len(sync_scores):.3f}")
    t2.add_row("Latency p50", f"{sorted(sync_latencies)[len(sync_latencies)//2]:.0f}ms")
    t2.add_row("Latency p95", f"{sorted(sync_latencies)[int(len(sync_latencies)*0.95)]:.0f}ms")
    badges = defaultdict(int)
    for r in valid:
        badges[r["sync_badge"]] += 1
    for badge in ["green", "yellow", "red", "gray"]:
        t2.add_row(f"Badge: {badge}", str(badges.get(badge, 0)))
    console.print(t2)

    # Agreement matrix (sync vs async flags)
    cell_a = sum(1 for r in valid if not r["sync_flags"] and not r["async_flags"])
    cell_b = sum(1 for r in valid if not r["sync_flags"] and r["async_flags"])
    cell_c = sum(1 for r in valid if r["sync_flags"] and not r["async_flags"])
    cell_d = sum(1 for r in valid if r["sync_flags"] and r["async_flags"])

    t3 = Table(title="Agreement Matrix (Sync x Async)")
    t3.add_column(""); t3.add_column("Async OK"); t3.add_column("Async Flags")
    t3.add_row("Sync Pass", str(cell_a), str(cell_b))
    t3.add_row("Sync Block", str(cell_c), str(cell_d))
    console.print(t3)

    # Detection recall per condition
    def detection_recall(flagged_key, subset):
        if not subset:
            return 0.0
        return sum(1 for r in subset if r[flagged_key]) / len(subset)

    sync_recall = detection_recall("sync_flags", ungrounded)
    async_recall = detection_recall("async_flags", ungrounded)
    dual_recall = detection_recall("dual_flags", ungrounded)

    t4 = Table(title="Detection Recall (on UNGROUNDED cases)")
    t4.add_column("Condition"); t4.add_column("Recall"); t4.add_column("N flagged"); t4.add_column("N total")
    t4.add_row("SYNC_ONLY", f"{sync_recall:.1%}", str(sum(1 for r in ungrounded if r["sync_flags"])), str(len(ungrounded)))
    t4.add_row("ASYNC_ONLY", f"{async_recall:.1%}", str(sum(1 for r in ungrounded if r["async_flags"])), str(len(ungrounded)))
    t4.add_row("DUAL_MODE", f"{dual_recall:.1%}", str(sum(1 for r in ungrounded if r["dual_flags"])), str(len(ungrounded)))
    console.print(t4)

    # AUROC
    if non_ambiguous:
        y_true = [1 if r["gt_label"] == "UNGROUNDED" else 0 for r in non_ambiguous]
        y_sync = [1.0 - (r["sync_faithfulness"] or 0.5) for r in non_ambiguous]
        sync_auroc = auroc(y_true, y_sync)
    else:
        sync_auroc = None

    # Cohen's kappa
    sync_labels = [1 if r["sync_flags"] else 0 for r in valid]
    async_labels = [1 if r["async_flags"] else 0 for r in valid]
    kappa = cohens_kappa(sync_labels, async_labels)

    # Stratify by difficulty
    tier_data = defaultdict(list)
    for r in valid:
        tier_data[r["difficulty_tier"]].append(r)

    t5 = Table(title="By Difficulty Tier")
    t5.add_column("Tier"); t5.add_column("N"); t5.add_column("Sync Recall"); t5.add_column("Dual Recall"); t5.add_column("Gap")
    for tier in ["easy", "medium", "hard"]:
        tier_ung = [r for r in tier_data.get(tier, []) if r["gt_label"] == "UNGROUNDED"]
        if tier_ung:
            sr = detection_recall("sync_flags", tier_ung)
            dr = detection_recall("dual_flags", tier_ung)
            t5.add_row(tier, str(len(tier_ung)), f"{sr:.1%}", f"{dr:.1%}", f"{(dr-sr)*100:.1f}pp")
        else:
            t5.add_row(tier, "0", "N/A", "N/A", "N/A")
    console.print(t5)

    # Real-time protection
    sync_blocks_ungrounded = sum(1 for r in ungrounded if r["sync_flags"]) if ungrounded else 0
    sync_block_rate = sync_blocks_ungrounded / len(ungrounded) if ungrounded else 0

    # Verdicts
    console.print("\n[bold]═══ RESEARCH GOAL VERDICTS ═══[/bold]\n")

    d1_pass = sync_auroc is not None and sync_auroc >= 0.70
    d1_fail = sync_auroc is not None and sync_auroc < 0.60
    d1_status = "[green]PASS[/green]" if d1_pass else ("[red]FAIL[/red]" if d1_fail else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"  D1  {d1_status}  Sync AUROC = {sync_auroc:.2f}" if sync_auroc else f"  D1  [yellow]INCONCLUSIVE[/yellow]  AUROC = N/A")

    sync_p95 = sorted(sync_latencies)[int(len(sync_latencies) * 0.95)] if sync_latencies else 0
    d2_pass = sync_p95 <= 3000
    d2_status = "[green]PASS[/green]" if d2_pass else "[red]FAIL[/red]"
    console.print(f"  D2  {d2_status}  Sync latency p95 = {sync_p95:.0f}ms (target <= 3000ms)")

    d3_pass = cell_b > 0
    d3_status = "[green]PASS[/green]" if d3_pass else "[red]FAIL[/red]"
    console.print(f"  D3  {d3_status}  Cell B (sync pass, async flags) = {cell_b}")

    d4_pass = 0.3 <= kappa <= 0.8
    d4_status = "[green]PASS[/green]" if d4_pass else "[red]FAIL[/red]"
    console.print(f"  D4  {d4_status}  Cohen's kappa = {kappa:.2f} (target 0.3-0.8)")

    dual_gap = (dual_recall - sync_recall) * 100
    d5_pass = dual_gap >= 5
    d5_status = "[green]PASS[/green]" if d5_pass else ("[red]FAIL[/red]" if dual_gap < 2 else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"  D5  {d5_status}  DUAL recall = {dual_recall:.1%} vs SYNC = {sync_recall:.1%} (gap = {dual_gap:.1f}pp)")

    d6_pass = sync_block_rate >= 0.80
    d6_status = "[green]PASS[/green]" if d6_pass else ("[red]FAIL[/red]" if sync_block_rate < 0.50 else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"  D6  {d6_status}  Sync blocks {sync_block_rate:.1%} of UNGROUNDED in real-time")

    # D7: hard gap > easy gap
    hard_ung = [r for r in tier_data.get("hard", []) if r["gt_label"] == "UNGROUNDED"]
    easy_ung = [r for r in tier_data.get("easy", []) if r["gt_label"] == "UNGROUNDED"]
    hard_gap = (detection_recall("dual_flags", hard_ung) - detection_recall("sync_flags", hard_ung)) * 100 if hard_ung else 0
    easy_gap = (detection_recall("dual_flags", easy_ung) - detection_recall("sync_flags", easy_ung)) * 100 if easy_ung else 0
    d7_pass = hard_gap > easy_gap
    d7_status = "[green]PASS[/green]" if d7_pass else "[yellow]INCONCLUSIVE[/yellow]"
    console.print(f"  D7  {d7_status}  Hard gap = {hard_gap:.1f}pp vs Easy gap = {easy_gap:.1f}pp")

    passed = sum([d1_pass, d2_pass, d3_pass, d4_pass, d5_pass, d6_pass, d7_pass])
    console.print(f"\n  OVERALL: {passed}/7 PASS\n")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "sync_auroc": sync_auroc,
        "sync_p95_ms": sync_p95,
        "kappa": kappa,
        "cell_a": cell_a, "cell_b": cell_b, "cell_c": cell_c, "cell_d": cell_d,
        "sync_recall": sync_recall, "async_recall": async_recall, "dual_recall": dual_recall,
        "sync_block_rate": sync_block_rate,
        "label_counts": {
            "GROUNDED": len(grounded), "UNGROUNDED": len(ungrounded),
            "AMBIGUOUS": sum(1 for r in valid if r["gt_label"] == "AMBIGUOUS"),
        },
        "verdicts": {
            "D1": d1_pass, "D2": d2_pass, "D3": d3_pass, "D4": d4_pass,
            "D5": d5_pass, "D6": d6_pass, "D7": d7_pass,
        },
        "records": results,
    }
    with open(RESULTS_DIR / "dual_mode_ablation.json", "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"[dim]Results saved to {RESULTS_DIR / 'dual_mode_ablation.json'}[/dim]")

    return output


if __name__ == "__main__":
    run()
