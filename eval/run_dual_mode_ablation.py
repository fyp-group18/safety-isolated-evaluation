import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.async_eval import evaluate_async
from src.config import EVAL_DATASET_PATH, MAX_WORKERS, RESULTS_DIR, TOP_K
from src.flat_rag import get_collection, retrieve
from src.generator import generate_repair_plan
from src.metrics import bootstrap_ci, cohens_kappa, precision_recall_f1
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
        f.write(json.dumps(record, default=str) + "\n")


def _process_record(r: dict, collection) -> dict:
    try:
        retrieved = retrieve(r["observation"], k=TOP_K, collection=collection)
        gt_ids = set(r.get("ground_truth_chunk_ids", [r["ground_truth_chunk_id"]]))
        ret_ids = [c.chunk_id for c in retrieved]
        hit = any(rid in gt_ids for rid in ret_ids)

        context = "\n\n---\n\n".join(
            f"[CHUNK-ID: {c.chunk_id}]\n{c.text}" for c in retrieved
        )

        plan = generate_repair_plan(r["observation"], context)
        if not plan:
            plan = {"repair_steps": [], "root_cause": "generation_failed", "confidence": 0.0}

        plan_text = "\n".join(
            s.get("text", str(s)) if isinstance(s, dict) else str(s)
            for s in plan.get("repair_steps", [])
        )

        sync_result = evaluate_sync(
            question=r["observation"],
            repair_plan=plan,
            context=context,
            chunk_ids=ret_ids,
        )

        async_result = evaluate_async(
            question=r["observation"],
            plan_text=plan_text,
            context=context,
        )

        fault_match = r["ground_truth_fault_code"].lower() in (plan.get("root_cause", "") or "").lower()
        if not hit and not fault_match:
            gt_label = "UNGROUNDED"
        elif hit and fault_match:
            gt_label = "GROUNDED"
        else:
            gt_label = "AMBIGUOUS"

        sync_flags = not sync_result.passed
        async_flags = (
            (async_result.context_relevance is not None and async_result.context_relevance < 0.5)
            or (async_result.completeness is not None and async_result.completeness < 0.5)
        )

        return {
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

    except Exception as e:
        logger.error(f"Case {r['id']} failed: {e}")
        return {
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


def run():
    with open(EVAL_DATASET_PATH) as f:
        records = json.load(f)

    collection = get_collection()
    checkpoint = _load_checkpoint()
    results = list(checkpoint.values())
    done_ids = set(checkpoint.keys())
    remaining = [r for r in records if r["id"] not in done_ids]

    console.print(f"[bold]Dual-mode ablation: {len(remaining)} remaining of {len(records)}[/bold]")

    if remaining:
        counts = defaultdict(int)
        rate_limit_hits = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_record, r, collection): r for r in remaining}
            pbar = tqdm(total=len(remaining), desc="[DUAL-MODE]")

            for future in as_completed(futures):
                record = future.result()
                results.append(record)
                _append_checkpoint(record)
                counts[record["gt_label"]] += 1
                pbar.set_postfix({
                    "GND": counts["GROUNDED"],
                    "UNG": counts["UNGROUNDED"],
                    "AMB": counts["AMBIGUOUS"],
                    "ERR": counts["ERROR"],
                })
                pbar.update(1)

            pbar.close()

    # === METRICS ===
    valid = [r for r in results if r["gt_label"] != "ERROR"]
    grounded = [r for r in valid if r["gt_label"] == "GROUNDED"]
    ungrounded = [r for r in valid if r["gt_label"] == "UNGROUNDED"]

    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(f"[bold]  DUAL-MODE ABLATION RESULTS ({len(valid)} valid / {len(results)} total)[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]\n")

    # Label distribution
    t = Table(title="Ground Truth Label Distribution")
    t.add_column("Label"); t.add_column("Count"); t.add_column("Pct")
    for label in ["GROUNDED", "UNGROUNDED", "AMBIGUOUS", "ERROR"]:
        count = sum(1 for r in results if r["gt_label"] == label)
        t.add_row(label, str(count), f"{count / len(results):.1%}")
    console.print(t)

    # Detection recall helper
    def detection_recall(flagged_key, subset):
        if not subset:
            return 0.0
        return sum(1 for r in subset if r[flagged_key]) / len(subset)

    def detection_precision(flagged_key, label_key, label_val, all_records):
        flagged = [r for r in all_records if r[flagged_key]]
        if not flagged:
            return 0.0
        true_pos = sum(1 for r in flagged if r[label_key] == label_val)
        return true_pos / len(flagged)

    sync_recall = detection_recall("sync_flags", ungrounded)
    async_recall = detection_recall("async_flags", ungrounded)
    dual_recall = detection_recall("dual_flags", ungrounded)

    sync_precision = detection_precision("sync_flags", "gt_label", "UNGROUNDED", valid)
    async_precision = detection_precision("async_flags", "gt_label", "UNGROUNDED", valid)
    dual_precision = detection_precision("dual_flags", "gt_label", "UNGROUNDED", valid)

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # === 3-Way Detection Recall by Difficulty ===
    tier_data = defaultdict(list)
    for r in valid:
        tier_data[r["difficulty_tier"]].append(r)

    t_recall = Table(title="3-Way Detection Recall (on UNGROUNDED)")
    t_recall.add_column("Difficulty"); t_recall.add_column("N (UNG)")
    t_recall.add_column("SYNC_ONLY"); t_recall.add_column("ASYNC_ONLY"); t_recall.add_column("DUAL_MODE")

    for tier in ["easy", "medium", "hard"]:
        tier_ung = [r for r in tier_data.get(tier, []) if r["gt_label"] == "UNGROUNDED"]
        if tier_ung:
            sr = detection_recall("sync_flags", tier_ung)
            ar = detection_recall("async_flags", tier_ung)
            dr = detection_recall("dual_flags", tier_ung)
            t_recall.add_row(f"{tier} ({len(tier_data.get(tier, []))})", str(len(tier_ung)),
                             f"{sr:.1%}", f"{ar:.1%}", f"{dr:.1%}")
        else:
            t_recall.add_row(f"{tier}", "0", "N/A", "N/A", "N/A")

    t_recall.add_row(f"All ({len(valid)})", str(len(ungrounded)),
                     f"{sync_recall:.1%}", f"{async_recall:.1%}", f"{dual_recall:.1%}",
                     style="bold")
    console.print(t_recall)

    # === Agreement Matrix ===
    cell_a = sum(1 for r in valid if not r["sync_flags"] and not r["async_flags"])
    cell_b = sum(1 for r in valid if not r["sync_flags"] and r["async_flags"])
    cell_c = sum(1 for r in valid if r["sync_flags"] and not r["async_flags"])
    cell_d = sum(1 for r in valid if r["sync_flags"] and r["async_flags"])

    t_agree = Table(title="Agreement Matrix (Sync x Async)")
    t_agree.add_column(""); t_agree.add_column("Async OK"); t_agree.add_column("Async Flags")
    t_agree.add_row("Sync Pass", f"A = {cell_a}", f"B = {cell_b}")
    t_agree.add_row("Sync Block", f"C = {cell_c}", f"D = {cell_d}")
    console.print(t_agree)

    # === Standalone Comparison ===
    sync_latencies = [r["sync_latency_ms"] for r in valid if r["sync_latency_ms"] > 0]
    async_latencies = [r["async_latency_ms"] for r in valid if r["async_latency_ms"] > 0]
    sync_p95 = sorted(sync_latencies)[int(len(sync_latencies) * 0.95)] if sync_latencies else 0
    async_p95 = sorted(async_latencies)[int(len(async_latencies) * 0.95)] if async_latencies else 0

    t_standalone = Table(title="Standalone Comparison")
    t_standalone.add_column("Metric"); t_standalone.add_column("SYNC_ONLY")
    t_standalone.add_column("ASYNC_ONLY"); t_standalone.add_column("DUAL_MODE")
    t_standalone.add_row("Recall", f"{sync_recall:.1%}", f"{async_recall:.1%}", f"{dual_recall:.1%}")
    t_standalone.add_row("Precision", f"{sync_precision:.1%}", f"{async_precision:.1%}", f"{dual_precision:.1%}")
    t_standalone.add_row("F1", f"{f1(sync_precision, sync_recall):.1%}",
                         f"{f1(async_precision, async_recall):.1%}",
                         f"{f1(dual_precision, dual_recall):.1%}")
    t_standalone.add_row("Latency p95", f"{sync_p95:.0f}ms", f"{async_p95:.0f}ms", "-")
    console.print(t_standalone)

    # === Unique Contribution ===
    only_sync = sum(1 for r in ungrounded if r["sync_flags"] and not r["async_flags"])
    only_async = sum(1 for r in ungrounded if r["async_flags"] and not r["sync_flags"])
    both_caught = sum(1 for r in ungrounded if r["sync_flags"] and r["async_flags"])
    neither = sum(1 for r in ungrounded if not r["sync_flags"] and not r["async_flags"])
    total_detected = only_sync + only_async + both_caught

    t_unique = Table(title="Unique Contribution (UNGROUNDED only)")
    t_unique.add_column(""); t_unique.add_column("Count"); t_unique.add_column("% of detections")
    t_unique.add_row("Only sync caught", str(only_sync),
                     f"{only_sync / total_detected:.1%}" if total_detected else "N/A")
    t_unique.add_row("Only async caught", str(only_async),
                     f"{only_async / total_detected:.1%}" if total_detected else "N/A")
    t_unique.add_row("Both caught", str(both_caught),
                     f"{both_caught / total_detected:.1%}" if total_detected else "N/A")
    t_unique.add_row("Neither caught", str(neither), "-")
    console.print(t_unique)

    # === Cohen's Kappa ===
    sync_labels = [1 if r["sync_flags"] else 0 for r in valid]
    async_labels = [1 if r["async_flags"] else 0 for r in valid]
    kappa = cohens_kappa(sync_labels, async_labels)

    # === Badge Distribution ===
    badges = defaultdict(int)
    for r in valid:
        badges[r["sync_badge"]] += 1
    t_badge = Table(title="Sync Badge Distribution")
    t_badge.add_column("Badge"); t_badge.add_column("Count"); t_badge.add_column("Pct")
    for badge in ["green", "yellow", "red", "gray"]:
        cnt = badges.get(badge, 0)
        t_badge.add_row(badge, str(cnt), f"{cnt / len(valid):.1%}" if valid else "0")
    console.print(t_badge)

    # ============================
    # RESEARCH GOAL VERDICTS D1-D9
    # ============================
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(f"[bold]  RESEARCH GOAL VERDICTS (D1-D9)[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]\n")

    # D1: Sync recall >= 50% AND p95 <= 3000ms
    d1_recall_pass = sync_recall >= 0.50
    d1_latency_pass = sync_p95 <= 3000
    d1_pass = d1_recall_pass and d1_latency_pass
    d1_status = "[green]PASS[/green]" if d1_pass else "[red]FAIL[/red]"
    console.print(f"  D1  {d1_status}  Sync recall={sync_recall:.1%} (>=50%), p95={sync_p95:.0f}ms (<=3000ms)")

    # D2: Async recall >= 40%
    d2_pass = async_recall >= 0.40
    d2_status = "[green]PASS[/green]" if d2_pass else "[red]FAIL[/red]"
    console.print(f"  D2  {d2_status}  Async recall={async_recall:.1%} (>=40%)")

    # D3: Cell B (sync pass + async flags) > 0
    d3_pass = cell_b > 0
    d3_status = "[green]PASS[/green]" if d3_pass else "[red]FAIL[/red]"
    console.print(f"  D3  {d3_status}  Cell B (sync pass, async flags) = {cell_b}")

    # D4: Cell C (sync block + async OK) > 0
    d4_pass = cell_c > 0
    d4_status = "[green]PASS[/green]" if d4_pass else "[red]FAIL[/red]"
    console.print(f"  D4  {d4_status}  Cell C (sync block, async OK) = {cell_c}")

    # D5: Cohen's kappa in [0.3, 0.8]
    d5_pass = 0.3 <= kappa <= 0.8
    d5_status = "[green]PASS[/green]" if d5_pass else "[red]FAIL[/red]"
    console.print(f"  D5  {d5_status}  Cohen's kappa = {kappa:.3f} (target [0.3, 0.8])")

    # D6: Each mode's unique contribution >= 10% of total detections
    sync_unique_pct = only_sync / total_detected if total_detected else 0
    async_unique_pct = only_async / total_detected if total_detected else 0
    d6_pass = sync_unique_pct >= 0.10 and async_unique_pct >= 0.10
    d6_status = "[green]PASS[/green]" if d6_pass else "[red]FAIL[/red]"
    console.print(f"  D6  {d6_status}  Sync unique={sync_unique_pct:.1%}, Async unique={async_unique_pct:.1%} (both >=10%)")

    # D7: DUAL_recall - SYNC_recall >= 5pp
    dual_sync_gap = (dual_recall - sync_recall) * 100
    d7_pass = dual_sync_gap >= 5.0
    d7_status = "[green]PASS[/green]" if d7_pass else "[red]FAIL[/red]"
    console.print(f"  D7  {d7_status}  DUAL={dual_recall:.1%} - SYNC={sync_recall:.1%} = {dual_sync_gap:.1f}pp (>=5pp)")

    # D8: DUAL_recall >= ASYNC_recall
    d8_pass = dual_recall >= async_recall
    d8_status = "[green]PASS[/green]" if d8_pass else "[red]FAIL[/red]"
    console.print(f"  D8  {d8_status}  DUAL={dual_recall:.1%} >= ASYNC={async_recall:.1%}")

    # D9: DUAL > SYNC and DUAL >= ASYNC on each tier
    d9_all_pass = True
    for tier in ["easy", "medium", "hard"]:
        tier_ung = [r for r in tier_data.get(tier, []) if r["gt_label"] == "UNGROUNDED"]
        if tier_ung:
            t_sr = detection_recall("sync_flags", tier_ung)
            t_ar = detection_recall("async_flags", tier_ung)
            t_dr = detection_recall("dual_flags", tier_ung)
            tier_ok = t_dr > t_sr and t_dr >= t_ar
            if not tier_ok:
                d9_all_pass = False
            mark = "[green]OK[/green]" if tier_ok else "[red]X[/red]"
            console.print(f"       {tier}: DUAL={t_dr:.1%} > SYNC={t_sr:.1%}, DUAL >= ASYNC={t_ar:.1%} {mark}")
        else:
            d9_all_pass = False
    d9_status = "[green]PASS[/green]" if d9_all_pass else "[red]FAIL[/red]"
    console.print(f"  D9  {d9_status}  DUAL wins all tiers")

    passed = sum([d1_pass, d2_pass, d3_pass, d4_pass, d5_pass, d6_pass, d7_pass, d8_pass, d9_all_pass])
    console.print(f"\n  [bold]DUAL-MODE OVERALL: {passed}/9 PASS[/bold]\n")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "total_records": len(results),
        "valid_records": len(valid),
        "sync_recall": sync_recall,
        "async_recall": async_recall,
        "dual_recall": dual_recall,
        "sync_precision": sync_precision,
        "async_precision": async_precision,
        "dual_precision": dual_precision,
        "sync_f1": f1(sync_precision, sync_recall),
        "async_f1": f1(async_precision, async_recall),
        "dual_f1": f1(dual_precision, dual_recall),
        "sync_p95_ms": sync_p95,
        "async_p95_ms": async_p95,
        "kappa": kappa,
        "cell_a": cell_a, "cell_b": cell_b, "cell_c": cell_c, "cell_d": cell_d,
        "only_sync": only_sync, "only_async": only_async,
        "both_caught": both_caught, "neither_caught": neither,
        "sync_unique_pct": sync_unique_pct,
        "async_unique_pct": async_unique_pct,
        "label_counts": {
            "GROUNDED": len(grounded),
            "UNGROUNDED": len(ungrounded),
            "AMBIGUOUS": sum(1 for r in valid if r["gt_label"] == "AMBIGUOUS"),
        },
        "verdicts": {
            "D1": d1_pass, "D2": d2_pass, "D3": d3_pass, "D4": d4_pass,
            "D5": d5_pass, "D6": d6_pass, "D7": d7_pass, "D8": d8_pass,
            "D9": d9_all_pass,
        },
        "records": results,
    }
    with open(RESULTS_DIR / "dual_mode_ablation.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"[dim]Results saved to {RESULTS_DIR / 'dual_mode_ablation.json'}[/dim]")

    return output


if __name__ == "__main__":
    run()
