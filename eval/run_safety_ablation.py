import json
import logging
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHUNKS_PATH, EVAL_DATASET_PATH, MAX_WORKERS, RESULTS_DIR, TOP_K
from src.flat_rag import get_collection, retrieve
from src.metrics import bootstrap_ci, precision_recall_f1, safety_coverage
from src.safety_evaluator import (
    _SAFETY_KEYWORDS,
    evaluate_s1_isolated,
    evaluate_s2_inline,
    evaluate_s3_system_prompt,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

CHECKPOINT_PATH = RESULTS_DIR / "safety_ablation_checkpoint.jsonl"

SAFETY_CATEGORIES = ["PPE", "LOTO", "HAZARD_WARNING"]
CATEGORY_PATTERNS = {
    "PPE": re.compile(r"(?i)\b(PPE|glove|goggles|helmet|hearing\s*protection|safety\s*glass|eye\s*protection|arc.flash|face\s*shield|respirator)\b"),
    "LOTO": re.compile(r"(?i)\b(lockout|tagout|LOTO|lock.out|tag.out|energy\s*isolation|de.energize|disconnect\s*power)\b"),
    "HAZARD_WARNING": re.compile(r"(?i)\b(DANGER|WARNING|CAUTION|NOTICE|HAZARD|high.voltage|electric.shock|burn|crush|pinch|toxic|fire|explosion)\b"),
}


def _extract_gt_safety(chunk_text: str) -> dict[str, list[str]]:
    gt = {}
    for line in chunk_text.split("\n"):
        line = line.strip()
        if len(line) < 10:
            continue
        for cat, pattern in CATEGORY_PATTERNS.items():
            if pattern.search(line):
                gt.setdefault(cat, []).append(line[:200])
    for cat in gt:
        gt[cat] = list(set(gt[cat]))
    return gt


def _categorize_protocols(protocols: list) -> dict[str, list]:
    categorized = defaultdict(list)
    for p in protocols:
        text = (p.text if hasattr(p, "text") else p.get("text", "")).lower()

        if CATEGORY_PATTERNS["PPE"].search(text):
            categorized["PPE"].append(p)
        if CATEGORY_PATTERNS["LOTO"].search(text):
            categorized["LOTO"].append(p)
        if CATEGORY_PATTERNS["HAZARD_WARNING"].search(text):
            categorized["HAZARD_WARNING"].append(p)
        if not any(CATEGORY_PATTERNS[c].search(text) for c in SAFETY_CATEGORIES):
            categorized["HAZARD_WARNING"].append(p)

    return dict(categorized)


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


def _process_record(r: dict, collection, chunk_map: dict) -> dict:
    try:
        retrieved = retrieve(r["observation"], k=TOP_K, collection=collection)
        context = "\n\n---\n\n".join(
            f"[CHUNK-ID: {c.chunk_id}]\n{c.text}" for c in retrieved
        )

        gt_safety = {}
        gt_chunk_texts = []
        for gid in r.get("ground_truth_chunk_ids", [r["ground_truth_chunk_id"]]):
            chunk = chunk_map.get(gid, {})
            chunk_text = chunk.get("text", "")
            if chunk_text:
                gt_chunk_texts.append(chunk_text)
                gt_cat = _extract_gt_safety(chunk_text)
                for cat, items in gt_cat.items():
                    gt_safety.setdefault(cat, []).extend(items)

        s1 = evaluate_s1_isolated(r["observation"], retrieved, collection=collection)
        s2 = evaluate_s2_inline(r["observation"], context)
        s3 = evaluate_s3_system_prompt(r["observation"], context)

        s1_cat = _categorize_protocols(s1.protocols)
        s2_cat = _categorize_protocols(s2.protocols)
        s3_cat = _categorize_protocols(s3.protocols)

        cat_metrics = {}
        for cat in SAFETY_CATEGORIES:
            gt_count = len(gt_safety.get(cat, []))
            if gt_count == 0:
                continue
            for label, ext_cat in [("S1", s1_cat), ("S2", s2_cat), ("S3", s3_cat)]:
                ext_count = len(ext_cat.get(cat, []))
                tp = min(ext_count, gt_count)
                fp = max(0, ext_count - gt_count)
                fn = max(0, gt_count - ext_count)
                p, rec, f1_val = precision_recall_f1(tp, fp, fn)
                cat_metrics.setdefault(cat, {})[label] = {
                    "precision": p, "recall": rec, "f1": f1_val,
                    "extracted": ext_count, "gt": gt_count,
                }

        gt_as_dicts = [{"text": item} for items in gt_safety.values() for item in items]
        s1_as_dicts = [{"text": p.text if hasattr(p, "text") else p.get("text", "")} for p in s1.protocols]
        s2_as_dicts = [{"text": p.text if hasattr(p, "text") else p.get("text", "")} for p in s2.protocols]
        s3_as_dicts = [{"text": p.text if hasattr(p, "text") else p.get("text", "")} for p in s3.protocols]

        s1_cov = safety_coverage(s1_as_dicts, gt_as_dicts) if gt_as_dicts else 1.0
        s2_cov = safety_coverage(s2_as_dicts, gt_as_dicts) if gt_as_dicts else 1.0
        s3_cov = safety_coverage(s3_as_dicts, gt_as_dicts) if gt_as_dicts else 1.0

        return {
            "id": r["id"],
            "difficulty_tier": r["difficulty_tier"],
            "s1_protocols": len(s1.protocols),
            "s2_protocols": len(s2.protocols),
            "s3_protocols": len(s3.protocols),
            "s1_confidence": s1.extraction_confidence,
            "s2_confidence": s2.extraction_confidence,
            "s3_confidence": s3.extraction_confidence,
            "s1_coverage": s1_cov,
            "s2_coverage": s2_cov,
            "s3_coverage": s3_cov,
            "s1_latency_ms": s1.latency_ms,
            "s2_latency_ms": s2.latency_ms,
            "s3_latency_ms": s3.latency_ms,
            "gt_safety_categories": {k: len(v) for k, v in gt_safety.items()},
            "cat_metrics": cat_metrics,
        }

    except Exception as e:
        logger.error(f"Safety case {r['id']} failed: {e}")
        return {
            "id": r["id"],
            "difficulty_tier": r["difficulty_tier"],
            "error": str(e),
            "s1_protocols": 0, "s2_protocols": 0, "s3_protocols": 0,
            "s1_confidence": 0, "s2_confidence": 0, "s3_confidence": 0,
            "s1_coverage": 0, "s2_coverage": 0, "s3_coverage": 0,
            "s1_latency_ms": 0, "s2_latency_ms": 0, "s3_latency_ms": 0,
            "gt_safety_categories": {}, "cat_metrics": {},
        }


def run():
    with open(EVAL_DATASET_PATH) as f:
        records = json.load(f)

    collection = get_collection()

    with open(CHUNKS_PATH) as f:
        all_chunks = json.load(f)
    chunk_map = {c["chunk_id"]: c for c in all_chunks}

    safety_records = []
    for r in records:
        gt_ids = r.get("ground_truth_chunk_ids", [r["ground_truth_chunk_id"]])
        has_safety = False
        for gid in gt_ids:
            chunk = chunk_map.get(gid, {})
            if _SAFETY_KEYWORDS.search(chunk.get("text", "")):
                has_safety = True
                break
        if has_safety:
            safety_records.append(r)

    console.print(f"[bold]Safety ablation: {len(safety_records)} records with safety content (out of {len(records)})[/bold]")

    if not safety_records:
        console.print("[red]No records with safety content found. Aborting.[/red]")
        return

    checkpoint = _load_checkpoint()
    results = list(checkpoint.values())
    done_ids = set(checkpoint.keys())
    remaining = [r for r in safety_records if r["id"] not in done_ids]

    console.print(f"[bold]{len(remaining)} remaining to process[/bold]")

    if remaining:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_record, r, collection, chunk_map): r for r in remaining}
            pbar = tqdm(total=len(remaining), desc="[SAFETY]")

            for future in as_completed(futures):
                record = future.result()
                results.append(record)
                _append_checkpoint(record)
                pbar.update(1)

            pbar.close()

    valid = [r for r in results if "error" not in r]

    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(f"[bold]  SAFETY ABLATION RESULTS ({len(valid)} valid / {len(results)} total)[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]\n")

    # === Overall Metrics ===
    s1_covs = [r["s1_coverage"] for r in valid]
    s2_covs = [r["s2_coverage"] for r in valid]
    s3_covs = [r["s3_coverage"] for r in valid]

    mean_s1_cov = sum(s1_covs) / len(s1_covs) if s1_covs else 0
    mean_s2_cov = sum(s2_covs) / len(s2_covs) if s2_covs else 0
    mean_s3_cov = sum(s3_covs) / len(s3_covs) if s3_covs else 0

    s1_ci = bootstrap_ci(s1_covs) if len(s1_covs) >= 5 else (mean_s1_cov, mean_s1_cov, mean_s1_cov)
    s2_ci = bootstrap_ci(s2_covs) if len(s2_covs) >= 5 else (mean_s2_cov, mean_s2_cov, mean_s2_cov)
    s3_ci = bootstrap_ci(s3_covs) if len(s3_covs) >= 5 else (mean_s3_cov, mean_s3_cov, mean_s3_cov)

    # Aggregate recalls and precisions
    s1_recalls, s2_recalls, s3_recalls = [], [], []
    s1_precisions, s2_precisions, s3_precisions = [], [], []
    for r in valid:
        for cat in r.get("cat_metrics", {}):
            if "S1" in r["cat_metrics"][cat]:
                s1_recalls.append(r["cat_metrics"][cat]["S1"]["recall"])
                s1_precisions.append(r["cat_metrics"][cat]["S1"]["precision"])
            if "S2" in r["cat_metrics"][cat]:
                s2_recalls.append(r["cat_metrics"][cat]["S2"]["recall"])
                s2_precisions.append(r["cat_metrics"][cat]["S2"]["precision"])
            if "S3" in r["cat_metrics"][cat]:
                s3_recalls.append(r["cat_metrics"][cat]["S3"]["recall"])
                s3_precisions.append(r["cat_metrics"][cat]["S3"]["precision"])

    mean_s1_rec = sum(s1_recalls) / len(s1_recalls) if s1_recalls else 0
    mean_s2_rec = sum(s2_recalls) / len(s2_recalls) if s2_recalls else 0
    mean_s3_rec = sum(s3_recalls) / len(s3_recalls) if s3_recalls else 0
    mean_s1_prec = sum(s1_precisions) / len(s1_precisions) if s1_precisions else 0
    mean_s2_prec = sum(s2_precisions) / len(s2_precisions) if s2_precisions else 0
    mean_s3_prec = sum(s3_precisions) / len(s3_precisions) if s3_precisions else 0

    def safe_f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    mean_s1_f1 = safe_f1(mean_s1_prec, mean_s1_rec)
    mean_s2_f1 = safe_f1(mean_s2_prec, mean_s2_rec)
    mean_s3_f1 = safe_f1(mean_s3_prec, mean_s3_rec)

    mean_s1_protos = sum(r["s1_protocols"] for r in valid) / len(valid) if valid else 0
    mean_s2_protos = sum(r["s2_protocols"] for r in valid) / len(valid) if valid else 0
    mean_s3_protos = sum(r["s3_protocols"] for r in valid) / len(valid) if valid else 0

    t_overall = Table(title="Overall Safety Metrics")
    t_overall.add_column("Metric")
    t_overall.add_column("S1 (Isolated)")
    t_overall.add_column("S2 (Inline)")
    t_overall.add_column("S3 (Sys Prompt)")
    t_overall.add_row("Recall", f"{mean_s1_rec:.3f}", f"{mean_s2_rec:.3f}", f"{mean_s3_rec:.3f}")
    t_overall.add_row("Precision", f"{mean_s1_prec:.3f}", f"{mean_s2_prec:.3f}", f"{mean_s3_prec:.3f}")
    t_overall.add_row("F1", f"{mean_s1_f1:.3f}", f"{mean_s2_f1:.3f}", f"{mean_s3_f1:.3f}")
    t_overall.add_row("Coverage", f"{mean_s1_cov:.3f}", f"{mean_s2_cov:.3f}", f"{mean_s3_cov:.3f}")
    t_overall.add_row("Mean protocols", f"{mean_s1_protos:.1f}", f"{mean_s2_protos:.1f}", f"{mean_s3_protos:.1f}")
    t_overall.add_row("95% CI (coverage)",
                      f"[{s1_ci[1]:.3f}, {s1_ci[2]:.3f}]",
                      f"[{s2_ci[1]:.3f}, {s2_ci[2]:.3f}]",
                      f"[{s3_ci[1]:.3f}, {s3_ci[2]:.3f}]")
    console.print(t_overall)

    # === Per-Category ===
    agg_cat = defaultdict(lambda: defaultdict(list))
    for r in valid:
        for cat, metrics in r.get("cat_metrics", {}).items():
            for label, m in metrics.items():
                agg_cat[cat][label].append(m)

    t_cat = Table(title="Per-Category Recall / Precision")
    t_cat.add_column("Category")
    t_cat.add_column("S1 Recall/Prec")
    t_cat.add_column("S2 Recall/Prec")
    t_cat.add_column("S3 Recall/Prec")

    for cat in SAFETY_CATEGORIES:
        if cat not in agg_cat:
            t_cat.add_row(cat, "N/A", "N/A", "N/A")
            continue
        vals = {}
        for label in ["S1", "S2", "S3"]:
            ml = agg_cat[cat].get(label, [])
            if ml:
                avg_r = sum(m["recall"] for m in ml) / len(ml)
                avg_p = sum(m["precision"] for m in ml) / len(ml)
                vals[label] = f"{avg_r:.3f} / {avg_p:.3f}"
            else:
                vals[label] = "N/A"
        t_cat.add_row(cat, vals["S1"], vals["S2"], vals["S3"])
    console.print(t_cat)

    # ============================
    # RESEARCH GOAL VERDICTS S1-S3
    # ============================
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(f"[bold]  RESEARCH GOAL VERDICTS (S1-S3)[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]\n")

    # S1: S1 recall > S2 AND S1 recall > S3
    s1_goal_pass = mean_s1_rec > mean_s2_rec and mean_s1_rec > mean_s3_rec
    s1_status = "[green]PASS[/green]" if s1_goal_pass else "[red]FAIL[/red]"
    console.print(f"  S1  {s1_status}  S1 Recall={mean_s1_rec:.3f} > S2={mean_s2_rec:.3f} AND > S3={mean_s3_rec:.3f}")

    # S2: S1 precision > S3 precision
    s2_goal_pass = mean_s1_prec > mean_s3_prec
    s2_status = "[green]PASS[/green]" if s2_goal_pass else "[red]FAIL[/red]"
    console.print(f"  S2  {s2_status}  S1 Precision={mean_s1_prec:.3f} > S3={mean_s3_prec:.3f}")

    # S3: S1 coverage >= 85%
    s3_goal_pass = mean_s1_cov >= 0.85
    s3_fail = mean_s1_cov < 0.70
    s3_status = "[green]PASS[/green]" if s3_goal_pass else ("[red]FAIL[/red]" if s3_fail else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"  S3  {s3_status}  S1 Coverage={mean_s1_cov:.3f} (target >= 0.85)")

    passed = sum([s1_goal_pass, s2_goal_pass, s3_goal_pass])
    console.print(f"\n  [bold]SAFETY OVERALL: {passed}/3 PASS[/bold]\n")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "n_safety": len(safety_records),
        "n_total": len(records),
        "mean_s1_coverage": mean_s1_cov,
        "mean_s2_coverage": mean_s2_cov,
        "mean_s3_coverage": mean_s3_cov,
        "mean_s1_precision": mean_s1_prec,
        "mean_s2_precision": mean_s2_prec,
        "mean_s3_precision": mean_s3_prec,
        "mean_s1_recall": mean_s1_rec,
        "mean_s2_recall": mean_s2_rec,
        "mean_s3_recall": mean_s3_rec,
        "mean_s1_f1": mean_s1_f1,
        "mean_s2_f1": mean_s2_f1,
        "mean_s3_f1": mean_s3_f1,
        "mean_s1_protocols": mean_s1_protos,
        "mean_s2_protocols": mean_s2_protos,
        "mean_s3_protocols": mean_s3_protos,
        "s1_ci": list(s1_ci),
        "s2_ci": list(s2_ci),
        "s3_ci": list(s3_ci),
        "verdicts": {
            "S1": s1_goal_pass,
            "S2": s2_goal_pass,
            "S3": s3_goal_pass,
        },
        "records": results,
    }
    with open(RESULTS_DIR / "safety_ablation.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"[dim]Results saved to {RESULTS_DIR / 'safety_ablation.json'}[/dim]")

    return output


if __name__ == "__main__":
    run()
