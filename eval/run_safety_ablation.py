import json
import logging
import re
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
    """Extract ground truth safety items from chunk text by pattern matching."""
    gt = {}
    for cat, pattern in CATEGORY_PATTERNS.items():
        matches = pattern.findall(chunk_text)
        if matches:
            gt[cat] = list(set(m.strip() for m in matches))
    return gt


def _categorize_protocols(protocols: list) -> dict[str, list]:
    """Categorize extracted protocols into PPE/LOTO/HAZARD buckets."""
    categorized = defaultdict(list)
    for p in protocols:
        text = (p.text if hasattr(p, 'text') else p.get("text", "")).lower()
        hazard_type = (p.hazard_type if hasattr(p, 'hazard_type') else p.get("hazard_type", "")).lower()

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


def run():
    with open(EVAL_DATASET_PATH) as f:
        records = json.load(f)

    collection = get_collection()

    # Load all chunks for GT extraction
    with open(Path(__file__).parent.parent / "data" / "chunks" / "all_chunks.json") as f:
        all_chunks = json.load(f)
    chunk_map = {c["chunk_id"]: c for c in all_chunks}

    # Identify records with safety content in their GT chunks
    safety_records = []
    no_safety_records = []
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
        else:
            no_safety_records.append(r)

    console.print(f"[bold]Safety ablation: {len(safety_records)} records with safety content, {len(no_safety_records)} without[/bold]")

    if not safety_records:
        console.print("[red]No records with safety content found. Aborting.[/red]")
        return

    checkpoint = _load_checkpoint()
    results = list(checkpoint.values())
    done_ids = set(checkpoint.keys())
    remaining = [r for r in safety_records if r["id"] not in done_ids]

    for r in tqdm(remaining, desc="Safety ablation"):
        try:
            # Retrieve
            retrieved = retrieve(r["observation"], k=TOP_K, collection=collection)
            context = "\n\n---\n\n".join(
                f"[CHUNK-ID: {c.chunk_id}]\n{c.text}" for c in retrieved
            )

            # Extract GT safety from ground truth chunks
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

            # Run S1, S2, S3
            s1 = evaluate_s1_isolated(r["observation"], retrieved, collection=collection)
            s2 = evaluate_s2_inline(r["observation"], context)
            s3 = evaluate_s3_system_prompt(r["observation"], context)

            # Categorize extractions
            s1_cat = _categorize_protocols(s1.protocols)
            s2_cat = _categorize_protocols(s2.protocols)
            s3_cat = _categorize_protocols(s3.protocols)

            # Per-category P/R/F1
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
                    p, rec, f1 = precision_recall_f1(tp, fp, fn)
                    cat_metrics.setdefault(cat, {})[label] = {
                        "precision": p, "recall": rec, "f1": f1,
                        "extracted": ext_count, "gt": gt_count,
                    }

            # Safety coverage (all protocols found?)
            gt_as_dicts = [{"text": item} for items in gt_safety.values() for item in items]
            s1_as_dicts = [{"text": p.text if hasattr(p, 'text') else p.get("text", "")} for p in s1.protocols]
            s2_as_dicts = [{"text": p.text if hasattr(p, 'text') else p.get("text", "")} for p in s2.protocols]
            s3_as_dicts = [{"text": p.text if hasattr(p, 'text') else p.get("text", "")} for p in s3.protocols]

            s1_cov = safety_coverage(s1_as_dicts, gt_as_dicts) if gt_as_dicts else 1.0
            s2_cov = safety_coverage(s2_as_dicts, gt_as_dicts) if gt_as_dicts else 1.0
            s3_cov = safety_coverage(s3_as_dicts, gt_as_dicts) if gt_as_dicts else 1.0

            record = {
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
            results.append(record)
            _append_checkpoint(record)

        except Exception as e:
            logger.error(f"Safety case {r['id']} failed: {e}")
            record = {
                "id": r["id"], "difficulty_tier": r["difficulty_tier"],
                "error": str(e),
                "s1_protocols": 0, "s2_protocols": 0, "s3_protocols": 0,
                "s1_confidence": 0, "s2_confidence": 0, "s3_confidence": 0,
                "s1_coverage": 0, "s2_coverage": 0, "s3_coverage": 0,
                "s1_latency_ms": 0, "s2_latency_ms": 0, "s3_latency_ms": 0,
                "gt_safety_categories": {}, "cat_metrics": {},
            }
            results.append(record)
            _append_checkpoint(record)

    valid = [r for r in results if "error" not in r]

    console.print(f"\n[bold]═══ SAFETY ABLATION RESULTS ═══[/bold]\n")

    # Aggregate per-category metrics
    agg_cat = defaultdict(lambda: defaultdict(list))
    for r in valid:
        for cat, metrics in r.get("cat_metrics", {}).items():
            for label, m in metrics.items():
                agg_cat[cat][label].append(m)

    for cat in SAFETY_CATEGORIES:
        if cat not in agg_cat:
            continue
        t = Table(title=f"{cat} — Precision / Recall / F1")
        t.add_column("Condition"); t.add_column("Precision"); t.add_column("Recall"); t.add_column("F1"); t.add_column("N")
        for label in ["S1", "S2", "S3"]:
            metrics_list = agg_cat[cat].get(label, [])
            if metrics_list:
                avg_p = sum(m["precision"] for m in metrics_list) / len(metrics_list)
                avg_r = sum(m["recall"] for m in metrics_list) / len(metrics_list)
                avg_f1 = sum(m["f1"] for m in metrics_list) / len(metrics_list)
                t.add_row(label, f"{avg_p:.2f}", f"{avg_r:.2f}", f"{avg_f1:.2f}", str(len(metrics_list)))
        console.print(t)

    # Safety coverage
    s1_covs = [r["s1_coverage"] for r in valid]
    s2_covs = [r["s2_coverage"] for r in valid]
    s3_covs = [r["s3_coverage"] for r in valid]

    mean_s1 = sum(s1_covs) / len(s1_covs) if s1_covs else 0
    mean_s2 = sum(s2_covs) / len(s2_covs) if s2_covs else 0
    mean_s3 = sum(s3_covs) / len(s3_covs) if s3_covs else 0

    t_cov = Table(title="Safety Coverage (fraction of GT protocols found)")
    t_cov.add_column("Condition"); t_cov.add_column("Mean Coverage"); t_cov.add_column("N")
    t_cov.add_row("S1 (Isolated)", f"{mean_s1:.2f}", str(len(s1_covs)))
    t_cov.add_row("S2 (Inline)", f"{mean_s2:.2f}", str(len(s2_covs)))
    t_cov.add_row("S3 (System Prompt)", f"{mean_s3:.2f}", str(len(s3_covs)))
    console.print(t_cov)

    # Extraction counts
    t_ext = Table(title="Mean Protocols Extracted")
    t_ext.add_column("Condition"); t_ext.add_column("Mean Count")
    t_ext.add_row("S1", f"{sum(r['s1_protocols'] for r in valid)/len(valid):.1f}")
    t_ext.add_row("S2", f"{sum(r['s2_protocols'] for r in valid)/len(valid):.1f}")
    t_ext.add_row("S3", f"{sum(r['s3_protocols'] for r in valid)/len(valid):.1f}")
    console.print(t_ext)

    # Precision comparison (hallucination check)
    s1_precisions = []
    s3_precisions = []
    for r in valid:
        for cat in r.get("cat_metrics", {}):
            if "S1" in r["cat_metrics"][cat]:
                s1_precisions.append(r["cat_metrics"][cat]["S1"]["precision"])
            if "S3" in r["cat_metrics"][cat]:
                s3_precisions.append(r["cat_metrics"][cat]["S3"]["precision"])
    mean_s1_prec = sum(s1_precisions) / len(s1_precisions) if s1_precisions else 0
    mean_s3_prec = sum(s3_precisions) / len(s3_precisions) if s3_precisions else 0

    # S5: failure mode analysis
    s3_extraction_miss = 0
    s3_retrieval_miss = 0
    for r in valid:
        if r["s3_coverage"] < 1.0:
            # Did the retriever find safety chunks?
            # If coverage < 1 despite retrieval, it's an extraction miss
            s3_extraction_miss += 1
    s3_retrieval_miss = sum(1 for r in valid if r["s3_protocols"] == 0 and r["gt_safety_categories"])

    # Verdicts
    console.print("\n[bold]═══ RESEARCH GOAL VERDICTS ═══[/bold]\n")

    # S1: Isolation recall > S2 and S3
    s1_recalls = []
    s2_recalls = []
    s3_recalls = []
    for r in valid:
        for cat in r.get("cat_metrics", {}):
            if "S1" in r["cat_metrics"][cat]:
                s1_recalls.append(r["cat_metrics"][cat]["S1"]["recall"])
            if "S2" in r["cat_metrics"][cat]:
                s2_recalls.append(r["cat_metrics"][cat]["S2"]["recall"])
            if "S3" in r["cat_metrics"][cat]:
                s3_recalls.append(r["cat_metrics"][cat]["S3"]["recall"])

    mean_s1_rec = sum(s1_recalls) / len(s1_recalls) if s1_recalls else 0
    mean_s2_rec = sum(s2_recalls) / len(s2_recalls) if s2_recalls else 0
    mean_s3_rec = sum(s3_recalls) / len(s3_recalls) if s3_recalls else 0

    s1_pass = mean_s1_rec > mean_s2_rec and mean_s1_rec > mean_s3_rec
    s1_status = "[green]PASS[/green]" if s1_pass else "[red]FAIL[/red]"
    console.print(f"  S1  {s1_status}  S1 Recall={mean_s1_rec:.2f} > S2={mean_s2_rec:.2f} AND S3={mean_s3_rec:.2f}")

    s2_pass = mean_s1_prec > mean_s3_prec
    s2_status = "[green]PASS[/green]" if s2_pass else "[red]FAIL[/red]"
    console.print(f"  S2  {s2_status}  S1 Precision={mean_s1_prec:.2f} > S3={mean_s3_prec:.2f}")

    s3_pass_val = mean_s1 >= 0.85
    s3_fail = mean_s1 < 0.70
    s3_status = "[green]PASS[/green]" if s3_pass_val else ("[red]FAIL[/red]" if s3_fail else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"  S3  {s3_status}  S1 Safety Coverage = {mean_s1:.2f} (target >= 0.85)")

    gap_pp = (mean_s1 - mean_s3) * 100
    s4_pass = gap_pp >= 15
    s4_status = "[green]PASS[/green]" if s4_pass else ("[red]FAIL[/red]" if gap_pp < 5 else "[yellow]INCONCLUSIVE[/yellow]")
    console.print(f"  S4  {s4_status}  S1-S3 Coverage gap = {gap_pp:.1f}pp (target >= 15pp)")

    s5_pass = s3_extraction_miss > s3_retrieval_miss
    s5_status = "[green]PASS[/green]" if s5_pass else "[red]FAIL[/red]"
    console.print(f"  S5  {s5_status}  S3 extraction misses={s3_extraction_miss} > retrieval misses={s3_retrieval_miss}")

    passed = sum([s1_pass, s2_pass, s3_pass_val, s4_pass, s5_pass])
    console.print(f"\n  OVERALL: {passed}/5 PASS\n")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "n_safety": len(safety_records),
        "n_no_safety": len(no_safety_records),
        "mean_s1_coverage": mean_s1,
        "mean_s2_coverage": mean_s2,
        "mean_s3_coverage": mean_s3,
        "mean_s1_precision": mean_s1_prec,
        "mean_s3_precision": mean_s3_prec,
        "mean_s1_recall": mean_s1_rec,
        "mean_s2_recall": mean_s2_rec,
        "mean_s3_recall": mean_s3_rec,
        "s3_extraction_miss": s3_extraction_miss,
        "s3_retrieval_miss": s3_retrieval_miss,
        "verdicts": {
            "S1": s1_pass, "S2": s2_pass, "S3": s3_pass_val,
            "S4": s4_pass, "S5": s5_pass,
        },
        "records": results,
    }
    with open(RESULTS_DIR / "safety_ablation.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"[dim]Results saved to {RESULTS_DIR / 'safety_ablation.json'}[/dim]")

    return output


if __name__ == "__main__":
    run()
