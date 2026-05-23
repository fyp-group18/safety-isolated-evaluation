import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    GOOGLE_CLOUD_PROJECT,
    MAX_WORKERS,
    MODEL_EMBEDDING,
    MODEL_FLASH,
    MODEL_FLASH_LITE,
    RESULTS_DIR,
    RETRY_LIMIT,
    TOP_K,
)

console = Console()


def run():
    console.print("\n[bold]" + "=" * 100 + "[/bold]")
    console.print("[bold]  P2 EVALUATION: Dual-Mode Confidence Gating + Safety-Isolated Extraction[/bold]")
    console.print(f"[bold]  GCP: {GOOGLE_CLOUD_PROJECT} | {MODEL_FLASH} + {MODEL_FLASH_LITE} | Workers: {MAX_WORKERS} | Retries: {RETRY_LIMIT}[/bold]")
    console.print("[bold]" + "=" * 100 + "[/bold]\n")

    total_start = time.time()

    # Phase 1: Dual-Mode Ablation
    console.print("[bold cyan]Phase 1/2: Dual-Mode Ablation (D1-D9)[/bold cyan]")
    t0 = time.time()
    from eval.run_dual_mode_ablation import run as run_dual
    dual_results = run_dual()
    t_dual = time.time() - t0
    console.print(f"[dim]Dual-mode ablation completed in {t_dual:.0f}s ({t_dual / 60:.1f} min)[/dim]\n")

    # Phase 2: Safety Ablation
    console.print("[bold cyan]Phase 2/2: Safety Ablation (S1-S3)[/bold cyan]")
    t0 = time.time()
    from eval.run_safety_ablation import run as run_safety
    safety_results = run_safety()
    t_safety = time.time() - t0
    console.print(f"[dim]Safety ablation completed in {t_safety:.0f}s ({t_safety / 60:.1f} min)[/dim]\n")

    total_time = time.time() - total_start

    # === FINAL SCORECARD ===
    dv = dual_results.get("verdicts", {})
    sv = safety_results.get("verdicts", {})

    d_pass = sum(1 for k in [f"D{i}" for i in range(1, 10)] if dv.get(k, False))
    s_pass = sum(1 for k in ["S1", "S2", "S3"] if sv.get(k, False))
    total_pass = d_pass + s_pass

    n_records = dual_results.get("total_records", 0)
    n_safety = safety_results.get("n_safety", 0)

    console.print("\n[bold]" + "=" * 100 + "[/bold]")
    console.print("[bold]  P2 EVALUATION SCORECARD[/bold]")
    console.print("[bold]  Dual-Mode Confidence Gating + Safety-Isolated Extraction[/bold]")
    console.print(f"[bold]  GCP: {GOOGLE_CLOUD_PROJECT} | {MODEL_FLASH} + {MODEL_FLASH_LITE} | {n_records} records | Workers: {MAX_WORKERS} | Retries: {RETRY_LIMIT}[/bold]")
    console.print("[bold]" + "=" * 100 + "[/bold]\n")

    t = Table(title="Research Goal Scorecard", show_lines=True)
    t.add_column("ID", width=4)
    t.add_column("Goal", width=26)
    t.add_column("What It Measures", width=38)
    t.add_column("Target", width=10)
    t.add_column("Achieved", width=10)
    t.add_column("P/F", width=4)

    def pf(v):
        return "[green]PASS[/green]" if v else "[red]FAIL[/red]"

    # D1-D9
    t.add_row("", "[bold]DUAL-MODE GATING[/bold]", "", "", "", "")

    sr = dual_results.get("sync_recall", 0)
    sp95 = dual_results.get("sync_p95_ms", 0)
    t.add_row("D1", "Sync-only performance",
              f"Sync recall + p95 latency",
              "Rec>=50%\np95<=3s",
              f"{sr:.1%}\n{sp95:.0f}ms",
              pf(dv.get("D1")))

    ar = dual_results.get("async_recall", 0)
    t.add_row("D2", "Async-only performance",
              "Async recall on UNGROUNDED",
              "Rec>=40%", f"{ar:.1%}", pf(dv.get("D2")))

    cb = dual_results.get("cell_b", 0)
    t.add_row("D3", "Async catches sync misses",
              "Cell B: sync pass + async flags",
              "B > 0", str(cb), pf(dv.get("D3")))

    cc = dual_results.get("cell_c", 0)
    t.add_row("D4", "Sync catches async misses",
              "Cell C: sync block + async OK",
              "C > 0", str(cc), pf(dv.get("D4")))

    kappa = dual_results.get("kappa", 0)
    t.add_row("D5", "Statistical complementarity",
              "Cohen's kappa (sync vs async)",
              "[0.3,0.8]", f"{kappa:.3f}", pf(dv.get("D5")))

    su = dual_results.get("sync_unique_pct", 0)
    au = dual_results.get("async_unique_pct", 0)
    t.add_row("D6", "Practical complementarity",
              "Each mode's unique contribution",
              "Both>=10%", f"S:{su:.0%} A:{au:.0%}", pf(dv.get("D6")))

    dr = dual_results.get("dual_recall", 0)
    gap = (dr - sr) * 100
    t.add_row("D7", "Dual > sync-only",
              "DUAL_recall - SYNC_recall",
              ">=5pp", f"{gap:.1f}pp", pf(dv.get("D7")))

    t.add_row("D8", "Dual >= async-only",
              "DUAL_recall >= ASYNC_recall",
              "DUAL>=ASY", f"{dr:.1%}/{ar:.1%}", pf(dv.get("D8")))

    t.add_row("D9", "Dual wins all tiers",
              "DUAL>SYNC & DUAL>=ASYNC per tier",
              "All 3", "", pf(dv.get("D9")))

    # S1-S3
    t.add_row("", "[bold]SAFETY ISOLATION[/bold]", "", "", "", "")

    s1r = safety_results.get("mean_s1_recall", 0)
    s2r = safety_results.get("mean_s2_recall", 0)
    s3r = safety_results.get("mean_s3_recall", 0)
    t.add_row("S1", "Isolated recall highest",
              f"S1 recall > S2 AND > S3",
              "S1 best",
              f"{s1r:.3f}>{s2r:.3f},{s3r:.3f}",
              pf(sv.get("S1")))

    s1p = safety_results.get("mean_s1_precision", 0)
    s3p = safety_results.get("mean_s3_precision", 0)
    t.add_row("S2", "Isolated precision > sys",
              "S1 precision > S3 precision",
              "S1>S3",
              f"{s1p:.3f}>{s3p:.3f}",
              pf(sv.get("S2")))

    s1c = safety_results.get("mean_s1_coverage", 0)
    t.add_row("S3", "S1 coverage >= 85%",
              f"Safety coverage ({n_safety} records)",
              ">=85%",
              f"{s1c:.1%}",
              pf(sv.get("S3")))

    # Totals
    t.add_row("", "[bold]TOTALS[/bold]", "", "", "", "")
    t.add_row("", "Dual-Mode (D1-D9)", "", "", "", f"[bold]{d_pass}/9[/bold]")
    t.add_row("", "Safety (S1-S3)", "", "", "", f"[bold]{s_pass}/3[/bold]")
    t.add_row("", "[bold]OVERALL[/bold]", "", "", "",
              f"[bold]{total_pass}/12[/bold]")

    console.print(t)

    target_met = total_pass >= 9
    console.print(f"\n  Minimum target: 9/12. {'[green]TARGET MET[/green]' if target_met else '[red]BELOW TARGET[/red]'}")
    console.print(f"\n[dim]Total execution time: {total_time:.0f}s ({total_time / 60:.1f} min)[/dim]")
    console.print(f"[dim]  Dual-mode: {t_dual:.0f}s | Safety: {t_safety:.0f}s[/dim]")

    # Save config
    RESULTS_DIR.mkdir(exist_ok=True)
    config = {
        "embedding_model": MODEL_EMBEDDING,
        "sync_gate_model": MODEL_FLASH_LITE,
        "async_eval_model": MODEL_FLASH,
        "safety_eval_model": MODEL_FLASH,
        "generator_model": MODEL_FLASH,
        "gcp_project": GOOGLE_CLOUD_PROJECT,
        "top_k": TOP_K,
        "max_workers": MAX_WORKERS,
        "retry_limit": RETRY_LIMIT,
        "random_seed": 42,
        "total_records": n_records,
        "safety_records": n_safety,
        "total_time_s": total_time,
        "dual_mode_time_s": t_dual,
        "safety_time_s": t_safety,
        "verdicts_dual": dv,
        "verdicts_safety": sv,
        "overall_pass": total_pass,
    }
    with open(RESULTS_DIR / "eval_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return {"dual": dual_results, "safety": safety_results, "config": config}


if __name__ == "__main__":
    run()
