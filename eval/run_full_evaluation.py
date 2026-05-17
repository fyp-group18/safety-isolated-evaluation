import json
import sys
import time
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR

console = Console()


def run():
    console.print("\n[bold]═══════════════════════════════════════════════════════════════[/bold]")
    console.print("[bold]  SAFETY-ISOLATED EVALUATION HARNESS — FULL RUN[/bold]")
    console.print("[bold]═══════════════════════════════════════════════════════════════[/bold]\n")

    total_start = time.time()

    # Phase 1: Retrieval Baseline
    console.print("[bold cyan]Phase 1/3: Retrieval Baseline[/bold cyan]")
    t0 = time.time()
    from eval.run_retrieval_baseline import run as run_retrieval
    retrieval_results = run_retrieval()
    t_retrieval = time.time() - t0
    console.print(f"[dim]Retrieval baseline completed in {t_retrieval:.0f}s[/dim]\n")

    # Phase 2: Dual-Mode Ablation
    console.print("[bold cyan]Phase 2/3: Dual-Mode Ablation[/bold cyan]")
    t0 = time.time()
    from eval.run_dual_mode_ablation import run as run_dual
    dual_results = run_dual()
    t_dual = time.time() - t0
    console.print(f"[dim]Dual-mode ablation completed in {t_dual:.0f}s[/dim]\n")

    # Phase 3: Safety Ablation
    console.print("[bold cyan]Phase 3/3: Safety Ablation[/bold cyan]")
    t0 = time.time()
    from eval.run_safety_ablation import run as run_safety
    safety_results = run_safety()
    t_safety = time.time() - t0
    console.print(f"[dim]Safety ablation completed in {t_safety:.0f}s[/dim]\n")

    total_time = time.time() - total_start

    # Final Scorecard
    rv = retrieval_results.get("verdicts", {})
    dv = dual_results.get("verdicts", {})
    sv = safety_results.get("verdicts", {})

    r_pass = sum(1 for k in ["R1", "R2", "R3"] if rv.get(k, {}).get("pass", False))
    d_pass = sum(1 for k in ["D1", "D2", "D3", "D4", "D5", "D6", "D7"] if dv.get(k, False))
    s_pass = sum(1 for k in ["S1", "S2", "S3", "S4", "S5"] if sv.get(k, False))
    total_pass = r_pass + d_pass + s_pass

    def _icon(v):
        return "PASS" if v else "FAIL"

    console.print("\n[bold]═══════════════════════════════════════════════════════════════[/bold]")
    console.print("[bold]  FINAL RESEARCH GOAL SCORECARD[/bold]")
    console.print("[bold]═══════════════════════════════════════════════════════════════[/bold]")

    r1 = _icon(rv.get("R1", {}).get("pass"))
    r2 = _icon(rv.get("R2", {}).get("pass"))
    r3 = _icon(rv.get("R3", {}).get("pass"))
    console.print(f"  RETRIEVAL:  R1 {r1}  R2 {r2}  R3 {r3}                  {r_pass}/3")

    d_icons = " ".join(f"D{i} {_icon(dv.get(f'D{i}'))}" for i in range(1, 5))
    d_icons2 = " ".join(f"D{i} {_icon(dv.get(f'D{i}'))}" for i in range(5, 8))
    console.print(f"  DUAL-MODE:  {d_icons}")
    console.print(f"              {d_icons2}          {d_pass}/7")

    s_icons = " ".join(f"S{i} {_icon(sv.get(f'S{i}'))}" for i in range(1, 6))
    console.print(f"  SAFETY:     {s_icons}  {s_pass}/5")

    console.print("[bold]─────────────────────────────────────────────────────────────[/bold]")
    console.print(f"  OVERALL:    {total_pass}/15 PASS")
    console.print("[bold]═══════════════════════════════════════════════════════════════[/bold]")

    # Bottom line
    hr = retrieval_results.get("overall_hit_rate", 0)
    dual_rec = dual_results.get("dual_recall", 0)
    sync_rec = dual_results.get("sync_recall", 0)
    gap = (dual_rec - sync_rec) * 100
    s1_cov = safety_results.get("mean_s1_coverage", 0)
    s3_cov = safety_results.get("mean_s3_coverage", 0)

    console.print(f"\n  BOTTOM LINE:")
    console.print(f"  The flat RAG retriever found the right manual section {hr:.0%} of the time.")
    console.print(f"  The dual-mode gate caught {dual_rec:.0%} of bad answers — {gap:.0f} percentage points more")
    console.print(f"  than the sync gate alone.")
    console.print(f"  The isolated safety node extracted all required PPE/LOTO/hazard items for")
    console.print(f"  {s1_cov:.0%} of cases, compared to {s3_cov:.0%} with system-prompt-only instructions.")
    console.print("[bold]═══════════════════════════════════════════════════════════════[/bold]")

    console.print(f"\n[dim]Total execution time: {total_time:.0f}s ({total_time/60:.1f} min)[/dim]")
    console.print(f"[dim]  Retrieval: {t_retrieval:.0f}s | Dual-mode: {t_dual:.0f}s | Safety: {t_safety:.0f}s[/dim]")

    # Save config
    RESULTS_DIR.mkdir(exist_ok=True)
    config = {
        "embedding_model": "all-MiniLM-L6-v2",
        "sync_gate_model": "gemini-2.5-flash-lite",
        "async_eval_model": "gemini-2.5-flash",
        "safety_eval_model": "gemini-2.5-flash",
        "top_k": 5,
        "random_seed": 42,
        "total_records": 150,
        "total_time_s": total_time,
    }
    with open(RESULTS_DIR / "eval_config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    run()
