import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def generate_report():
    retrieval = _load("retrieval_baseline.json")
    dual = _load("dual_mode_ablation.json")
    safety = _load("safety_ablation.json")

    lines = ["# Evaluation Report\n"]

    lines.append("## 1. Retrieval Baseline\n")
    if retrieval:
        lines.append(f"- Hit Rate @5: {retrieval['overall_hit_rate']:.1%}")
        lines.append(f"- MRR: {retrieval['overall_mrr']:.3f}")
        lines.append(f"- Miss Rate: {retrieval['miss_rate']:.1%}")
        lines.append(f"- 95% CI: [{retrieval['ci_95'][0]:.1%}, {retrieval['ci_95'][1]:.1%}]")

        for tier, m in sorted(retrieval.get("tier_metrics", {}).items()):
            lines.append(f"  - {tier}: {m['hit_rate']:.1%} (N={m['count']})")

    lines.append("\n## 2. Dual-Mode Ablation\n")
    if dual:
        lines.append(f"- Sync AUROC: {dual.get('sync_auroc', 'N/A')}")
        lines.append(f"- Sync Latency p95: {dual.get('sync_p95_ms', 0):.0f}ms")
        lines.append(f"- Cohen's kappa: {dual.get('kappa', 0):.2f}")
        lines.append(f"- Sync Recall: {dual.get('sync_recall', 0):.1%}")
        lines.append(f"- Async Recall: {dual.get('async_recall', 0):.1%}")
        lines.append(f"- Dual Recall: {dual.get('dual_recall', 0):.1%}")
        lines.append(f"- Agreement Matrix: A={dual.get('cell_a')}, B={dual.get('cell_b')}, C={dual.get('cell_c')}, D={dual.get('cell_d')}")

    lines.append("\n## 3. Safety Ablation\n")
    if safety:
        lines.append(f"- S1 (Isolated) Coverage: {safety.get('mean_s1_coverage', 0):.2f}")
        lines.append(f"- S2 (Inline) Coverage: {safety.get('mean_s2_coverage', 0):.2f}")
        lines.append(f"- S3 (System Prompt) Coverage: {safety.get('mean_s3_coverage', 0):.2f}")
        lines.append(f"- S1 Precision: {safety.get('mean_s1_precision', 0):.2f}")
        lines.append(f"- S3 Precision: {safety.get('mean_s3_precision', 0):.2f}")

    report = "\n".join(lines)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "final_report.md", "w") as f:
        f.write(report)
    return report


def _load(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    print(generate_report())
