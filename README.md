# Safety-Isolated Evaluation Harness

Standalone evaluation harness for two P2 paper novelty claims:

1. **Dual-mode sync + async confidence gating** outperforms either mode alone
2. **Safety-isolated evaluation** outperforms inline and system-prompt-only safety extraction

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env` with Google Cloud credentials.

## Running

```bash
# Individual experiments
python -m eval.run_retrieval_baseline
python -m eval.run_dual_mode_ablation
python -m eval.run_safety_ablation

# Full evaluation (all experiments + report)
python -m eval.run_full_evaluation
```

## Structure

- `data/` — eval dataset + chunked manual data
- `src/` — core modules (flat RAG, sync gate, async eval, safety evaluator, metrics)
- `eval/` — evaluation scripts
- `results/` — output metrics, checkpoints, reports
