# Safety-Isolated Evaluation Harness

Standalone evaluation harness for two research claims on LLM-based aircraft maintenance decision support:

1. **Dual-mode confidence gating** — combining synchronous (inline) and asynchronous (shadow) quality gates outperforms either mode alone at detecting ungrounded repair plans.
2. **Safety-isolated extraction** — a dedicated retrieval + extraction path for safety protocols outperforms inline and system-prompt-only approaches.

The harness runs against 300 corrective maintenance records from two aircraft maintenance manuals (SC10000AMM: 146, AS-AMM-01-000: 154), using Gemini 2.5 Flash (generator and asynchronous judge) and Gemini 2.5 Flash-Lite (synchronous judge) via Vertex AI.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_CLOUD_PROJECT=your-project-id
EMBEDDING_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=credentials.json
```

Place your GCP service account key as `credentials.json` in the project root.

## Data

| File | Records | Description |
|---|---|---|
| `data/eval_dataset.json` | 300 | Aircraft maintenance fault scenarios with ground-truth chunk IDs, difficulty tiers (easy: 84, medium: 133, hard: 83), and observation types |
| `data/chunks/all_chunks.json` | 5,659 | Leaf-level text chunks from two aircraft maintenance manuals. 363 chunks (6.4%) flagged with safety content |

Records are auto-labeled GROUNDED, UNGROUNDED or AMBIGUOUS at run time, based on retrieval hit and fault-code match. Labels are not manually validated. AMBIGUOUS records are excluded from recall. The safety subset (55 records) contains PPE, LOTO or hazard content; its ground-truth protocols are keyword-extracted and not manually validated.

The vector store is built automatically on the first retrieval call via ChromaDB + `text-embedding-004`. This takes ~10 minutes and requires Vertex AI embedding quota.

## Scripts

### Core Modules (`src/`)

| Module | Description |
|---|---|
| `config.py` | Paths, model IDs, tuning knobs |
| `schemas.py` | Shared dataclasses and enums |
| `llm.py` | Vertex AI generate/embed with exponential backoff retry |
| `flat_rag.py` | ChromaDB-backed vector retrieval (cosine similarity, top-k). Auto-builds index if missing |
| `generator.py` | LLM repair plan generation from retrieved context |
| `sync_gate.py` | Synchronous quality gate — per-step faithfulness + grounding labels |
| `async_eval.py` | Asynchronous evaluator — context relevance + completeness scores |
| `safety_evaluator.py` | Three safety extraction conditions: S1 (isolated), S2 (inline), S3 (system-prompt) |
| `metrics.py` | Deterministic metrics: hit rate, MRR, precision/recall/F1, AUROC, ECE, Cohen's kappa, Krippendorff's alpha, bootstrap CI |

### Evaluation Scripts (`eval/`)

| Script | Command | Output |
|---|---|---|
| Retrieval baseline | `python -m eval.run_retrieval_baseline` | `results/retrieval_baseline.json` |
| Dual-mode ablation (D1–D9) | `python -m eval.run_dual_mode_ablation` | `results/dual_mode_ablation.json` |
| Safety ablation (S1–S3) | `python -m eval.run_safety_ablation` | `results/safety_ablation.json` |
| Full evaluation | `python -m eval.run_full_evaluation` | `results/eval_config.json` |
| Threshold calibration | `python -m eval.threshold_calibration` | `results/threshold_analysis/` |
| Report generator | `python -m eval.report_generator` | `results/final_report.md` |

### Reproducing

```bash
# Full pipeline (dual-mode + safety ablation + 12-goal scorecard)
python -m eval.run_full_evaluation

# Or run experiments individually
python -m eval.run_retrieval_baseline
python -m eval.run_dual_mode_ablation
python -m eval.run_safety_ablation

# Post-hoc threshold analysis (requires dual_mode_ablation.json)
python -m eval.threshold_calibration
```

Both ablation scripts support checkpointing — if interrupted, they resume from the last completed record.

## Paper Results

Results as reported in the paper. LLM outputs are non-deterministic, and GROUNDED/UNGROUNDED labels are assigned per run, so re-running the scripts will produce different numbers.

### Threshold configurations

| Config | Faith. | Ans. Rel. | Ctx. Rel. | Compl. |
| --- | --- | --- | --- | --- |
| Production | 0.30 | 0.15 | 0.30 | 0.40 |
| Manual | 0.40 | 0.30 | 0.50 | 0.50 |
| F1-optimal | 0.90 | 1.00 | 0.75 | 0.75 |

F1-optimal thresholds are found by sweeping each metric in 0.05 steps and choosing the best F1 (`eval/threshold_calibration.py`). They are calibrated in-sample on all 300 records, so F1-optimal recall is an upper estimate.

### Dual-mode detection across configurations

| Goal | Metric | Criterion | Prod. | Manual | F1-Opt. |
| --- | --- | --- | --- | --- | --- |
| D1 | Sync-only recall | ≥ 50% | 3.9% | 7.8% | 80.3% |
| D2 | Async-only recall | ≥ 40% | 1.1% | 10.6% | 69.1% |
| — | Dual-mode recall | — | 5.1% | 19.0% | 92.1% |
| D3 | Async-unique cases | > 0 | 4 | 16 | 21 |
| D4 | Sync-unique cases | > 0 | 11 | 18 | 41 |
| D6 | Sync-unique share | ≥ 10% | 77.8% | 52.9% | 25.0% |
| D6 | Async-unique share | ≥ 10% | 22.2% | 47.1% | 12.8% |
| D5 | Cohen's κ | < 0.40 | −0.02 | 0.20 | 0.13 |
| D7 | Dual highest in all difficulty tiers | 3 of 3 | 1 of 3 | 3 of 3 | 3 of 3 |
| — | Records flagged RED (% of 300) | — | 22.0% | 44.3% | 77.7% |
| — | Goals met (of 7) | — | 4 | 5 | 7 |

Recall is over the 178 UNGROUNDED records. Unique-detection counts (D3, D4) and κ (D5) are computed over all valid records. Manual-configuration sync detection also flags a plan when ≥ 30% of its steps are unfaithful.

### Goal numbering: paper vs. scripts

The scripts print nine goals (D1–D9); the paper reports seven.

| Paper | Script | Difference |
| --- | --- | --- |
| D1 | D1 | Script also requires sync P95 latency ≤ 3,000 ms |
| D2 | D2 | Same |
| D3 | D3 | Same |
| D4 | D4 | Same |
| D5 | D5 | Paper criterion κ < 0.40; script criterion κ in [0.3, 0.8] |
| D6 | D6 | Same |
| D7 | D9 | Same |
| — | D7, D8 | Not reported in the paper |

### Complementary detection at F1-optimal (178 UNGROUNDED records)

| | Async detects | Async misses |
| --- | --- | --- |
| Sync detects | 102 (62.2%) | 41 (25.0%) |
| Sync misses | 21 (12.8%) | 14 (not detected) |

Inter-mode Cohen's κ = 0.128.

### Recall by difficulty tier (F1-optimal)

| Mode | Easy | Medium | Hard | Overall |
| --- | --- | --- | --- | --- |
| SYNC_ONLY | 63.4% | 85.5% | 85.2% | 80.3% |
| ASYNC_ONLY | 56.1% | 66.3% | 83.3% | 69.1% |
| DUAL_MODE | 80.5% | 94.0% | 98.1% | 92.1% |

### Safety extraction ablation (55 records)

| Condition | Recall | Precision | Protocols / record |
| --- | --- | --- | --- |
| S1 (safety-isolated retrieval) | 79.1% | 44.2% | 5.5 |
| S2 (shared retrieval, explicit prompt) | 42.2% | 40.0% | 2.4 |
| S3 (shared retrieval, implicit instruction) | 37.0% | 37.2% | 1.1 |

### Scope of this harness

- The synchronous gate emits red / yellow / green; yellow counts as a pass, so detection is binary (RED = flagged), matching the paper's GREEN/RED signal.
- The retry-on-RED escalation policy used for the paper (Section III-B and Table VII) is not included in the current public release.

## Project Structure

```
├── src/               Core pipeline modules
├── eval/              Evaluation scripts
├── data/
│   ├── eval_dataset.json      300 evaluation records
│   └── chunks/
│       └── all_chunks.json    5,659 text chunks
├── results/           Output directory (gitignored)
├── requirements.txt
└── README.md
```
