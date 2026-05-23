# Safety-Isolated Evaluation Harness

Standalone evaluation harness for two research claims on LLM-based aircraft maintenance decision support:

1. **Dual-mode confidence gating** — combining synchronous (inline) and asynchronous (shadow) quality gates outperforms either mode alone at detecting ungrounded repair plans.
2. **Safety-isolated extraction** — a dedicated retrieval + extraction path for safety protocols outperforms inline and system-prompt-only approaches.

The harness runs against 300 aircraft maintenance fault scenarios drawn from two real maintenance manuals, using Gemini 2.5 Flash via Vertex AI.

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

## Benchmark Results

Results from a single run using `gemini-2.5-flash` (generation/eval) and `gemini-2.5-flash-lite` (sync gate) on 300 records. LLM outputs are non-deterministic; exact numbers will vary across runs.

### Dual-Mode Ablation (D1–D9)

| Goal | Criterion | Result | Verdict |
|---|---|---|---|
| D1 | Sync recall ≥ 60% | 7.8% | FAIL |
| D2 | Async recall ≥ 60% | 10.6% | FAIL |
| D3 | Dual recall > sync | 16.2% > 7.8% | PASS |
| D4 | Dual recall > async | 16.2% > 10.6% | PASS |
| D5 | Sync P95 latency < 3s | 2,886 ms | FAIL |
| D6 | Cohen's kappa < 0.4 (complementary modes) | κ = 0.20 | PASS |
| D7 | Cell B+C ≥ 10% of flagged | 34 / 300 | PASS |
| D8 | Sync unique contribution > 0% | 34.5% | PASS |
| D9 | Async unique contribution > 0% | 51.7% | PASS |

**6 / 9 pass**

| Metric | Sync | Async | Dual |
|---|---|---|---|
| Recall | 7.8% | 10.6% | 16.2% |
| Precision | 58.3% | 86.4% | 72.5% |
| F1 | 0.138 | 0.189 | 0.265 |

### Safety Ablation (S1–S3)

Evaluated on 55 records containing safety-relevant content.

| Goal | Criterion | Result | Verdict |
|---|---|---|---|
| S1 | S1 recall > S2 and S3 | 79.1% vs 42.2% vs 37.0% | PASS |
| S2 | S1 precision > S3 | 44.2% vs 37.2% | PASS |
| S3 | S1 coverage ≥ 85% | 23.6% | FAIL |

**2 / 3 pass**

| Condition | Precision | Recall | F1 | Avg Protocols |
|---|---|---|---|---|
| S1 (isolated) | 44.2% | 79.1% | 0.567 | 5.5 |
| S2 (inline) | 37.3% | 42.2% | 0.396 | 2.4 |
| S3 (system-prompt) | 37.2% | 37.0% | 0.371 | 1.1 |

### Overall: 8 / 12 goals pass

### Threshold Calibration (Post-Hoc)

F1-optimal thresholds (faithfulness=0.9, answer_relevance=1.0, context_relevance=0.75, completeness=0.75) boost sync recall to 88.3% and dual recall to 95.0%, achieving 6/9 D-goals under optimized thresholds.

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
