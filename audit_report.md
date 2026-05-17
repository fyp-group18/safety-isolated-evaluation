# Audit Report: hitl-dss-react

## 1. Eval Dataset

- **File**: `backend/eval/test_set_data/synthetic_eval/datasets/eval_dataset.bak.pre_multichunk_1778635134.json`
- **Records**: 150
- **Schema fields**: `id`, `observation`, `ground_truth_fault_code`, `manual_source`, `section_system`, `page_reference`, `rationale`, `difficulty_tier`, `observation_type`, `ground_truth_chunk_id`, `origin`, `ground_truth_chunk_id_original`, `ground_truth_chunk_ids`
- **Difficulty tiers**: easy (39), medium (73), hard (38)
- **Observation types**: symptom_report (43), inspection_finding (20), operational_anomaly (42), part_replacement (45)
- **Origins**: all `table` (150/150)
- **Multi-chunk IDs**: all 150 records have `ground_truth_chunk_ids` (list of UUIDs)
- **Ambiguous alternatives**: 18 records have `ambiguous_alternatives` field

## 2. Chunk Data (Vector Store Source)

- **Storage**: PostgreSQL table `document_chunks_multimodal` with pgvector `Vector(3072)`
- **Total chunks**: 5770 (5659 leaf-level, level=0)
- **Documents**: 2 distinct documents
- **Exported**: 5659 non-blacklisted leaf chunks to `data/chunks/all_chunks.json` (4.9 MB)
- **Safety chunks**: 363/5659 (6.4%) contain safety signal words

### Chunk Fields (from DB)
| Field | Type | Notes |
|---|---|---|
| id | String (UUID) | Primary key |
| document_id | Integer | FK to documents_multimodal |
| text | Text | Chunk content |
| level | Integer | 0 = leaf, >0 = RAPTOR summary |
| pages | String | Human-readable page label |
| page_start | Integer | Numeric page bound |
| page_end | Integer | Numeric page bound |
| sequence_index | Integer | Leaf order within document (0-based) |
| section_header | Text | Procedure/section name from PDF headings |
| section_code | String | ATA chapter/section identifier |
| parent_id | String | Self-referencing FK (table-row children → full-table parent) |
| images | JSONB | Associated diagram images |
| blacklisted | Boolean | Admin blacklist flag |

### Safety Content Tagging (computed at export, not in DB)
- Signal word regex: `DANGER|WARNING|CAUTION|NOTICE|HAZARD|LETHAL|HIGH VOLTAGE|PPE|LOCKOUT|TAGOUT|LOTO`
- Fields added: `has_safety_content` (bool), `safety_signal_words` (list)

## 3. Embedding Model

- **Model**: `gemini-embedding-2-preview`
- **Dimensions**: 3072-D
- **Client**: `google.genai.Client` (Vertex AI, `us-central1`)
- **Batch embedding**: `ThreadPoolExecutor` with 10 concurrent workers
- **Retry**: Exponential backoff, 6 max retries, base 2s

## 4. Sync Gate (Inline Evaluator)

### Architecture
- **Single unified call** via `evaluate_unified()` → `UnifiedInlineEvalResult`
- **Model**: `gemini-2.5-flash-lite` (constant `INLINE_EVAL_MODEL`)
- **Metrics computed**: Per-step faithfulness + answer relevance (2-metric inline tier)

### Gate Logic
1. Each repair step evaluated for faithfulness with grounding labels: `verbatim`, `paraphrased`, `synthesized`, `ungrounded`
2. Whole-plan faithfulness = `faithful_steps / total_steps` (arithmetic from step verdicts)
3. Answer relevance: 0.0–1.0 semantic alignment score

### Badge Computation (`compute_quality_badge`)
- **green**: all metrics ≥ green thresholds
- **yellow**: not red, but at least one metric below green threshold; OR any unfaithful step
- **red**: any metric below red threshold; OR ≥30% unfaithful steps
- **gray**: any metric is None (LLM/JSON failure)

### Retry Logic
- REPAIR_PLAN artifact + non-green badge + retry_count==0 → loop back to RepairPlanner
- retry_count≥1 → forced pass, flagged_for_review=True

### Prompt
- `PROMPT_UNIFIED_INLINE_EVAL` in `prompts/system_prompts.py` (line 334)
- Dual-task: per-step faithfulness + answer relevance in one pass
- Structured JSON output via Gemini's `response_schema`

## 5. Async Evaluation (Shadow Evaluator)

### Architecture
- Runs asynchronously after inline gate via Cloud Tasks / BackgroundTask
- **Model**: `gemini-2.5-flash` for all metrics
- **Budget**: 180s total, 90s per-call timeout, 3 attempts with exponential backoff

### Metrics (3 phases, sequential + parallel)
Faithfulness and answer relevance are already computed by the sync gate — removed from async to avoid redundancy.

1. **Phase 1**: Context relevance + Completeness (parallel) — `PROMPT_ASYNC_CONTEXT_RELEVANCE`, `PROMPT_ASYNC_COMPLETENESS`
2. **Phase 2**: Per-step faithfulness cross-validation (serial) — `PROMPT_SYNC_PER_STEP_FAITHFULNESS`
3. **Phase 3**: Safety coverage (arithmetic, not LLM) — `protocol_survives()` from safety_judge_gate

### Output Schema
```python
ASYNC_METRIC_KEYS = ("context_relevance", "completeness")
```
Each metric: `{score: float|None, reason: str}`
Plus: `evidence`, `step_verdicts`, `duration_ms`, `safety_coverage`

## 6. Safety Evaluator

### Architecture — Dual-Path Retrieval
- **Path A**: Regex filter on already-retrieved chunks (from KnowledgeRetriever)
  - Pattern: `DANGER|WARNING|CAUTION|NOTICE|HAZARD|PPE|lockout|tagout|high.voltage|electric.shock|burn|crush|pinch|entangle|toxic|fire.hazard|explosion|asphyxiation`
- **Path B**: Fresh vector search with `has_safety_content = TRUE` filter
  - Query: `"safety warnings hazards precautions {equipment_type}"`
  - Cosine distance, LIMIT 10, filtered to compatible device models
- **Merge**: Deduplicate (Path B skips IDs already in Path A), cap at 15 chunks

### Extraction
- **Model**: `gemini-2.5-flash` (MODEL_FLASH)
- **Prompt**: `_SAFETY_EXTRACTION_PROMPT` — extracts JSON with `protocols[{severity, text, hazard_type, source_chunk_ids}]`
- **Temperature**: 0.0
- **Output MIME**: `application/json`

### Output Schema
```json
{
  "protocols": [{"severity": "DANGER|WARNING|CAUTION|NOTICE", "text": "...", "hazard_type": "...", "source_chunk_ids": [...]}],
  "extraction_confidence": 0.0-1.0,
  "no_safety_content_found": true/false
}
```

### Safety Schemas (Pydantic)
- `SafetyCategory` enum: LOTO, PPE, HAZARD_WARNING, ELECTRICAL, CHEMICAL, MECHANICAL, THERMAL, PRESSURE, GENERAL
- `SafetySeverity` enum: DANGER, WARNING, CAUTION, NOTICE
- `SafetyProtocol`: id, category, severity, instruction, source_chunk_ids, source_text_excerpt, applies_before_step, applies_during_step, applies_throughout

## 7. Safety Judge Gate

### Architecture — Deterministic/Lexical (no LLM)
- **Function**: `protocol_survives(protocol, plan_text)` — keyword overlap check
- **Logic**: Normalize text → extract significant words (≥5 chars) → overlap rate ≥ 0.5 → survived
- **Gate threshold**: `_SURVIVAL_THRESHOLD = 0.6` (survival_rate = survived/total)
- **PASS**: Routes to InlineEvaluator
- **FAIL**: Routes back to RepairPlanner with feedback (first fail only)
- **Used by shadow eval**: Phase 5 safety_coverage metric

## 8. LangGraph Workflow

### Node Sequence (for troubleshoot/replace_part paths)
```
IntentRouter → DeterministicRuleChecker → SymptomAnalyzer/DirectReplacementAnalyzer
→ KnowledgeRetriever → SafetyEvaluator → SafetyExtractor → RepairPlanner
→ SafetyJudgeGate → InlineEvaluator → END
```

### Retry Loop
- InlineEvaluator (fail, retry_count=0) → RepairPlanner → SafetyJudgeGate → InlineEvaluator
- InlineEvaluator (fail, retry_count≥1) → END (forced, flagged_for_review)

## 9. API Keys & Configuration

### Environment Variables (backend/.env)
- `DATABASE_URL` — PostgreSQL connection string
- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS`
- `GCS_BUCKET_NAME`, `STORAGE_BACKEND`
- No OpenAI or Anthropic keys — system is Gemini-only

### Model Constants (core/config.py)
| Constant | Value | Used By |
|---|---|---|
| MODEL_PRO | `gemini-2.5-pro` | Legacy individual metric runners |
| MODEL_FLASH | `gemini-2.5-flash` | Shadow evaluator, safety evaluator |
| MODEL_EMBEDDING | `gemini-embedding-2-preview` | Embedding module |
| INLINE_EVAL_MODEL | `gemini-2.5-flash-lite` | Unified inline evaluator |

## 10. Key Files for Re-implementation

| Component | Source File | Key Functions |
|---|---|---|
| Sync gate prompt | `prompts/system_prompts.py:334` | `PROMPT_UNIFIED_INLINE_EVAL` |
| Sync gate logic | `modules/graph/eval_logic.py:324` | `evaluate_unified()`, `compute_quality_badge()` |
| Async eval | `modules/graph/custom_ragas.py:260` | `run_shadow_evaluation()` |
| Async prompts | `prompts/system_prompts.py:918+` | `PROMPT_SYNC_FAITHFULNESS`, `PROMPT_SYNC_ANSWER_RELEVANCE`, `PROMPT_ASYNC_CONTEXT_RELEVANCE`, `PROMPT_ASYNC_COMPLETENESS` |
| Safety evaluator | `modules/graph/safety_evaluator.py` | `safety_evaluator()` |
| Safety judge | `modules/graph/safety_judge_gate.py` | `protocol_survives()`, `safety_judge_gate()` |
| Eval schemas | `modules/graph/eval_schemas.py` | All Pydantic schemas |
| Safety schemas | `modules/graph/safety_schemas.py` | `SafetyProtocol`, `SafetyCategory`, `SafetySeverity` |
| Embeddings | `modules/embeddings.py` | `embed()`, `embed_batch()` |

## 11. Data Exported

| File | Description | Location |
|---|---|---|
| `all_chunks.json` | 5659 leaf chunks from DB | `data/chunks/all_chunks.json` |
| `eval_dataset.json` | 150-record eval dataset | To copy from hitl-dss-react |
