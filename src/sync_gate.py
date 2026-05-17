import json
import logging
import time

from src.config import MODEL_FLASH_LITE
from src.generator import generate_with_retry
from src.schemas import GroundingLabel, StepVerdict, SyncGateResult

logger = logging.getLogger(__name__)

PROMPT_UNIFIED_INLINE_EVAL = """You are an impartial quality judge evaluating a repair plan generated from retrieved technical documentation.

You will perform TWO evaluation tasks in a SINGLE pass:

═══════════════════════════════════════════════════════
TASK 1: PER-STEP FAITHFULNESS
═══════════════════════════════════════════════════════

For EACH repair step, determine whether it is grounded in the retrieved context.

A step is "faithful" if its action, component, measurement, tool, or procedure is explicitly
stated in (or directly paraphrased from) at least one retrieved chunk.

For each step, provide:
- step_id: "s-0", "s-1", etc. (zero-indexed)
- faithful: true/false
- reason: one sentence explaining the verdict (max 100 chars)
- source_chunk_ids: list of chunk IDs that support this step (empty if ungrounded)
- grounding_label: one of:
    "verbatim" — step text closely matches source wording
    "paraphrased" — step conveys same fact in different words
    "synthesized" — step combines facts from multiple chunks correctly
    "ungrounded" — step introduces content not found in any chunk (automatically unfaithful)

═══════════════════════════════════════════════════════
TASK 2: ANSWER RELEVANCE
═══════════════════════════════════════════════════════

Determine how well the ENTIRE plan addresses the user's original question.

Scoring guide:
- 1.0: Plan directly addresses the exact fault/target the user asked about
- 0.75: Plan addresses the same subsystem/symptom but slightly different interpretation
- 0.5: Plan is about related equipment but a different fault mechanism
- 0.25: Weak topical overlap only
- 0.0: Plan is unrelated to the user's question

═══════════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════════

<UserQuestion>
{question}
</UserQuestion>

<RepairSteps>
{steps}
</RepairSteps>

<RetrievedContext>
{context}
</RetrievedContext>

═══════════════════════════════════════════════════════
OUTPUT FORMAT (strict JSON)
═══════════════════════════════════════════════════════

Return JSON:
{{
  "steps": [
    {{
      "step_id": "s-0",
      "faithful": true,
      "reason": "short reason",
      "source_chunk_ids": ["chunk-id"],
      "grounding_label": "verbatim|paraphrased|synthesized|ungrounded"
    }}
  ],
  "answer_relevance_score": 0.0-1.0,
  "answer_relevance_reason": "short reason"
}}
"""

BADGE_THRESHOLDS = {
    "red": {"faithfulness": 0.4, "answer_relevance": 0.3},
    "green": {"faithfulness": 0.7, "answer_relevance": 0.6},
    "max_unfaithful_step_pct": 0.3,
}


def compute_quality_badge(
    faithfulness_score: float | None,
    answer_relevance_score: float | None,
    step_verdicts: list[dict] | None = None,
) -> str:
    if faithfulness_score is None or answer_relevance_score is None:
        return "gray"

    scores = {"faithfulness": faithfulness_score, "answer_relevance": answer_relevance_score}
    red = BADGE_THRESHOLDS["red"]
    green = BADGE_THRESHOLDS["green"]

    if any(scores[k] < red[k] for k in scores):
        badge = "red"
    elif any(scores[k] < green[k] for k in scores):
        badge = "yellow"
    else:
        badge = "green"

    if step_verdicts:
        total = len(step_verdicts)
        unfaithful = sum(1 for v in step_verdicts if not v.get("faithful", True))
        if total > 0 and unfaithful > 0:
            pct = unfaithful / total
            if pct >= BADGE_THRESHOLDS["max_unfaithful_step_pct"]:
                badge = "red"
            elif badge == "green":
                badge = "yellow"

    return badge


def evaluate_sync(
    question: str,
    repair_plan: dict,
    context: str,
    chunk_ids: list[str] | None = None,
) -> SyncGateResult:
    t0 = time.perf_counter()

    steps = repair_plan.get("repair_steps", [])
    step_lines = []
    for s in steps:
        sid = s.get("step_id", "") if isinstance(s, dict) else ""
        text = s.get("text", "") if isinstance(s, dict) else str(s)
        step_lines.append(f"[{sid}] {text}")

    prompt = PROMPT_UNIFIED_INLINE_EVAL.format(
        question=question,
        steps="\n".join(step_lines) if step_lines else "(No steps to evaluate)",
        context=context,
    )

    response_text = generate_with_retry(prompt, model=MODEL_FLASH_LITE)
    latency_ms = (time.perf_counter() - t0) * 1000

    if not response_text:
        return SyncGateResult(
            passed=False,
            faithfulness_score=None,
            answer_relevance_score=None,
            badge="gray",
            step_verdicts=[],
            latency_ms=latency_ms,
            reasoning="LLM call failed",
        )

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return SyncGateResult(
            passed=False,
            faithfulness_score=None,
            answer_relevance_score=None,
            badge="gray",
            step_verdicts=[],
            latency_ms=latency_ms,
            reasoning=f"JSON parse error: {response_text[:100]}",
        )

    raw_steps = data.get("steps", [])
    step_verdicts = []
    for sv in raw_steps:
        step_verdicts.append(StepVerdict(
            step_id=sv.get("step_id", ""),
            faithful=sv.get("faithful", False),
            reason=sv.get("reason", ""),
            source_chunk_ids=sv.get("source_chunk_ids", []),
            grounding_label=sv.get("grounding_label", "ungrounded"),
        ))

    total_steps = len(step_verdicts)
    faithful_steps = sum(1 for v in step_verdicts if v.faithful)
    faithfulness_score = faithful_steps / total_steps if total_steps > 0 else None
    answer_relevance_score = data.get("answer_relevance_score")

    badge = compute_quality_badge(
        faithfulness_score,
        answer_relevance_score,
        [{"faithful": v.faithful} for v in step_verdicts],
    )

    passed = badge in ("green", "yellow")

    return SyncGateResult(
        passed=passed,
        faithfulness_score=faithfulness_score,
        answer_relevance_score=answer_relevance_score,
        badge=badge,
        step_verdicts=step_verdicts,
        latency_ms=latency_ms,
        reasoning=data.get("answer_relevance_reason", ""),
    )
