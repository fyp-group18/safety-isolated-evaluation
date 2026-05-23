import json
import logging
import time

from src.config import MODEL_FLASH_LITE
from src.llm import generate_with_retry
from src.schemas import StepVerdict, SyncGateResult

logger = logging.getLogger(__name__)

BADGE_THRESHOLDS = {
    "red": {"faithfulness": 0.4, "answer_relevance": 0.3},
    "green": {"faithfulness": 0.7, "answer_relevance": 0.6},
    "max_unfaithful_step_pct": 0.3,
}

_SYNC_EVAL_PROMPT = """\
Evaluate this repair plan for faithfulness to the source context and relevance to the question.

QUESTION: {question}

SOURCE CONTEXT:
{context}

REPAIR PLAN STEPS:
{steps}

For EACH step, determine:
1. faithful: Can this step be verified from the source context? (true/false)
2. grounding_label: One of "verbatim" (exact or near-exact match), "paraphrased" (same meaning, different wording), "synthesized" (reasonable inference from context), "ungrounded" (not supported by context)
3. reason: Brief explanation (one sentence)

Also rate overall answer_relevance: How well does this repair plan address the original question? (0.0-1.0)

Return JSON:
{{
  "step_verdicts": [
    {{"step_id": "s-0", "faithful": true, "grounding_label": "paraphrased", "reason": "Step matches procedure in context paragraph 2"}}
  ],
  "answer_relevance": 0.85
}}"""


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
    if not steps:
        latency_ms = (time.perf_counter() - t0) * 1000
        return SyncGateResult(
            passed=False,
            faithfulness_score=0.0,
            answer_relevance_score=0.0,
            badge="red",
            step_verdicts=[],
            latency_ms=latency_ms,
            reasoning="no_steps",
        )

    steps_formatted = "\n".join(
        f"  {s.get('step_id', f's-{i}')}: {s.get('text', str(s))}"
        for i, s in enumerate(steps)
    )

    prompt = _SYNC_EVAL_PROMPT.format(
        question=question,
        context=context,
        steps=steps_formatted,
    )

    raw = generate_with_retry(
        prompt=prompt,
        model=MODEL_FLASH_LITE,
        temperature=0.0,
        response_mime_type="application/json",
    )

    step_verdicts = []
    answer_relevance_score = None

    if raw:
        try:
            result = json.loads(raw)
            answer_relevance_score = float(result.get("answer_relevance", 0.5))

            for v in result.get("step_verdicts", []):
                step_verdicts.append(StepVerdict(
                    step_id=v.get("step_id", ""),
                    faithful=bool(v.get("faithful", False)),
                    reason=v.get("reason", ""),
                    source_chunk_ids=[],
                    grounding_label=v.get("grounding_label", "ungrounded"),
                ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Sync gate JSON parse error: {e}")

    if not step_verdicts:
        for i, s in enumerate(steps):
            step_verdicts.append(StepVerdict(
                step_id=s.get("step_id", f"s-{i}") if isinstance(s, dict) else f"s-{i}",
                faithful=False,
                reason="llm_parse_failed",
                source_chunk_ids=[],
                grounding_label="ungrounded",
            ))

    total_steps = len(step_verdicts)
    faithful_steps = sum(1 for v in step_verdicts if v.faithful)
    faithfulness_score = faithful_steps / total_steps if total_steps > 0 else 0.0

    badge = compute_quality_badge(
        faithfulness_score,
        answer_relevance_score,
        [{"faithful": v.faithful} for v in step_verdicts],
    )

    passed = badge in ("green", "yellow")
    latency_ms = (time.perf_counter() - t0) * 1000

    return SyncGateResult(
        passed=passed,
        faithfulness_score=faithfulness_score,
        answer_relevance_score=answer_relevance_score,
        badge=badge,
        step_verdicts=step_verdicts,
        latency_ms=latency_ms,
        reasoning=f"llm_faithfulness={faithfulness_score:.2f}, llm_relevance={answer_relevance_score}",
    )
