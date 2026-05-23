import json
import logging
import time

from src.config import MODEL_FLASH
from src.llm import generate_with_retry

logger = logging.getLogger(__name__)

_PLAN_PROMPT = """\
You are an aircraft maintenance repair planner. Given a technician's observation and retrieved manual context, generate a detailed repair plan.

OBSERVATION:
{observation}

RETRIEVED CONTEXT:
{context}

Generate a JSON repair plan with the following structure:
{{
  "repair_steps": [
    {{"step_id": "s-0", "text": "Detailed actionable repair step..."}},
    {{"step_id": "s-1", "text": "..."}},
    ...
  ],
  "root_cause": "One-line description of the most likely root cause",
  "confidence": 0.0
}}

Rules:
- Generate 3-8 actionable repair steps based on the context
- Each step must be grounded in the retrieved context but written in your own words
- Include specific part numbers, torque values, tools, and procedures from the context where available
- Steps should be in logical execution order
- confidence: how well the context supports this plan (0.0-1.0)
- If the context is insufficient, still provide the best plan possible but set confidence low"""


def generate_repair_plan(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> dict:
    t0 = time.perf_counter()

    prompt = _PLAN_PROMPT.format(observation=observation, context=context)
    raw = generate_with_retry(
        prompt=prompt,
        model=MODEL_FLASH,
        temperature=0.1,
        response_mime_type="application/json",
    )

    if not raw:
        logger.warning("Plan generation returned None")
        return {"repair_steps": [], "root_cause": "generation_failed", "confidence": 0.0}

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse plan JSON: {raw[:200]}")
        return {"repair_steps": [], "root_cause": "json_parse_failed", "confidence": 0.0}

    if "repair_steps" not in plan:
        plan["repair_steps"] = []
    if "root_cause" not in plan:
        plan["root_cause"] = "unknown"
    if "confidence" not in plan:
        plan["confidence"] = 0.5

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.debug(f"Plan generated in {latency_ms:.0f}ms with {len(plan['repair_steps'])} steps")

    return plan
