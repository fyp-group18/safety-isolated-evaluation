import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import MODEL_FLASH
from src.generator import generate_with_retry
from src.schemas import AsyncEvalResult

logger = logging.getLogger(__name__)

PROMPT_ASYNC_CONTEXT_RELEVANCE = """You are a retrieval-quality auditor. Mark every sentence in the CONTEXT and explain why each is or is not relevant.

INSTRUCTIONS:
1. Segment CONTEXT into sentences (cap at 50).
2. For each segment: relevance verdict + a short reason (max 30 chars).
3. score = relevant / total.

<USER_QUESTION>
{question}
</USER_QUESTION>

<CONTEXT>
{context}
</CONTEXT>

Return JSON: {{
  "score": float|null,
  "reason": "justification",
  "evidence": {{"sentences": [{{"text": "...", "relevant": true, "why": "..."}}]}}
}}
"""

PROMPT_ASYNC_COMPLETENESS = """You are a completeness auditor. Identify every key fact required to answer the USER_QUESTION (using the CONTEXT) and verify presence in the PLAN.

INSTRUCTIONS:
1. List up to 12 key facts from the USER_QUESTION and CONTEXT that a complete plan must contain.
2. For each, decide whether it is present in the PLAN. If present, quote the supporting plan text.
3. score = present / total.

<USER_QUESTION>
{question}
</USER_QUESTION>

<CONTEXT>
{context}
</CONTEXT>

<PLAN>
{plan}
</PLAN>

Return JSON: {{
  "score": float|null,
  "reason": "justification",
  "evidence": {{"key_facts": [{{"fact": "...", "present_in_plan": true, "evidence_quote": "..."}}]}}
}}
"""

_SURVIVAL_THRESHOLD = 0.5


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def protocol_survives(protocol: dict, plan_text: str) -> bool:
    protocol_text = protocol.get("text", "")
    if not protocol_text:
        return True

    norm_plan = _normalize_text(plan_text)
    norm_protocol = _normalize_text(protocol_text)

    protocol_words = set(re.findall(r"\b\w{5,}\b", norm_protocol))
    if not protocol_words:
        return norm_protocol in norm_plan

    plan_words = set(re.findall(r"\b\w{5,}\b", norm_plan))
    overlap = protocol_words & plan_words
    overlap_rate = len(overlap) / len(protocol_words)
    return overlap_rate >= _SURVIVAL_THRESHOLD


def evaluate_async(
    question: str,
    plan_text: str,
    context: str,
    safety_protocols: list[dict] | None = None,
) -> AsyncEvalResult:
    t0 = time.perf_counter()

    ctx_relevance_score = None
    ctx_relevance_reason = ""
    completeness_score = None
    completeness_reason = ""
    safety_coverage = None

    with ThreadPoolExecutor(max_workers=2) as ex:
        future_ctx = ex.submit(
            generate_with_retry,
            PROMPT_ASYNC_CONTEXT_RELEVANCE.format(question=question, context=context),
            MODEL_FLASH,
        )
        future_comp = ex.submit(
            generate_with_retry,
            PROMPT_ASYNC_COMPLETENESS.format(question=question, context=context, plan=plan_text),
            MODEL_FLASH,
        )

        for fut in as_completed([future_ctx, future_comp], timeout=120):
            try:
                result_text = fut.result()
                if result_text:
                    data = json.loads(result_text)
                    if fut is future_ctx:
                        ctx_relevance_score = data.get("score")
                        ctx_relevance_reason = data.get("reason", "")
                    else:
                        completeness_score = data.get("score")
                        completeness_reason = data.get("reason", "")
            except Exception as e:
                metric = "context_relevance" if fut is future_ctx else "completeness"
                logger.error(f"Async {metric} failed: {e}")

    if safety_protocols and plan_text:
        survived = sum(1 for p in safety_protocols if protocol_survives(p, plan_text))
        safety_coverage = survived / len(safety_protocols) if safety_protocols else None

    latency_ms = (time.perf_counter() - t0) * 1000

    return AsyncEvalResult(
        context_relevance=ctx_relevance_score,
        completeness=completeness_score,
        safety_coverage=safety_coverage,
        latency_ms=latency_ms,
        context_relevance_reason=ctx_relevance_reason,
        completeness_reason=completeness_reason,
    )
