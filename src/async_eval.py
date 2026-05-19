import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import MODEL_FLASH
from src.llm import generate_with_retry
from src.schemas import AsyncEvalResult

logger = logging.getLogger(__name__)

_SURVIVAL_THRESHOLD = 0.5

_CONTEXT_RELEVANCE_PROMPT = """\
Rate the relevance of the retrieved context to the technician's question.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

Score from 0.0 to 1.0:
- 1.0: Every retrieved chunk is directly relevant to diagnosing or resolving the issue
- 0.7: Most chunks are relevant with minor noise
- 0.5: Mixed relevance — some chunks are useful, others are unrelated
- 0.3: Mostly irrelevant with a few useful fragments
- 0.0: Context is completely irrelevant to the question

Return JSON: {{"score": 0.0, "reason": "one sentence explanation"}}"""

_COMPLETENESS_PROMPT = """\
Rate how completely the repair plan addresses the technician's question given the available context.

QUESTION: {question}

REPAIR PLAN:
{plan_text}

AVAILABLE CONTEXT:
{context}

Score from 0.0 to 1.0:
- 1.0: Plan thoroughly covers every aspect the context supports — diagnosis, procedure, parts, verification
- 0.7: Plan covers the main repair procedure but misses some secondary details
- 0.5: Plan addresses the core issue but omits important steps or safety considerations
- 0.3: Plan is incomplete — misses critical steps or procedures available in the context
- 0.0: Plan fails to address the core issue or is entirely wrong

Return JSON: {{"score": 0.0, "reason": "one sentence explanation"}}"""


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


def _eval_context_relevance(question: str, context: str) -> tuple[float | None, str]:
    prompt = _CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
    raw = generate_with_retry(
        prompt=prompt,
        model=MODEL_FLASH,
        temperature=0.0,
        response_mime_type="application/json",
    )
    if raw:
        try:
            result = json.loads(raw)
            return float(result["score"]), result.get("reason", "")
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(f"Context relevance parse error: {raw[:200]}")
    return None, "llm_failed"


def _eval_completeness(question: str, plan_text: str, context: str) -> tuple[float | None, str]:
    prompt = _COMPLETENESS_PROMPT.format(
        question=question, plan_text=plan_text, context=context
    )
    raw = generate_with_retry(
        prompt=prompt,
        model=MODEL_FLASH,
        temperature=0.0,
        response_mime_type="application/json",
    )
    if raw:
        try:
            result = json.loads(raw)
            return float(result["score"]), result.get("reason", "")
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(f"Completeness parse error: {raw[:200]}")
    return None, "llm_failed"


def evaluate_async(
    question: str,
    plan_text: str,
    context: str,
    safety_protocols: list[dict] | None = None,
) -> AsyncEvalResult:
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_cr = executor.submit(_eval_context_relevance, question, context)
        future_comp = executor.submit(_eval_completeness, question, plan_text, context)

        ctx_relevance, cr_reason = future_cr.result()
        completeness, comp_reason = future_comp.result()

    safety_cov = None
    if safety_protocols and plan_text:
        survived = sum(1 for p in safety_protocols if protocol_survives(p, plan_text))
        safety_cov = survived / len(safety_protocols) if safety_protocols else None

    latency_ms = (time.perf_counter() - t0) * 1000

    return AsyncEvalResult(
        context_relevance=ctx_relevance,
        completeness=completeness,
        safety_coverage=safety_cov,
        latency_ms=latency_ms,
        context_relevance_reason=cr_reason,
        completeness_reason=comp_reason,
    )
