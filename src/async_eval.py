import logging
import re
import time

from src.flat_rag import _get_model
from src.schemas import AsyncEvalResult

logger = logging.getLogger(__name__)

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


def _context_relevance_heuristic(question: str, context: str) -> float:
    """Compute context relevance via embedding similarity between question and each context sentence."""
    model = _get_model()
    sentences = re.split(r'(?<=[.!?])\s+', context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return 0.0

    q_emb = model.encode(question, normalize_embeddings=True)
    s_embs = model.encode(sentences, normalize_embeddings=True)
    sims = (s_embs @ q_emb).tolist()

    relevant = sum(1 for s in sims if s > 0.3)
    return relevant / len(sentences)


def _completeness_heuristic(question: str, plan_text: str, context: str) -> float:
    """Compute completeness: what fraction of context key-terms appear in the plan."""
    context_words = set(re.findall(r"\b\w{6,}\b", _normalize_text(context)))
    plan_words = set(re.findall(r"\b\w{6,}\b", _normalize_text(plan_text)))
    question_words = set(re.findall(r"\b\w{5,}\b", _normalize_text(question)))

    key_terms = context_words & question_words
    if not key_terms:
        key_terms = context_words

    if not key_terms:
        return 0.0

    covered = key_terms & plan_words
    return len(covered) / len(key_terms)


def evaluate_async(
    question: str,
    plan_text: str,
    context: str,
    safety_protocols: list[dict] | None = None,
) -> AsyncEvalResult:
    """Evaluate async metrics using heuristic methods (no LLM needed)."""
    t0 = time.perf_counter()

    ctx_relevance = _context_relevance_heuristic(question, context)
    completeness = _completeness_heuristic(question, plan_text, context)

    safety_coverage = None
    if safety_protocols and plan_text:
        survived = sum(1 for p in safety_protocols if protocol_survives(p, plan_text))
        safety_coverage = survived / len(safety_protocols) if safety_protocols else None

    latency_ms = (time.perf_counter() - t0) * 1000

    return AsyncEvalResult(
        context_relevance=ctx_relevance,
        completeness=completeness,
        safety_coverage=safety_coverage,
        latency_ms=latency_ms,
        context_relevance_reason="embedding_similarity",
        completeness_reason="key_term_coverage",
    )
