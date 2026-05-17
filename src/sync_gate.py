import logging
import re
import time

from src.flat_rag import _get_model
from src.schemas import StepVerdict, SyncGateResult

logger = logging.getLogger(__name__)

BADGE_THRESHOLDS = {
    "red": {"faithfulness": 0.4, "answer_relevance": 0.3},
    "green": {"faithfulness": 0.7, "answer_relevance": 0.6},
    "max_unfaithful_step_pct": 0.3,
}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower())


def _ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute n-gram overlap ratio between two texts."""
    words_a = _normalize(text_a).split()
    words_b = _normalize(text_b).split()
    if len(words_a) < n or len(words_b) < n:
        word_set_a = set(words_a)
        word_set_b = set(words_b)
        if not word_set_a:
            return 0.0
        return len(word_set_a & word_set_b) / len(word_set_a)

    ngrams_a = set(tuple(words_a[i:i+n]) for i in range(len(words_a) - n + 1))
    ngrams_b = set(tuple(words_b[i:i+n]) for i in range(len(words_b) - n + 1))
    if not ngrams_a:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(ngrams_a)


def _classify_grounding(overlap: float) -> str:
    if overlap >= 0.6:
        return "verbatim"
    elif overlap >= 0.3:
        return "paraphrased"
    elif overlap >= 0.1:
        return "synthesized"
    return "ungrounded"


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
    """Evaluate sync gate using n-gram overlap and embedding similarity.

    Faithfulness: n-gram overlap between each step and the context.
    Answer relevance: cosine similarity between question and plan text.
    """
    t0 = time.perf_counter()
    model = _get_model()

    steps = repair_plan.get("repair_steps", [])
    step_verdicts = []

    for s in steps:
        step_text = s.get("text", "") if isinstance(s, dict) else str(s)
        sid = s.get("step_id", "") if isinstance(s, dict) else ""

        overlap = _ngram_overlap(step_text, context)
        faithful = overlap >= 0.1
        label = _classify_grounding(overlap)

        step_verdicts.append(StepVerdict(
            step_id=sid,
            faithful=faithful,
            reason=f"ngram_overlap={overlap:.2f}",
            source_chunk_ids=[],
            grounding_label=label,
        ))

    total_steps = len(step_verdicts)
    faithful_steps = sum(1 for v in step_verdicts if v.faithful)
    faithfulness_score = faithful_steps / total_steps if total_steps > 0 else None

    # Answer relevance via embedding cosine similarity
    plan_text = " ".join(
        s.get("text", str(s)) if isinstance(s, dict) else str(s)
        for s in steps
    )
    if plan_text and question:
        embs = model.encode([question, plan_text], normalize_embeddings=True)
        answer_relevance_score = float(embs[0] @ embs[1])
    else:
        answer_relevance_score = None

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
        reasoning=f"ngram_faithfulness={faithfulness_score}, emb_relevance={answer_relevance_score}",
    )
