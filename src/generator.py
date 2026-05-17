import json
import logging
import re
import time

from src.flat_rag import _get_model

logger = logging.getLogger(__name__)


def _extract_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def generate_repair_plan(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> dict:
    """Generate a repair plan by extracting relevant sentences from context.

    Uses embedding similarity to select the most relevant sentences
    from the retrieved context as repair steps.
    """
    model = _get_model()
    obs_emb = model.encode(observation, normalize_embeddings=True)

    sentences = _extract_sentences(context)
    if not sentences:
        return {"repair_steps": [], "root_cause": "no_context", "confidence": 0.0}

    sent_embs = model.encode(sentences, normalize_embeddings=True)
    scores = (sent_embs @ obs_emb).tolist()

    ranked = sorted(zip(scores, sentences), reverse=True)
    top_sentences = ranked[:min(6, len(ranked))]

    steps = []
    for i, (score, sent) in enumerate(top_sentences):
        clean = re.sub(r'\[CHUNK-ID: [^\]]+\]', '', sent).strip()
        if len(clean) > 15:
            steps.append({"step_id": f"s-{i}", "text": clean, "score": score})

    root_cause = steps[0]["text"][:100] if steps else "unknown"

    return {
        "repair_steps": steps,
        "root_cause": root_cause,
        "confidence": top_sentences[0][0] if top_sentences else 0.0,
    }
