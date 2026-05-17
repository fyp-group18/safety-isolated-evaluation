import json
import logging
import time

from google import genai
from google.genai import types

from src.config import (
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    MODEL_FLASH,
)

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
        )
    return _client


def generate_with_retry(
    prompt: str,
    model: str = MODEL_FLASH,
    temperature: float = 0.0,
    response_mime_type: str = "application/json",
    max_retries: int = 3,
) -> str | None:
    client = _get_genai_client()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type=response_mime_type,
                    temperature=temperature,
                ),
            )
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            is_quota = "429" in err_str or "quota" in err_str or "resource" in err_str
            if attempt < max_retries - 1 and is_quota:
                wait = min(2 ** (attempt + 1), 60)
                logger.warning(f"LLM 429, backing off {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"LLM error: {e}, retrying in {wait}s")
                time.sleep(wait)
                continue
            logger.error(f"LLM call failed after {max_retries} attempts: {e}")
            return None


def generate_repair_plan(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> dict | None:
    prompt = f"""You are an expert industrial maintenance diagnostician. Based on the technician's observation and the retrieved maintenance manual context, generate a structured repair plan.

<Observation>
{observation}
</Observation>

<RetrievedContext>
{context}
</RetrievedContext>

Equipment type: {equipment_type}

Generate a JSON response with:
{{
  "root_cause": "Most likely root cause based on the context",
  "repair_steps": [
    {{"step_id": "s-0", "text": "Step description grounded in the manual context"}},
    {{"step_id": "s-1", "text": "Next step..."}}
  ],
  "confidence": 0.0-1.0
}}

Rules:
- Ground every step in the retrieved context. Do not invent procedures.
- If context is insufficient, say so rather than guessing.
- Include specific part numbers, measurements, and tool references from the context.
"""
    result = generate_with_retry(prompt)
    if result:
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse repair plan JSON: {result[:200]}")
    return None
