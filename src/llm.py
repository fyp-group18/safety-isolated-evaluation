import logging
import time

from google import genai
from google.genai import types

from src.config import (
    EMBEDDING_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    MODEL_EMBEDDING,
    RETRY_LIMIT,
)

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=EMBEDDING_LOCATION,
        )
        logger.info(f"Initialized genai client for project={GOOGLE_CLOUD_PROJECT}, location={EMBEDDING_LOCATION}")
    return _client


def generate_with_retry(
    prompt: str,
    model: str,
    max_retries: int = RETRY_LIMIT,
    temperature: float = 0.0,
    response_mime_type: str | None = None,
) -> str | None:
    client = _get_client()
    config_kwargs = {"temperature": temperature}
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    config = types.GenerateContentConfig(**config_kwargs)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "resource" in err or "rate" in err:
                wait = min(2 ** attempt, 60)
                logger.warning(f"Rate limited, backing off {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = min(2 ** min(attempt, 5), 60)
                logger.warning(f"Error: {e}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                return None
    return None


def embed_with_retry(
    texts: str | list[str],
    model: str = MODEL_EMBEDDING,
    max_retries: int = RETRY_LIMIT,
) -> list[list[float]]:
    client = _get_client()
    if isinstance(texts, str):
        texts = [texts]

    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model,
                contents=texts,
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            err = str(e).lower()
            if "400" in err or "invalid_argument" in err:
                logger.error(f"Embed input error (not retryable): {e}")
                raise
            if "429" in err or "quota" in err or "resource" in err or "rate" in err:
                wait = min(2 ** attempt, 60)
                logger.warning(f"Embed rate limited, backing off {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = min(2 ** min(attempt, 5), 60)
                logger.warning(f"Embed error: {e}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                logger.error(f"Embed failed after {max_retries} attempts: {e}")
                raise
    raise RuntimeError(f"embed_with_retry exhausted {max_retries} retries")
