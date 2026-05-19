import json
import logging
import re
import time

from src.config import MODEL_FLASH
from src.flat_rag import retrieve, retrieve_safety
from src.llm import generate_with_retry
from src.schemas import SafetyExtractionResult, SafetyProtocol

logger = logging.getLogger(__name__)

_SAFETY_KEYWORDS = re.compile(
    r"\b(DANGER|WARNING|CAUTION|NOTICE|HAZARD|PPE|lockout|tagout|"
    r"high.voltage|electric.shock|burn|crush|pinch|entangle|toxic|"
    r"fire.hazard|explosion|asphyxiation)\b",
    re.IGNORECASE,
)

_S1_EXTRACTION_PROMPT = """\
Extract safety protocols from the following aircraft maintenance manual text.
Only extract EXPLICIT safety warnings — statements that begin with or are labeled as DANGER, WARNING, CAUTION, or NOTICE, plus PPE requirements and lockout/tagout procedures.

TEXT:
{text}

Return JSON:
{{
  "protocols": [
    {{
      "severity": "DANGER or WARNING or CAUTION or NOTICE",
      "text": "The complete safety instruction text",
      "hazard_type": "electrical or thermal or mechanical or chemical or other",
      "source_chunk_ids": ["{chunk_ids}"]
    }}
  ],
  "extraction_confidence": 0.0,
  "no_safety_content_found": false
}}

Rules:
- Only extract statements explicitly marked as DANGER, WARNING, CAUTION, or NOTICE
- Also extract explicit PPE requirements and lockout/tagout (LOTO) procedures
- Do NOT extract general maintenance instructions or procedures that are not safety-specific
- Maximum 10 protocols — prioritize by severity (DANGER > WARNING > CAUTION > NOTICE)
- severity must be one of: DANGER, WARNING, CAUTION, NOTICE
- hazard_type must be one of: electrical, thermal, mechanical, chemical, other
- Set no_safety_content_found to true if there are no explicit safety protocols"""

_S2_INLINE_PROMPT = """\
Given this aircraft maintenance observation and retrieved context, extract any safety protocols present in the context.

OBSERVATION: {observation}

CONTEXT:
{context}

Return JSON:
{{
  "protocols": [
    {{
      "severity": "DANGER or WARNING or CAUTION or NOTICE",
      "text": "The safety instruction",
      "hazard_type": "electrical or thermal or mechanical or chemical or other",
      "source_chunk_ids": []
    }}
  ],
  "extraction_confidence": 0.0,
  "no_safety_content_found": false
}}

Extract all safety warnings, cautions, PPE requirements, lockout/tagout procedures, and hazard notices from the context."""

_S3_SYSTEM_PROMPT = """\
You are an aircraft maintenance assistant. When responding to maintenance queries, you should note any relevant safety information from the provided context.

A technician reports: {observation}

Here is the relevant maintenance manual context:
{context}

Based on the context, what safety considerations should the technician be aware of? If there are safety warnings (DANGER, WARNING, CAUTION, NOTICE), PPE requirements, or hazard notices in the context, list them.

Return JSON:
{{
  "protocols": [
    {{
      "severity": "WARNING",
      "text": "example safety note",
      "hazard_type": "other",
      "source_chunk_ids": []
    }}
  ],
  "extraction_confidence": 0.0,
  "no_safety_content_found": false
}}"""


def _parse_safety_response(raw: str | None) -> tuple[list[SafetyProtocol], float, bool]:
    if not raw:
        return [], 0.0, True

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Safety JSON parse error: {raw[:200]}")
        return [], 0.0, True

    protocols = []
    for p in result.get("protocols", []):
        severity = p.get("severity", "CAUTION")
        if severity not in ("DANGER", "WARNING", "CAUTION", "NOTICE"):
            severity = "CAUTION"
        hazard_type = p.get("hazard_type", "other")
        if hazard_type not in ("electrical", "thermal", "mechanical", "chemical", "other"):
            hazard_type = "other"

        text = p.get("text", "").strip()
        if len(text) > 10:
            protocols.append(SafetyProtocol(
                severity=severity,
                text=text[:200],
                hazard_type=hazard_type,
                source_chunk_ids=p.get("source_chunk_ids", []),
            ))

    confidence = float(result.get("extraction_confidence", 0.5))
    no_safety = bool(result.get("no_safety_content_found", len(protocols) == 0))

    return protocols, confidence, no_safety


def _deduplicate_protocols(protocols: list[SafetyProtocol]) -> list[SafetyProtocol]:
    seen = set()
    unique = []
    for p in protocols:
        key = p.text[:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def evaluate_s1_isolated(
    observation: str,
    retrieved_chunks: list,
    equipment_type: str = "heavy_industrial",
    collection=None,
) -> SafetyExtractionResult:
    t0 = time.perf_counter()

    # Path A: filter already-retrieved chunks for safety content
    path_a_texts = []
    path_a_ids = set()
    for chunk in retrieved_chunks:
        if _SAFETY_KEYWORDS.search(chunk.text):
            path_a_texts.append(f"[CHUNK: {chunk.chunk_id}]\n{chunk.text}")
            path_a_ids.add(chunk.chunk_id)

    # Path B: dedicated safety retrieval
    path_b_texts = []
    try:
        safety_results = retrieve_safety(equipment_type, k=10, collection=collection)
        for chunk in safety_results:
            if chunk.chunk_id not in path_a_ids:
                path_b_texts.append(f"[CHUNK: {chunk.chunk_id}]\n{chunk.text}")
                path_a_ids.add(chunk.chunk_id)
    except Exception as e:
        logger.warning(f"S1 Path B retrieval failed: {e}")

    all_safety_text = "\n\n---\n\n".join(path_a_texts + path_b_texts)
    chunk_ids_str = ", ".join(path_a_ids)

    if not all_safety_text.strip():
        latency_ms = (time.perf_counter() - t0) * 1000
        return SafetyExtractionResult(
            protocols=[],
            extraction_confidence=1.0,
            no_safety_content_found=True,
            latency_ms=latency_ms,
            condition="S1",
        )

    prompt = _S1_EXTRACTION_PROMPT.format(text=all_safety_text, chunk_ids=chunk_ids_str)
    raw = generate_with_retry(
        prompt=prompt,
        model=MODEL_FLASH,
        temperature=0.0,
        response_mime_type="application/json",
    )

    protocols, confidence, no_safety = _parse_safety_response(raw)
    protocols = _deduplicate_protocols(protocols)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SafetyExtractionResult(
        protocols=protocols,
        extraction_confidence=confidence,
        no_safety_content_found=no_safety,
        latency_ms=latency_ms,
        condition="S1",
    )


def evaluate_s2_inline(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> SafetyExtractionResult:
    t0 = time.perf_counter()

    prompt = _S2_INLINE_PROMPT.format(observation=observation, context=context)
    raw = generate_with_retry(
        prompt=prompt,
        model=MODEL_FLASH,
        temperature=0.0,
        response_mime_type="application/json",
    )

    protocols, confidence, no_safety = _parse_safety_response(raw)
    protocols = _deduplicate_protocols(protocols)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SafetyExtractionResult(
        protocols=protocols,
        extraction_confidence=confidence,
        no_safety_content_found=no_safety,
        latency_ms=latency_ms,
        condition="S2",
    )


def evaluate_s3_system_prompt(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> SafetyExtractionResult:
    t0 = time.perf_counter()

    prompt = _S3_SYSTEM_PROMPT.format(observation=observation, context=context)
    raw = generate_with_retry(
        prompt=prompt,
        model=MODEL_FLASH,
        temperature=0.0,
        response_mime_type="application/json",
    )

    protocols, confidence, no_safety = _parse_safety_response(raw)
    protocols = _deduplicate_protocols(protocols)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SafetyExtractionResult(
        protocols=protocols,
        extraction_confidence=confidence,
        no_safety_content_found=no_safety,
        latency_ms=latency_ms,
        condition="S3",
    )
