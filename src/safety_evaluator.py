import json
import logging
import re
import time

from src.config import MODEL_FLASH
from src.flat_rag import retrieve, retrieve_safety
from src.generator import generate_with_retry
from src.schemas import SafetyExtractionResult, SafetyProtocol

logger = logging.getLogger(__name__)

_SAFETY_KEYWORDS = re.compile(
    r"\b(DANGER|WARNING|CAUTION|NOTICE|HAZARD|PPE|lockout|tagout|"
    r"high.voltage|electric.shock|burn|crush|pinch|entangle|toxic|"
    r"fire.hazard|explosion|asphyxiation)\b",
    re.IGNORECASE,
)

SAFETY_EXTRACTION_PROMPT = """You are a safety protocol extractor for industrial equipment maintenance.

Given the following documentation chunks that may contain safety information, extract ALL safety warnings, hazards, and required precautions.

Equipment type: {equipment_type}
Context: Technician is performing maintenance/troubleshooting

<DOCUMENTATION>
{chunks_text}
</DOCUMENTATION>

Extract safety protocols in this JSON format:
{{
  "protocols": [
    {{
      "severity": "DANGER|WARNING|CAUTION|NOTICE",
      "text": "Clear, actionable safety instruction",
      "hazard_type": "electrical|mechanical|chemical|thermal|other",
      "source_chunk_ids": ["chunk-id-1"]
    }}
  ],
  "extraction_confidence": 0.0-1.0,
  "no_safety_content_found": true/false
}}

If no safety content is found in the documentation, return:
{{"protocols": [], "extraction_confidence": 1.0, "no_safety_content_found": true}}

Rules:
- Extract ONLY safety protocols explicitly stated in the documentation
- Do NOT invent or hallucinate safety warnings
- Include source_chunk_ids for traceability
- Severity must match the signal word used in the source (DANGER > WARNING > CAUTION > NOTICE)
"""

INLINE_SAFETY_PROMPT = """Based on the following documentation, provide a repair plan for the observation below.
Additionally, extract ALL PPE, LOTO, and hazard requirements. Output as JSON.

<Observation>
{observation}
</Observation>

<Documentation>
{context}
</Documentation>

Return JSON:
{{
  "repair_steps": [{{"step_id": "s-0", "text": "..."}}],
  "protocols": [
    {{
      "severity": "DANGER|WARNING|CAUTION|NOTICE",
      "text": "safety instruction",
      "hazard_type": "electrical|mechanical|chemical|thermal|other",
      "source_chunk_ids": []
    }}
  ],
  "extraction_confidence": 0.0-1.0,
  "no_safety_content_found": true/false
}}
"""

SYSTEM_PROMPT_SAFETY = """You are a safety-aware repair planning assistant. When generating repair plans, you MUST also extract all safety requirements from the context.

SAFETY EXTRACTION REQUIREMENTS:
For any maintenance procedure, identify and extract:
- PPE requirements (gloves, eye protection, hearing protection, arc-flash gear)
- LOTO (Lockout/Tagout) procedures or energy isolation steps
- WARNING/CAUTION/DANGER/NOTICE signal-word messages
- Electrical, mechanical, chemical, thermal, and pressure hazards
- Weight/lifting warnings and qualified-personnel requirements

EXAMPLE 1:
Context: "WARNING: Disconnect battery before servicing. Wear safety glasses."
Extract: [{{"severity": "WARNING", "text": "Disconnect battery before servicing", "hazard_type": "electrical"}},
          {{"severity": "CAUTION", "text": "Wear safety glasses", "hazard_type": "mechanical"}}]

EXAMPLE 2:
Context: "DANGER: High voltage present. Lockout all power sources before opening panel."
Extract: [{{"severity": "DANGER", "text": "High voltage present - lockout all power sources before opening panel", "hazard_type": "electrical"}}]

EXAMPLE 3:
Context: "CAUTION: Hot surfaces. Allow engine to cool before removing oil filter."
Extract: [{{"severity": "CAUTION", "text": "Allow engine to cool before removing oil filter - hot surfaces", "hazard_type": "thermal"}}]
"""

SYSTEM_PROMPT_SAFETY_FULL = SYSTEM_PROMPT_SAFETY + """

Now process this maintenance case:

<Observation>
{observation}
</Observation>

<Documentation>
{context}
</Documentation>

Return JSON:
{{
  "repair_steps": [{{"step_id": "s-0", "text": "..."}}],
  "protocols": [
    {{
      "severity": "DANGER|WARNING|CAUTION|NOTICE",
      "text": "safety instruction",
      "hazard_type": "electrical|mechanical|chemical|thermal|other",
      "source_chunk_ids": []
    }}
  ],
  "extraction_confidence": 0.0-1.0,
  "no_safety_content_found": true/false
}}
"""


def _parse_protocols(data: dict) -> list[SafetyProtocol]:
    protocols = []
    for p in data.get("protocols", []):
        protocols.append(SafetyProtocol(
            severity=p.get("severity", "WARNING"),
            text=p.get("text", ""),
            hazard_type=p.get("hazard_type", "other"),
            source_chunk_ids=p.get("source_chunk_ids", []),
        ))
    return protocols


def evaluate_s1_isolated(
    observation: str,
    retrieved_chunks: list,
    equipment_type: str = "heavy_industrial",
    collection=None,
) -> SafetyExtractionResult:
    """S1 — Full Isolation: dedicated safety retrieval + separate LLM extraction."""
    t0 = time.perf_counter()

    # Path A: filter already-retrieved chunks for safety content
    path_a_chunks = []
    path_a_ids = set()
    for chunk in retrieved_chunks:
        if _SAFETY_KEYWORDS.search(chunk.text):
            path_a_chunks.append(chunk)
            path_a_ids.add(chunk.chunk_id)

    # Path B: dedicated safety retrieval
    path_b_chunks = []
    try:
        safety_results = retrieve_safety(equipment_type, k=10, collection=collection)
        for chunk in safety_results:
            if chunk.chunk_id not in path_a_ids:
                path_b_chunks.append(chunk)
    except Exception as e:
        logger.warning(f"S1 Path B retrieval failed: {e}")

    all_safety_chunks = path_a_chunks + path_b_chunks

    if not all_safety_chunks:
        return SafetyExtractionResult(
            protocols=[],
            extraction_confidence=1.0,
            no_safety_content_found=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            condition="S1",
        )

    chunks_text_parts = []
    for chunk in all_safety_chunks[:15]:
        chunks_text_parts.append(f"[CHUNK-ID: {chunk.chunk_id}]\n{chunk.text}")
    chunks_combined = "\n\n---\n\n".join(chunks_text_parts)

    prompt = SAFETY_EXTRACTION_PROMPT.format(
        equipment_type=equipment_type,
        chunks_text=chunks_combined,
    )

    result_text = generate_with_retry(prompt, model=MODEL_FLASH)
    latency_ms = (time.perf_counter() - t0) * 1000

    if not result_text:
        return SafetyExtractionResult(
            protocols=[], extraction_confidence=0.0,
            no_safety_content_found=False, latency_ms=latency_ms, condition="S1",
        )

    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return SafetyExtractionResult(
            protocols=[], extraction_confidence=0.0,
            no_safety_content_found=False, latency_ms=latency_ms, condition="S1",
        )

    return SafetyExtractionResult(
        protocols=_parse_protocols(data),
        extraction_confidence=data.get("extraction_confidence", 0.0),
        no_safety_content_found=data.get("no_safety_content_found", False),
        latency_ms=latency_ms,
        condition="S1",
    )


def evaluate_s2_inline(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> SafetyExtractionResult:
    """S2 — Inline Safety: same retriever, same LLM, appended extraction instruction."""
    t0 = time.perf_counter()

    prompt = INLINE_SAFETY_PROMPT.format(
        observation=observation,
        context=context,
    )

    result_text = generate_with_retry(prompt, model=MODEL_FLASH)
    latency_ms = (time.perf_counter() - t0) * 1000

    if not result_text:
        return SafetyExtractionResult(
            protocols=[], extraction_confidence=0.0,
            no_safety_content_found=False, latency_ms=latency_ms, condition="S2",
        )

    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return SafetyExtractionResult(
            protocols=[], extraction_confidence=0.0,
            no_safety_content_found=False, latency_ms=latency_ms, condition="S2",
        )

    return SafetyExtractionResult(
        protocols=_parse_protocols(data),
        extraction_confidence=data.get("extraction_confidence", 0.0),
        no_safety_content_found=data.get("no_safety_content_found", False),
        latency_ms=latency_ms,
        condition="S2",
    )


def evaluate_s3_system_prompt(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> SafetyExtractionResult:
    """S3 — System Prompt Only: same retriever, same LLM, system prompt with few-shot examples."""
    t0 = time.perf_counter()

    prompt = SYSTEM_PROMPT_SAFETY_FULL.format(
        observation=observation,
        context=context,
    )

    result_text = generate_with_retry(prompt, model=MODEL_FLASH)
    latency_ms = (time.perf_counter() - t0) * 1000

    if not result_text:
        return SafetyExtractionResult(
            protocols=[], extraction_confidence=0.0,
            no_safety_content_found=False, latency_ms=latency_ms, condition="S3",
        )

    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return SafetyExtractionResult(
            protocols=[], extraction_confidence=0.0,
            no_safety_content_found=False, latency_ms=latency_ms, condition="S3",
        )

    return SafetyExtractionResult(
        protocols=_parse_protocols(data),
        extraction_confidence=data.get("extraction_confidence", 0.0),
        no_safety_content_found=data.get("no_safety_content_found", False),
        latency_ms=latency_ms,
        condition="S3",
    )
