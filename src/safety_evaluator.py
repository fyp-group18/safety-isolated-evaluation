import logging
import re
import time
from collections import defaultdict

from src.flat_rag import retrieve, retrieve_safety
from src.schemas import SafetyExtractionResult, SafetyProtocol

logger = logging.getLogger(__name__)

_SAFETY_KEYWORDS = re.compile(
    r"\b(DANGER|WARNING|CAUTION|NOTICE|HAZARD|PPE|lockout|tagout|"
    r"high.voltage|electric.shock|burn|crush|pinch|entangle|toxic|"
    r"fire.hazard|explosion|asphyxiation)\b",
    re.IGNORECASE,
)

_SEVERITY_PATTERN = re.compile(r"\b(DANGER|WARNING|CAUTION|NOTICE)\b")

_HAZARD_TYPE_MAP = {
    "voltage": "electrical", "electric": "electrical", "shock": "electrical",
    "arc": "electrical", "energize": "electrical", "power": "electrical",
    "burn": "thermal", "hot": "thermal", "heat": "thermal", "cool": "thermal",
    "crush": "mechanical", "pinch": "mechanical", "entangle": "mechanical",
    "rotating": "mechanical", "spring": "mechanical", "weight": "mechanical",
    "toxic": "chemical", "chemical": "chemical", "solvent": "chemical",
    "fume": "chemical", "asbestos": "chemical",
    "pressure": "other", "hydraulic": "other", "pneumatic": "other",
    "PPE": "other", "lockout": "other", "tagout": "other",
    "hazard": "other", "explosion": "other", "fire": "other",
}


def _detect_hazard_type(text: str) -> str:
    text_lower = text.lower()
    for keyword, htype in _HAZARD_TYPE_MAP.items():
        if keyword.lower() in text_lower:
            return htype
    return "other"


def _extract_safety_from_text(text: str, chunk_id: str = "") -> list[SafetyProtocol]:
    """Extract safety protocols from text using regex pattern matching."""
    protocols = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue

        severity_match = _SEVERITY_PATTERN.search(line)
        if severity_match or _SAFETY_KEYWORDS.search(line):
            severity = severity_match.group(1) if severity_match else "CAUTION"
            hazard_type = _detect_hazard_type(line)

            clean_text = re.sub(r'\[CHUNK-ID: [^\]]+\]', '', line).strip()
            clean_text = re.sub(r'^(DANGER|WARNING|CAUTION|NOTICE)[:\s]*', '', clean_text).strip()

            if len(clean_text) > 10:
                protocols.append(SafetyProtocol(
                    severity=severity,
                    text=clean_text[:200],
                    hazard_type=hazard_type,
                    source_chunk_ids=[chunk_id] if chunk_id else [],
                ))

    return protocols


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
    """S1 — Full Isolation: dedicated safety retrieval + regex extraction."""
    t0 = time.perf_counter()

    # Path A: filter already-retrieved chunks for safety content
    path_a_protocols = []
    path_a_ids = set()
    for chunk in retrieved_chunks:
        if _SAFETY_KEYWORDS.search(chunk.text):
            path_a_protocols.extend(_extract_safety_from_text(chunk.text, chunk.chunk_id))
            path_a_ids.add(chunk.chunk_id)

    # Path B: dedicated safety retrieval
    path_b_protocols = []
    try:
        safety_results = retrieve_safety(equipment_type, k=10, collection=collection)
        for chunk in safety_results:
            if chunk.chunk_id not in path_a_ids:
                path_b_protocols.extend(_extract_safety_from_text(chunk.text, chunk.chunk_id))
    except Exception as e:
        logger.warning(f"S1 Path B retrieval failed: {e}")

    all_protocols = _deduplicate_protocols(path_a_protocols + path_b_protocols)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SafetyExtractionResult(
        protocols=all_protocols,
        extraction_confidence=0.9 if all_protocols else 1.0,
        no_safety_content_found=len(all_protocols) == 0,
        latency_ms=latency_ms,
        condition="S1",
    )


def evaluate_s2_inline(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> SafetyExtractionResult:
    """S2 — Inline Safety: extract from the same context (no separate retrieval)."""
    t0 = time.perf_counter()

    protocols = _extract_safety_from_text(context)
    protocols = _deduplicate_protocols(protocols)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SafetyExtractionResult(
        protocols=protocols,
        extraction_confidence=0.7 if protocols else 1.0,
        no_safety_content_found=len(protocols) == 0,
        latency_ms=latency_ms,
        condition="S2",
    )


def evaluate_s3_system_prompt(
    observation: str,
    context: str,
    equipment_type: str = "heavy_industrial",
) -> SafetyExtractionResult:
    """S3 — System Prompt Only: extract from context with stricter matching.

    Simulates weaker extraction by requiring explicit signal words
    (DANGER/WARNING/CAUTION/NOTICE) rather than the broader keyword set.
    """
    t0 = time.perf_counter()

    protocols = []
    lines = context.split("\n")
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        severity_match = _SEVERITY_PATTERN.search(line)
        if severity_match:
            clean_text = re.sub(r'\[CHUNK-ID: [^\]]+\]', '', line).strip()
            clean_text = re.sub(r'^(DANGER|WARNING|CAUTION|NOTICE)[:\s]*', '', clean_text).strip()
            if len(clean_text) > 10:
                protocols.append(SafetyProtocol(
                    severity=severity_match.group(1),
                    text=clean_text[:200],
                    hazard_type=_detect_hazard_type(line),
                    source_chunk_ids=[],
                ))

    protocols = _deduplicate_protocols(protocols)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SafetyExtractionResult(
        protocols=protocols,
        extraction_confidence=0.5 if protocols else 1.0,
        no_safety_content_found=len(protocols) == 0,
        latency_ms=latency_ms,
        condition="S3",
    )
