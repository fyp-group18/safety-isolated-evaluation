from dataclasses import dataclass, field
from enum import Enum


class SafetyCategory(str, Enum):
    LOTO = "LOTO"
    PPE = "PPE"
    HAZARD_WARNING = "HAZARD_WARNING"
    ELECTRICAL = "ELECTRICAL"
    CHEMICAL = "CHEMICAL"
    MECHANICAL = "MECHANICAL"
    THERMAL = "THERMAL"
    PRESSURE = "PRESSURE"
    GENERAL = "GENERAL"


class SafetySeverity(str, Enum):
    DANGER = "DANGER"
    WARNING = "WARNING"
    CAUTION = "CAUTION"
    NOTICE = "NOTICE"


class GroundingLabel(str, Enum):
    VERBATIM = "verbatim"
    PARAPHRASED = "paraphrased"
    SYNTHESIZED = "synthesized"
    UNGROUNDED = "ungrounded"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    cosine_score: float


@dataclass
class StepVerdict:
    step_id: str
    faithful: bool
    reason: str
    source_chunk_ids: list[str]
    grounding_label: str


@dataclass
class SyncGateResult:
    passed: bool
    faithfulness_score: float | None
    answer_relevance_score: float | None
    badge: str
    step_verdicts: list[StepVerdict]
    latency_ms: float
    reasoning: str


@dataclass
class AsyncEvalResult:
    context_relevance: float | None
    completeness: float | None
    safety_coverage: float | None
    latency_ms: float
    context_relevance_reason: str = ""
    completeness_reason: str = ""


@dataclass
class SafetyProtocol:
    severity: str
    text: str
    hazard_type: str
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class SafetyExtractionResult:
    protocols: list[SafetyProtocol]
    extraction_confidence: float
    no_safety_content_found: bool
    latency_ms: float
    condition: str  # S1, S2, S3


@dataclass
class EvalRecord:
    id: str
    observation: str
    ground_truth_fault_code: str
    manual_source: str
    section_system: str
    page_reference: str
    rationale: str
    difficulty_tier: str
    observation_type: str
    ground_truth_chunk_id: str
    origin: str
    ground_truth_chunk_ids: list[str] = field(default_factory=list)
