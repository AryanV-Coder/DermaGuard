"""
DermaGuard Unified State & Data Contracts (Pydantic & TypedDict)
================================================================
Defines strict type-safe schemas for each agent's inputs and outputs,
and the shared AgentState passed across the LangGraph StateGraph.
"""

from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel, Field


# ─── Individual Agent Pydantic Models ──────────────────────────────────────────

class QualityAssessment(BaseModel):
    status: str = Field(..., description="'PASS', 'FAIL', or 'ERROR'")
    score: float = Field(..., ge=0.0, le=1.0, description="Overall weighted quality score")
    recommendation: str
    hard_fail_triggered: bool = False
    hard_fail_reason: Optional[str] = None
    failed_checks: List[str] = Field(default_factory=list)


class PatientContextInput(BaseModel):
    age: Optional[float] = Field(None, ge=0.0, le=125.0)
    sex: Optional[str] = Field(None, description="'male', 'female', or 'unknown'")
    localization: Optional[str] = Field(None, description="Anatomical site")


class PatientContextOutput(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_indicator: str = Field(..., description="'LOW', 'MODERATE', or 'HIGH'")
    missing_fields: List[str] = Field(default_factory=list)
    clinical_notes: List[str] = Field(default_factory=list)


class DiagnosisOutput(BaseModel):
    predicted_class: str = Field(..., description="7-class HAM10000 code")
    predicted_label: str = Field(..., description="Human-readable diagnosis name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    melanoma_flag: bool = False
    melanoma_prob: float = 0.0
    models_used: List[str] = Field(default_factory=list)


class LesionMorphologyOutput(BaseModel):
    asymmetry_index: float = Field(..., ge=0.0, le=1.0)
    border_irregularity: float = Field(..., ge=0.0, le=1.0)
    color_variation: float = Field(..., ge=0.0, le=1.0)
    lesion_area_fraction: float = Field(..., ge=0.0, le=1.0)
    equivalent_diameter_px: float = 0.0
    max_dimension_px: float = 0.0
    suspicion_score: float = Field(..., ge=0.0, le=1.0)
    suspicion_level: str = Field(..., description="'LOW', 'MODERATE', or 'HIGH'")
    reliability_flags: List[str] = Field(default_factory=list)
    segmentation_backend: str = "otsu_fallback"


class SafetyReviewOutput(BaseModel):
    veto: bool = False
    safety_flag: bool = False
    severity: str = Field("CLEAR", description="'CLEAR', 'WARN', or 'VETO'")
    triggers: List[str] = Field(default_factory=list)
    justification: str = ""


class CoordinatorDecisionOutput(BaseModel):
    image_id: str
    prediction: str
    prediction_label: str
    decision: str = Field(..., description="'AUTO_TRIAGE', 'REFER', or 'RE_IMAGE'")
    decision_rule: str
    confidence_tier: str = Field(..., description="'HIGH', 'MODERATE', or 'LOW'")
    rationale: str


# ─── LangGraph Shared State (TypedDict) ───────────────────────────────────────

class DermaGuardState(TypedDict, total=False):
    """
    Unified state schema passed through the LangGraph StateGraph.
    Every node reads needed inputs and updates its corresponding fields.
    """
    # ── Initial Inputs ──
    image_id: str
    image_path: str
    age: Optional[float]
    sex: Optional[str]
    localization: Optional[str]

    # ── Agent 1: Image Quality ──
    quality_status: str                     # "PASS" | "FAIL" | "ERROR"
    quality_score: float
    quality_recommendation: str
    quality_failed_checks: List[str]
    quality_report: Dict[str, Any]

    # ── Agent 4: Patient Context ──
    patient_risk_score: float
    patient_risk_indicator: str            # "LOW" | "MODERATE" | "HIGH"
    patient_missing_fields: List[str]
    patient_context_evidence: Dict[str, Any]

    # ── Agent 2: Diagnosis ──
    predicted_class: str
    predicted_label: str
    diagnosis_confidence: float
    class_probabilities: Dict[str, float]
    melanoma_flag: bool
    melanoma_prob: float
    diagnosis_models: List[str]

    # ── Agent 3: Lesion Analysis (ABCD) ──
    asymmetry_index: float
    border_irregularity: float
    color_variation: float
    lesion_area_fraction: float
    equivalent_diameter_px: float
    max_dimension_px: float
    morphology_suspicion_score: float
    morphology_suspicion_level: str        # "LOW" | "MODERATE" | "HIGH"
    morphology_flags: List[str]
    segmentation_backend: str

    # ── Agent 5: Safety Review ──
    safety_veto: bool
    safety_flag: bool
    safety_severity: str                   # "CLEAR" | "WARN" | "VETO"
    safety_triggers: List[str]
    safety_justification: str

    # ── Agent 6: Coordinator Decision ──
    final_decision: str                    # "AUTO_TRIAGE" | "REFER" | "RE_IMAGE"
    decision_rule: str
    confidence_tier: str                   # "HIGH" | "MODERATE" | "LOW"
    rationale: str
    evidence_summary: Dict[str, Any]

    # ── Explanation Layer: Gemini API ──
    clinical_explanation: str
    llm_error: Optional[str]
