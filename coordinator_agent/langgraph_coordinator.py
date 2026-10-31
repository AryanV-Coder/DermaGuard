"""
DermaGuard LangGraph Multi-Agent Orchestrator
==============================================
Implements the 6-agent collaborative StateGraph workflow:
  Node 1: Quality Supervisor (Agent 1)
  Node 2: Reasoning / Context Agent (Agent 4)
  Node 3: Diagnosis Agent (Agent 2)
  Node 4: Lesion Analysis Agent with U-Net + ABCD (Agent 3)
  Node 5: Safety Agent & Veto Engine (Agent 5)
  Node 6: Coordinator Agent (Agent 6)
  Node 7: Gemini Clinical Explainer (LLM)

Features:
  - Parallel fan-out for Quality & Context
  - Conditional short-circuit routing on Image Quality failure
  - Parallel fan-out for Diagnosis & Lesion Analysis on Quality PASS
  - Immutable typed state updates (DermaGuardState)
  - Post-coordinator LLM explanation generation
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from langgraph.graph import StateGraph, START, END

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "image_quality_agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "image_quality_agent"))

from schema.dermaguard_state import DermaGuardState
from image_quality_agent.supervisor.quality_supervisor import QualitySupervisor
from lesion_analysis_agent.lesion_analysis_agent import LesionAnalysisAgent
from reasoning_context_agent.reasoning_context_agent import ReasoningContextAgent
from safety_agent.safety_agent import SafetyAgent
from coordinator_agent.coordinator_agent import CoordinatorAgent
from explanation.gemini_explainer import GeminiExplainer

# Optional DiagnosisAgent import (handles missing torch gracefully)
try:
    from diagnosis_agent.diagnosis_agent import DiagnosisAgent
    DIAGNOSIS_AVAILABLE = True
except Exception:
    DIAGNOSIS_AVAILABLE = False


class DermaGuardGraph:
    """
    StateGraph wrapper for the complete multi-agent system.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        unet_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        print("[DermaGuardGraph] Initializing 6 specialized agents and explainer...")
        self.a1_quality = QualitySupervisor()
        self.a4_context = ReasoningContextAgent()
        self.a3_lesion = LesionAnalysisAgent(model_path=unet_path)
        self.a5_safety = SafetyAgent()
        self.a6_coordinator = CoordinatorAgent()
        self.explainer = GeminiExplainer(api_key=gemini_api_key)

        self.a2_diagnosis = None
        if DIAGNOSIS_AVAILABLE and model_path and Path(model_path).exists():
            try:
                self.a2_diagnosis = DiagnosisAgent(model_path=model_path)
                print(f"  [Agent 2] Diagnosis model loaded from: {model_path}")
            except Exception as e:
                print(f"  [Agent 2] Could not load diagnosis model: {e}. Fallback active.")

        self.graph = self._build_graph()
        self.app = self.graph.compile()
        print("[DermaGuardGraph] LangGraph StateGraph compiled successfully.\n")

    # ── Node Implementations ──────────────────────────────────────────────────

    def node_quality(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 1: Evaluates technical image quality (blur, lighting, glare, obstruction)."""
        img_path = state.get("image_path")
        report = self.a1_quality.evaluate(img_path)
        overall = report.get("overall", {})

        return {
            "quality_status": overall.get("status", "ERROR"),
            "quality_score": round(float(overall.get("score", 0.0)), 4),
            "quality_recommendation": overall.get("recommendation", ""),
            "quality_failed_checks": [
                k for k, v in report.get("agent_statuses", {}).items() if v == "FAIL"
            ],
            "quality_report": report,
        }

    def node_context(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 2: Evaluates patient demographic priors (age, sex, anatomical location)."""
        c_report = self.a4_context.evaluate(
            age=state.get("age"),
            sex=state.get("sex"),
            localization=state.get("localization"),
        )
        return {
            "patient_risk_score": round(float(c_report.get("risk_score", 0.0)), 4),
            "patient_risk_indicator": c_report.get("risk_indicator", "LOW"),
            "patient_missing_fields": c_report.get("missing_field_flags", []),
            "patient_context_evidence": c_report,
        }

    def node_diagnosis(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 3: 7-class lesion diagnostic prediction."""
        img_path = state.get("image_path")
        if self.a2_diagnosis is not None:
            res = self.a2_diagnosis.predict(img_path)
            pred_class = res.get("predicted_class") or "nv"
            pred_label = res.get("predicted_label") or "Melanocytic Nevi"
            conf = res.get("confidence", 0.0)
            probs = res.get("probabilities", {})
            mel_prob = float(probs.get("mel", 0.0))
            models = ["EfficientNet-B0"]
        else:
            # Fallback if diagnosis checkpoint is not present in local runtime
            pred_class = "nv"
            pred_label = "Melanocytic Nevi (Baseline)"
            conf = 0.50
            probs = {"nv": 0.50, "mel": 0.10, "bkl": 0.10, "bcc": 0.10, "akiec": 0.10, "vasc": 0.05, "df": 0.05}
            mel_prob = 0.10
            models = ["Heuristic Baseline"]

        mel_flag = (pred_class == "mel") or (mel_prob > 0.30)
        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "diagnosis_confidence": round(float(conf), 4),
            "class_probabilities": probs,
            "melanoma_flag": mel_flag,
            "melanoma_prob": round(mel_prob, 4),
            "diagnosis_models": models,
        }

    def node_lesion_analysis(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 4: U-Net Lesion Segmentation + ABCD Morphometrics."""
        img_path = state.get("image_path")
        m_report = self.a3_lesion.analyze(img_path)
        fe = m_report.get("feature_evidence", {})

        return {
            "asymmetry_index": fe.get("asymmetry_index", 0.0),
            "border_irregularity": fe.get("border_irregularity", 0.0),
            "color_variation": fe.get("color_variation", 0.0),
            "lesion_area_fraction": fe.get("lesion_area_fraction", 0.0),
            "equivalent_diameter_px": fe.get("equivalent_diameter_px", 0.0),
            "max_dimension_px": fe.get("max_dimension_px", 0.0),
            "morphology_suspicion_score": m_report.get("suspicion_score", 0.0),
            "morphology_suspicion_level": m_report.get("suspicion_level", "LOW"),
            "morphology_flags": m_report.get("reliability_flags", []),
            "segmentation_backend": fe.get("segmentation_backend", "otsu_fallback"),
        }

    def node_safety(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 5: Multi-agent conflict checking, vetoes, and clinical safety flags."""
        a1_report = state.get("quality_report", {"overall": {"status": state.get("quality_status", "PASS"), "score": state.get("quality_score", 1.0)}})
        a2_result = {
            "predicted_class": state.get("predicted_class", "nv"),
            "confidence": state.get("diagnosis_confidence", 0.50),
            "probabilities": state.get("class_probabilities", {}),
        }
        a3_result = {
            "suspicion_score": state.get("morphology_suspicion_score", 0.0),
            "suspicion_level": state.get("morphology_suspicion_level", "LOW"),
            "reliability_flags": state.get("morphology_flags", []),
            "segmentation_success": True,
        }
        a4_result = state.get("patient_context_evidence", {
            "risk_score": state.get("patient_risk_score", 0.0),
            "risk_indicator": state.get("patient_risk_indicator", "LOW"),
            "missing_field_flags": state.get("patient_missing_fields", []),
        })

        s_report = self.a5_safety.review(a1_report, a2_result, a3_result, a4_result)

        return {
            "safety_veto": s_report.get("veto", False),
            "safety_flag": s_report.get("safety_flag", False),
            "safety_severity": s_report.get("severity", "CLEAR"),
            "safety_triggers": s_report.get("triggers", []),
            "safety_justification": s_report.get("justification", ""),
        }

    def node_coordinator(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 6: Final disposition arbitration (AUTO_TRIAGE, REFER, or RE_IMAGE)."""
        image_id = state.get("image_id", "IMAGE_001")
        q_status = state.get("quality_status", "PASS")

        # Reconstruct standard agent reports
        a1_report = state.get("quality_report", {"overall": {"status": q_status, "score": state.get("quality_score", 1.0)}})
        a2_result = {
            "predicted_class": state.get("predicted_class", "nv"),
            "confidence": state.get("diagnosis_confidence", 0.0),
            "probabilities": state.get("class_probabilities", {}),
        }
        a3_result = {
            "suspicion_score": state.get("morphology_suspicion_score", 0.0),
            "suspicion_level": state.get("morphology_suspicion_level", "LOW"),
            "reliability_flags": state.get("morphology_flags", []),
        }
        a4_result = {
            "risk_score": state.get("patient_risk_score", 0.0),
            "risk_indicator": state.get("patient_risk_indicator", "LOW"),
            "missing_field_flags": state.get("patient_missing_fields", []),
        }
        a5_result = {
            "veto": state.get("safety_veto", False),
            "severity": state.get("safety_severity", "CLEAR"),
            "triggers": state.get("safety_triggers", []),
            "justification": state.get("safety_justification", ""),
        }

        c_out = self.a6_coordinator.decide(image_id, a1_report, a2_result, a3_result, a4_result, a5_result)

        return {
            "final_decision": c_out["decision"],
            "decision_rule": c_out["decision_rule"],
            "confidence_tier": c_out["confidence_tier"],
            "rationale": c_out["rationale"],
            "evidence_summary": c_out["evidence_summary"],
        }

    def node_gemini_explainer(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 7: Post-decision Gemini Clinical Explanation."""
        findings_payload = {
            "image_id": state.get("image_id"),
            "final_decision": state.get("final_decision"),
            "decision_rule": state.get("decision_rule"),
            "confidence_tier": state.get("confidence_tier"),
            "image_quality": {
                "status": state.get("quality_status"),
                "score": state.get("quality_score"),
                "failed_checks": state.get("quality_failed_checks", []),
            },
            "diagnosis": {
                "predicted_class": state.get("predicted_class"),
                "predicted_label": state.get("predicted_label"),
                "confidence": state.get("diagnosis_confidence"),
                "melanoma_flag": state.get("melanoma_flag"),
                "melanoma_prob": state.get("melanoma_prob"),
            },
            "lesion_analysis": {
                "asymmetry_index": state.get("asymmetry_index"),
                "border_irregularity": state.get("border_irregularity"),
                "color_variation": state.get("color_variation"),
                "suspicion_level": state.get("morphology_suspicion_level"),
                "suspicion_score": state.get("morphology_suspicion_score"),
                "segmentation_backend": state.get("segmentation_backend"),
            },
            "patient_context": {
                "age": state.get("age"),
                "sex": state.get("sex"),
                "localization": state.get("localization"),
                "risk_indicator": state.get("patient_risk_indicator"),
            },
            "safety": {
                "severity": state.get("safety_severity"),
                "veto": state.get("safety_veto"),
                "triggers": state.get("safety_triggers", []),
            },
        }

        explanation = self.explainer.explain(findings_payload)
        return {
            "clinical_explanation": explanation
        }

    # ── Graph Construction ────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(DermaGuardState)

        # 1. Register Nodes
        workflow.add_node("quality_agent", self.node_quality)
        workflow.add_node("context_agent", self.node_context)
        workflow.add_node("diagnosis_agent", self.node_diagnosis)
        workflow.add_node("lesion_agent", self.node_lesion_analysis)
        workflow.add_node("safety_agent", self.node_safety)
        workflow.add_node("coordinator", self.node_coordinator)
        workflow.add_node("gemini_explainer", self.node_gemini_explainer)

        # 2. Parallel fan-out from START to Quality & Context
        workflow.add_edge(START, "quality_agent")
        workflow.add_edge(START, "context_agent")

        # 3. Conditional Quality Router
        def quality_router(state: DermaGuardState) -> str:
            if state.get("quality_status") in ("FAIL", "ERROR"):
                return "coordinator"  # Short-circuit to RE_IMAGE
            return "diagnosis_agent"  # Proceed to diagnosis & visual analysis

        workflow.add_conditional_edges(
            "quality_agent",
            quality_router,
            {
                "coordinator": "coordinator",
                "diagnosis_agent": "diagnosis_agent",
            }
        )

        # When quality PASSes, Diagnosis and Lesion analysis run
        workflow.add_edge("diagnosis_agent", "lesion_agent")
        workflow.add_edge("lesion_agent", "safety_agent")
        workflow.add_edge("context_agent", "safety_agent")

        # Safety reviews both diagnosis + lesion analysis + context
        workflow.add_edge("safety_agent", "coordinator")

        # Coordinator triggers Gemini Explainer
        workflow.add_edge("coordinator", "gemini_explainer")
        workflow.add_edge("gemini_explainer", END)

        return workflow

    def run(
        self,
        image_path: Union[str, Path],
        image_id: str = "CASE_001",
        age: Optional[float] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
    ) -> DermaGuardState:
        """
        Executes the compiled multi-agent StateGraph on a patient case.
        """
        initial_state: DermaGuardState = {
            "image_id": image_id,
            "image_path": str(image_path),
            "age": age,
            "sex": sex,
            "localization": localization,
        }
        return self.app.invoke(initial_state)
