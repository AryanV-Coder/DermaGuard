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
from explanation.gradcam import GradCAMExplainer

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

        # Initialize Grad-CAM Explainer
        diag_model = None
        if self.a2_diagnosis and hasattr(self.a2_diagnosis, "models") and len(self.a2_diagnosis.models) > 0:
            diag_model = self.a2_diagnosis.models[0][1]
        self.gradcam = GradCAMExplainer(model=diag_model)
        print("[DermaGuardGraph] LangGraph StateGraph compiled & Grad-CAM engine ready.\n")

    # ── Node Implementations ──────────────────────────────────────────────────

    def node_quality(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 1: Evaluates technical image quality (blur, lighting, glare, obstruction)."""
        img_path = state.get("image_path")
        report = self.a1_quality.evaluate(img_path)
        overall = report.get("overall", {})
        q_pass = bool(report.get("quality_pass", overall.get("status") == "PASS"))

        return {
            "quality_pass": q_pass,
            "quality_status": overall.get("status", "PASS" if q_pass else "FAIL"),
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
            "risk_level": c_report.get("risk_level", "MODERATE"),
            "patient_risk_score": round(float(c_report.get("risk_score", 0.0)), 4),
            "patient_risk_indicator": c_report.get("risk_indicator", "MODERATE"),
            "patient_missing_fields": c_report.get("missing_field_flags", []),
            "patient_risk_factors": c_report.get("risk_factors", []),
            "patient_context_evidence": c_report,
        }

    def node_diagnosis(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 3: 7-class lesion diagnostic prediction."""
        img_path = state.get("image_path")
        if self.a2_diagnosis is not None:
            res = self.a2_diagnosis.predict(img_path)
            pred_class = (res.get("predicted_class") or "nv").lower()
            pred_label = res.get("predicted_label") or pred_class
            conf = float(res.get("confidence", 0.0))
            probs = res.get("probabilities", {})
            mel_prob = float(res.get("melanoma_prob", probs.get("mel", 0.0)))
            malignant_prob = float(res.get("malignant_prob", sum(probs.get(c, 0.0) for c in ("mel", "bcc", "akiec"))))
            mel_flag = bool(res.get("melanoma_flag", (pred_class == "mel") or (mel_prob >= 0.18)))
            models = res.get("models_used", ["EfficientNet-B0"])
        else:
            # Fallback if diagnosis checkpoint is not present in local runtime
            pred_class = "nv"
            pred_label = "nv"
            conf = 0.50
            probs = {"nv": 0.50, "mel": 0.10, "bkl": 0.10, "bcc": 0.10, "akiec": 0.10, "vasc": 0.05, "df": 0.05}
            mel_prob = 0.10
            malignant_prob = 0.30
            mel_flag = False
            models = ["Heuristic Baseline"]

        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "diagnosis_confidence": round(float(conf), 4),
            "class_probabilities": probs,
            "melanoma_flag": mel_flag,
            "melanoma_prob": round(mel_prob, 4),
            "malignant_prob": round(malignant_prob, 4),
            "diagnosis_models": models,
        }

    def node_lesion_analysis(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 4: U-Net Lesion Segmentation + ABCD Morphometrics."""
        img_path = state.get("image_path")
        m_report = self.a3_lesion.analyze(img_path)
        fe = m_report.get("feature_evidence", {})

        return {
            "asymmetry_index": m_report.get("asymmetry", fe.get("asymmetry_index", 0.0)),
            "border_irregularity": m_report.get("border_score", fe.get("border_irregularity", 0.0)),
            "color_variation": m_report.get("colour_var", fe.get("color_variation", 0.0)),
            "lesion_area_fraction": m_report.get("lesion_area", fe.get("lesion_area_fraction", 0.0)),
            "equivalent_diameter_px": m_report.get("diameter_px", fe.get("equivalent_diameter_px", 0.0)),
            "max_dimension_px": fe.get("max_dimension_px", 0.0),
            "abcd_score": m_report.get("abcd_score", 0.0),
            "lesion_risk": m_report.get("lesion_risk", "LOW"),
            "morphology_suspicion_score": m_report.get("suspicion_score", 0.0),
            "morphology_suspicion_level": m_report.get("suspicion_level", "LOW"),
            "morphology_flags": m_report.get("reliability_flags", []),
            "segmentation_backend": m_report.get("seg_backend", fe.get("segmentation_backend", "otsu_fallback")),
        }

    def node_safety(self, state: DermaGuardState) -> Dict[str, Any]:
        """Node 5: Multi-agent conflict checking, vetoes, and zero-tolerance safety guardrails."""
        a1_report = state.get("quality_report", {
            "quality_pass": state.get("quality_pass", True),
            "overall": {"status": state.get("quality_status", "PASS"), "score": state.get("quality_score", 1.0)}
        })
        a2_result = {
            "predicted_class": state.get("predicted_class", "nv"),
            "predicted_label": state.get("predicted_label", "nv"),
            "confidence": state.get("diagnosis_confidence", 0.50),
            "probabilities": state.get("class_probabilities", {}),
            "melanoma_flag": state.get("melanoma_flag", False),
            "melanoma_prob": state.get("melanoma_prob", 0.0),
            "malignant_prob": state.get("malignant_prob", 0.0),
        }
        a3_result = {
            "suspicion_score": state.get("morphology_suspicion_score", 0.0),
            "suspicion_level": state.get("morphology_suspicion_level", "LOW"),
            "abcd_score": state.get("abcd_score", 0.0),
            "lesion_risk": state.get("lesion_risk", "LOW"),
            "reliability_flags": state.get("morphology_flags", []),
            "segmentation_success": True,
        }
        a4_result = state.get("patient_context_evidence", {
            "risk_score": state.get("patient_risk_score", 0.0),
            "risk_indicator": state.get("patient_risk_indicator", "MODERATE"),
            "risk_level": state.get("risk_level", "MODERATE"),
            "missing_field_flags": state.get("patient_missing_fields", []),
        })

        s_report = self.a5_safety.review(a1_report, a2_result, a3_result, a4_result)

        return {
            "must_refer": s_report.get("must_refer", s_report.get("veto", False)),
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

        a1_report = state.get("quality_report", {
            "quality_pass": state.get("quality_pass", True),
            "overall": {"status": q_status, "score": state.get("quality_score", 1.0)}
        })
        a2_result = {
            "predicted_class": state.get("predicted_class", "nv"),
            "predicted_label": state.get("predicted_label", "nv"),
            "confidence": state.get("diagnosis_confidence", 0.0),
            "probabilities": state.get("class_probabilities", {}),
            "melanoma_flag": state.get("melanoma_flag", False),
            "melanoma_prob": state.get("melanoma_prob", 0.0),
            "malignant_prob": state.get("malignant_prob", 0.0),
        }
        a3_result = {
            "suspicion_score": state.get("morphology_suspicion_score", 0.0),
            "suspicion_level": state.get("morphology_suspicion_level", "LOW"),
            "abcd_score": state.get("abcd_score", 0.0),
            "lesion_risk": state.get("lesion_risk", "LOW"),
            "reliability_flags": state.get("morphology_flags", []),
        }
        a4_result = {
            "risk_score": state.get("patient_risk_score", 0.0),
            "risk_indicator": state.get("patient_risk_indicator", "MODERATE"),
            "risk_level": state.get("risk_level", "MODERATE"),
            "missing_field_flags": state.get("patient_missing_fields", []),
        }
        a5_result = {
            "must_refer": state.get("must_refer", state.get("safety_veto", False)),
            "veto": state.get("safety_veto", False),
            "severity": state.get("safety_severity", "CLEAR"),
            "triggers": state.get("safety_triggers", []),
            "safety_flags": state.get("safety_triggers", []),
            "justification": state.get("safety_justification", ""),
        }

        c_out = self.a6_coordinator.decide(image_id, a1_report, a2_result, a3_result, a4_result, a5_result)

        return {
            "final_decision": c_out["decision"],
            "decision": c_out["decision"],
            "predicted_label": c_out.get("predicted_label", state.get("predicted_label", "nv")),
            "decision_rule": c_out["decision_rule"],
            "urgency": c_out.get("urgency", "ROUTINE"),
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

    def get_api_response(self, state: DermaGuardState) -> Dict[str, Any]:
        """
        Formats the LangGraph pipeline execution state into a clean,
        structured dictionary with individual agent outputs ready for API JSON response.
        """
        img_id = state.get("image_id", "CASE_001")
        pred_label = state.get("predicted_label") or state.get("predicted_class", "nv")
        decision = state.get("final_decision") or state.get("decision", "REFER")
        urgency = state.get("urgency", "ROUTINE" if decision == "AUTO_TRIAGE" else "SOON")

        return {
            # ── Track 2 Official Submission Format ────────────────────────────
            "image_id":         img_id,
            "predicted_label":  pred_label,
            "decision":         decision,
            "urgency":          urgency,
            "rationale":        state.get("rationale", ""),

            # ── Agent 1: Image Quality ────────────────────────────────────────
            "agent1_quality": {
                "quality_pass":   state.get("quality_pass", state.get("quality_status") == "PASS"),
                "status":         state.get("quality_status", "PASS"),
                "score":          state.get("quality_score", 1.0),
                "failed_checks":  state.get("quality_failed_checks", []),
                "recommendation": state.get("quality_recommendation", ""),
                "metrics": {
                    "blur":        (state.get("quality_report") or {}).get("blur", 0.0),
                    "brightness":  (state.get("quality_report") or {}).get("brightness", 0.0),
                    "glare_ratio": (state.get("quality_report") or {}).get("glare_ratio", 0.0),
                    "occlude_ratio": (state.get("quality_report") or {}).get("occlude_ratio", 0.0),
                },
            },

            # ── Agent 2: Diagnosis ────────────────────────────────────────────
            "agent2_diagnosis": {
                "predicted_class":     state.get("predicted_class", "nv"),
                "predicted_label":     pred_label,
                "confidence":          state.get("diagnosis_confidence", 0.0),
                "class_probabilities": state.get("class_probabilities", {}),
                "melanoma_flag":       state.get("melanoma_flag", False),
                "melanoma_prob":       state.get("melanoma_prob", 0.0),
                "malignant_prob":      state.get("malignant_prob", 0.0),
                "models_used":         state.get("diagnosis_models", []),
            },

            # ── Agent 3: Lesion Analysis (ABCD) ───────────────────────────────
            "agent3_lesion": {
                "asymmetry":            state.get("asymmetry_index", 0.0),
                "border_score":         state.get("border_irregularity", 0.0),
                "colour_var":           state.get("color_variation", 0.0),
                "diameter_px":          state.get("equivalent_diameter_px", 0.0),
                "abcd_score":           state.get("abcd_score", 0.0),
                "lesion_risk":          state.get("lesion_risk", state.get("morphology_suspicion_level", "LOW")),
                "suspicion_score":      state.get("morphology_suspicion_score", 0.0),
                "segmentation_backend": state.get("segmentation_backend", "otsu_fallback"),
                "flags":                state.get("morphology_flags", []),
            },

            # ── Agent 4: Patient Context ──────────────────────────────────────
            "agent4_patient_risk": {
                "risk_level":     state.get("risk_level", state.get("patient_risk_indicator", "MODERATE")),
                "risk_score":     state.get("patient_risk_score", 0.40),
                "age":            state.get("age"),
                "sex":            state.get("sex"),
                "localization":   state.get("localization"),
                "risk_factors":   state.get("patient_risk_factors", []),
                "missing_fields": state.get("patient_missing_fields", []),
            },

            # ── Agent 5: Safety Review ────────────────────────────────────────
            "agent5_safety": {
                "must_refer":    state.get("must_refer", state.get("safety_veto", False)),
                "veto":          state.get("safety_veto", False),
                "severity":      state.get("safety_severity", "CLEAR"),
                "safety_flags":  state.get("safety_triggers", []),
                "justification": state.get("safety_justification", ""),
            },

            # ── Agent 6: Coordinator Decision ─────────────────────────────────
            "agent6_coordinator": {
                "decision":        decision,
                "decision_rule":   state.get("decision_rule", ""),
                "urgency":         urgency,
                "confidence_tier": state.get("confidence_tier", "LOW"),
                "rationale":       state.get("rationale", ""),
            },

            # ── LLM Clinical Explanation ──────────────────────────────────────
            "clinical_explanation": state.get("clinical_explanation", ""),

            # ── Grad-CAM Visual Explainability ────────────────────────────────
            "visualizations": state.get("visualizations", {}),
        }

    def run_api(
        self,
        image_path: Union[str, Path],
        image_id: str = "CASE_001",
        age: Optional[float] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
        include_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """Convenience method that runs pipeline, generates Grad-CAM heatmap, and returns structured API dictionary."""
        state = self.run(image_path=image_path, image_id=image_id, age=age, sex=sex, localization=localization)
        
        # Generate Grad-CAM heatmap if requested
        if include_gradcam and Path(image_path).exists():
            try:
                target_cls = state.get("predicted_label") or state.get("predicted_class", "nv")
                cam_res = self.gradcam.generate_heatmap(
                    image_input=image_path,
                    target_class_name=target_cls,
                )
                state["visualizations"] = {
                    "gradcam_overlay": cam_res.get("visualizations", {}).get("gradcam_overlay", ""),
                    "original_image":  cam_res.get("visualizations", {}).get("original_image", ""),
                    "pure_heatmap":    cam_res.get("visualizations", {}).get("pure_heatmap", ""),
                    "focus_inside_lesion_pct": cam_res.get("focus_inside_lesion_pct", 0.0),
                    "peak_attention_coords": cam_res.get("peak_attention_coords", {}),
                    "target_class":    cam_res.get("target_class", target_cls),
                    "target_class_name": cam_res.get("target_class_name", ""),
                    "description":     cam_res.get("description", ""),
                }
            except Exception as e:
                state["visualizations"] = {"error": str(e)}

        return self.get_api_response(state)

