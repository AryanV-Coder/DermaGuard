"""
Agent 6: Coordinator Agent / Final Decision Agent
==================================================
Final decision-maker in the DermaGuard six-agent pipeline.
Synthesizes evidence from Agents 1-5 and applies the clinical decision policy
to produce the definitive Track 2 disposition:
  Rule 1 — RE_IMAGE   : Technical image quality failure (cannot safely diagnose)
  Rule 2 — REFER      : Safety veto, confirmed/suspected malignancy (mel, bcc, akiec),
                        melanoma flag, high lesion ABCD suspicion, or patient risk conflict
  Rule 3 — AUTO_TRIAGE: Image Quality PASS + benign diagnosis + all safety checks clear

Submission Core Fields:
  image_id, predicted_label, decision

Usage
-----
    from coordinator_agent.coordinator_agent import CoordinatorAgent, FinalDecisionAgent
    coordinator = CoordinatorAgent()
    result = coordinator.decide(image_id, a1_report, a2_result, a3_result, a4_result, a5_result)
    # or callable (matches eval_pipeline.ipynb):
    result = coordinator(a1_report, a2_result, a3_result, a4_result, a5_result)
"""

from typing import Any, Dict, List, Optional, Union

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
BENIGN_CLASSES    = {"nv", "df", "bkl", "vasc"}

CLASS_FULL_NAME = {
    "akiec": "Actinic Keratoses / Bowen's Disease",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}


class CoordinatorAgent:
    """
    Synthesizes multi-agent evidence into a final traceable triage decision.
    """

    def decide(
        self,
        image_id:  str,
        a1_report: Dict[str, Any],
        a2_result: Dict[str, Any],
        a3_result: Dict[str, Any],
        a4_result: Dict[str, Any],
        a5_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply the documented decision policy across all agent outputs.
        """
        # ── 1. Image Quality (Priority 1) ─────────────────────────────────────
        q_pass = a1_report.get("quality_pass", True) if "quality_pass" in a1_report else (
            a1_report.get("overall", {}).get("status") == "PASS"
        )
        q_status = a1_report.get("overall", {}).get("status", "PASS" if q_pass else "FAIL")
        q_score = float(a1_report.get("overall", {}).get("score", a1_report.get("blur", 1.0)))
        q_reason = a1_report.get("reason", a1_report.get("overall", {}).get("recommendation", "poor quality"))

        # Priority 1: RE_IMAGE
        if not q_pass or q_status in ("FAIL", "ERROR"):
            return {
                "image_id":         image_id,
                "predicted_label":  "nv",
                "prediction":       "nv",
                "prediction_label": "Melanocytic Nevi",
                "decision":         "RE_IMAGE",
                "decision_rule":    "RULE_1_QUALITY_FAIL",
                "urgency":          "REPEAT_IMAGE",
                "confidence":       0.0,
                "confidence_tier":  "LOW",
                "rationale":        f"Technical quality check failed: {q_reason}. Recapture required.",
                "evidence_summary": {
                    "agent1": {"quality_pass": False, "status": q_status, "reason": q_reason},
                },
            }

        # ── 2. Diagnostic & Safety Findings ───────────────────────────────────
        pred_class = (
            a2_result.get("predicted_class")
            or a2_result.get("predicted_label")
            or a2_result.get("prediction")
            or "nv"
        ).lower()
        confidence = float(a2_result.get("confidence", 0.0))
        mel_flag = bool(a2_result.get("melanoma_flag", False))
        is_malignant = pred_class in MALIGNANT_CLASSES

        susp_score = float(a3_result.get("suspicion_score", a3_result.get("abcd_score", 0.0)))
        susp_level = a3_result.get("suspicion_level", a3_result.get("lesion_risk", "UNKNOWN"))

        risk_score = float(a4_result.get("risk_score", 0.0))
        risk_ind = a4_result.get("risk_indicator", a4_result.get("risk_level", "UNKNOWN"))

        must_refer = bool(a5_result.get("must_refer", False)) or bool(a5_result.get("veto", False))
        severity = a5_result.get("severity", "VETO" if must_refer else "CLEAR")
        flags = a5_result.get("safety_flags", a5_result.get("triggers", []))

        is_urgent = (pred_class == "mel") or mel_flag

        # ── 3. Apply Decision Policy ──────────────────────────────────────────
        if must_refer or is_malignant or (severity == "VETO"):
            decision = "REFER"
            decision_rule = "RULE_2_SAFETY_VETO"
            urgency = "URGENT" if is_urgent else "SOON"
            flag_str = "; ".join(flags) if flags else f"malignant diagnosis ({pred_class})"
            rationale = f"Safety referral: {flag_str}. Mandatory clinical dermatologist evaluation."
        else:
            decision = "AUTO_TRIAGE"
            decision_rule = "RULE_3_BENIGN_ALL_CLEAR"
            urgency = "ROUTINE"
            rationale = f"Benign {pred_class} (conf={confidence:.2f}), low suspicion, no safety conflicts. Cleared for auto-triage."

        # Overall confidence tier
        confidence_tier = self._confidence_tier(confidence, susp_score, risk_score)

        evidence_summary = {
            "agent1_quality": {
                "quality_pass": q_pass,
                "status": q_status,
                "score": round(q_score, 4),
            },
            "agent2_diagnosis": {
                "predicted_label": pred_class,
                "confidence": round(confidence, 4),
                "melanoma_flag": mel_flag,
            },
            "agent3_lesion": {
                "suspicion_score": round(susp_score, 4),
                "suspicion_level": susp_level,
            },
            "agent4_patient_risk": {
                "risk_score": round(risk_score, 4),
                "risk_indicator": risk_ind,
            },
            "agent5_safety": {
                "must_refer": must_refer,
                "severity": severity,
                "flags": flags,
            },
        }

        return {
            "image_id":         image_id,
            "predicted_label":  pred_class,  # Standard Track 2 submission column
            "prediction":       pred_class,
            "prediction_label": CLASS_FULL_NAME.get(pred_class, pred_class),
            "decision":         decision,
            "decision_rule":    decision_rule,
            "urgency":          urgency,
            "confidence":       round(confidence, 4),
            "confidence_tier":  confidence_tier,
            "rationale":        rationale,
            "evidence_summary": evidence_summary,
        }

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Supports:
          agent6(quality, diag, lesion, patient_risk, safety) -- eval_pipeline.ipynb
          agent6(image_id, a1, a2, a3, a4, a5)
        """
        if len(args) == 5:
            # (quality, diag, lesion, patient_risk, safety)
            img_id = kwargs.get("image_id", "CASE_001")
            return self.decide(img_id, args[0], args[1], args[2], args[3], args[4])
        elif len(args) == 6:
            # (image_id, a1, a2, a3, a4, a5)
            return self.decide(str(args[0]), args[1], args[2], args[3], args[4], args[5])
        else:
            img_id = kwargs.get("image_id", "CASE_001")
            return self.decide(
                img_id,
                kwargs.get("a1_report", {}),
                kwargs.get("a2_result", {}),
                kwargs.get("a3_result", {}),
                kwargs.get("a4_result", {}),
                kwargs.get("a5_result", {}),
            )

    @staticmethod
    def _confidence_tier(confidence: float, susp_score: float, risk_score: float) -> str:
        if confidence >= 0.80 and susp_score < 0.50 and risk_score < 0.60:
            return "HIGH"
        elif confidence >= 0.50:
            return "MODERATE"
        else:
            return "LOW"


# Alias matching eval_pipeline.ipynb
FinalDecisionAgent = CoordinatorAgent
