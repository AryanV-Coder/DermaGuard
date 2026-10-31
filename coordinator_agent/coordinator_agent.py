"""
Agent 6: Coordinator Agent
==========================
Final decision-maker in the DermaGuard six-agent pipeline.

Receives structured outputs from Agents 1-5 and applies a documented
decision policy to produce the definitive triage outcome.

Decision Policy (in priority order):
  Rule 1 — RE_IMAGE  : Agent 1 status is FAIL or ERROR
  Rule 2 — REFER     : Agent 5 has issued a VETO
  Rule 3 — REFER     : Agent 5 WARN + predicted class is malignant
  Rule 4 — REFER     : Agent 3 HIGH suspicion + Agent 4 HIGH risk
                        (even without explicit veto)
  Rule 5 — AUTO_TRIAGE: All other cases — Agent 1 PASS + Agent 5 CLEAR/WARN

Outputs:
  - prediction       : str   — final lesion class (from Agent 2, or 'nv' fallback)
  - decision         : str   — 'AUTO_TRIAGE' | 'REFER' | 'RE_IMAGE'
  - rationale        : str   — human-readable step-by-step reasoning trail
  - decision_rule    : str   — which rule triggered the final decision
  - evidence_summary : Dict  — structured snapshot of all agent contributions
  - confidence_tier  : str   — 'HIGH' | 'MODERATE' | 'LOW' (overall pipeline confidence)
"""

from typing import Any, Dict, List, Optional


_MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
_BENIGN_CLASSES    = {"nv", "df", "bkl", "vasc"}

_CLASS_FULL_NAME = {
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
    Synthesises evidence from all five agents into a final traceable decision.

    Usage
    -----
        coordinator = CoordinatorAgent()
        result = coordinator.decide(
            image_id   = "ISIC_0029176",
            a1_report  = ...,   # QualitySupervisor output
            a2_result  = ...,   # DiagnosisAgent output
            a3_result  = ...,   # LesionAnalysisAgent output
            a4_result  = ...,   # ReasoningContextAgent output
            a5_result  = ...,   # SafetyAgent output
        )
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

        Returns
        -------
        {
            image_id         : str
            prediction       : str   — final lesion class code
            prediction_label : str   — full lesion name
            decision         : str   — 'AUTO_TRIAGE' | 'REFER' | 'RE_IMAGE'
            decision_rule    : str   — which rule triggered the decision
            confidence_tier  : str   — 'HIGH' | 'MODERATE' | 'LOW'
            rationale        : str   — traceable reasoning chain
            evidence_summary : Dict  — structured snapshot of all agents
        }
        """
        # ── Extract key values from each agent ────────────────────────────────
        q_status   = a1_report.get("overall", {}).get("status", "ERROR")
        q_score    = float(a1_report.get("overall", {}).get("score", 0.0))
        q_rec      = a1_report.get("overall", {}).get("recommendation", "")

        pred_class  = a2_result.get("predicted_class") or "nv"
        confidence  = float(a2_result.get("confidence", 0.0))
        probs       = a2_result.get("probabilities", {})

        susp_score  = float(a3_result.get("suspicion_score", 0.0))
        susp_level  = a3_result.get("suspicion_level", "UNKNOWN")
        rel_flags   = a3_result.get("reliability_flags", [])

        risk_score  = float(a4_result.get("risk_score", 0.0))
        risk_ind    = a4_result.get("risk_indicator", "UNKNOWN")
        miss_fields = a4_result.get("missing_field_flags", [])

        veto        = a5_result.get("veto", False)
        severity    = a5_result.get("severity", "CLEAR")
        triggers    = a5_result.get("triggers", [])
        a5_just     = a5_result.get("justification", "")

        is_malignant = pred_class in _MALIGNANT_CLASSES

        # ── Apply decision policy ─────────────────────────────────────────────
        decision, rule = self._apply_policy(
            q_status, veto, severity, is_malignant,
            susp_level, susp_score, risk_ind, risk_score,
            confidence,
        )

        # ── Compute overall confidence tier ───────────────────────────────────
        confidence_tier = self._confidence_tier(
            confidence, susp_score, risk_score, len(miss_fields), len(rel_flags)
        )

        # ── Build traceable rationale ─────────────────────────────────────────
        rationale = self._build_rationale(
            image_id, decision, rule, pred_class, confidence,
            q_status, q_score, susp_level, susp_score,
            risk_ind, risk_score, severity, triggers,
            miss_fields, rel_flags, a5_just,
        )

        # ── Evidence summary ──────────────────────────────────────────────────
        evidence_summary = {
            "agent1": {
                "quality_status": q_status,
                "quality_score":  round(q_score, 4),
            },
            "agent2": {
                "predicted_class": pred_class,
                "confidence":      round(confidence, 4),
                "top_probabilities": self._top3(probs),
            },
            "agent3": {
                "suspicion_score": round(susp_score, 4),
                "suspicion_level": susp_level,
                "reliability_flags": rel_flags,
            },
            "agent4": {
                "risk_score":         round(risk_score, 4),
                "risk_indicator":     risk_ind,
                "missing_fields":     miss_fields,
            },
            "agent5": {
                "severity":     severity,
                "veto":         veto,
                "triggers":     triggers,
            },
        }

        return {
            "image_id":         image_id,
            "prediction":       pred_class,
            "prediction_label": _CLASS_FULL_NAME.get(pred_class, pred_class),
            "decision":         decision,
            "decision_rule":    rule,
            "confidence_tier":  confidence_tier,
            "rationale":        rationale,
            "evidence_summary": evidence_summary,
        }

    # ── Decision policy ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_policy(
        q_status:     str,
        veto:         bool,
        severity:     str,
        is_malignant: bool,
        susp_level:   str,
        susp_score:   float,
        risk_ind:     str,
        risk_score:   float,
        confidence:   float,
    ):
        """
        Priority-ordered decision rules.
        Returns (decision, rule_label).
        """

        # Rule 1: Image quality failure — cannot proceed with diagnosis
        if q_status in ("FAIL", "ERROR"):
            return "RE_IMAGE", "RULE_1_QUALITY_FAIL"

        # Rule 2: Safety agent has issued a hard veto
        if veto:
            return "REFER", "RULE_2_SAFETY_VETO"

        # Rule 3: Safety warning + malignant class → still refer
        if severity == "WARN" and is_malignant:
            return "REFER", "RULE_3_WARN_MALIGNANT"

        # Rule 4: Convergent risk from visual + context (belt-and-suspenders)
        if (
            susp_level == "HIGH"
            and risk_ind == "HIGH"
        ):
            return "REFER", "RULE_4_CONVERGENT_RISK"

        # Rule 5: Malignant prediction (Melanoma, BCC, AKIEC) — must ALWAYS be referred
        if is_malignant:
            if confidence >= 0.75:
                return "REFER", "RULE_5_CONFIRMED_MALIGNANT_REFER"
            else:
                return "REFER", "RULE_5_SUSPECTED_MALIGNANT_REFER"

        # Rule 6: Benign class with all safety checks cleared — safe for automated triage
        return "AUTO_TRIAGE", "RULE_6_BENIGN_ALL_CLEAR"

    # ── Confidence tier ───────────────────────────────────────────────────────

    @staticmethod
    def _confidence_tier(
        confidence:   float,
        susp_score:   float,
        risk_score:   float,
        n_missing:    int,
        n_rel_flags:  int,
    ) -> str:
        penalty = 0
        if confidence < 0.70:
            penalty += 2
        elif confidence < 0.85:
            penalty += 1
        if susp_score > 0.50:
            penalty += 1
        if risk_score > 0.65:
            penalty += 1
        if n_missing >= 2:
            penalty += 1
        if n_rel_flags > 0:
            penalty += 1

        if penalty == 0:
            return "HIGH"
        elif penalty <= 2:
            return "MODERATE"
        else:
            return "LOW"

    # ── Rationale builder ─────────────────────────────────────────────────────

    @staticmethod
    def _build_rationale(
        image_id:    str,
        decision:    str,
        rule:        str,
        pred_class:  str,
        confidence:  float,
        q_status:    str,
        q_score:     float,
        susp_level:  str,
        susp_score:  float,
        risk_ind:    str,
        risk_score:  float,
        severity:    str,
        triggers:    List[str],
        miss_fields: List[str],
        rel_flags:   List[str],
        a5_just:     str,
    ) -> str:
        lines = [
            f"[COORDINATOR] Image: {image_id}  Final Decision: {decision}  "
            f"(via {rule})",
            f"",
            f"  A1 Image Quality  : {q_status} (score={q_score:.3f})",
            f"  A2 Diagnosis      : {pred_class} (confidence={confidence:.3f})",
            f"  A3 Lesion Analysis: suspicion={susp_level} (score={susp_score:.3f})",
        ]
        if rel_flags:
            lines.append(f"                    Reliability flags: {'; '.join(rel_flags)}")
        lines += [
            f"  A4 Patient Context: risk={risk_ind} (score={risk_score:.3f})",
        ]
        if miss_fields:
            lines.append(f"                    Missing fields: {'; '.join(f.split(':')[0] for f in miss_fields)}")
        lines += [
            f"  A5 Safety Review  : severity={severity}",
        ]
        if triggers:
            lines.append(f"                    Triggers: {', '.join(triggers)}")

        lines += [
            f"",
            f"  Decision rationale:",
        ]

        if rule == "RULE_1_QUALITY_FAIL":
            lines.append(
                f"    Image quality check failed (score={q_score:.3f}). "
                f"Diagnosis cannot proceed on a degraded image. Request re-capture."
            )
        elif rule == "RULE_2_SAFETY_VETO":
            lines.append(
                f"    Safety Agent issued a hard VETO. "
                f"One or more critical conflicts detected between agent outputs. "
                f"Expert clinician review is required before any triage action."
            )
        elif rule == "RULE_3_WARN_MALIGNANT":
            lines.append(
                f"    Safety Agent raised a warning and predicted class '{pred_class}' is "
                f"potentially malignant. Precautionary REFER to avoid missing a critical case."
            )
        elif rule == "RULE_4_CONVERGENT_RISK":
            lines.append(
                f"    Both visual lesion analysis (suspicion={susp_level}) and patient "
                f"context (risk={risk_ind}) independently signal elevated malignancy concern. "
                f"Convergent multi-agent risk warrants expert review."
            )
        elif rule == "RULE_5_MALIGNANT_MODERATE_CONF":
            lines.append(
                f"    Predicted class '{pred_class}' is malignant but model confidence "
                f"({confidence:.3f}) is below safe threshold (0.75). "
                f"Insufficient certainty to auto-triage a potentially life-threatening lesion."
            )
        elif rule == "RULE_6_ALL_CLEAR":
            lines.append(
                f"    All agents are in agreement. Image quality is adequate, model is "
                f"confident (conf={confidence:.3f}), no safety conflicts detected. "
                f"Prediction '{pred_class}' is cleared for automated triage."
            )

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _top3(probs: Dict[str, float]) -> Dict[str, float]:
        """Return top-3 class probabilities."""
        sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return {k: round(v, 4) for k, v in sorted_p[:3]}

