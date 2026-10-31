"""
Agent 5: Safety Agent / Safety Review Agent
============================================
Zero-tolerance clinical safety policy enforcer.
Ensures zero missed malignancies in automated triage:
  1. Predicted malignant (mel, bcc, akiec) → Always REFER / VETO
  2. Melanoma prob >= 0.18 OR melanoma flag → Always REFER / VETO
  3. Aggregate malignancy prob (BCC + Mel + AKIEC) >= 0.25 → Always REFER / VETO
  4. High visual ABCD morphometric suspicion (ABCD > 1.50 or HIGH) → Always REFER / VETO
  5. Convergent risk: elevated lesion suspicion + elevated patient risk → Always REFER / VETO
  6. Low confidence (< 0.40) on any diagnosis → Always REFER / VETO

Usage
-----
    from safety_agent.safety_agent import SafetyAgent, SafetyReviewAgent
    agent = SafetyAgent()
    # Direct review:
    report = agent.review(a1_report, a2_result, a3_result, a4_result)
    # Or callable format matching eval_pipeline.ipynb:
    report = agent(a2_result, a3_result, a4_result)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

# ── Malignant class codes ─────────────────────────────────────────────────────
MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
BENIGN_CLASSES    = {"nv", "df", "bkl", "vasc"}

# ── Calibrated Safety Thresholds (matches eval_pipeline.ipynb) ────────────────
MEL_THRESH    = 0.18   # Lowered from 0.30 to ensure 0 missed melanomas in Auto-Triage
MALIGN_THRESH = 0.25   # Aggregate malignancy threshold for early cancer safety
CONF_THRESH   = 0.40   # Minimum acceptable confidence for autonomous clearance
SUSPICION_HIGH = 1.50   # ABCD composite threshold for high visual suspicion


class SafetyAgent:
    """
    Zero-tolerance clinical safety reviewer.
    Guarantees no suspected or confirmed skin cancer is cleared for auto-triage.
    """

    MALIGNANT = MALIGNANT_CLASSES
    MEL_THRESH = MEL_THRESH
    MALIGN_THRESH = MALIGN_THRESH
    CONF_THRESH = CONF_THRESH

    def review(
        self,
        agent1_report: Optional[Dict[str, Any]],
        agent2_result: Dict[str, Any],
        agent3_result: Dict[str, Any],
        agent4_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run all safety checks across agent outputs.
        """
        q_status = (agent1_report or {}).get("overall", {}).get("status", "PASS")
        q_score = float((agent1_report or {}).get("overall", {}).get("score", 1.0))

        pred_class = (
            agent2_result.get("predicted_class")
            or agent2_result.get("predicted_label")
            or agent2_result.get("prediction")
            or "nv"
        ).lower()
        confidence = float(agent2_result.get("confidence", 0.0))
        mel_flag = bool(agent2_result.get("melanoma_flag", False))
        mel_prob = float(agent2_result.get("melanoma_prob", 0.0))
        malign_prob = float(agent2_result.get("malignant_prob", 0.0))
        if malign_prob == 0.0 and "probabilities" in agent2_result:
            probs = agent2_result["probabilities"]
            malign_prob = float(sum(probs.get(c, 0.0) for c in MALIGNANT_CLASSES))

        # Agent 3: Lesion Morphometrics
        suspicion_score = float(agent3_result.get("suspicion_score", agent3_result.get("abcd_score", 0.0)))
        suspicion_level = agent3_result.get("suspicion_level", agent3_result.get("lesion_risk", "UNKNOWN")).upper()
        abcd_sc = float(agent3_result.get("abcd_score", suspicion_score))

        # Agent 4: Patient Context
        risk_score = float(agent4_result.get("risk_score", 0.0))
        risk_indicator = agent4_result.get("risk_indicator", agent4_result.get("risk_level", "UNKNOWN")).upper()

        veto_triggers: List[Tuple[str, str]] = []
        warn_triggers: List[Tuple[str, str]] = []
        checks: Dict[str, Dict[str, Any]] = {}

        # ── 1. Malignant Prediction (Zero-Tolerance) ───────────────────────────
        if pred_class in MALIGNANT_CLASSES:
            reason = f"Direct malignant prediction '{pred_class}' (confidence={confidence:.3f}). Immediate clinical referral mandatory."
            veto_triggers.append(("MALIGNANT_PREDICTION", reason))
            checks["malignant_prediction"] = {"fired": True, "reason": reason}
        else:
            checks["malignant_prediction"] = {"fired": False, "reason": "OK"}

        # ── 2. Elevated Melanoma Risk ──────────────────────────────────────────
        if mel_flag or (mel_prob >= self.MEL_THRESH):
            reason = f"Melanoma probability ({mel_prob:.3f}) exceeds safety limit ({self.MEL_THRESH}) or melanoma flag set."
            veto_triggers.append(("MELANOMA_SAFETY_BARRIER", reason))
            checks["melanoma_safety_barrier"] = {"fired": True, "reason": reason}
        else:
            checks["melanoma_safety_barrier"] = {"fired": False, "reason": "OK"}

        # ── 3. Aggregate Malignancy Probability (BCC + Mel + AKIEC) ───────────
        if malign_prob >= self.MALIGN_THRESH:
            reason = f"Aggregate malignancy probability ({malign_prob:.3f}) exceeds threshold ({self.MALIGN_THRESH})."
            veto_triggers.append(("AGGREGATE_MALIGNANCY_ELEVATED", reason))
            checks["aggregate_malignancy"] = {"fired": True, "reason": reason}
        else:
            checks["aggregate_malignancy"] = {"fired": False, "reason": "OK"}

        # ── 4. High Visual ABCD Morphometric Suspicion ────────────────────────
        if suspicion_level == "HIGH" or abcd_sc > SUSPICION_HIGH:
            reason = f"Agent 3 reports HIGH ABCD morphometric suspicion (ABCD={abcd_sc:.2f}, level={suspicion_level}). Visual cues conflict with benign automated clearance."
            veto_triggers.append(("HIGH_VISUAL_SUSPICION", reason))
            checks["high_visual_suspicion"] = {"fired": True, "reason": reason}
        else:
            checks["high_visual_suspicion"] = {"fired": False, "reason": "OK"}

        # ── 5. Convergent Multi-Agent Risk ─────────────────────────────────────
        if suspicion_level in ("MODERATE", "HIGH") and risk_indicator in ("MODERATE", "HIGH"):
            reason = f"Convergent risk: visual suspicion is {suspicion_level} and patient context risk is {risk_indicator}."
            veto_triggers.append(("CONVERGENT_RISK", reason))
            checks["convergent_risk"] = {"fired": True, "reason": reason}
        else:
            checks["convergent_risk"] = {"fired": False, "reason": "OK"}

        # ── 6. Low Confidence Prediction ───────────────────────────────────────
        low_conf = confidence < self.CONF_THRESH
        if low_conf:
            reason = f"Diagnostic confidence ({confidence:.3f}) is below minimum threshold ({self.CONF_THRESH}). Prediction is uncertain."
            veto_triggers.append(("LOW_CONFIDENCE", reason))
            checks["low_confidence"] = {"fired": True, "reason": reason}
        else:
            checks["low_confidence"] = {"fired": False, "reason": "OK"}

        # ── Aggregate Decision ────────────────────────────────────────────────
        has_veto = len(veto_triggers) > 0
        must_refer = has_veto

        all_triggers = [n for n, _ in veto_triggers] + [n for n, _ in warn_triggers]
        all_reasons  = [r for _, r in veto_triggers] + [r for _, r in warn_triggers]

        if has_veto:
            severity = "VETO"
            justification = f"Clinical Safety VETO triggered: {'; '.join(all_triggers)}. Specialist referral required."
        else:
            severity = "CLEAR"
            justification = f"All safety checks passed. Benign {pred_class} (conf={confidence:.2f}), low suspicion, no safety conflicts. Safe for auto-triage."

        return {
            # eval_pipeline.ipynb keys
            "must_refer":     must_refer,
            "low_confidence": low_conf,
            "safety_flags":   all_reasons,

            # LangGraph & structured API keys
            "veto":           has_veto,
            "safety_flag":    has_veto or len(warn_triggers) > 0,
            "severity":       severity,
            "triggers":       all_triggers,
            "justification":  justification,
            "checks":         checks,
        }

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Supports:
          agent5(diag, lesion, patient_risk)  -- 3 args (eval_pipeline.ipynb)
          agent5(quality, diag, lesion, patient_risk) -- 4 args
        """
        if len(args) == 3:
            return self.review(None, args[0], args[1], args[2])
        elif len(args) == 4:
            return self.review(args[0], args[1], args[2], args[3])
        else:
            return self.review(
                kwargs.get("agent1_report"),
                kwargs.get("agent2_result", {}),
                kwargs.get("agent3_result", {}),
                kwargs.get("agent4_result", {}),
            )


# Alias matching eval_pipeline.ipynb
SafetyReviewAgent = SafetyAgent
