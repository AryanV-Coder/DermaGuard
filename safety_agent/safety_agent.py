"""
Agent 5: Safety Agent
=====================
Reviews outputs from Agents 1-4 for conflicts or unsafe conditions
that would make an automated triage (AUTO_TRIAGE) decision dangerous.

Inputs:
  - agent1_report  : dict  — output of QualitySupervisor.evaluate()
  - agent2_result  : dict  — output of DiagnosisAgent.predict()
  - agent3_result  : dict  — output of LesionAnalysisAgent.analyze()
  - agent4_result  : dict  — output of ReasoningContextAgent.evaluate()

Outputs:
  - safety_flag    : bool  — True if any safety concern was triggered
  - veto           : bool  — True if AUTO_TRIAGE must be blocked
  - triggers       : List[str]  — which rules fired
  - justification  : str   — human-readable explanation for the veto/flag
  - severity       : str   — 'CLEAR' | 'WARN' | 'VETO'
"""

from typing import Any, Dict, List, Tuple

# ── Malignant class codes (require extra scrutiny) ─────────────────────────────
_MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
_BENIGN_CLASSES    = {"nv", "df", "bkl", "vasc"}

# ── Thresholds ─────────────────────────────────────────────────────────────────
_CONF_VETO_MALIGNANT   = 0.65   # Confidence below this for a malignant pred → VETO
_CONF_WARN_MALIGNANT   = 0.80   # Below this is still a soft warning
_CONF_VETO_ANY         = 0.45   # Below this for ANY class → VETO (too uncertain)
_SUSPICION_HIGH        = 0.60   # Agent 3 suspicion_score threshold for HIGH
_SUSPICION_MODERATE    = 0.35   # Agent 3 suspicion_score threshold for MODERATE
_RISK_HIGH             = 0.60   # Agent 4 risk_score threshold for HIGH
_QUALITY_MARGINAL      = 0.68   # Agent 1 score just above threshold = marginal quality


class SafetyAgent:
    """
    Cross-agent safety reviewer.

    All conflict-check methods return (triggered: bool, trigger_name: str, reason: str).
    A veto is raised when any veto-level check fires.
    A safety flag is raised when any warn-level check fires (no veto).
    """

    def review(
        self,
        agent1_report: Dict[str, Any],
        agent2_result: Dict[str, Any],
        agent3_result: Dict[str, Any],
        agent4_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run all safety checks across agent outputs.

        Returns
        -------
        {
            safety_flag   : bool
            veto          : bool
            triggers      : List[str]
            justification : str
            severity      : 'CLEAR' | 'WARN' | 'VETO'
            checks        : Dict[str, dict]   — per-check detail
        }
        """
        # Extract key values up front for readability
        q_status   = agent1_report.get("overall", {}).get("status", "ERROR")
        q_score    = float(agent1_report.get("overall", {}).get("score", 0.0))

        pred_class = agent2_result.get("predicted_class") or ""
        confidence = float(agent2_result.get("confidence", 0.0))

        suspicion_score = float(agent3_result.get("suspicion_score", 0.0))
        suspicion_level = agent3_result.get("suspicion_level", "UNKNOWN")
        seg_success     = agent3_result.get("segmentation_success", False)
        reliability_flags = agent3_result.get("reliability_flags", [])

        risk_score     = float(agent4_result.get("risk_score", 0.0))
        risk_indicator = agent4_result.get("risk_indicator", "UNKNOWN")
        missing_fields = agent4_result.get("missing_field_flags", [])

        veto_triggers: List[Tuple[str, str]] = []   # (trigger_name, reason)
        warn_triggers: List[Tuple[str, str]] = []

        # ── Run all checks ────────────────────────────────────────────────────
        checks = {}

        # 1. Malignant prediction with low confidence
        fired, name, reason = self._check_malignant_low_confidence(pred_class, confidence)
        checks["malignant_low_confidence"] = {"fired": fired, "reason": reason}
        if fired:
            veto_triggers.append((name, reason))

        # 2. Very low confidence on any class
        fired, name, reason = self._check_very_low_confidence(confidence)
        checks["very_low_confidence"] = {"fired": fired, "reason": reason}
        if fired:
            veto_triggers.append((name, reason))

        # 3. High visual suspicion vs. benign prediction
        fired, name, reason = self._check_suspicion_benign_conflict(
            pred_class, suspicion_score, suspicion_level
        )
        checks["suspicion_benign_conflict"] = {"fired": fired, "reason": reason}
        if fired:
            veto_triggers.append((name, reason))

        # 4. High patient risk + uncertain model
        fired, name, reason = self._check_high_risk_low_confidence(
            risk_score, risk_indicator, confidence, pred_class
        )
        checks["high_risk_low_confidence"] = {"fired": fired, "reason": reason}
        if fired:
            veto_triggers.append((name, reason))

        # 5. Convergent risk: both Agent 3 MODERATE+ and Agent 4 HIGH → benign pred
        fired, name, reason = self._check_convergent_risk(
            pred_class, suspicion_score, risk_score
        )
        checks["convergent_risk_benign_pred"] = {"fired": fired, "reason": reason}
        if fired:
            veto_triggers.append((name, reason))

        # 6. Segmentation failure on malignant prediction (warn only)
        fired, name, reason = self._check_seg_failure_malignant(
            pred_class, seg_success, reliability_flags
        )
        checks["seg_failure_malignant"] = {"fired": fired, "reason": reason}
        if fired:
            warn_triggers.append((name, reason))

        # 7. Marginal quality image with malignant prediction (warn)
        fired, name, reason = self._check_marginal_quality_malignant(
            q_score, pred_class
        )
        checks["marginal_quality_malignant"] = {"fired": fired, "reason": reason}
        if fired:
            warn_triggers.append((name, reason))

        # 8. Missing critical metadata fields with high-risk prediction (warn)
        fired, name, reason = self._check_missing_metadata_risk(
            missing_fields, pred_class, confidence
        )
        checks["missing_metadata_risk"] = {"fired": fired, "reason": reason}
        if fired:
            warn_triggers.append((name, reason))

        # ── Aggregate decision ────────────────────────────────────────────────
        all_triggers  = [n for n, _ in veto_triggers] + [n for n, _ in warn_triggers]
        all_reasons   = [r for _, r in veto_triggers] + [r for _, r in warn_triggers]

        has_veto = len(veto_triggers) > 0
        has_warn = len(warn_triggers) > 0

        if has_veto:
            severity = "VETO"
            justification = self._build_justification(
                "VETO", pred_class, confidence, suspicion_level,
                risk_indicator, veto_triggers, warn_triggers
            )
        elif has_warn:
            severity = "WARN"
            justification = self._build_justification(
                "WARN", pred_class, confidence, suspicion_level,
                risk_indicator, [], warn_triggers
            )
        else:
            severity = "CLEAR"
            justification = (
                f"All safety checks passed. Diagnosis: {pred_class} "
                f"(conf={confidence:.2f}), suspicion: {suspicion_level}, "
                f"patient risk: {risk_indicator}. Safe for AUTO_TRIAGE."
            )

        return {
            "safety_flag":    has_veto or has_warn,
            "veto":           has_veto,
            "triggers":       all_triggers,
            "justification":  justification,
            "severity":       severity,
            "checks":         checks,
        }

    # ── Individual safety checks ──────────────────────────────────────────────

    @staticmethod
    def _check_malignant_low_confidence(
        pred_class: str, confidence: float
    ) -> Tuple[bool, str, str]:
        if pred_class in _MALIGNANT_CLASSES and confidence < _CONF_VETO_MALIGNANT:
            return (
                True,
                "MALIGNANT_LOW_CONFIDENCE",
                f"Model predicts malignant class '{pred_class}' with low confidence "
                f"({confidence:.2f} < {_CONF_VETO_MALIGNANT}). "
                f"Insufficient certainty to auto-triage a potentially life-threatening lesion.",
            )
        return False, "MALIGNANT_LOW_CONFIDENCE", "OK"

    @staticmethod
    def _check_very_low_confidence(confidence: float) -> Tuple[bool, str, str]:
        if confidence < _CONF_VETO_ANY:
            return (
                True,
                "VERY_LOW_CONFIDENCE",
                f"Model confidence ({confidence:.2f}) is below minimum threshold "
                f"({_CONF_VETO_ANY}) for any class. Prediction is unreliable.",
            )
        return False, "VERY_LOW_CONFIDENCE", "OK"

    @staticmethod
    def _check_suspicion_benign_conflict(
        pred_class: str, suspicion_score: float, suspicion_level: str
    ) -> Tuple[bool, str, str]:
        if pred_class in _BENIGN_CLASSES and suspicion_score >= _SUSPICION_HIGH:
            return (
                True,
                "SUSPICION_BENIGN_CONFLICT",
                f"Agent 3 reports HIGH morphological suspicion (score={suspicion_score:.2f}) "
                f"but Agent 2 predicts benign class '{pred_class}'. "
                f"Visual cues and model prediction are in conflict.",
            )
        return False, "SUSPICION_BENIGN_CONFLICT", "OK"

    @staticmethod
    def _check_high_risk_low_confidence(
        risk_score: float, risk_indicator: str, confidence: float, pred_class: str
    ) -> Tuple[bool, str, str]:
        if risk_indicator == "HIGH" and confidence < 0.70:
            return (
                True,
                "HIGH_RISK_LOW_CONFIDENCE",
                f"Patient context indicates HIGH risk (score={risk_score:.2f}) "
                f"but model confidence is low ({confidence:.2f}). "
                f"Cannot auto-triage a high-risk patient with an uncertain prediction '{pred_class}'.",
            )
        return False, "HIGH_RISK_LOW_CONFIDENCE", "OK"

    @staticmethod
    def _check_convergent_risk(
        pred_class: str, suspicion_score: float, risk_score: float
    ) -> Tuple[bool, str, str]:
        """Both visual and contextual evidence suggest malignancy but model says benign."""
        if (
            pred_class in _BENIGN_CLASSES
            and suspicion_score >= _SUSPICION_MODERATE
            and risk_score >= _RISK_HIGH
        ):
            return (
                True,
                "CONVERGENT_RISK_BENIGN_PRED",
                f"Agent 3 suspicion ({suspicion_score:.2f}) and Agent 4 patient risk "
                f"({risk_score:.2f}) both indicate elevated malignancy concern, "
                f"but Agent 2 predicts benign class '{pred_class}'. Expert review required.",
            )
        return False, "CONVERGENT_RISK_BENIGN_PRED", "OK"

    @staticmethod
    def _check_seg_failure_malignant(
        pred_class: str, seg_success: bool, reliability_flags: List[str]
    ) -> Tuple[bool, str, str]:
        if pred_class in _MALIGNANT_CLASSES and (not seg_success or len(reliability_flags) > 0):
            detail = "; ".join(reliability_flags) if reliability_flags else "segmentation failed"
            return (
                True,
                "SEG_FAILURE_MALIGNANT",
                f"Malignant class '{pred_class}' predicted but lesion analysis has "
                f"reliability issues: {detail}. Visual evidence unreliable.",
            )
        return False, "SEG_FAILURE_MALIGNANT", "OK"

    @staticmethod
    def _check_marginal_quality_malignant(
        q_score: float, pred_class: str
    ) -> Tuple[bool, str, str]:
        if pred_class in _MALIGNANT_CLASSES and 0.60 <= q_score <= _QUALITY_MARGINAL:
            return (
                True,
                "MARGINAL_QUALITY_MALIGNANT",
                f"Image quality is marginal (score={q_score:.3f}, just above PASS "
                f"threshold) while predicting malignant class '{pred_class}'. "
                f"Image quality may be affecting diagnostic reliability.",
            )
        return False, "MARGINAL_QUALITY_MALIGNANT", "OK"

    @staticmethod
    def _check_missing_metadata_risk(
        missing_fields: List[str], pred_class: str, confidence: float
    ) -> Tuple[bool, str, str]:
        n_missing = len(missing_fields)
        if n_missing >= 2 and pred_class in _MALIGNANT_CLASSES and confidence < 0.80:
            return (
                True,
                "MISSING_METADATA_RISK",
                f"{n_missing} patient metadata fields are missing "
                f"({', '.join(f.split(':')[0] for f in missing_fields)}) "
                f"while predicting malignant class '{pred_class}' with confidence {confidence:.2f}. "
                f"Context-free malignant prediction warrants expert review.",
            )
        return False, "MISSING_METADATA_RISK", "OK"

    # ── Justification builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_justification(
        level: str,
        pred_class: str,
        confidence: float,
        suspicion_level: str,
        risk_indicator: str,
        veto_triggers: List[Tuple[str, str]],
        warn_triggers: List[Tuple[str, str]],
    ) -> str:
        lines = [
            f"[{level}] Prediction='{pred_class}' conf={confidence:.2f} | "
            f"Suspicion={suspicion_level} | PatientRisk={risk_indicator}",
        ]
        if veto_triggers:
            lines.append("VETO triggers:")
            for name, reason in veto_triggers:
                lines.append(f"  [VETO] [{name}] {reason}")
        if warn_triggers:
            lines.append("Warning triggers:")
            for name, reason in warn_triggers:
                lines.append(f"  [WARN] [{name}] {reason}")
        return " | ".join(lines)
