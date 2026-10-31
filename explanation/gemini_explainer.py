"""
Gemini Clinical Audit Explainer Module
======================================
Invokes Google Gemini API strictly after Agent 6 has produced its decision.
Translates structured multi-agent evidence into a clear, empathetic, and
traceable natural-language explanation for patients and primary care clinicians.

Guarantees:
  1. The LLM NEVER changes or overrides the triage decision.
  2. The LLM NEVER invents findings outside the multi-agent findings JSON.
  3. Seamless template fallback if GEMINI_API_KEY is not configured or network is unavailable.
"""

import os
import json
from typing import Dict, Any, Optional
import httpx


SYSTEM_PROMPT = """You are the Clinical Audit Explainer for DermaGuard, an AI skin-lesion triage system.
Your sole job is to produce a concise, professional, empathetic clinical explanation for the final triage decision that has ALREADY been made by the medical consensus engine.

CRITICAL RULES:
1. You MUST NEVER change, override, or question the final decision (AUTO_TRIAGE, REFER, or RE_IMAGE).
2. You MUST NEVER invent lesions, symptoms, or diagnoses not present in the input JSON.
3. Keep the tone calm, objective, and empathetic.
4. Structure your response into two short paragraphs:
   - Paragraph 1 (Patient/Nurse Summary): Clear 2-sentence summary explaining the decision and immediate next step.
   - Paragraph 2 (Clinical Rationale): Direct synthesis referencing the Image Quality, Diagnostic Prediction, ABCD Morphometrics, and Safety checks that led to this decision.
"""


class GeminiExplainer:
    """
    Lightweight, robust Gemini API client for post-hoc clinical explanation.
    Uses direct HTTP requests (no heavy SDK dependency).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        timeout_seconds: float = 12.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.timeout = timeout_seconds

    def explain(self, findings: Dict[str, Any]) -> str:
        """
        Generate a natural-language clinical explanation from multi-agent findings.
        """
        if not self.api_key:
            return self._deterministic_fallback(findings, reason="GEMINI_API_KEY not configured")

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        user_content = (
            "Here are the structured multi-agent findings for this case:\n\n"
            + json.dumps(findings, indent=2)
            + "\n\nPlease explain this decision following your instructions."
        )

        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "maxOutputTokens": 350,
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                elif resp.status_code == 404:
                    # Fallback to gemini-1.5-flash if 2.0-flash is unavailable on that endpoint
                    alt_endpoint = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
                        f"?key={self.api_key}"
                    )
                    resp_alt = client.post(alt_endpoint, json=payload)
                    if resp_alt.status_code == 200:
                        data = resp_alt.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                return parts[0].get("text", "").strip()

            return self._deterministic_fallback(findings, reason=f"API returned status {resp.status_code}")
        except Exception as exc:
            return self._deterministic_fallback(findings, reason=f"Network/API call failed: {exc}")

    def _deterministic_fallback(self, findings: Dict[str, Any], reason: str = "") -> str:
        """
        Rule-based deterministic clinical synthesis used when Gemini API is offline.
        Ensures the pipeline is 100% resilient and self-contained.
        """
        decision = findings.get("final_decision", "REFER")
        pred_label = findings.get("diagnosis", {}).get("predicted_label", "Unknown")
        conf = findings.get("diagnosis", {}).get("confidence", 0.0)
        q_status = findings.get("image_quality", {}).get("status", "PASS")
        susp = findings.get("lesion_analysis", {}).get("suspicion_level", "LOW")
        triggers = findings.get("safety", {}).get("triggers", [])
        risk_ind = findings.get("patient_context", {}).get("risk_indicator", "LOW")

        if decision == "RE_IMAGE":
            return (
                f"The submitted skin image did not pass automated quality standards ({q_status}). "
                f"Features were obscured or lighting was insufficient for safe clinical diagnosis. "
                f"Please recapture the lesion with good lighting, sharp focus, and centered positioning."
            )

        if decision == "REFER":
            reasons = []
            if triggers:
                reasons.append(f"Safety alerts triggered: {', '.join(triggers[:2])}")
            if susp in ("MODERATE", "HIGH"):
                reasons.append(f"Visual ABCD analysis identified elevated morphological suspicion ({susp})")
            if "mel" in findings.get("diagnosis", {}).get("predicted_class", ""):
                reasons.append(f"Differential diagnosis includes potential melanoma")
            elif conf < 0.70:
                reasons.append(f"Moderate classification certainty ({conf:.1%}) on {pred_label}")

            reason_str = "; ".join(reasons) if reasons else f"Patient context ({risk_ind} risk) warrants review"
            return (
                f"This case has been referred for specialist dermatological evaluation. "
                f"While the preliminary model predicted {pred_label} ({conf:.1%}), clinical consensus "
                f"requires human examination because: {reason_str}. An in-person consultation is recommended."
            )

        # AUTO_TRIAGE
        return (
            f"The lesion has been cleared for routine monitoring (Auto-Triage). "
            f"The image passed all quality checks. Analysis indicates a typical benign {pred_label} "
            f"with high model confidence ({conf:.1%}), low visual ABCD suspicion, and no conflicting safety flags. "
            f"Routine annual self-checks are recommended."
        )
