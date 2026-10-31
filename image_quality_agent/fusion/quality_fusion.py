"""
Quality Fusion Engine — aggregates results from all assessment agents
into a single overall quality score and PASS / FAIL decision.

Design decisions
----------------
1. Weighted score
   Each agent contributes a normalised score (0.0 – 1.0).
   The weighted sum forms the overall score.
   Weights are configurable in config/thresholds.py.

2. Hard FAIL
   If any participating agent reports a *severe* issue (is_severe=True),
   the image is marked FAIL regardless of the weighted score.
   This prevents a single catastrophic defect from being averaged away.

3. PASS threshold
   Images with an overall score >= FUSION["pass_threshold"] and no hard
   fail are marked PASS.
"""

from typing import Any, Dict, Optional

try:
    from config.thresholds import FUSION
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config.thresholds import FUSION

# Agents whose is_severe flag triggers an unconditional FAIL
_HARD_FAIL_AGENTS = {"blur", "glare", "obstruction"}


class QualityFusionEngine:
    """
    Combines per-agent results into an overall quality decision.

    Usage
    -----
        engine = QualityFusionEngine()
        result = engine.fuse(blur_r, lighting_r, glare_r, obstruction_r)
    """

    def __init__(self):
        self.weights        = FUSION["weights"]
        self.pass_threshold = FUSION["pass_threshold"]

    # ── Public API ──────────────────────────────────────────────────────────

    def fuse(
        self,
        blur_result:        Dict[str, Any],
        lighting_result:    Dict[str, Any],
        glare_result:       Dict[str, Any],
        obstruction_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        blur_result        : output of BlurAgent.assess()
        lighting_result    : output of LightingAgent.assess()
        glare_result       : output of GlareAgent.assess()
        obstruction_result : output of ObstructionAgent.assess()

        Returns
        -------
        {
            overall_status      : "PASS" | "FAIL"
            overall_score       : float
            weighted_scores     : Dict[str, float]   — per-agent weighted contribution
            hard_fail_triggered : bool
            hard_fail_reason    : str | None
            recommendation      : str
            agent_statuses      : Dict[str, str]     — raw PASS/FAIL per agent
        }
        """
        agent_results = {
            "blur":        blur_result,
            "lighting":    lighting_result,
            "glare":       glare_result,
            "obstruction": obstruction_result,
        }

        # ── 1. Weighted score ────────────────────────────────────────────────
        weighted_scores: Dict[str, float] = {}
        overall_score = 0.0
        for name, res in agent_results.items():
            w = self.weights.get(name, 0.0)
            s = float(res.get("score", 0.0))
            weighted_scores[name] = round(s * w, 4)
            overall_score += s * w
        overall_score = round(overall_score, 4)

        # ── 2. Hard-fail check ───────────────────────────────────────────────
        hard_fail        = False
        hard_fail_reason: Optional[str] = None
        for name in _HARD_FAIL_AGENTS:
            res = agent_results[name]
            if res.get("is_severe", False):
                hard_fail        = True
                issues_text      = "; ".join(res.get("issues", ["(no detail)"]))
                hard_fail_reason = f"Severe issue in [{name}]: {issues_text}"
                break

        # ── 3. Final decision ────────────────────────────────────────────────
        passed = (not hard_fail) and (overall_score >= self.pass_threshold)
        overall_status = "PASS" if passed else "FAIL"

        recommendation = self._build_recommendation(
            overall_status, hard_fail, hard_fail_reason, agent_results
        )

        return {
            "overall_status":       overall_status,
            "overall_score":        overall_score,
            "weighted_scores":      weighted_scores,
            "hard_fail_triggered":  hard_fail,
            "hard_fail_reason":     hard_fail_reason,
            "recommendation":       recommendation,
            "agent_statuses":       {k: v.get("status", "UNKNOWN") for k, v in agent_results.items()},
        }

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_recommendation(
        status:           str,
        hard_fail:        bool,
        hard_fail_reason: Optional[str],
        agent_results:    Dict[str, Dict[str, Any]],
    ) -> str:
        if status == "PASS":
            return "Image quality is acceptable.  Proceed with diagnosis."

        failing_agents = [
            name for name, r in agent_results.items()
            if r.get("status") == "FAIL"
        ]
        parts = []
        if hard_fail and hard_fail_reason:
            parts.append(f"HARD FAIL — {hard_fail_reason}.")
        if failing_agents:
            parts.append(f"Failed checks: {', '.join(failing_agents)}.")
        parts.append("Please recapture the image with better conditions and resubmit.")
        return "  ".join(parts)
