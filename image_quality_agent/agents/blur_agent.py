"""
Blur Agent — evaluates sharpness of a dermoscopic image via Laplacian variance.

The Laplacian operator highlights edges; its variance measures how much
high-frequency (sharp) content the image contains.  A low variance indicates
a blurry image lacking sharp edges.

⚠️  The default threshold is a starting point.  Validate it on a sample of
    labelled dermoscopic images before relying on it in production.
"""

from typing import Any, Dict

import cv2
import numpy as np

# Use a relative import so the module can be run from the project root
# with `python -m image_quality_agent.agents.blur_agent`.
try:
    from config.thresholds import BLUR
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config.thresholds import BLUR


class BlurAgent:
    """
    Assesses image sharpness using Laplacian variance.

    Higher variance → sharper image.
    Lower variance  → blurrier image.
    """

    def __init__(self):
        self.blur_threshold:   float = BLUR["laplacian_threshold"]
        self.severe_threshold: float = BLUR["severe_threshold"]

    # ── Public API ──────────────────────────────────────────────────────────

    def assess(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Parameters
        ----------
        gray : np.ndarray (H, W) uint8
            Grayscale version of the image.

        Returns
        -------
        {
            status       : "PASS" | "FAIL"
            score        : float  0.0 – 1.0  (higher = sharper)
            laplacian_var: float              (raw Laplacian variance)
            is_severe    : bool               (True → hard FAIL in fusion)
            issues       : List[str]
            explanation  : str
        }
        """
        lap_var    = self._laplacian_variance(gray)
        is_severe  = lap_var < self.severe_threshold
        is_blurry  = lap_var < self.blur_threshold

        # Normalised score: clamp to [0, 1] over a 3× range above threshold
        score = min(lap_var / (3.0 * self.blur_threshold), 1.0)

        issues = []
        if is_severe:
            issues.append(
                f"Severe blur detected (Laplacian var {lap_var:.1f} < "
                f"severe threshold {self.severe_threshold}).  Image is unusable."
            )
        elif is_blurry:
            issues.append(
                f"Image appears blurry (Laplacian var {lap_var:.1f} < "
                f"threshold {self.blur_threshold})."
            )

        status = "FAIL" if is_blurry else "PASS"

        explanation = (
            f"Laplacian variance: {lap_var:.2f} "
            f"(threshold {self.blur_threshold}, severe {self.severe_threshold}).  "
            + (
                f"Image is {'severely ' if is_severe else ''}blurry."
                if is_blurry
                else "Sharpness is acceptable."
            )
        )

        return {
            "status":        status,
            "score":         round(score, 4),
            "laplacian_var": round(lap_var, 4),
            "is_severe":     is_severe,
            "issues":        issues,
            "explanation":   explanation,
        }

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Compute variance of the Laplacian of the grayscale image."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
