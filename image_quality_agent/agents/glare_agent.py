"""
Glare Agent — detects specular reflections and excessive bright spots
in dermoscopic images using HSV thresholding.

Strategy
--------
Glare pixels are characterised by:
  - Very HIGH brightness (V channel > value_min)
  - Very LOW saturation (S channel < saturation_max)
    → they appear white / washed out rather than coloured

This two-condition filter avoids mis-classifying legitimately bright but
coloured skin regions as glare.

A small morphological opening step removes isolated noise pixels before
computing the coverage ratio.
"""

from typing import Any, Dict, List

import cv2
import numpy as np

try:
    from config.thresholds import GLARE
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config.thresholds import GLARE


class GlareAgent:
    """
    Detects specular reflections (glare) using HSV thresholding.
    Returns a glare coverage ratio and a PASS / FAIL decision.
    """

    def __init__(self):
        self.cfg = GLARE

    # ── Public API ──────────────────────────────────────────────────────────

    def assess(self, hsv: np.ndarray) -> Dict[str, Any]:
        """
        Parameters
        ----------
        hsv : np.ndarray (H, W, 3) uint8

        Returns
        -------
        {
            status          : "PASS" | "FAIL"
            score           : float 0.0 – 1.0
            glare_pixel_count: int
            glare_ratio     : float
            is_severe       : bool   (True → hard FAIL in fusion)
            issues          : List[str]
            explanation     : str
        }
        """
        glare_mask   = self._build_glare_mask(hsv)
        total_px     = hsv.shape[0] * hsv.shape[1]
        glare_px     = int(np.sum(glare_mask))
        glare_ratio  = glare_px / max(total_px, 1)

        is_severe = glare_ratio > self.cfg["severe_glare_ratio"]
        issues: List[str] = []

        if is_severe:
            issues.append(
                f"Severe glare: {glare_ratio * 100:.1f}% of image affected "
                f"(severe limit {self.cfg['severe_glare_ratio'] * 100:.0f}%)."
            )
        elif glare_ratio > self.cfg["glare_pixel_ratio"]:
            issues.append(
                f"Glare detected: {glare_ratio * 100:.1f}% of pixels affected "
                f"(limit {self.cfg['glare_pixel_ratio'] * 100:.0f}%)."
            )

        score  = self._compute_score(glare_ratio)
        status = "FAIL" if issues else "PASS"

        explanation = (
            f"Glare-affected pixels: {glare_px} ({glare_ratio:.2%} of image).  "
            f"Threshold: {self.cfg['glare_pixel_ratio'] * 100:.0f}% / "
            f"severe: {self.cfg['severe_glare_ratio'] * 100:.0f}%.  "
            + ("Glare within acceptable limits." if not issues else "Excessive glare present.")
        )

        return {
            "status":            status,
            "score":             round(score, 4),
            "glare_pixel_count": glare_px,
            "glare_ratio":       round(glare_ratio, 4),
            "is_severe":         is_severe,
            "issues":            issues,
            "explanation":       explanation,
        }

    # ── Private ─────────────────────────────────────────────────────────────

    def _build_glare_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask marking glare pixels:
            S < saturation_max  AND  V > value_min
        A morphological opening removes single-pixel noise.
        """
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        raw_mask = ((s < self.cfg["saturation_max"]) & (v > self.cfg["value_min"])).astype(np.uint8) * 255

        # Remove isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        return cleaned > 0

    def _compute_score(self, glare_ratio: float) -> float:
        """Linear interpolation between the two glare thresholds."""
        lo = self.cfg["glare_pixel_ratio"]
        hi = self.cfg["severe_glare_ratio"]

        if glare_ratio <= lo:
            return 1.0
        if glare_ratio >= hi:
            return 0.0
        return 1.0 - (glare_ratio - lo) / max(hi - lo, 1e-9)
