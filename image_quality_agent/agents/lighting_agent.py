"""
Lighting Agent — evaluates exposure and illumination uniformity.

Strategy
--------
Work in HSV colour space, focusing on the V (value / brightness) channel:

  1. Underexposure  — too many pixels with very low brightness.
  2. Overexposure   — too many pixels with very high brightness.
  3. Uneven illumination — high variance in per-patch mean brightness
                           (grid-based analysis).

No model is used; all checks are threshold-based and fully configurable.
"""

from typing import Any, Dict, List

import numpy as np

try:
    from config.thresholds import LIGHTING
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config.thresholds import LIGHTING


class LightingAgent:
    """
    Checks for underexposure, overexposure, and uneven illumination
    using the V channel of an HSV image.
    """

    def __init__(self):
        self.cfg = LIGHTING

    # ── Public API ──────────────────────────────────────────────────────────

    def assess(self, hsv: np.ndarray) -> Dict[str, Any]:
        """
        Parameters
        ----------
        hsv : np.ndarray (H, W, 3) uint8
            HSV representation of the image.

        Returns
        -------
        {
            status          : "PASS" | "FAIL"
            score           : float 0.0 – 1.0
            mean_brightness : float   — mean V-channel value (0-255)
            dark_pixel_ratio: float
            bright_pixel_ratio: float
            illumination_std: float   — std of patch-level mean brightness
            issues          : List[str]
            explanation     : str
        }
        """
        v = hsv[:, :, 2].astype(np.float32)

        mean_v       = float(np.mean(v))
        dark_ratio   = float(np.mean(v < self.cfg["dark_threshold"]))
        bright_ratio = float(np.mean(v > self.cfg["bright_threshold"]))
        uneven_std   = self._patch_illumination_std(v)

        issues: List[str] = []

        if dark_ratio > self.cfg["dark_pixel_ratio"]:
            issues.append(
                f"Underexposed: {dark_ratio * 100:.1f}% pixels are too dark "
                f"(limit {self.cfg['dark_pixel_ratio'] * 100:.0f}%)."
            )
        if bright_ratio > self.cfg["bright_pixel_ratio"]:
            issues.append(
                f"Overexposed: {bright_ratio * 100:.1f}% pixels are too bright "
                f"(limit {self.cfg['bright_pixel_ratio'] * 100:.0f}%)."
            )
        if uneven_std > self.cfg["uneven_std_threshold"]:
            issues.append(
                f"Uneven illumination: patch brightness std = {uneven_std:.1f} "
                f"(limit {self.cfg['uneven_std_threshold']})."
            )

        score  = self._compute_score(dark_ratio, bright_ratio, uneven_std)
        status = "FAIL" if issues else "PASS"

        explanation = (
            f"Mean brightness (V): {mean_v:.1f}/255.  "
            f"Dark ratio: {dark_ratio:.2%}.  "
            f"Bright ratio: {bright_ratio:.2%}.  "
            f"Illumination std: {uneven_std:.1f}.  "
            + ("Issues found." if issues else "Lighting is acceptable.")
        )

        return {
            "status":            status,
            "score":             round(score, 4),
            "mean_brightness":   round(mean_v, 2),
            "dark_pixel_ratio":  round(dark_ratio, 4),
            "bright_pixel_ratio": round(bright_ratio, 4),
            "illumination_std":  round(uneven_std, 2),
            "issues":            issues,
            "explanation":       explanation,
        }

    # ── Private ─────────────────────────────────────────────────────────────

    def _patch_illumination_std(self, v: np.ndarray, grid: int = 4) -> float:
        """
        Divide the V-channel into grid × grid patches.
        Return the std of the per-patch mean brightness.
        High std → spatially non-uniform illumination.
        """
        h, w = v.shape
        ph, pw = h // grid, w // grid
        if ph == 0 or pw == 0:
            return 0.0

        means: List[float] = []
        for i in range(grid):
            for j in range(grid):
                patch = v[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
                means.append(float(np.mean(patch)))

        return float(np.std(means))

    def _compute_score(
        self,
        dark_ratio: float,
        bright_ratio: float,
        uneven_std: float,
    ) -> float:
        """
        Penalty-based score.  Each issue reduces the score independently.
        Returns a value in [0.0, 1.0].
        """
        # Normalise each metric against its threshold, cap at 1.0
        dark_penalty   = min(dark_ratio   / max(self.cfg["dark_pixel_ratio"],   1e-9), 1.0)
        bright_penalty = min(bright_ratio / max(self.cfg["bright_pixel_ratio"], 1e-9), 1.0)
        uneven_penalty = min(uneven_std   / max(self.cfg["uneven_std_threshold"] * 2, 1e-9), 1.0)

        # Weighted contribution of each penalty
        total_penalty = 0.35 * dark_penalty + 0.35 * bright_penalty + 0.30 * uneven_penalty
        return max(1.0 - total_penalty, 0.0)
