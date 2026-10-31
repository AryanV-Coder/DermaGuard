"""
Obstruction Agent — detects hair and image artifacts in dermoscopic images.

Two complementary techniques are used:

1. Black-hat morphological transform
   Highlights dark, thin structures (hair) against a lighter background.
   The kernel size is chosen to be larger than typical hair width.

2. Connected-component analysis on Canny edges
   Detects large, irregular blobs that may indicate ruler markers,
   stickers, ink marks, or other non-biological artifacts.

⚠️  These are heuristic, approximate methods.  They will not catch every
    obstruction, and may occasionally raise false positives.  Validate
    thoroughly on real dermoscopic images before clinical use.
"""

from typing import Any, Dict, List

import cv2
import numpy as np

try:
    from config.thresholds import OBSTRUCTION
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config.thresholds import OBSTRUCTION


class ObstructionAgent:
    """
    Detects hair coverage and suspicious artifacts using morphological
    operations and connected-component analysis.
    """

    def __init__(self):
        self.cfg = OBSTRUCTION

    # ── Public API ──────────────────────────────────────────────────────────

    def assess(self, gray: np.ndarray, rgb: np.ndarray) -> Dict[str, Any]:
        """
        Parameters
        ----------
        gray : np.ndarray (H, W) uint8
        rgb  : np.ndarray (H, W, 3) uint8  (kept for potential future use)

        Returns
        -------
        {
            status          : "PASS" | "FAIL"
            score           : float 0.0 – 1.0
            hair_pixel_count: int
            hair_ratio      : float
            artifact_ratio  : float
            is_severe       : bool   (True → hard FAIL in fusion)
            issues          : List[str]
            explanation     : str
        }
        """
        total_px = gray.shape[0] * gray.shape[1]

        # ── Hair detection ────────────────────────────────────────────────
        hair_mask = self._detect_hair(gray)
        hair_px   = int(np.sum(hair_mask))
        hair_ratio = hair_px / max(total_px, 1)

        # ── Artifact detection ────────────────────────────────────────────
        artifact_ratio = self._detect_artifacts(gray, total_px)

        # ── Issue classification ──────────────────────────────────────────
        is_severe = hair_ratio > self.cfg["severe_hair_ratio"]
        issues: List[str] = []

        if is_severe:
            issues.append(
                f"Severe hair obstruction: {hair_ratio * 100:.1f}% of image covered "
                f"(severe limit {self.cfg['severe_hair_ratio'] * 100:.0f}%)."
            )
        elif hair_ratio > self.cfg["hair_pixel_ratio"]:
            issues.append(
                f"Hair obstruction: {hair_ratio * 100:.1f}% of image affected "
                f"(limit {self.cfg['hair_pixel_ratio'] * 100:.0f}%)."
            )

        if artifact_ratio > self.cfg["artifact_area_ratio"]:
            issues.append(
                f"Suspicious artifact region: {artifact_ratio * 100:.1f}% of image area."
            )

        score  = self._compute_score(hair_ratio, artifact_ratio)
        status = "FAIL" if issues else "PASS"

        explanation = (
            f"Hair pixels: {hair_px} ({hair_ratio:.2%}).  "
            f"Artifact area ratio: {artifact_ratio:.2%}.  "
            + ("No significant obstruction." if not issues else "Obstruction detected.")
        )

        return {
            "status":            status,
            "score":             round(score, 4),
            "hair_pixel_count":  hair_px,
            "hair_ratio":        round(hair_ratio, 4),
            "artifact_ratio":    round(artifact_ratio, 4),
            "is_severe":         is_severe,
            "issues":            issues,
            "explanation":       explanation,
        }

    # ── Private ─────────────────────────────────────────────────────────────

    def _detect_hair(self, gray: np.ndarray) -> np.ndarray:
        """
        Black-hat morphological transform:
          result = closing(src) − src
        Highlights dark structures (hair) smaller than the kernel.

        Returns a boolean mask.
        """
        ksize  = self.cfg["blackhat_kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        _, hair_mask = cv2.threshold(
            blackhat,
            self.cfg["blackhat_threshold"],
            255,
            cv2.THRESH_BINARY,
        )

        # Remove tiny noise clusters (keep elongated hair-like structures)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, clean_kernel)
        return hair_mask > 0

    def _detect_artifacts(self, gray: np.ndarray, total_px: int) -> float:
        """
        Use Canny edge detection + dilation + connected-component analysis
        to identify large, irregular blobs that may be non-biological artifacts.

        Returns the fraction of the image occupied by suspicious components.
        """
        edges  = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        # An "artifact" component is one that covers more than artifact_area_ratio
        # of the total image on its own.
        component_threshold = total_px * self.cfg["artifact_area_ratio"]
        suspicious_area = sum(
            stats[label, cv2.CC_STAT_AREA]
            for label in range(1, num_labels)           # label 0 = background
            if stats[label, cv2.CC_STAT_AREA] > component_threshold
        )
        return min(suspicious_area / max(total_px, 1), 1.0)

    def _compute_score(self, hair_ratio: float, artifact_ratio: float) -> float:
        """
        Score in [0.0, 1.0].
        Hair coverage contributes 70 % of the penalty; artifacts 30 %.
        """
        hair_penalty     = min(hair_ratio     / max(self.cfg["severe_hair_ratio"],    1e-9), 1.0)
        artifact_penalty = min(artifact_ratio / max(self.cfg["artifact_area_ratio"] * 5, 1e-9), 1.0)
        total_penalty    = 0.70 * hair_penalty + 0.30 * artifact_penalty
        return max(1.0 - total_penalty, 0.0)
