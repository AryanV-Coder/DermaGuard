"""
Agent 3: Lesion Analysis Agent
==============================
Analyzes visual dermoscopic features following clinical ABCD criteria:
  - Asymmetry (A)
  - Border irregularity (B)
  - Color variation / variegation (C)
  - Structural / Diameter cues (D)

Outputs:
  - feature_evidence: dictionary of measured quantitative features
  - suspicion_score: float in [0.0, 1.0] indicating morphological risk
  - suspicion_level: 'LOW', 'MODERATE', 'HIGH'
  - reliability_flags: list of warnings (e.g. border cutoff, tiny lesion)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from lesion_analysis_agent.unet import UNetSegmenter
except ImportError:
    from unet import UNetSegmenter


class LesionAnalysisAgent:
    """
    Extracts clinical morphometric cues from dermoscopic images.
    Uses U-Net deep learning segmentation with OpenCV ABCD feature extraction.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        asymmetry_weight: float = 0.35,
        border_weight: float = 0.35,
        color_weight: float = 0.30,
        high_suspicion_threshold: float = 0.60,
        moderate_suspicion_threshold: float = 0.35,
    ):
        self.w_a = asymmetry_weight
        self.w_b = border_weight
        self.w_c = color_weight
        self.high_thresh = high_suspicion_threshold
        self.mod_thresh = moderate_suspicion_threshold
        self.segmenter = UNetSegmenter(model_path=model_path)

    def analyze(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a skin lesion image following ABCD criteria.
        
        Parameters
        ----------
        source : str | Path | np.ndarray (RGB or BGR image)

        Returns
        -------
        dict with:
          - feature_evidence: Dict[str, Any]
          - suspicion_score: float (0.0 to 1.0)
          - suspicion_level: "LOW" | "MODERATE" | "HIGH"
          - reliability_flags: List[str]
          - segmentation_success: bool
        """
        img_bgr = self._load_image(source)
        if img_bgr is None:
            return self._fallback_result(["Failed to load or decode image"])

        h, w = img_bgr.shape[:2]
        reliability_flags: List[str] = []

        # 1. Segment lesion using U-Net (with adaptive Otsu fallback)
        mask, seg_flags, seg_backend = self.segmenter.segment(img_bgr)
        reliability_flags.extend(seg_flags)

        # Check border touch (lesion truncated by image boundary)
        border_pixels = np.concatenate([
            mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]
        ])
        if np.any(border_pixels > 0):
            reliability_flags.append("Lesion touches image border (possible truncation)")

        lesion_area = int(np.count_nonzero(mask))
        total_area = h * w
        area_ratio = lesion_area / total_area if total_area > 0 else 0.0

        if lesion_area < 50:
            reliability_flags.append("Lesion area too small for reliable ABCD analysis")
            return {
                "feature_evidence": {
                    "asymmetry_index": 0.0,
                    "border_irregularity": 0.0,
                    "color_variation": 0.0,
                    "lesion_area_fraction": round(area_ratio, 4),
                    "equivalent_diameter_px": 0.0,
                    "segmentation_backend": seg_backend,
                },
                "suspicion_score": 0.20,
                "suspicion_level": "LOW",
                "reliability_flags": reliability_flags,
                "segmentation_success": False,
            }

        # 2. Asymmetry (A)
        asymmetry_score, a_flags = self._compute_asymmetry(mask)
        reliability_flags.extend(a_flags)

        # 3. Border irregularity (B)
        border_score, b_flags = self._compute_border_irregularity(mask)
        reliability_flags.extend(b_flags)

        # 4. Color variation (C)
        color_score, c_flags = self._compute_color_variegation(img_bgr, mask)
        reliability_flags.extend(c_flags)

        # 5. Diameter / Scale (D)
        eq_diameter = float(np.sqrt(4.0 * lesion_area / np.pi))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_dim = 0.0
        if contours:
            _, _, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            max_dim = float(max(bw, bh))

        # 6. Composite suspicion score
        raw_suspicion = (
            self.w_a * asymmetry_score +
            self.w_b * border_score +
            self.w_c * color_score
        )
        suspicion_score = float(np.clip(raw_suspicion, 0.0, 1.0))

        if suspicion_score >= self.high_thresh:
            suspicion_level = "HIGH"
        elif suspicion_score >= self.mod_thresh:
            suspicion_level = "MODERATE"
        else:
            suspicion_level = "LOW"

        feature_evidence = {
            "asymmetry_index": round(float(asymmetry_score), 4),
            "border_irregularity": round(float(border_score), 4),
            "color_variation": round(float(color_score), 4),
            "lesion_area_fraction": round(float(area_ratio), 4),
            "equivalent_diameter_px": round(eq_diameter, 2),
            "max_dimension_px": round(max_dim, 2),
            "segmentation_backend": seg_backend,
        }

        return {
            "feature_evidence": feature_evidence,
            "suspicion_score": round(suspicion_score, 4),
            "suspicion_level": suspicion_level,
            "reliability_flags": reliability_flags,
            "segmentation_success": True,
        }

    # ───────────────── Helper Methods ─────────────────

    @staticmethod
    def _load_image(source: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            return img
        elif isinstance(source, np.ndarray):
            if source.ndim == 3 and source.shape[2] == 3:
                return source.copy()
        return None

    def _segment_lesion(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Segments the darker skin lesion from background skin using Otsu on inverted blue/gray channel.
        """
        flags = []
        h, w = img_bgr.shape[:2]

        # Use Blue channel or grayscale because melanin strongly absorbs blue
        b_channel = img_bgr[:, :, 0]
        blurred = cv2.GaussianBlur(b_channel, (11, 11), 0)

        # Otsu thresholding (lesion is darker than normal skin)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to clean hair/noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Keep connected component closest to center
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((h, w), dtype=np.uint8), ["No contour found in segmentation"]

        center = (w / 2.0, h / 2.0)
        best_cnt = None
        min_dist = float("inf")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                dist = np.hypot(cx - center[0], cy - center[1])
                # Prioritize central lesions with decent size
                weighted_dist = dist / np.sqrt(area)
                if weighted_dist < min_dist:
                    min_dist = weighted_dist
                    best_cnt = cnt

        if best_cnt is None:
            best_cnt = max(contours, key=cv2.contourArea)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [best_cnt], -1, 255, -1)

        # Check border touch (lesion truncated by image boundary)
        border_pixels = np.concatenate([
            mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]
        ])
        if np.any(border_pixels > 0):
            flags.append("Lesion touches image border (possible truncation)")

        return mask, flags

    def _compute_asymmetry(self, mask: np.ndarray) -> Tuple[float, List[str]]:
        """
        Computes asymmetry along principal inertia axes of the binary lesion mask.
        """
        flags = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, ["Cannot compute asymmetry (empty mask)"]

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area <= 0:
            return 0.0, []

        # Find orientation via PCA / minAreaRect
        rect = cv2.minAreaRect(cnt)
        center, size, angle = rect

        # Rotate mask to align principal axes with coordinate axes
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        h, w = mask.shape
        rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        # Flip horizontally and vertically relative to center
        # Bounding box around rotated lesion
        r_cnts, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not r_cnts:
            return 0.0, []
        x, y, bw, bh = cv2.boundingRect(max(r_cnts, key=cv2.contourArea))
        crop = rotated[y:y+bh, x:x+bw]

        if crop.size == 0 or np.count_nonzero(crop) == 0:
            return 0.0, []

        # Compare crop with its flipped versions
        flip_h = cv2.flip(crop, 1)
        flip_v = cv2.flip(crop, 0)

        diff_h = np.logical_xor(crop > 0, flip_h > 0)
        diff_v = np.logical_xor(crop > 0, flip_v > 0)

        crop_area = np.count_nonzero(crop > 0)
        asym_h = np.count_nonzero(diff_h) / (2.0 * crop_area) if crop_area > 0 else 0.0
        asym_v = np.count_nonzero(diff_v) / (2.0 * crop_area) if crop_area > 0 else 0.0

        asymmetry_index = float(np.clip((asym_h + asym_v) / 2.0 * 2.0, 0.0, 1.0))
        return asymmetry_index, flags

    def _compute_border_irregularity(self, mask: np.ndarray) -> Tuple[float, List[str]]:
        """
        Computes border irregularity using compactness / isoperimetric quotient:
        Circular lesion = 1.0 (lowest irregularity)
        Irregular border gives high perimeter relative to area.
        """
        flags = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, ["Cannot compute border (empty mask)"]

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area <= 0 or perimeter <= 0:
            return 0.0, []

        # Circularity = 4 * pi * Area / Perimeter^2 (1.0 for perfect circle, < 0.5 for jagged/irregular)
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        # Invert to define irregularity: 0.0 = smooth/circular, 1.0 = highly irregular
        irregularity = float(np.clip(1.0 - circularity, 0.0, 1.0))

        # Also measure hull indentation (convex hull ratio)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0
        indentation = float(np.clip(1.0 - solidity, 0.0, 1.0))

        border_score = 0.6 * irregularity + 0.4 * indentation
        return float(np.clip(border_score, 0.0, 1.0)), flags

    def _compute_color_variegation(self, img_bgr: np.ndarray, mask: np.ndarray) -> Tuple[float, List[str]]:
        """
        Measures intra-lesion color variation in RGB and HSV color space.
        Higher variance / spread of distinct shades increases suspicion.
        """
        flags = []
        lesion_pixels_bgr = img_bgr[mask > 0]
        if len(lesion_pixels_bgr) < 50:
            return 0.0, []

        # Convert to HSV to check Hue/Saturation spread
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lesion_pixels_hsv = img_hsv[mask > 0]

        # Standard deviations in R, G, B, and Value
        std_bgr = np.std(lesion_pixels_bgr, axis=0)  # [std_b, std_g, std_r]
        std_val = np.std(lesion_pixels_hsv[:, 2])

        avg_std = np.mean(std_bgr)
        # Normalization (typical dermoscopic std ranges between 10 and 55)
        norm_std = np.clip((avg_std - 12.0) / 38.0, 0.0, 1.0)
        norm_val = np.clip((std_val - 15.0) / 40.0, 0.0, 1.0)

        color_score = float(np.clip(0.6 * norm_std + 0.4 * norm_val, 0.0, 1.0))
        return color_score, flags

    @staticmethod
    def _fallback_result(flags: List[str]) -> Dict[str, Any]:
        return {
            "feature_evidence": {
                "asymmetry_index": 0.0,
                "border_irregularity": 0.0,
                "color_variation": 0.0,
                "lesion_area_fraction": 0.0,
            },
            "suspicion_score": 0.0,
            "suspicion_level": "UNKNOWN",
            "reliability_flags": flags,
            "segmentation_success": False,
        }

