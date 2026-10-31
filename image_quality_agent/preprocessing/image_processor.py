"""
Image preprocessing module for the Image Quality Agent.

Responsibilities:
  - Accept a file path or numpy BGR array
  - Validate the file / array
  - Convert to RGB, grayscale, and HSV colour spaces
  - Resize to a standard long-side length (aspect ratio preserved)
  - Return a ready-to-use dict for all downstream assessment agents
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default max dimension for internal processing
_DEFAULT_LONG_SIDE = 512
_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ImageProcessor:
    """
    Loads, validates, and prepares a dermoscopic image for quality assessment.

    Parameters
    ----------
    target_long_side : int
        Images larger than this (on their longest side) are down-scaled
        proportionally before processing.  Set to a large value to disable.
    """

    def __init__(self, target_long_side: int = _DEFAULT_LONG_SIDE):
        self.target_long_side = target_long_side

    # ── Public API ──────────────────────────────────────────────────────────

    def load(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Load and preprocess an image.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path or a pre-loaded numpy BGR image.

        Returns
        -------
        dict with keys:
            valid          : bool
            error          : str | None          — set when valid is False
            rgb            : np.ndarray (H,W,3)  uint8
            gray           : np.ndarray (H,W)    uint8
            hsv            : np.ndarray (H,W,3)  uint8
            original_shape : Tuple[int,int,int]  — shape before resizing
        """
        result: Dict[str, Any] = {
            "valid": False,
            "error": None,
            "rgb": None,
            "gray": None,
            "hsv": None,
            "original_shape": None,
        }

        # ── 1. Acquire raw BGR array ────────────────────────────────────────
        if isinstance(source, np.ndarray):
            bgr = source.copy()
        else:
            bgr = self._load_from_path(source)
            if bgr is None:
                result["error"] = f"Could not read image from: {source}"
                return result

        # ── 2. Basic shape validation ───────────────────────────────────────
        if bgr is None or bgr.size == 0 or bgr.ndim not in (2, 3):
            result["error"] = "Image array is empty or has an unsupported shape."
            return result

        # Ensure 3-channel BGR
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

        result["original_shape"] = bgr.shape  # (H, W, 3)

        # ── 3. Resize ───────────────────────────────────────────────────────
        bgr_resized = self._resize(bgr)

        # ── 4. Colour-space conversions ─────────────────────────────────────
        result["rgb"]  = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
        result["gray"] = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2GRAY)
        result["hsv"]  = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2HSV)
        result["valid"] = True
        return result

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load_from_path(self, path: Union[str, Path]) -> Optional[np.ndarray]:
        path = Path(path)
        if not path.exists():
            logger.error("File not found: %s", path)
            return None
        if path.suffix.lower() not in _SUPPORTED_EXTS:
            logger.error("Unsupported file extension: %s", path.suffix)
            return None
        img = cv2.imread(str(path))
        if img is None:
            logger.error("OpenCV failed to decode image: %s", path)
        return img

    def _resize(self, bgr: np.ndarray) -> np.ndarray:
        """Down-scale so the longest side equals target_long_side (no up-scaling)."""
        h, w = bgr.shape[:2]
        long_side = max(h, w)
        if long_side <= self.target_long_side:
            return bgr
        scale = self.target_long_side / long_side
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
