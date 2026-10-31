"""
Visualization utilities for the Image Quality Agent.

Provides helpers to overlay quality assessment results on images for
debugging, reporting, and manual validation.

Functions
---------
draw_quality_report  — Annotate an RGB image with the overall report.
draw_glare_mask      — Overlay glare regions in red.
draw_hair_mask       — Overlay hair/obstruction regions in yellow.
save_annotated       — Save an annotated image to disk.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


# ── Internal helpers ──────────────────────────────────────────────────────────

def _wrap_text(text: str, max_chars: int = 55) -> list[str]:
    """Wrap text at word boundaries."""
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 > max_chars:
            if line:
                lines.append(line)
            line = word
        else:
            line += (" " if line else "") + word
    if line:
        lines.append(line)
    return lines


# ── Public API ────────────────────────────────────────────────────────────────

def draw_quality_report(
    rgb: np.ndarray,
    report: Dict[str, Any],
    scale: float = 1.0,
) -> np.ndarray:
    """
    Overlay a quality summary panel below the image.

    Parameters
    ----------
    rgb    : RGB image (H, W, 3) uint8.
    report : Output of QualitySupervisor.evaluate().
    scale  : Optional display scale factor.

    Returns
    -------
    Annotated BGR image (image + panel stacked vertically).
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    h, w = bgr.shape[:2]

    overall  = report.get("overall", {})
    status   = overall.get("status", "UNKNOWN")
    score    = overall.get("score", 0.0)
    statuses = report.get("agent_statuses", {})
    rec      = overall.get("recommendation", "")

    # Status colour: green for PASS, red for FAIL/ERROR
    colour = (0, 200, 80) if status == "PASS" else (30, 60, 230)

    # Panel dimensions
    panel_h = 50 + len(statuses) * 22 + len(_wrap_text(rec)) * 18 + 20
    panel   = np.zeros((panel_h, w, 3), dtype=np.uint8)

    # Overall status + score
    cv2.putText(
        panel,
        f"Quality: {status}   Score: {score:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2,
    )

    # Per-agent statuses
    y = 55
    for agent, st in statuses.items():
        c = (0, 200, 80) if st == "PASS" else (30, 60, 230)
        cv2.putText(panel, f"  {agent}: {st}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        y += 22

    # Recommendation (word-wrapped)
    for ln in _wrap_text(rec):
        cv2.putText(panel, ln, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y += 18

    return np.vstack([bgr, panel])


def draw_glare_mask(rgb: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
    """
    Overlay glare regions in semi-transparent red.

    Parameters
    ----------
    rgb        : RGB image (H, W, 3) uint8.
    glare_mask : Boolean mask (H, W).

    Returns BGR image.
    """
    bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[glare_mask] = [0, 0, 220]
    return cv2.addWeighted(bgr, 0.65, overlay, 0.35, 0)


def draw_hair_mask(rgb: np.ndarray, hair_mask: np.ndarray) -> np.ndarray:
    """
    Overlay hair / obstruction regions in semi-transparent yellow.

    Parameters
    ----------
    rgb       : RGB image (H, W, 3) uint8.
    hair_mask : Boolean mask (H, W).

    Returns BGR image.
    """
    bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[hair_mask] = [0, 220, 220]
    return cv2.addWeighted(bgr, 0.65, overlay, 0.35, 0)


def save_annotated(image: np.ndarray, output_path: str) -> None:
    """
    Save an annotated BGR image to disk.

    Parameters
    ----------
    image       : BGR image array.
    output_path : Destination file path (will overwrite if exists).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
