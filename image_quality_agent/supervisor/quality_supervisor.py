"""
Image Quality Supervisor — top-level orchestrator for the quality pipeline.

Coordinates:
  1. Image preprocessing (ImageProcessor)
  2. Four independent assessment agents
     - BlurAgent
     - LightingAgent
     - GlareAgent
     - ObstructionAgent
  3. Quality Fusion Engine (weighted score + hard-FAIL logic)

Returns a single, JSON-serialisable quality report.

Usage
-----
    from supervisor.quality_supervisor import QualitySupervisor
    import json

    supervisor = QualitySupervisor()
    report = supervisor.evaluate("path/to/lesion.jpg")
    print(json.dumps(report, indent=2))
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

try:
    from preprocessing.image_processor import ImageProcessor
    from agents.blur_agent         import BlurAgent
    from agents.lighting_agent     import LightingAgent
    from agents.glare_agent        import GlareAgent
    from agents.obstruction_agent  import ObstructionAgent
    from fusion.quality_fusion     import QualityFusionEngine
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from preprocessing.image_processor import ImageProcessor
    from agents.blur_agent         import BlurAgent
    from agents.lighting_agent     import LightingAgent
    from agents.glare_agent        import GlareAgent
    from agents.obstruction_agent  import ObstructionAgent
    from fusion.quality_fusion     import QualityFusionEngine


class QualitySupervisor:
    """
    Orchestrates the full Image Quality Agent pipeline.

    All sub-components are instantiated once and reused across multiple
    calls to evaluate(), making the supervisor efficient in batch scenarios.
    """

    def __init__(self):
        self.processor          = ImageProcessor()
        self.blur_agent         = BlurAgent()
        self.lighting_agent     = LightingAgent()
        self.glare_agent        = GlareAgent()
        self.obstruction_agent  = ObstructionAgent()
        self.fusion_engine      = QualityFusionEngine()

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run the complete quality assessment pipeline on one image.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path to an image, or a pre-loaded numpy BGR array.

        Returns
        -------
        A fully-structured, JSON-serialisable quality report:
        {
            image: {
                source         : str
                original_shape : [H, W, C]
            }
            overall: {
                status             : "PASS" | "FAIL" | "ERROR"
                score              : float   0.0 – 1.0
                recommendation     : str
                hard_fail_triggered: bool
                hard_fail_reason   : str | None
            }
            agent_scores   : { blur, lighting, glare, obstruction }  — weighted contributions
            agent_statuses : { blur, lighting, glare, obstruction }  — raw PASS/FAIL
            details: {
                blur, lighting, glare, obstruction               — full agent output
            }
            processing_time_s : float
        }
        """
        t_start = time.perf_counter()

        # ── Step 1: Preprocessing ────────────────────────────────────────────
        img_data = self.processor.load(source)
        if not img_data["valid"]:
            return self._error_report(source, img_data["error"])

        rgb  = img_data["rgb"]
        gray = img_data["gray"]
        hsv  = img_data["hsv"]

        # ── Step 2: Individual agent assessments ─────────────────────────────
        blur_result        = self.blur_agent.assess(gray)
        lighting_result    = self.lighting_agent.assess(hsv)
        glare_result       = self.glare_agent.assess(hsv)
        obstruction_result = self.obstruction_agent.assess(gray, rgb)

        # ── Step 3: Fusion ───────────────────────────────────────────────────
        fusion_result = self.fusion_engine.fuse(
            blur_result,
            lighting_result,
            glare_result,
            obstruction_result,
        )

        elapsed = round(time.perf_counter() - t_start, 4)

        # ── Step 4: Assemble final report ────────────────────────────────────
        source_str = (
            str(source)
            if not isinstance(source, np.ndarray)
            else "<numpy array>"
        )

        return {
            "image": {
                "source":         source_str,
                "original_shape": list(img_data["original_shape"]),
            },
            "overall": {
                "status":              fusion_result["overall_status"],
                "score":               fusion_result["overall_score"],
                "recommendation":      fusion_result["recommendation"],
                "hard_fail_triggered": fusion_result["hard_fail_triggered"],
                "hard_fail_reason":    fusion_result["hard_fail_reason"],
            },
            "agent_scores":   fusion_result["weighted_scores"],
            "agent_statuses": fusion_result["agent_statuses"],
            "details": {
                "blur":        blur_result,
                "lighting":    lighting_result,
                "glare":       glare_result,
                "obstruction": obstruction_result,
            },
            "processing_time_s": elapsed,
        }

    def evaluate_and_print(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Convenience wrapper: evaluate + pretty-print JSON to stdout."""
        report = self.evaluate(source)
        print(json.dumps(report, indent=2))
        return report

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _error_report(
        source: Union[str, Path, np.ndarray],
        error:  str,
    ) -> Dict[str, Any]:
        source_str = str(source) if not isinstance(source, np.ndarray) else "<numpy array>"
        return {
            "image":   {"source": source_str, "original_shape": None},
            "overall": {
                "status":              "ERROR",
                "score":               0.0,
                "recommendation":      f"Could not process image: {error}",
                "hard_fail_triggered": False,
                "hard_fail_reason":    None,
            },
            "agent_scores":    {},
            "agent_statuses":  {},
            "details":         {},
            "processing_time_s": 0.0,
        }
