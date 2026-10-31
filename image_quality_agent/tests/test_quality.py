"""
Unit tests for the Image Quality Agent.

Tests every assessment module independently using synthetic images.
No real dermoscopy dataset required.

Run:
    cd image_quality_agent
    python -m pytest tests/test_quality.py -v

Or from the repo root:
    python -m pytest DermaGuard/image_quality_agent/tests/test_quality.py -v
"""

import os
import sys

# Ensure the package root is on the path regardless of how pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest

from preprocessing.image_processor import ImageProcessor
from agents.blur_agent              import BlurAgent
from agents.lighting_agent          import LightingAgent
from agents.glare_agent             import GlareAgent
from agents.obstruction_agent       import ObstructionAgent
from fusion.quality_fusion          import QualityFusionEngine
from supervisor.quality_supervisor  import QualitySupervisor


# ─── Synthetic image factories ────────────────────────────────────────────────

def sharp_gray(h: int = 256, w: int = 256) -> np.ndarray:
    """High-frequency checkerboard — high Laplacian variance."""
    img = np.zeros((h, w), dtype=np.uint8)
    block = 8
    for i in range(0, h, block):
        for j in range(0, w, block):
            if (i // block + j // block) % 2 == 0:
                img[i : i + block, j : j + block] = 255
    return img


def blurry_gray(h: int = 256, w: int = 256, ksize: int = 51) -> np.ndarray:
    return cv2.GaussianBlur(sharp_gray(h, w), (ksize, ksize), 0)


def dark_gray(value: int = 15, h: int = 256, w: int = 256) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def bright_gray(value: int = 245, h: int = 256, w: int = 256) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def mid_gray(value: int = 128, h: int = 256, w: int = 256) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def skin_rgb(h: int = 256, w: int = 256) -> np.ndarray:
    """Brownish skin tone — no glare."""
    return np.full((h, w, 3), [110, 75, 55], dtype=np.uint8)


def glare_rgb(frac: float = 0.20, h: int = 256, w: int = 256) -> np.ndarray:
    """Skin image with a white glare patch covering frac of area."""
    img = skin_rgb(h, w)
    side = max(int((h * w * frac) ** 0.5), 1)
    cx, cy = w // 2, h // 2
    half = side // 2
    img[cy - half : cy + half, cx - half : cx + half] = [255, 255, 255]
    return img


def hairy_gray(n: int = 30, h: int = 256, w: int = 256) -> np.ndarray:
    """Mid-grey image with n synthetic dark hair-like lines."""
    img = mid_gray(h=h, w=w)
    rng = np.random.default_rng(0)
    for _ in range(n):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, 5, thickness=2)
    return img


# ─── Colour-space helpers ─────────────────────────────────────────────────────

def gray_to_hsv(gray: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# ═════════════════════════════════════════════════════════════════════════════
# ImageProcessor tests
# ═════════════════════════════════════════════════════════════════════════════

class TestImageProcessor:

    def setup_method(self):
        self.proc = ImageProcessor()

    def test_valid_bgr_array(self):
        bgr = np.zeros((300, 400, 3), dtype=np.uint8)
        r = self.proc.load(bgr)
        assert r["valid"] is True
        assert r["rgb"] is not None
        assert r["gray"] is not None
        assert r["hsv"] is not None

    def test_invalid_path_returns_error(self):
        r = self.proc.load("/no/such/file.jpg")
        assert r["valid"] is False
        assert r["error"] is not None

    def test_large_image_is_resized(self):
        bgr = np.zeros((1024, 768, 3), dtype=np.uint8)
        r = self.proc.load(bgr)
        assert r["valid"] is True
        h, w = r["rgb"].shape[:2]
        assert max(h, w) <= self.proc.target_long_side

    def test_aspect_ratio_preserved(self):
        # 2:1 wide image
        bgr = np.zeros((256, 512, 3), dtype=np.uint8)
        r = self.proc.load(bgr)
        assert r["valid"] is True
        h, w = r["rgb"].shape[:2]
        # ratio should remain close to 0.5
        assert abs(h / w - 0.5) < 0.05

    def test_grayscale_input_accepted(self):
        gray = np.zeros((256, 256), dtype=np.uint8)
        r = self.proc.load(gray)
        assert r["valid"] is True
        assert r["rgb"].shape[2] == 3

    def test_original_shape_recorded(self):
        bgr = np.zeros((200, 300, 3), dtype=np.uint8)
        r = self.proc.load(bgr)
        assert r["original_shape"] == (200, 300, 3)

    def test_small_image_not_upscaled(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        r = self.proc.load(bgr)
        h, w = r["rgb"].shape[:2]
        assert h == 64 and w == 64


# ═════════════════════════════════════════════════════════════════════════════
# BlurAgent tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBlurAgent:

    def setup_method(self):
        self.agent = BlurAgent()

    def test_sharp_image_passes(self):
        r = self.agent.assess(sharp_gray())
        assert r["status"] == "PASS"

    def test_blurry_image_fails(self):
        r = self.agent.assess(blurry_gray(ksize=71))
        assert r["status"] == "FAIL"

    def test_severe_blur_sets_flag(self):
        r = self.agent.assess(blurry_gray(ksize=101))
        assert r["is_severe"] is True

    def test_sharp_not_severe(self):
        r = self.agent.assess(sharp_gray())
        assert r["is_severe"] is False

    def test_score_between_0_and_1(self):
        for g in [sharp_gray(), blurry_gray()]:
            r = self.agent.assess(g)
            assert 0.0 <= r["score"] <= 1.0

    def test_sharp_score_gt_blurry(self):
        sharp_score = self.agent.assess(sharp_gray())["score"]
        blurry_score = self.agent.assess(blurry_gray(ksize=71))["score"]
        assert sharp_score > blurry_score

    def test_laplacian_var_returned(self):
        r = self.agent.assess(sharp_gray())
        assert "laplacian_var" in r
        assert r["laplacian_var"] > 0

    def test_issues_list_empty_on_pass(self):
        r = self.agent.assess(sharp_gray())
        assert r["issues"] == []

    def test_issues_list_nonempty_on_fail(self):
        r = self.agent.assess(blurry_gray(ksize=71))
        assert len(r["issues"]) > 0

    def test_explanation_present(self):
        r = self.agent.assess(sharp_gray())
        assert isinstance(r["explanation"], str) and len(r["explanation"]) > 0


# ═════════════════════════════════════════════════════════════════════════════
# LightingAgent tests
# ═════════════════════════════════════════════════════════════════════════════

class TestLightingAgent:

    def setup_method(self):
        self.agent = LightingAgent()

    def test_normal_brightness_passes(self):
        r = self.agent.assess(gray_to_hsv(mid_gray()))
        assert r["status"] == "PASS"

    def test_dark_image_fails(self):
        r = self.agent.assess(gray_to_hsv(dark_gray()))
        assert r["status"] == "FAIL"
        assert any("dark" in i.lower() or "underexposed" in i.lower() for i in r["issues"])

    def test_bright_image_fails(self):
        r = self.agent.assess(gray_to_hsv(bright_gray()))
        assert r["status"] == "FAIL"
        assert any("overexposed" in i.lower() or "bright" in i.lower() for i in r["issues"])

    def test_score_between_0_and_1(self):
        for g in [dark_gray(), bright_gray(), mid_gray()]:
            r = self.agent.assess(gray_to_hsv(g))
            assert 0.0 <= r["score"] <= 1.0

    def test_normal_has_higher_score_than_dark(self):
        normal = self.agent.assess(gray_to_hsv(mid_gray()))["score"]
        dark   = self.agent.assess(gray_to_hsv(dark_gray()))["score"]
        assert normal > dark

    def test_measurements_present(self):
        r = self.agent.assess(gray_to_hsv(mid_gray()))
        for key in ("mean_brightness", "dark_pixel_ratio", "bright_pixel_ratio", "illumination_std"):
            assert key in r

    def test_mean_brightness_range(self):
        r = self.agent.assess(gray_to_hsv(mid_gray()))
        assert 0.0 <= r["mean_brightness"] <= 255.0

    def test_issues_empty_on_pass(self):
        r = self.agent.assess(gray_to_hsv(mid_gray()))
        assert r["issues"] == []


# ═════════════════════════════════════════════════════════════════════════════
# GlareAgent tests
# ═════════════════════════════════════════════════════════════════════════════

class TestGlareAgent:

    def setup_method(self):
        self.agent = GlareAgent()

    def test_clean_skin_passes(self):
        r = self.agent.assess(rgb_to_hsv(skin_rgb()))
        assert r["status"] == "PASS"

    def test_large_glare_patch_fails(self):
        r = self.agent.assess(rgb_to_hsv(glare_rgb(frac=0.30)))
        assert r["status"] == "FAIL"

    def test_severe_glare_sets_flag(self):
        r = self.agent.assess(rgb_to_hsv(glare_rgb(frac=0.60)))
        assert r["is_severe"] is True

    def test_small_glare_passes(self):
        # 1 % glare — below the 5 % threshold
        r = self.agent.assess(rgb_to_hsv(glare_rgb(frac=0.01)))
        assert r["status"] == "PASS"

    def test_score_between_0_and_1(self):
        for frac in (0.01, 0.10, 0.50):
            r = self.agent.assess(rgb_to_hsv(glare_rgb(frac=frac)))
            assert 0.0 <= r["score"] <= 1.0

    def test_more_glare_lower_score(self):
        s_low  = self.agent.assess(rgb_to_hsv(glare_rgb(frac=0.01)))["score"]
        s_high = self.agent.assess(rgb_to_hsv(glare_rgb(frac=0.60)))["score"]
        assert s_low > s_high

    def test_glare_ratio_returned(self):
        r = self.agent.assess(rgb_to_hsv(glare_rgb()))
        assert "glare_ratio" in r
        assert r["glare_ratio"] >= 0

    def test_glare_pixel_count_nonnegative(self):
        r = self.agent.assess(rgb_to_hsv(skin_rgb()))
        assert r["glare_pixel_count"] >= 0


# ═════════════════════════════════════════════════════════════════════════════
# ObstructionAgent tests
# ═════════════════════════════════════════════════════════════════════════════

class TestObstructionAgent:

    def setup_method(self):
        self.agent = ObstructionAgent()

    def test_clean_image_passes(self):
        g = mid_gray()
        r = self.agent.assess(g, gray_to_rgb(g))
        assert r["status"] == "PASS"

    def test_result_has_required_keys(self):
        g = mid_gray()
        r = self.agent.assess(g, gray_to_rgb(g))
        for key in ("status", "score", "hair_pixel_count", "hair_ratio",
                    "artifact_ratio", "is_severe", "issues", "explanation"):
            assert key in r

    def test_score_between_0_and_1(self):
        for g in [mid_gray(), hairy_gray(n=80)]:
            r = self.agent.assess(g, gray_to_rgb(g))
            assert 0.0 <= r["score"] <= 1.0

    def test_hair_pixel_count_nonnegative(self):
        g = mid_gray()
        r = self.agent.assess(g, gray_to_rgb(g))
        assert r["hair_pixel_count"] >= 0

    def test_hairy_image_has_higher_hair_ratio(self):
        clean_r = self.agent.assess(mid_gray(), gray_to_rgb(mid_gray()))
        hairy_r = self.agent.assess(hairy_gray(n=80), gray_to_rgb(hairy_gray(n=80)))
        assert hairy_r["hair_ratio"] >= clean_r["hair_ratio"]

    def test_artifact_ratio_nonnegative(self):
        g = mid_gray()
        r = self.agent.assess(g, gray_to_rgb(g))
        assert r["artifact_ratio"] >= 0.0

    def test_explanation_is_string(self):
        g = mid_gray()
        r = self.agent.assess(g, gray_to_rgb(g))
        assert isinstance(r["explanation"], str)


# ═════════════════════════════════════════════════════════════════════════════
# QualityFusionEngine tests
# ═════════════════════════════════════════════════════════════════════════════

def _pass(score: float = 0.90) -> dict:
    return {"status": "PASS", "score": score, "issues": [], "is_severe": False}


def _fail(score: float = 0.20, severe: bool = False) -> dict:
    return {"status": "FAIL", "score": score, "issues": ["Issue found."], "is_severe": severe}


class TestQualityFusionEngine:

    def setup_method(self):
        self.eng = QualityFusionEngine()

    def test_all_pass_gives_pass(self):
        r = self.eng.fuse(_pass(), _pass(), _pass(), _pass())
        assert r["overall_status"] == "PASS"

    def test_all_fail_gives_fail(self):
        r = self.eng.fuse(_fail(), _fail(), _fail(), _fail())
        assert r["overall_status"] == "FAIL"

    def test_severe_blur_triggers_hard_fail(self):
        r = self.eng.fuse(_fail(severe=True), _pass(), _pass(), _pass())
        assert r["hard_fail_triggered"] is True
        assert r["overall_status"] == "FAIL"

    def test_severe_glare_triggers_hard_fail(self):
        r = self.eng.fuse(_pass(), _pass(), _fail(severe=True), _pass())
        assert r["hard_fail_triggered"] is True

    def test_severe_obstruction_triggers_hard_fail(self):
        r = self.eng.fuse(_pass(), _pass(), _pass(), _fail(severe=True))
        assert r["hard_fail_triggered"] is True

    def test_lighting_severe_does_not_trigger_hard_fail(self):
        # Lighting is not in _HARD_FAIL_AGENTS
        r = self.eng.fuse(_pass(), _fail(severe=True), _pass(), _pass())
        assert r["hard_fail_triggered"] is False

    def test_score_in_valid_range(self):
        r = self.eng.fuse(_pass(0.8), _pass(0.7), _fail(0.3), _pass(0.9))
        assert 0.0 <= r["overall_score"] <= 1.0

    def test_weighted_scores_present(self):
        r = self.eng.fuse(_pass(), _pass(), _pass(), _pass())
        for k in ("blur", "lighting", "glare", "obstruction"):
            assert k in r["weighted_scores"]

    def test_agent_statuses_present(self):
        r = self.eng.fuse(_pass(), _fail(), _pass(), _pass())
        assert r["agent_statuses"]["lighting"] == "FAIL"
        assert r["agent_statuses"]["blur"]     == "PASS"

    def test_recommendation_is_nonempty_string(self):
        r = self.eng.fuse(_pass(), _pass(), _pass(), _pass())
        assert isinstance(r["recommendation"], str) and len(r["recommendation"]) > 0


# ═════════════════════════════════════════════════════════════════════════════
# QualitySupervisor integration tests
# ═════════════════════════════════════════════════════════════════════════════

class TestQualitySupervisor:

    def setup_method(self):
        self.sv = QualitySupervisor()

    def _bgr(self, value: int = 128) -> np.ndarray:
        return np.full((256, 256, 3), value, dtype=np.uint8)

    def test_valid_image_produces_report(self):
        r = self.sv.evaluate(self._bgr())
        assert r["overall"]["status"] in ("PASS", "FAIL")

    def test_invalid_path_returns_error_status(self):
        r = self.sv.evaluate("/does/not/exist.jpg")
        assert r["overall"]["status"] == "ERROR"

    def test_report_top_level_keys(self):
        r = self.sv.evaluate(self._bgr())
        for key in ("image", "overall", "agent_scores", "agent_statuses", "details", "processing_time_s"):
            assert key in r

    def test_all_four_agents_in_details(self):
        r = self.sv.evaluate(self._bgr())
        for agent in ("blur", "lighting", "glare", "obstruction"):
            assert agent in r["details"]

    def test_overall_score_range(self):
        r = self.sv.evaluate(self._bgr())
        assert 0.0 <= r["overall"]["score"] <= 1.0

    def test_dark_image_fails(self):
        dark_bgr = np.full((256, 256, 3), 8, dtype=np.uint8)
        r = self.sv.evaluate(dark_bgr)
        assert r["overall"]["status"] == "FAIL"

    def test_processing_time_nonnegative(self):
        r = self.sv.evaluate(self._bgr())
        assert r["processing_time_s"] >= 0.0

    def test_recommendation_string_present(self):
        r = self.sv.evaluate(self._bgr())
        assert isinstance(r["overall"]["recommendation"], str)

    def test_original_shape_recorded(self):
        bgr = np.full((300, 400, 3), 128, dtype=np.uint8)
        r = self.sv.evaluate(bgr)
        assert r["image"]["original_shape"] == [300, 400, 3]

    def test_report_is_json_serialisable(self):
        import json
        r = self.sv.evaluate(self._bgr())
        # Should not raise
        json.dumps(r)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
