"""
Configurable thresholds for the Image Quality Agent.

⚠️  These are initial experimental values.
    Validate against a representative set of dermoscopic images
    before using in any medical screening pipeline.
"""

# ─── Blur Agent ───────────────────────────────────────────────────────────────
BLUR = {
    # Laplacian variance: lower = blurrier
    # Calibrated for dermoscopy: normal smooth skin nevi (25-50) pass; severe blur < 18-20 fails
    "laplacian_threshold": 30.0,   # Soft threshold
    "severe_threshold":     18.0,   # Severe out-of-focus / motion shake (hard FAIL)
}

# ─── Lighting Agent ───────────────────────────────────────────────────────────
LIGHTING = {
    # V-channel (HSV) brightness, range 0-255
    "dark_threshold":        25,    # Truly dark pixels
    "bright_threshold":     240,    # Completely blown-out pixels

    # Ratio of image pixels that must be dark/bright before raising an issue
    # Accommodates circular dermatoscope vignette lens edges which cover 25-35%
    "dark_pixel_ratio":    0.50,    # Over 50% pitch black fails
    "bright_pixel_ratio":  0.35,    # > 35 % blown-out pixels → overexposed

    # Patch-based illumination uniformity (std of patch mean brightness)
    "uneven_std_threshold": 48.0,   # High std → uneven lighting
}

# ─── Glare Agent ──────────────────────────────────────────────────────────────
GLARE = {
    # HSV thresholds to isolate specular-reflection pixels
    "saturation_max":  35,          # Low saturation  (desaturated = white-ish)
    "value_min":      240,          # High brightness in V channel (> 240)

    # Fraction of image pixels classified as glare
    # Normal LED/fluid gel flash spots (3-8%) PASS; severe specular patches fail
    "glare_pixel_ratio":   0.12,    # > 12 % → glare issue
    "severe_glare_ratio":  0.22,    # > 22 % → severe glare (hard FAIL)
}

# ─── Obstruction Agent ────────────────────────────────────────────────────────
OBSTRUCTION = {
    # Black-hat morphological kernel (should be larger than hair width)
    "blackhat_kernel_size": (17, 17),
    "blackhat_threshold":   10,     # Threshold on black-hat response

    # Hair coverage
    "hair_pixel_ratio":   0.14,     # > 14 % → hair obstruction
    "severe_hair_ratio":  0.28,     # > 28 % → severe obstruction (hard FAIL)

    # Artifact blobs (relative to total image area)
    "artifact_area_ratio": 0.03,    # Suspicious connected region > 3 % of image
}

# ─── Quality Fusion Engine ────────────────────────────────────────────────────
FUSION = {
    # Must sum to 1.0
    "weights": {
        "blur":        0.30,
        "lighting":    0.25,
        "glare":       0.25,
        "obstruction": 0.20,
    },
    "pass_threshold": 0.50,         # Weighted score >= 0.50 → PASS (calibrated for clinical dermoscopy)
}

