"""
Configurable thresholds for the Image Quality Agent.

⚠️  These are initial experimental values.
    Validate against a representative set of dermoscopic images
    before using in any medical screening pipeline.
"""

# ─── Blur Agent ───────────────────────────────────────────────────────────────
BLUR = {
    # Laplacian variance: lower = blurrier
    "laplacian_threshold": 60.0,   # Adjusted from 100.0 for smooth skin tissue
    "severe_threshold":     25.0,   # Severe blur (hard FAIL)
}

# ─── Lighting Agent ───────────────────────────────────────────────────────────
LIGHTING = {
    # V-channel (HSV) brightness, range 0-255
    "dark_threshold":        40,    # Pixels below this count as "dark"
    "bright_threshold":     210,    # Pixels above this count as "bright"

    # Ratio of image pixels that must be dark/bright before raising an issue
    "dark_pixel_ratio":    0.38,    # Accounts for dermatoscope vignette lens edges
    "bright_pixel_ratio":  0.35,    # > 35 % bright pixels → overexposed

    # Patch-based illumination uniformity (std of patch mean brightness)
    "uneven_std_threshold": 45.0,   # High std → uneven lighting
}

# ─── Glare Agent ──────────────────────────────────────────────────────────────
GLARE = {
    # HSV thresholds to isolate specular-reflection pixels
    "saturation_max":  30,          # Low saturation  (desaturated = white-ish)
    "value_min":      220,          # High brightness in V channel

    # Fraction of image pixels classified as glare
    "glare_pixel_ratio":   0.06,    # > 6 %  → glare issue
    "severe_glare_ratio":  0.18,    # > 18 % → severe glare (hard FAIL)
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
    "pass_threshold": 0.60,         # Weighted score >= 0.60 → PASS
}
