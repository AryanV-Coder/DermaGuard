"""
Configurable thresholds for the Image Quality Agent.

⚠️  These are initial experimental values.
    Validate against a representative set of dermoscopic images
    before using in any medical screening pipeline.
"""

# ─── Blur Agent ───────────────────────────────────────────────────────────────
BLUR = {
    # Laplacian variance: lower = blurrier
    # Calibrated for dermoscopy: normal smooth skin nevi pass; only severe out-of-focus fails
    "laplacian_threshold": 16.0,    # Reduced threshold for clinical dermoscopy
    "severe_threshold":     8.0,    # Truly severe out-of-focus / motion shake (hard FAIL)
}

# ─── Lighting Agent ───────────────────────────────────────────────────────────
LIGHTING = {
    # V-channel (HSV) brightness, range 0-255
    "dark_threshold":        15,    # Truly dark pixels
    "bright_threshold":     248,    # Completely blown-out pixels

    # Ratio of image pixels that must be dark/bright before raising an issue
    # Accommodates circular dermatoscope vignette lens edges which cover 35-50%
    "dark_pixel_ratio":    0.70,    # Over 70% pitch black fails
    "bright_pixel_ratio":  0.50,    # Over 50% blown-out pixels fails

    # Patch-based illumination uniformity (std of patch mean brightness)
    "uneven_std_threshold": 65.0,   # High std → uneven lighting
}

# ─── Glare Agent ──────────────────────────────────────────────────────────────
GLARE = {
    # HSV thresholds to isolate specular-reflection pixels
    "saturation_max":  30,          # Low saturation  (desaturated = white-ish)
    "value_min":      245,          # High brightness in V channel (> 245)

    # Fraction of image pixels classified as glare
    # Normal LED/fluid gel flash spots (5-15%) PASS; severe specular patches fail
    "glare_pixel_ratio":   0.22,    # > 22 % → glare issue
    "severe_glare_ratio":  0.40,    # > 40 % → severe glare (hard FAIL)
}

# ─── Obstruction Agent ────────────────────────────────────────────────────────
OBSTRUCTION = {
    # Black-hat morphological kernel (should be larger than hair width)
    "blackhat_kernel_size": (17, 17),
    "blackhat_threshold":   15,     # Threshold on black-hat response

    # Hair coverage
    "hair_pixel_ratio":   0.28,     # > 28 % → hair obstruction
    "severe_hair_ratio":  0.45,     # > 45 % → severe obstruction (hard FAIL)

    # Artifact blobs (relative to total image area)
    "artifact_area_ratio": 0.08,    # Suspicious connected region > 8 % of image
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
    "pass_threshold": 0.30,         # Reduced to 0.30 so typical clinical dermoscopy easily clears to diagnosis
}

