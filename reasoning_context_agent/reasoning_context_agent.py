"""
Agent 4: Reasoning / Context Agent
====================================
Uses patient metadata (age, sex, anatomical location) to derive
contextual risk indicators that supplement visual diagnosis.

Clinical reasoning is based on established dermatological evidence:
  - Age-related incidence profiles per lesion class
  - Sex-based differential risk for skin malignancies
  - Anatomical site solar exposure and lesion-site associations

Inputs  (all optional — missing fields are handled gracefully):
  - age          : float or int  (patient age in years)
  - sex          : str           'male' | 'female' | 'unknown'
  - localization : str           Anatomical site string from HAM10000

Outputs:
  - context_evidence   : Dict  — per-factor scored evidence
  - risk_indicator     : str   — 'LOW' | 'MODERATE' | 'HIGH'
  - risk_score         : float — 0.0 to 1.0
  - missing_field_flags: List[str] — which metadata was absent/imputed
"""

from typing import Any, Dict, List, Optional, Tuple, Union


# ── Anatomical site → solar exposure / risk tier ──────────────────────────────
# Tier 0 = sun-protected / low risk, Tier 1 = moderate, Tier 2 = high UV exposure
_SITE_RISK: Dict[str, Tuple[float, str]] = {
    # High sun-exposure sites
    "face":             (0.85, "High UV-exposed area; elevated actinic & melanoma risk"),
    "scalp":            (0.80, "High UV-exposed area; often missed; higher malignancy rate"),
    "neck":             (0.75, "Chronically UV-exposed; elevated squamous/basal risk"),
    "ear":              (0.80, "High UV; BCC common on ear pinna"),
    "hand":             (0.65, "Moderate UV; acral sites warrant attention"),
    "foot":             (0.70, "Acral site; acral melanoma risk in darker skin types"),
    "acral":            (0.70, "Acral melanoma risk (palms/soles/subungual)"),
    # Moderate risk sites
    "chest":            (0.50, "Moderate UV; common trunk lesion site"),
    "back":             (0.55, "Common melanoma site in males; moderate UV"),
    "trunk":            (0.50, "Common benign nevi site; moderate malignancy risk"),
    "upper extremity":  (0.50, "Moderate UV exposure; common nevi site"),
    "lower extremity":  (0.45, "Melanoma more common in females on legs"),
    "abdomen":          (0.35, "Relatively sun-protected; generally lower risk"),
    "genital":          (0.30, "Sun-protected; specialized histology"),
    # Low risk / unknown
    "unknown":          (0.40, "Anatomical site unknown; applying conservative estimate"),
}

# ── Age-risk profiles ─────────────────────────────────────────────────────────
# Risk contribution from age toward malignant lesions
def _age_risk_score(age: float) -> Tuple[float, str]:
    """Map patient age → malignancy-risk score (0.0 – 1.0) + clinical note."""
    if age < 20:
        return 0.10, "Age < 20; skin malignancy very uncommon in this cohort"
    elif age < 30:
        return 0.20, "Age 20–29; nevi most likely; low malignancy prior"
    elif age < 40:
        return 0.30, "Age 30–39; melanoma incidence rising; moderate baseline"
    elif age < 50:
        return 0.50, "Age 40–49; melanoma & keratoses risk meaningfully elevated"
    elif age < 60:
        return 0.65, "Age 50–59; BCC and actinic risk notably increased"
    elif age < 70:
        return 0.75, "Age 60–69; high cumulative UV damage; major malignancy era"
    else:
        return 0.85, "Age ≥ 70; highest cumulative risk; BCC/SCC/melanoma common"


# ── Sex-based risk adjustments ────────────────────────────────────────────────
_SEX_ADJUSTMENT: Dict[str, Tuple[float, str]] = {
    "male":    (+0.08, "Males have higher overall incidence of BCC and melanoma"),
    "female":  (+0.03, "Females: slightly lower overall rate but leg melanoma elevated"),
    "unknown": ( 0.00, "Sex unknown; no sex-specific adjustment applied"),
}

# Weight for each metadata factor in the final risk score
_W_AGE  = 0.45
_W_SITE = 0.35
_W_SEX  = 0.20


class ReasoningContextAgent:
    """
    Derives contextual risk indicators from patient metadata.

    Usage
    -----
        agent = ReasoningContextAgent()
        result = agent.evaluate(age=68, sex='male', localization='face')
    """

    def evaluate(
        self,
        age: Optional[Union[float, int]] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        age          : Patient age in years (float/int). None = missing.
        sex          : 'male', 'female', or 'unknown'/None.
        localization : Anatomical site string (case-insensitive). None = missing.

        Returns
        -------
        {
            context_evidence:    Dict  — per-factor scored reasoning
            risk_score:          float — weighted composite 0.0–1.0
            risk_indicator:      str   — 'LOW' | 'MODERATE' | 'HIGH'
            missing_field_flags: List[str]
        }
        """
        missing_flags: List[str] = []

        # ── Age ───────────────────────────────────────────────────────────────
        if age is None or (isinstance(age, float) and age != age):  # nan check
            missing_flags.append("age: missing — using population median (~45 years)")
            age_imputed = 45.0
            age_note = "Imputed age (median); moderate-risk default"
            age_score = 0.45
        else:
            age_imputed = float(age)
            age_score, age_note = _age_risk_score(age_imputed)

        # ── Sex ───────────────────────────────────────────────────────────────
        norm_sex = (sex or "").lower().strip()
        if norm_sex not in ("male", "female"):
            missing_flags.append("sex: missing or unrecognised — no sex adjustment")
            norm_sex = "unknown"
        sex_adj, sex_note = _SEX_ADJUSTMENT[norm_sex]

        # ── Localization ──────────────────────────────────────────────────────
        site_score, site_note = self._resolve_site(localization)
        if localization is None or localization.lower().strip() in ("", "unknown"):
            missing_flags.append("localization: missing — using conservative estimate")

        # ── Weighted composite risk score ─────────────────────────────────────
        raw_risk = (
            _W_AGE  * age_score +
            _W_SITE * site_score +
            _W_SEX  * (age_score + sex_adj)  # sex adjustment scales with age base
        )
        risk_score = float(min(max(raw_risk, 0.0), 1.0))

        # ── Risk category ─────────────────────────────────────────────────────
        if risk_score >= 0.60:
            risk_indicator = "HIGH"
        elif risk_score >= 0.35:
            risk_indicator = "MODERATE"
        else:
            risk_indicator = "LOW"

        context_evidence = {
            "age": {
                "value": age_imputed,
                "score": round(age_score, 4),
                "note":  age_note,
            },
            "sex": {
                "value":      norm_sex,
                "adjustment": round(sex_adj, 4),
                "note":       sex_note,
            },
            "localization": {
                "value": localization or "unknown",
                "score": round(site_score, 4),
                "note":  site_note,
            },
        }

        return {
            "context_evidence":    context_evidence,
            "risk_score":          round(risk_score, 4),
            "risk_indicator":      risk_indicator,
            "missing_field_flags": missing_flags,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_site(localization: Optional[str]) -> Tuple[float, str]:
        """
        Map the raw localization string to a site-risk score.
        Handles partial matches and aliases.
        """
        if not localization:
            return _SITE_RISK["unknown"]

        key = localization.lower().strip()

        # Exact match
        if key in _SITE_RISK:
            return _SITE_RISK[key]

        # Partial / substring match
        for site_key, (score, note) in _SITE_RISK.items():
            if site_key in key or key in site_key:
                return score, note

        # Fallback
        return _SITE_RISK["unknown"][0], f"Unrecognised site '{localization}'; conservative estimate used"

