"""
DermaGuard Diagnosis Agent (Agent 2)
7-class lesion diagnostic prediction with CNN ensemble, TTA, and temperature scaling.
"""

from .diagnosis_agent import DiagnosisAgent, CLASS_NAMES, CLASS_FULL, MALIGNANT_CLASSES, LABEL_MAP, IDX_TO_CLASS

__all__ = [
    "DiagnosisAgent",
    "CLASS_NAMES",
    "CLASS_FULL",
    "MALIGNANT_CLASSES",
    "LABEL_MAP",
    "IDX_TO_CLASS",
]
