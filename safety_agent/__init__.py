"""
DermaGuard Safety Agent (Agent 5)
Clinical safety guardrails and multi-agent conflict arbitration.
"""

from .safety_agent import SafetyAgent, SafetyReviewAgent, MALIGNANT_CLASSES, BENIGN_CLASSES

__all__ = ["SafetyAgent", "SafetyReviewAgent", "MALIGNANT_CLASSES", "BENIGN_CLASSES"]
