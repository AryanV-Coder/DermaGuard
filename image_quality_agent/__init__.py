"""
DermaGuard Image Quality Agent (Agent 1)
Evaluates technical image quality for dermoscopic imaging.
"""

from .supervisor.quality_supervisor import QualitySupervisor, ImageQualityAgent

__all__ = ["QualitySupervisor", "ImageQualityAgent"]
