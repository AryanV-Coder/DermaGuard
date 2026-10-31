"""
DermaGuard Coordinator Agent (Agent 6)
Synthesizes findings from all agents and decides final disposition (AUTO_TRIAGE, REFER, RE_IMAGE).
"""

from .coordinator_agent import CoordinatorAgent, FinalDecisionAgent
from .langgraph_coordinator import DermaGuardGraph

__all__ = ["CoordinatorAgent", "FinalDecisionAgent", "DermaGuardGraph"]
