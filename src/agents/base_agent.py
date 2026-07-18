"""
src/agents/base_agent.py
------------------------
Base contract for all orchestrated agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    agent: str
    status: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "status": self.status,
            "summary": self.summary,
            "data": self.data,
            "artifacts": self.artifacts,
            "next_actions": self.next_actions,
            "error_message": self.error_message,
        }


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, context: dict[str, Any]) -> AgentResult:
        """Execute the agent and return a structured result."""
