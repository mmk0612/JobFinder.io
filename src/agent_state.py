"""
src/agent_state.py
------------------
Typed workflow state models for the AI agent runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

RunStatus = Literal["pending", "running", "completed", "failed"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentRunRecord:
    agent_name: str
    status: RunStatus = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    summary: str = ""
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
            "error_message": self.error_message,
        }


@dataclass
class WorkflowState:
    run_id: str
    intent: str
    status: RunStatus = "running"
    started_at: str = field(default_factory=_utc_now_iso)
    completed_at: str | None = None
    steps: list[AgentRunRecord] = field(default_factory=list)

    def start_step(self, agent_name: str) -> AgentRunRecord:
        step = AgentRunRecord(
            agent_name=agent_name,
            status="running",
            started_at=_utc_now_iso(),
        )
        self.steps.append(step)
        return step

    def complete_step(self, step: AgentRunRecord, summary: str) -> None:
        step.status = "completed"
        step.summary = summary
        step.completed_at = _utc_now_iso()

    def fail_step(self, step: AgentRunRecord, error_message: str) -> None:
        step.status = "failed"
        step.error_message = error_message
        step.completed_at = _utc_now_iso()
        self.status = "failed"
        self.completed_at = step.completed_at

    def mark_completed(self) -> None:
        if self.status != "failed":
            self.status = "completed"
        self.completed_at = _utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "intent": self.intent,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": [step.to_dict() for step in self.steps],
        }
