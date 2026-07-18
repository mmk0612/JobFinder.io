"""
src/agent_orchestrator.py
-------------------------
Orchestrator runtime for coordinating specialized JobFinder agents.
"""

from __future__ import annotations

import logging
from uuid import uuid4
from typing import Any

from src.agent_state import WorkflowState
from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.registry import ALL_AGENT_NAMES, build_default_agent_registry

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Coordinates execution across specialized agents and tracks run state.

    Context keys:
      intent: str (optional)
      execution_plan: list[str] (optional)
      fail_fast: bool (optional, default True)
      plus any keys needed by individual agents.
    """

    def __init__(self, agents: dict[str, BaseAgent] | None = None) -> None:
        self._agents = agents or build_default_agent_registry()

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(context, dict):
            raise ValueError("Orchestrator context must be a dictionary.")

        intent = str(context.get("intent") or "auto").strip() or "auto"
        workflow = WorkflowState(run_id=str(uuid4()), intent=intent)
        fail_fast = bool(context.get("fail_fast", True))

        plan = self._resolve_plan(context, intent)
        if not plan:
            raise ValueError("Unable to derive an execution plan from context.")

        outputs: dict[str, dict[str, Any]] = {}
        aggregated_next_actions: list[str] = []

        mutable_context = dict(context)
        mutable_context["agent_outputs"] = outputs

        for agent_name in plan:
            agent = self._agents.get(agent_name)
            if agent is None:
                raise ValueError(f"Unknown agent in plan: {agent_name}")

            step = workflow.start_step(agent_name)
            try:
                result = agent.run(mutable_context)
                self._apply_result_to_context(result, mutable_context, outputs)
                workflow.complete_step(step, result.summary)
                aggregated_next_actions.extend(result.next_actions)
            except Exception as exc:
                message = f"{agent_name} failed: {exc}"
                logger.error(message)
                workflow.fail_step(step, message)
                if fail_fast:
                    break

        workflow.mark_completed()
        return {
            "orchestrator": "jobfinder_orchestrator",
            "status": workflow.status,
            "run": workflow.to_dict(),
            "execution_plan": plan,
            "available_agents": self.available_agents(),
            "outputs": outputs,
            "next_actions": list(dict.fromkeys(aggregated_next_actions)),
        }

    def _resolve_plan(self, context: dict[str, Any], intent: str) -> list[str]:
        explicit = context.get("execution_plan")
        if isinstance(explicit, list):
            return [str(step).strip() for step in explicit if str(step).strip()]

        if intent == "analyze_resume":
            return ["resume_analysis"]
        if intent == "discover_jobs":
            return ["job_collection"]
        if intent == "tailor_resume":
            return ["resume_tailoring"]
        if intent == "optimize_ats":
            return ["ats_optimization"]
        if intent == "research_company":
            return ["company_research"]
        if intent == "track_application":
            return ["application_tracker"]
        if intent == "prepare_interview":
            return ["interview_prep"]
        if intent == "career_coaching":
            return ["career_coach"]
        if intent == "bootstrap":
            return ["resume_analysis", "job_collection"]
        if intent == "full_assistant":
            return list(ALL_AGENT_NAMES)

        plan: list[str] = []
        if "structured_resume" in context or "resume_text" in context:
            plan.append("resume_analysis")
        if "keywords" in context or "target_roles" in context:
            plan.append("job_collection")
        return plan

    def available_agents(self) -> list[str]:
        return sorted(self._agents.keys())

    def _apply_result_to_context(
        self,
        result: AgentResult,
        context: dict[str, Any],
        outputs: dict[str, dict[str, Any]],
    ) -> None:
        outputs[result.agent] = result.to_dict()
        for key, value in result.data.items():
            context[key] = value


def run_orchestrator(context: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper for one-shot orchestrator execution."""
    orchestrator = OrchestratorAgent()
    return orchestrator.run(context)


def list_available_agents() -> list[str]:
    orchestrator = OrchestratorAgent()
    return orchestrator.available_agents()
