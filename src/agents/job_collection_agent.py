"""
src/agents/job_collection_agent.py
----------------------------------
Phase-1 job collection agent that reuses the existing scraper orchestrator.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentResult, BaseAgent
from src.scrapers.orchestrator import run_all_scrapers


class JobCollectionAgent(BaseAgent):
    name = "job_collection"

    def run(self, context: dict[str, Any]) -> AgentResult:
        keywords = _resolve_keywords(context)
        location = str(context.get("location", "remote")).strip() or "remote"
        sources = context.get("sources")
        max_results = int(context.get("max_results_per_source", 25))
        save_to_db = bool(context.get("save_to_db", True))

        summary = run_all_scrapers(
            keywords=keywords,
            location=location,
            sources=sources,
            max_results_per_source=max_results,
            save_to_db=save_to_db,
        )

        status = "completed" if not summary.get("errors") else "completed_with_errors"
        message = (
            f"Collected {summary['total_unique']} unique jobs "
            f"({summary['total_scraped']} total scraped) for '{keywords}'."
        )

        next_actions = ["Run ranking/matching on the refreshed job set."]
        if summary.get("errors"):
            next_actions.insert(0, "Inspect source-level scraper errors in summary.errors.")

        return AgentResult(
            agent=self.name,
            status=status,
            summary=message,
            data={"job_collection_summary": summary},
            next_actions=next_actions,
        )


def _resolve_keywords(context: dict[str, Any]) -> str:
    explicit = str(context.get("keywords") or "").strip()
    if explicit:
        return explicit

    target_roles = context.get("target_roles")
    if isinstance(target_roles, list):
        cleaned = [str(role).strip() for role in target_roles if str(role).strip()]
        if cleaned:
            return cleaned[0]

    raise ValueError(
        "JobCollectionAgent requires context['keywords'] or context['target_roles']."
    )
