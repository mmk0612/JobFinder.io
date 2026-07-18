"""
src/agents/career_coach_agent.py
--------------------------------
Builds a skill-gap and prioritization plan using resume + market demand data.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.context_helpers import extract_resume_skills, top_missing_skills
from src.db.db import get_matchable_jobs
from src.llm_client import call_llm_for_json


class CareerCoachAgent(BaseAgent):
    name = "career_coach"

    def run(self, context: dict[str, Any]) -> AgentResult:
        structured_resume = context.get("structured_resume")
        if not isinstance(structured_resume, dict):
            return AgentResult(
                agent=self.name,
                status="skipped",
                summary="Skipped: structured_resume is required.",
                next_actions=["Run resume_analysis before career_coach."],
            )

        source = str(context.get("source") or "").strip() or None
        limit = max(25, int(context.get("career_market_sample_limit", 400)))
        jobs = get_matchable_jobs(source=source, limit=limit)
        if not jobs:
            return AgentResult(
                agent=self.name,
                status="skipped",
                summary="Skipped: no matchable jobs available for market analysis.",
                next_actions=["Run job_collection and processing before career_coach."],
            )

        resume_skills = extract_resume_skills(structured_resume)
        gaps = top_missing_skills(resume_skills=resume_skills, jobs=jobs, limit=12)

        plan: dict[str, Any] = {
            "resume_skills_count": len(resume_skills),
            "market_sample_size": len(jobs),
            "top_skill_gaps": gaps,
            "90_day_plan": _build_90_day_plan(gaps),
        }

        if bool(context.get("use_llm", False)):
            plan["llm_career_plan"] = _llm_career_plan(
                structured_resume=structured_resume,
                market_sample_size=len(jobs),
                top_gaps=gaps[:10],
            )

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=f"Generated career coaching plan from {len(jobs)} market jobs.",
            data={"career_coach": plan},
            next_actions=[
                "Pick 2 high-frequency gaps for focused upskilling this month.",
                "Update resume/projects with evidence for newly covered skills.",
            ],
        )


def _build_90_day_plan(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    top = gaps[:6]
    weeks_1_4 = [item["skill"] for item in top[:2]]
    weeks_5_8 = [item["skill"] for item in top[2:4]]
    weeks_9_12 = [item["skill"] for item in top[4:6]]
    return [
        {
            "window": "weeks_1_4",
            "focus_skills": weeks_1_4,
            "outcome": "Build one portfolio-quality project artifact using these skills.",
        },
        {
            "window": "weeks_5_8",
            "focus_skills": weeks_5_8,
            "outcome": "Demonstrate production-style depth and measurable impact.",
        },
        {
            "window": "weeks_9_12",
            "focus_skills": weeks_9_12,
            "outcome": "Translate outcomes into resume bullets and interview stories.",
        },
    ]


def _llm_career_plan(
    *,
    structured_resume: dict[str, Any],
    market_sample_size: int,
    top_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = f"""
You are a career coach for software roles.
Return ONLY JSON:
{{
  "positioning_summary": "string",
  "priority_skill_investments": ["string"],
  "portfolio_project_ideas": ["string"],
  "application_strategy": ["string"]
}}

Resume summary: {str(structured_resume.get("summary", "") or "")[:1200]}
Resume skills: {structured_resume.get("skills", []) or []}
Market sample size: {market_sample_size}
Top gaps: {top_gaps}
"""
    return call_llm_for_json(prompt)
