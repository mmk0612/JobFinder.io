"""
src/agents/company_research_agent.py
------------------------------------
Builds a company brief from internal job-market signals already in the system.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.context_helpers import (
    compact_job_summary,
    filter_jobs_for_company,
    jobs_recent_count,
    resolve_target_job,
    source_breakdown,
)
from src.db.db import get_matchable_jobs
from src.llm_client import call_llm_for_json


class CompanyResearchAgent(BaseAgent):
    name = "company_research"

    def run(self, context: dict[str, Any]) -> AgentResult:
        target_job = resolve_target_job(context)
        company = str(context.get("company") or "").strip()
        if target_job and not company:
            company = str(target_job.get("company") or "").strip()
        if not company:
            return AgentResult(
                agent=self.name,
                status="skipped",
                summary="Skipped: company (or resolvable target job) is required.",
                next_actions=["Provide company name or target_job_url."],
            )

        source = str(context.get("source") or "").strip() or None
        limit = max(10, int(context.get("company_research_limit", 500)))
        jobs = get_matchable_jobs(source=source, limit=limit)
        company_jobs = filter_jobs_for_company(jobs, company)
        recent_count = jobs_recent_count(company_jobs, days=7)
        skills_counter = Counter()
        for job in company_jobs:
            for skill in list(job.get("processed_skills") or []):
                normalized = str(skill).strip().lower()
                if normalized:
                    skills_counter[normalized] += 1

        payload: dict[str, Any] = {
            "company": company,
            "target_job": compact_job_summary(target_job) if target_job else None,
            "open_roles_count": len(company_jobs),
            "recent_7d_postings": recent_count,
            "source_breakdown": source_breakdown(company_jobs),
            "top_hiring_skills": [
                {"skill": skill, "frequency": count}
                for skill, count in skills_counter.most_common(12)
            ],
            "hiring_signal": _hiring_signal_label(total=len(company_jobs), recent=recent_count),
        }

        if bool(context.get("use_llm", False)):
            payload["llm_company_brief"] = _llm_company_brief(
                company=company,
                target_job=target_job,
                open_roles=len(company_jobs),
                recent_7d=recent_count,
                top_skills=payload["top_hiring_skills"],
            )

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=f"Generated company research brief for {company}.",
            data={"company_research": payload},
            next_actions=[
                "Use top_hiring_skills to tailor resume bullets and interview examples.",
                "Prioritize applications if hiring_signal is high.",
            ],
        )


def _hiring_signal_label(*, total: int, recent: int) -> str:
    if total >= 20 or recent >= 8:
        return "high"
    if total >= 8 or recent >= 3:
        return "medium"
    return "low"


def _llm_company_brief(
    *,
    company: str,
    target_job: dict[str, Any] | None,
    open_roles: int,
    recent_7d: int,
    top_skills: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = f"""
You are a company research analyst for job seekers.
Return ONLY JSON:
{{
  "company_hypothesis": "string",
  "likely_interview_focus": ["string"],
  "candidate_positioning_advice": ["string"],
  "risk_flags": ["string"]
}}

Company: {company}
Target job: {target_job.get("job_title", "") if target_job else ""}
Open roles in current dataset: {open_roles}
Recent 7-day postings: {recent_7d}
Top observed skills: {top_skills}
"""
    return call_llm_for_json(prompt)
