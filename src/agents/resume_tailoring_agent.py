"""
src/agents/resume_tailoring_agent.py
------------------------------------
Generates role-specific resume tailoring guidance from resume + target job.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.context_helpers import (
    compact_job_summary,
    extract_resume_bullets,
    extract_resume_skills,
    resolve_target_job,
)
from src.llm_client import call_llm_for_json


class ResumeTailoringAgent(BaseAgent):
    name = "resume_tailoring"

    def run(self, context: dict[str, Any]) -> AgentResult:
        structured_resume = context.get("structured_resume")
        if not isinstance(structured_resume, dict):
            return AgentResult(
                agent=self.name,
                status="skipped",
                summary="Skipped: structured_resume is required.",
                next_actions=["Run resume_analysis before resume_tailoring."],
            )

        target_job = resolve_target_job(context)
        if not target_job:
            return AgentResult(
                agent=self.name,
                status="skipped",
                summary="Skipped: no target job found in context or DB.",
                next_actions=["Provide target_job, target_job_url, or job_keyword."],
            )

        resume_skills = extract_resume_skills(structured_resume)
        job_skills = [str(skill).strip().lower() for skill in target_job.get("processed_skills", []) or [] if str(skill).strip()]
        missing = [skill for skill in job_skills if skill not in set(resume_skills)]
        highlights = [skill for skill in job_skills if skill in set(resume_skills)]
        bullets = extract_resume_bullets(structured_resume)

        guidance = {
            "target_job": compact_job_summary(target_job),
            "recommended_headline": f"{target_job.get('job_title', '')} candidate with strong delivery track record",
            "prioritize_skills": highlights[:12],
            "add_or_emphasize_keywords": missing[:12],
            "experience_bullets_to_reposition": bullets[:6],
            "section_order_recommendation": [
                "Summary",
                "Skills",
                "Experience",
                "Projects",
                "Education",
            ],
        }

        if bool(context.get("use_llm", False)):
            guidance["llm_variant"] = _llm_tailoring_guidance(
                structured_resume=structured_resume,
                target_job=target_job,
                highlights=highlights[:10],
                missing=missing[:10],
                bullets=bullets[:6],
            )

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=(
                f"Generated tailoring guidance for {target_job.get('job_title', '')} "
                f"at {target_job.get('company', '')}."
            ),
            data={"resume_tailoring": guidance},
            next_actions=[
                "Apply keyword and bullet-positioning changes to a role-specific resume copy.",
                "Run ats_optimization on the tailored version.",
            ],
        )


def _llm_tailoring_guidance(
    *,
    structured_resume: dict[str, Any],
    target_job: dict[str, Any],
    highlights: list[str],
    missing: list[str],
    bullets: list[str],
) -> dict[str, Any]:
    prompt = f"""
You are a resume tailoring assistant.
Return ONLY valid JSON with this exact structure:
{{
  "summary_rewrite": "string",
  "top_skill_order": ["skill1", "skill2"],
  "bullet_rewrites": [
    {{"original": "string", "rewrite": "string", "reason": "string"}}
  ],
  "missing_keyword_integration_tips": ["string"]
}}

Target job:
{{
  "job_title": "{target_job.get("job_title", "")}",
  "company": "{target_job.get("company", "")}",
  "description": "{str(target_job.get("description", "") or "")[:1800]}",
  "processed_summary": "{str(target_job.get("processed_summary", "") or "")[:1000]}",
  "processed_skills": {target_job.get("processed_skills", []) or []}
}}

Resume summary:
{str(structured_resume.get("summary", "") or "")[:1200]}

Resume highlights skills:
{highlights}

Missing skills:
{missing}

Candidate bullets:
{bullets}
"""
    return call_llm_for_json(prompt)
