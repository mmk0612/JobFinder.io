"""
src/agents/ats_optimization_agent.py
------------------------------------
Runs ATS-style keyword and alignment checks for resume vs target job.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.context_helpers import compact_job_summary, extract_resume_skills, resolve_target_job
from src.llm_client import call_llm_for_json


class AtsOptimizationAgent(BaseAgent):
    name = "ats_optimization"

    def run(self, context: dict[str, Any]) -> AgentResult:
        structured_resume = context.get("structured_resume")
        if not isinstance(structured_resume, dict):
            return AgentResult(
                agent=self.name,
                status="skipped",
                summary="Skipped: structured_resume is required.",
                next_actions=["Run resume_analysis before ats_optimization."],
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
        resume_set = set(resume_skills)
        job_skills = [
            str(skill).strip().lower()
            for skill in target_job.get("processed_skills", []) or []
            if str(skill).strip()
        ]
        unique_job_skills = list(dict.fromkeys(job_skills))

        matched = [skill for skill in unique_job_skills if skill in resume_set]
        missing = [skill for skill in unique_job_skills if skill not in resume_set]
        coverage = (len(matched) / len(unique_job_skills)) if unique_job_skills else 0.0
        ats_score = int(round(max(0.0, min(1.0, coverage)) * 100))

        payload: dict[str, Any] = {
            "target_job": compact_job_summary(target_job),
            "ats_score": ats_score,
            "coverage_ratio": round(coverage, 4),
            "matched_keywords": matched,
            "missing_keywords": missing,
            "priority_missing_keywords": missing[:10],
            "suggested_actions": [
                "Mirror exact keyword casing from job description in skills and experience bullets.",
                "Add at least 3 missing high-priority keywords with evidence bullets.",
                "Place strongest matching skills in the top-third of the resume.",
            ],
        }

        if bool(context.get("use_llm", False)):
            payload["llm_recommendations"] = _llm_ats_recommendations(
                structured_resume=structured_resume,
                target_job=target_job,
                matched=matched[:12],
                missing=missing[:12],
            )

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=(
                f"ATS optimization completed for {target_job.get('job_title', '')} "
                f"at {target_job.get('company', '')}: score={ats_score}."
            ),
            data={"ats_optimization": payload},
            next_actions=[
                "Incorporate priority missing keywords with concrete project/impact evidence.",
                "Re-run ATS optimization after edits.",
            ],
        )


def _llm_ats_recommendations(
    *,
    structured_resume: dict[str, Any],
    target_job: dict[str, Any],
    matched: list[str],
    missing: list[str],
) -> dict[str, Any]:
    prompt = f"""
You are an ATS optimization expert.
Return ONLY valid JSON:
{{
  "rewrite_tips": ["string"],
  "section_specific_changes": [
    {{"section": "summary|skills|experience|projects", "change": "string", "why": "string"}}
  ],
  "keyword_insertion_examples": ["string"]
}}

Job title: {target_job.get("job_title", "")}
Company: {target_job.get("company", "")}
Job summary: {str(target_job.get("processed_summary", "") or "")[:1400]}
Job skills: {target_job.get("processed_skills", []) or []}

Resume summary: {str(structured_resume.get("summary", "") or "")[:1200]}
Resume skills: {structured_resume.get("skills", []) or []}
Matched keywords: {matched}
Missing keywords: {missing}
"""
    return call_llm_for_json(prompt)
