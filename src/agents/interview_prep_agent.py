"""
src/agents/interview_prep_agent.py
----------------------------------
Creates interview preparation packs from target job + resume profile.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.context_helpers import compact_job_summary, extract_resume_skills, resolve_target_job
from src.llm_client import call_llm_for_json


class InterviewPrepAgent(BaseAgent):
    name = "interview_prep"

    def run(self, context: dict[str, Any]) -> AgentResult:
        structured_resume = context.get("structured_resume")
        if not isinstance(structured_resume, dict):
            return AgentResult(
                agent=self.name,
                status="skipped",
                summary="Skipped: structured_resume is required.",
                next_actions=["Run resume_analysis before interview_prep."],
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
        job_skills = [
            str(skill).strip().lower()
            for skill in target_job.get("processed_skills", []) or []
            if str(skill).strip()
        ]
        overlap = [skill for skill in job_skills if skill in set(resume_skills)]
        gaps = [skill for skill in job_skills if skill not in set(resume_skills)]

        pack: dict[str, Any] = {
            "target_job": compact_job_summary(target_job),
            "likely_questions": _heuristic_questions(
                title=str(target_job.get("job_title") or ""),
                seniority=str(target_job.get("processed_seniority") or ""),
                skills=job_skills[:6],
            ),
            "talking_points": [
                f"Lead with impact examples tied to {skill}."
                for skill in overlap[:6]
            ],
            "gap_mitigation_points": [
                f"Prepare a bridge narrative for {skill} with adjacent experience."
                for skill in gaps[:5]
            ],
            "star_story_prompts": _star_prompts(structured_resume),
        }

        if bool(context.get("use_llm", False)):
            pack["llm_interview_pack"] = _llm_interview_pack(
                structured_resume=structured_resume,
                target_job=target_job,
                overlap=overlap[:8],
                gaps=gaps[:8],
            )

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=(
                f"Generated interview prep pack for {target_job.get('job_title', '')} "
                f"at {target_job.get('company', '')}."
            ),
            data={"interview_prep": pack},
            next_actions=[
                "Draft 2 STAR stories aligned to top job skills.",
                "Rehearse answers for likely technical and behavioral questions.",
            ],
        )


def _heuristic_questions(*, title: str, seniority: str, skills: list[str]) -> list[str]:
    questions = [
        f"Walk me through a project most relevant to this {title} role.",
        "How do you prioritize trade-offs between speed, quality, and reliability?",
        "Describe a production issue you debugged and resolved.",
    ]
    if seniority in {"senior", "staff", "lead"}:
        questions.append("How do you mentor engineers and influence architecture decisions?")
    for skill in skills:
        questions.append(f"How have you used {skill} in a production setting?")
    return questions[:10]


def _star_prompts(structured_resume: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    for item in structured_resume.get("experience", []) or []:
        company = str(item.get("company") or "").strip()
        title = str(item.get("title") or "").strip()
        if company or title:
            prompts.append(
                f"STAR story from {title or 'role'} at {company or 'company'}: "
                "problem, action, measurable impact."
            )
    return prompts[:6]


def _llm_interview_pack(
    *,
    structured_resume: dict[str, Any],
    target_job: dict[str, Any],
    overlap: list[str],
    gaps: list[str],
) -> dict[str, Any]:
    prompt = f"""
You are an interview coach.
Return ONLY JSON:
{{
  "technical_questions": ["string"],
  "behavioral_questions": ["string"],
  "best_talking_points": ["string"],
  "risk_questions_to_prepare": ["string"]
}}

Job:
{{
  "title": "{target_job.get("job_title", "")}",
  "company": "{target_job.get("company", "")}",
  "summary": "{str(target_job.get("processed_summary", "") or "")[:1500]}",
  "skills": {target_job.get("processed_skills", []) or []},
  "seniority": "{target_job.get("processed_seniority", "")}"
}}

Resume summary: {str(structured_resume.get("summary", "") or "")[:1200]}
Resume skills: {structured_resume.get("skills", []) or []}
Overlap skills: {overlap}
Gap skills: {gaps}
"""
    return call_llm_for_json(prompt)
