"""
src/agents/resume_analysis_agent.py
-----------------------------------
Phase-1 resume analysis agent that reuses the existing resume parser pipeline.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentResult, BaseAgent
from src.normalizer import normalize_skills
from src.resume_parser import extract_sections


class ResumeAnalysisAgent(BaseAgent):
    name = "resume_analysis"

    def run(self, context: dict[str, Any]) -> AgentResult:
        structured_resume = context.get("structured_resume")
        resume_text = context.get("resume_text")

        if structured_resume is not None and not isinstance(structured_resume, dict):
            raise ValueError("context['structured_resume'] must be a dict when provided.")
        if structured_resume is None and not isinstance(resume_text, str):
            raise ValueError(
                "ResumeAnalysisAgent requires either context['structured_resume'] (dict) "
                "or context['resume_text'] (str)."
            )

        resume = structured_resume if structured_resume is not None else extract_sections(resume_text)
        if bool(context.get("normalize_resume_skills", True)):
            resume = normalize_skills(resume)

        skills = resume.get("skills", [])
        experience = resume.get("experience", [])
        projects = resume.get("projects", [])
        contact = resume.get("contact", {})
        candidate_name = str(contact.get("name") or "").strip() or "candidate"

        summary = (
            f"Analyzed resume for {candidate_name}: "
            f"{len(skills)} normalized skills, {len(experience)} experience entries, "
            f"{len(projects)} projects."
        )

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=summary,
            data={
                "structured_resume": resume,
                "resume_profile": {
                    "skills": skills,
                    "experience_count": len(experience),
                    "project_count": len(projects),
                },
            },
            next_actions=[
                "Run job collection for target roles.",
                "Use resume profile to drive tailoring and ATS checks.",
            ],
        )
