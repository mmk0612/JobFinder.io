"""Core service layer for the JobFinder microservices architecture.

These functions replace the old agent/orchestrator layer with direct service
operations that the FastAPI API can call.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from src.db.db import (
    get_job_by_url,
    get_matchable_jobs,
    list_application_records,
    update_application_record_status,
    upsert_application_record,
)
from src.llm_client import call_llm_for_json
from src.normalizer import normalize_skills
from src.resume_parser import extract_sections
from src.scrapers.orchestrator import run_all_scrapers


def analyze_resume_service(
    *,
    structured_resume: dict[str, Any] | None = None,
    resume_text: str | None = None,
    normalize_resume_skills: bool = True,
) -> dict[str, Any]:
    if structured_resume is not None and not isinstance(structured_resume, dict):
        raise ValueError("structured_resume must be a dictionary when provided.")
    if structured_resume is None and not isinstance(resume_text, str):
        raise ValueError("Either structured_resume or resume_text is required.")

    resume = structured_resume if structured_resume is not None else extract_sections(resume_text or "")
    if normalize_resume_skills:
        resume = normalize_skills(resume)

    skills = resume.get("skills", []) or []
    experience = resume.get("experience", []) or []
    projects = resume.get("projects", []) or []
    contact = resume.get("contact", {}) or {}
    candidate_name = str(contact.get("name") or "").strip() or "candidate"

    return {
        "status": "completed",
        "summary": (
            f"Analyzed resume for {candidate_name}: "
            f"{len(skills)} normalized skills, {len(experience)} experience entries, "
            f"{len(projects)} projects."
        ),
        "data": {
            "structured_resume": resume,
            "resume_profile": {
                "skills": skills,
                "experience_count": len(experience),
                "project_count": len(projects),
            },
        },
        "next_actions": [
            "Run job discovery for target roles.",
            "Use resume profile to drive tailoring and ATS checks.",
        ],
    }


def discover_jobs_service(
    *,
    keywords: str,
    location: str = "remote",
    sources: list[str] | None = None,
    max_results_per_source: int = 25,
    save_to_db: bool = True,
) -> dict[str, Any]:
    summary = run_all_scrapers(
        keywords=keywords,
        location=location,
        sources=sources,
        max_results_per_source=max_results_per_source,
        save_to_db=save_to_db,
    )
    return {
        "status": "completed" if not summary.get("errors") else "completed_with_errors",
        "summary": (
            f"Collected {summary['total_unique']} unique jobs "
            f"({summary['total_scraped']} total scraped) for '{keywords}'."
        ),
        "data": {"job_collection_summary": summary},
        "next_actions": ["Run ranking/matching on the refreshed job set."],
    }


def _job_matches(job: dict[str, Any], *, keyword: str, company: str, title: str) -> bool:
    haystack = " ".join(
        [
            str(job.get("job_title") or ""),
            str(job.get("company") or ""),
            str(job.get("location") or ""),
            str(job.get("description") or ""),
            str(job.get("processed_summary") or ""),
        ]
    ).lower()

    if keyword:
        terms = [term for term in keyword.split() if term]
        if not terms or not all(term in haystack for term in terms):
            return False
    if company and company not in str(job.get("company") or "").strip().lower():
        return False
    if title and title not in str(job.get("job_title") or "").strip().lower():
        return False
    return True


def _resolve_target_job(context: dict[str, Any]) -> dict[str, Any] | None:
    target_job = context.get("target_job")
    if isinstance(target_job, dict):
        return target_job

    target_url = str(context.get("target_job_url") or "").strip()
    if target_url:
        return get_job_by_url(target_url)

    keyword = str(context.get("job_keyword") or "").strip().lower()
    company = str(context.get("company") or "").strip().lower()
    title = str(context.get("job_title") or "").strip().lower()
    source = str(context.get("source") or "").strip() or None
    limit = max(1, int(context.get("job_search_limit", 250)))

    jobs = get_matchable_jobs(source=source, limit=limit)
    filtered = [job for job in jobs if _job_matches(job, keyword=keyword, company=company, title=title)]
    return filtered[0] if filtered else None


def _compact_job_summary(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    return {
        "job_title": str(job.get("job_title") or ""),
        "company": str(job.get("company") or ""),
        "location": str(job.get("location") or ""),
        "source": str(job.get("source") or ""),
        "url": str(job.get("url") or ""),
        "salary": str(job.get("salary") or ""),
        "seniority": str(job.get("processed_seniority") or ""),
        "processed_skills": list(job.get("processed_skills") or []),
        "processed_tech_stack": list(job.get("processed_tech_stack") or []),
        "processed_summary": str(job.get("processed_summary") or ""),
    }


def _extract_resume_skills(structured_resume: dict[str, Any]) -> list[str]:
    raw = structured_resume.get("skills", []) or []
    return list(dict.fromkeys(str(skill).strip().lower() for skill in raw if str(skill).strip()))


def _extract_resume_bullets(structured_resume: dict[str, Any], *, limit: int = 8) -> list[str]:
    bullets: list[str] = []
    for item in structured_resume.get("experience", []) or []:
        for bullet in item.get("bullets", []) or []:
            text = str(bullet).strip()
            if text:
                bullets.append(text)
    for item in structured_resume.get("projects", []) or []:
        text = str(item.get("description") or "").strip()
        if text:
            bullets.append(text)
    return bullets[: max(1, limit)]


def _top_missing_skills(*, resume_skills: list[str], jobs: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    resume_set = set(resume_skills)
    counter: Counter[str] = Counter()
    for job in jobs:
        for skill in list(job.get("processed_skills") or []):
            normalized = str(skill).strip().lower()
            if normalized and normalized not in resume_set:
                counter[normalized] += 1

    return [{"skill": skill, "job_frequency": count} for skill, count in counter.most_common(max(1, limit))]


def research_company_service(
    *,
    company: str,
    target_job_url: str | None = None,
    source: str | None = None,
    limit: int = 500,
    use_llm: bool = False,
) -> dict[str, Any]:
    target_job = get_job_by_url(target_job_url) if target_job_url else None
    if target_job and not company:
        company = str(target_job.get("company") or "").strip()
    if not company:
        raise ValueError("company name or target_job_url is required.")

    jobs = get_matchable_jobs(source=source, limit=limit)
    company_jobs = [job for job in jobs if str(job.get("company") or "").strip().lower() == company.strip().lower()]
    recent_count = sum(1 for job in company_jobs if isinstance(job.get("scraped_at"), datetime) and (datetime.now().timestamp() - job["scraped_at"].timestamp()) <= 7 * 86400)
    skills_counter = Counter()
    for job in company_jobs:
        for skill in list(job.get("processed_skills") or []):
            normalized = str(skill).strip().lower()
            if normalized:
                skills_counter[normalized] += 1

    payload: dict[str, Any] = {
        "company": company,
        "target_job": _compact_job_summary(target_job) if target_job else None,
        "open_roles_count": len(company_jobs),
        "recent_7d_postings": recent_count,
        "source_breakdown": dict(Counter(str(job.get("source") or "unknown").strip().lower() or "unknown" for job in company_jobs)),
        "top_hiring_skills": [{"skill": skill, "frequency": count} for skill, count in skills_counter.most_common(12)],
        "hiring_signal": "high" if len(company_jobs) >= 20 or recent_count >= 8 else "medium" if len(company_jobs) >= 8 or recent_count >= 3 else "low",
    }

    if use_llm:
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
Target job: {target_job.get('job_title', '') if target_job else ''}
Open roles in current dataset: {len(company_jobs)}
Recent 7-day postings: {recent_count}
Top observed skills: {payload['top_hiring_skills']}
"""
        payload["llm_company_brief"] = call_llm_for_json(prompt)

    return {
        "status": "completed",
        "summary": f"Generated company research brief for {company}.",
        "data": {"company_research": payload},
        "next_actions": [
            "Use top_hiring_skills to tailor resume bullets and interview examples.",
            "Prioritize applications if hiring_signal is high.",
        ],
    }


def tailor_resume_service(
    *,
    structured_resume: dict[str, Any],
    target_job_url: str | None = None,
    job_keyword: str | None = None,
    company: str | None = None,
    job_title: str | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    if not isinstance(structured_resume, dict):
        raise ValueError("structured_resume is required.")
    target_job = _resolve_target_job(
        {
            "target_job_url": target_job_url,
            "job_keyword": job_keyword,
            "company": company,
            "job_title": job_title,
        }
    )
    if not target_job:
        raise ValueError("No target job could be resolved.")

    resume_skills = _extract_resume_skills(structured_resume)
    job_skills = [str(skill).strip().lower() for skill in target_job.get("processed_skills", []) or [] if str(skill).strip()]
    missing = [skill for skill in job_skills if skill not in set(resume_skills)]
    highlights = [skill for skill in job_skills if skill in set(resume_skills)]
    bullets = _extract_resume_bullets(structured_resume)

    guidance = {
        "target_job": _compact_job_summary(target_job),
        "recommended_headline": f"{target_job.get('job_title', '')} candidate with strong delivery track record",
        "prioritize_skills": highlights[:12],
        "add_or_emphasize_keywords": missing[:12],
        "experience_bullets_to_reposition": bullets[:6],
        "section_order_recommendation": ["Summary", "Skills", "Experience", "Projects", "Education"],
    }

    if use_llm:
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
  "job_title": "{target_job.get('job_title', '')}",
  "company": "{target_job.get('company', '')}",
  "description": "{str(target_job.get('description', '') or '')[:1800]}",
  "processed_summary": "{str(target_job.get('processed_summary', '') or '')[:1000]}",
  "processed_skills": {target_job.get('processed_skills', []) or []}
}}

Resume summary:
{str(structured_resume.get('summary', '') or '')[:1200]}

Resume highlights skills:
{highlights}

Missing skills:
{missing}

Candidate bullets:
{bullets}
"""
        guidance["llm_variant"] = call_llm_for_json(prompt)

    return {
        "status": "completed",
        "summary": f"Generated tailoring guidance for {target_job.get('job_title', '')} at {target_job.get('company', '')}.",
        "data": {"resume_tailoring": guidance},
        "next_actions": [
            "Apply keyword and bullet-positioning changes to a role-specific resume copy.",
            "Run ats_optimization on the tailored version.",
        ],
    }


def ats_optimization_service(
    *,
    structured_resume: dict[str, Any],
    target_job_url: str | None = None,
    job_keyword: str | None = None,
    company: str | None = None,
    job_title: str | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    if not isinstance(structured_resume, dict):
        raise ValueError("structured_resume is required.")
    target_job = _resolve_target_job(
        {
            "target_job_url": target_job_url,
            "job_keyword": job_keyword,
            "company": company,
            "job_title": job_title,
        }
    )
    if not target_job:
        raise ValueError("No target job could be resolved.")

    resume_skills = _extract_resume_skills(structured_resume)
    resume_set = set(resume_skills)
    job_skills = [str(skill).strip().lower() for skill in target_job.get("processed_skills", []) or [] if str(skill).strip()]
    unique_job_skills = list(dict.fromkeys(job_skills))
    matched = [skill for skill in unique_job_skills if skill in resume_set]
    missing = [skill for skill in unique_job_skills if skill not in resume_set]
    coverage = (len(matched) / len(unique_job_skills)) if unique_job_skills else 0.0
    ats_score = int(round(max(0.0, min(1.0, coverage)) * 100))

    payload: dict[str, Any] = {
        "target_job": _compact_job_summary(target_job),
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

    if use_llm:
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

Job title: {target_job.get('job_title', '')}
Company: {target_job.get('company', '')}
Job summary: {str(target_job.get('processed_summary', '') or '')[:1400]}
Job skills: {target_job.get('processed_skills', []) or []}

Resume summary: {str(structured_resume.get('summary', '') or '')[:1200]}
Resume skills: {structured_resume.get('skills', []) or []}
Matched keywords: {matched}
Missing keywords: {missing}
"""
        payload["llm_recommendations"] = call_llm_for_json(prompt)

    return {
        "status": "completed",
        "summary": (
            f"ATS optimization completed for {target_job.get('job_title', '')} "
            f"at {target_job.get('company', '')}: score={ats_score}."
        ),
        "data": {"ats_optimization": payload},
        "next_actions": [
            "Incorporate priority missing keywords with concrete project/impact evidence.",
            "Re-run ATS optimization after edits.",
        ],
    }


def interview_prep_service(
    *,
    structured_resume: dict[str, Any],
    target_job_url: str | None = None,
    job_keyword: str | None = None,
    company: str | None = None,
    job_title: str | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    if not isinstance(structured_resume, dict):
        raise ValueError("structured_resume is required.")
    target_job = _resolve_target_job(
        {
            "target_job_url": target_job_url,
            "job_keyword": job_keyword,
            "company": company,
            "job_title": job_title,
        }
    )
    if not target_job:
        raise ValueError("No target job could be resolved.")

    resume_skills = _extract_resume_skills(structured_resume)
    job_skills = [str(skill).strip().lower() for skill in target_job.get("processed_skills", []) or [] if str(skill).strip()]
    overlap = [skill for skill in job_skills if skill in set(resume_skills)]
    gaps = [skill for skill in job_skills if skill not in set(resume_skills)]

    pack: dict[str, Any] = {
        "target_job": _compact_job_summary(target_job),
        "likely_questions": [
            f"Walk me through a project most relevant to this {target_job.get('job_title', '')} role.",
            "How do you prioritize trade-offs between speed, quality, and reliability?",
            "Describe a production issue you debugged and resolved.",
        ] + (["How do you mentor engineers and influence architecture decisions?"] if str(target_job.get("processed_seniority") or "").strip().lower() in {"senior", "staff", "lead"} else []),
        "talking_points": [f"Lead with impact examples tied to {skill}." for skill in overlap[:6]],
        "gap_mitigation_points": [f"Prepare a bridge narrative for {skill} with adjacent experience." for skill in gaps[:5]],
        "star_story_prompts": [
            f"STAR story from {str(item.get('title') or '').strip() or 'role'} at {str(item.get('company') or '').strip() or 'company'}: problem, action, measurable impact."
            for item in structured_resume.get("experience", []) or []
        ][:6],
    }

    if use_llm:
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
  "title": "{target_job.get('job_title', '')}",
  "company": "{target_job.get('company', '')}",
  "summary": "{str(target_job.get('processed_summary', '') or '')[:1500]}",
  "skills": {target_job.get('processed_skills', []) or []},
  "seniority": "{target_job.get('processed_seniority', '')}"
}}

Resume summary: {str(structured_resume.get('summary', '') or '')[:1200]}
Resume skills: {structured_resume.get('skills', []) or []}
Overlap skills: {overlap}
Gap skills: {gaps}
"""
        pack["llm_interview_pack"] = call_llm_for_json(prompt)

    return {
        "status": "completed",
        "summary": f"Generated interview prep pack for {target_job.get('job_title', '')} at {target_job.get('company', '')}.",
        "data": {"interview_prep": pack},
        "next_actions": [
            "Draft 2 STAR stories aligned to top job skills.",
            "Rehearse answers for likely technical and behavioral questions.",
        ],
    }


def career_coach_service(
    *,
    structured_resume: dict[str, Any],
    source: str | None = None,
    use_llm: bool = False,
    market_sample_limit: int = 400,
) -> dict[str, Any]:
    if not isinstance(structured_resume, dict):
        raise ValueError("structured_resume is required.")

    jobs = get_matchable_jobs(source=source, limit=max(25, int(market_sample_limit)))
    if not jobs:
        raise ValueError("No matchable jobs available for market analysis.")

    resume_skills = _extract_resume_skills(structured_resume)
    gaps = _top_missing_skills(resume_skills=resume_skills, jobs=jobs, limit=12)

    plan: dict[str, Any] = {
        "resume_skills_count": len(resume_skills),
        "market_sample_size": len(jobs),
        "top_skill_gaps": gaps,
        "90_day_plan": [
            {
                "window": "weeks_1_4",
                "focus_skills": [item["skill"] for item in gaps[:2]],
                "outcome": "Build one portfolio-quality project artifact using these skills.",
            },
            {
                "window": "weeks_5_8",
                "focus_skills": [item["skill"] for item in gaps[2:4]],
                "outcome": "Demonstrate production-style depth and measurable impact.",
            },
            {
                "window": "weeks_9_12",
                "focus_skills": [item["skill"] for item in gaps[4:6]],
                "outcome": "Translate outcomes into resume bullets and interview stories.",
            },
        ],
    }

    if use_llm:
        prompt = f"""
You are a career coach for software roles.
Return ONLY JSON:
{{
  "positioning_summary": "string",
  "priority_skill_investments": ["string"],
  "portfolio_project_ideas": ["string"],
  "application_strategy": ["string"]
}}

Resume summary: {str(structured_resume.get('summary', '') or '')[:1200]}
Resume skills: {structured_resume.get('skills', []) or []}
Market sample size: {len(jobs)}
Top gaps: {gaps}
"""
        plan["llm_career_plan"] = call_llm_for_json(prompt)

    return {
        "status": "completed",
        "summary": f"Generated career coaching plan from {len(jobs)} market jobs.",
        "data": {"career_coach": plan},
        "next_actions": [
            "Pick 2 high-frequency gaps for focused upskilling this month.",
            "Update resume/projects with evidence for newly covered skills.",
        ],
    }


def recommend_service_plan(context: dict[str, Any]) -> dict[str, Any]:
    intent = str(context.get("intent") or "auto").strip() or "auto"
    outputs: dict[str, Any] = {}
    next_actions: list[str] = []

    if intent in {"auto", "bootstrap", "full_assistant"}:
        if "resume_text" in context or "structured_resume" in context:
            outputs["resume_analysis"] = analyze_resume_service(
                structured_resume=context.get("structured_resume"),
                resume_text=context.get("resume_text"),
                normalize_resume_skills=bool(context.get("normalize_resume_skills", True)),
            )
        if context.get("keywords") or context.get("target_roles"):
            outputs["job_discovery"] = discover_jobs_service(
                keywords=str(context.get("keywords") or (context.get("target_roles") or ["software engineer"])[0]),
                location=str(context.get("location") or "remote"),
                sources=context.get("sources"),
                max_results_per_source=int(context.get("max_results_per_source", 25)),
                save_to_db=bool(context.get("save_to_db", True)),
            )

    elif intent == "analyze_resume":
        outputs["resume_analysis"] = analyze_resume_service(
            structured_resume=context.get("structured_resume"),
            resume_text=context.get("resume_text"),
            normalize_resume_skills=bool(context.get("normalize_resume_skills", True)),
        )
    elif intent == "discover_jobs":
        outputs["job_discovery"] = discover_jobs_service(
            keywords=str(context.get("keywords") or "software engineer"),
            location=str(context.get("location") or "remote"),
            sources=context.get("sources"),
            max_results_per_source=int(context.get("max_results_per_source", 25)),
            save_to_db=bool(context.get("save_to_db", True)),
        )
    elif intent == "research_company":
        outputs["company_research"] = research_company_service(
            company=str(context.get("company") or "").strip(),
            target_job_url=context.get("target_job_url"),
            source=str(context.get("source") or "").strip() or None,
            limit=int(context.get("company_research_limit", 500)),
            use_llm=bool(context.get("use_llm", False)),
        )
    elif intent == "track_application":
        outputs["application_tracker"] = application_tracker_service(context)
    elif intent == "tailor_resume":
        resume = context.get("structured_resume")
        if not isinstance(resume, dict):
            raise ValueError("structured_resume is required for resume tailoring.")
        outputs["resume_tailoring"] = tailor_resume_service(
            structured_resume=resume,
            target_job_url=context.get("target_job_url"),
            job_keyword=context.get("job_keyword"),
            company=context.get("company"),
            job_title=context.get("job_title"),
            use_llm=bool(context.get("use_llm", False)),
        )
    elif intent == "optimize_ats":
        resume = context.get("structured_resume")
        if not isinstance(resume, dict):
            raise ValueError("structured_resume is required for ATS optimization.")
        outputs["ats_optimization"] = ats_optimization_service(
            structured_resume=resume,
            target_job_url=context.get("target_job_url"),
            job_keyword=context.get("job_keyword"),
            company=context.get("company"),
            job_title=context.get("job_title"),
            use_llm=bool(context.get("use_llm", False)),
        )
    elif intent == "prepare_interview" or intent == "interview_prep":
        resume = context.get("structured_resume")
        if not isinstance(resume, dict):
            raise ValueError("structured_resume is required for interview prep.")
        outputs["interview_prep"] = interview_prep_service(
            structured_resume=resume,
            target_job_url=context.get("target_job_url"),
            job_keyword=context.get("job_keyword"),
            company=context.get("company"),
            job_title=context.get("job_title"),
            use_llm=bool(context.get("use_llm", False)),
        )
    elif intent == "career_coaching" or intent == "career_coach":
        resume = context.get("structured_resume")
        if not isinstance(resume, dict):
            raise ValueError("structured_resume is required for career coaching.")
        outputs["career_coach"] = career_coach_service(
            structured_resume=resume,
            source=str(context.get("source") or "").strip() or None,
            use_llm=bool(context.get("use_llm", False)),
            market_sample_limit=int(context.get("career_market_sample_limit", 400)),
        )

    for value in outputs.values():
        next_actions.extend(value.get("next_actions", []))

    status = "completed" if outputs else "skipped"
    return {
        "status": status,
        "execution_plan": list(outputs.keys()),
        "available_services": [
            "resume_analysis",
            "job_discovery",
            "company_research",
            "application_tracker",
            "resume_tailoring",
            "ats_optimization",
            "interview_prep",
            "career_coach",
        ],
        "outputs": outputs,
        "next_actions": list(dict.fromkeys(next_actions)),
    }


def application_tracker_service(context: dict[str, Any]) -> dict[str, Any]:
    action = str(context.get("application_action", "list")).strip().lower()

    if action in {"track", "upsert", "save"}:
        email = str(context.get("email") or "").strip().lower()
        if not email:
            raise ValueError("application tracking requires email.")
        target_job = _resolve_target_job(context)
        if not target_job:
            raise ValueError("application tracking requires resolvable target_job/target_job_url/job_keyword.")

        status = str(context.get("application_status", "saved")).strip().lower() or "saved"
        follow_up_due_at = _parse_optional_datetime(context.get("follow_up_due_at"))
        notes = str(context.get("application_notes") or "").strip()

        record_id = upsert_application_record(
            email=email,
            job_url=str(target_job.get("url") or ""),
            company=str(target_job.get("company") or ""),
            job_title=str(target_job.get("job_title") or ""),
            status=status,
            follow_up_due_at=follow_up_due_at,
            notes=notes,
        )

        return {
            "status": "completed",
            "summary": f"Tracked application record #{record_id} for {email}.",
            "data": {
                "application_record_id": record_id,
                "application_record": {
                    "email": email,
                    "status": status,
                    "follow_up_due_at": follow_up_due_at,
                    "notes": notes,
                    "target_job": _compact_job_summary(target_job),
                },
            },
            "next_actions": [
                "Update status to interviewing/offer/rejected as process moves.",
                "Set follow_up_due_at for pending recruiter responses.",
            ],
        }

    if action in {"update", "update_status"}:
        record_id_raw = context.get("application_record_id")
        if record_id_raw is None:
            raise ValueError("update_status requires application_record_id.")
        record_id = int(record_id_raw)
        status = str(context.get("application_status") or "").strip().lower()
        if not status:
            raise ValueError("update_status requires application_status.")

        notes = str(context.get("application_notes") or "").strip()
        follow_up_due_at = _parse_optional_datetime(context.get("follow_up_due_at"))
        updated = update_application_record_status(
            record_id=record_id,
            status=status,
            notes=notes,
            follow_up_due_at=follow_up_due_at,
        )
        return {
            "status": "completed",
            "summary": f"Updated {updated} application record(s) for record_id={record_id}.",
            "data": {"updated_records": updated},
            "next_actions": ["List records to confirm next follow-up queue."],
        }

    if action == "list":
        email = str(context.get("email") or "").strip().lower() or None
        status = str(context.get("application_status") or "").strip().lower() or None
        limit = max(1, int(context.get("application_limit", 50)))
        records = list_application_records(email=email, status=status, limit=limit)
        return {
            "status": "completed",
            "summary": f"Fetched {len(records)} tracked application record(s).",
            "data": {"application_records": records},
            "next_actions": ["Use record_id + update_status to progress each application stage."],
        }

    raise ValueError("application_action must be one of: list, track, upsert, save, update, update_status.")


def _parse_optional_datetime(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    datetime.fromisoformat(raw)
    return raw
