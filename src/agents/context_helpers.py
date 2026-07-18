"""
src/agents/context_helpers.py
-----------------------------
Shared helpers for resolving resume/job context in specialized agents.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from src.db.db import get_job_by_url, get_matchable_jobs


def resolve_target_job(context: dict[str, Any]) -> dict[str, Any] | None:
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


def compact_job_summary(job: dict[str, Any]) -> dict[str, Any]:
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


def extract_resume_skills(structured_resume: dict[str, Any]) -> list[str]:
    raw = structured_resume.get("skills", []) or []
    return _dedupe([str(skill).strip().lower() for skill in raw if str(skill).strip()])


def extract_resume_bullets(structured_resume: dict[str, Any], *, limit: int = 8) -> list[str]:
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


def top_missing_skills(
    *,
    resume_skills: list[str],
    jobs: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    resume_set = set(resume_skills)
    counter: Counter[str] = Counter()
    for job in jobs:
        for skill in list(job.get("processed_skills") or []):
            normalized = str(skill).strip().lower()
            if normalized and normalized not in resume_set:
                counter[normalized] += 1

    return [
        {"skill": skill, "job_frequency": count}
        for skill, count in counter.most_common(max(1, limit))
    ]


def filter_jobs_for_company(jobs: list[dict[str, Any]], company: str) -> list[dict[str, Any]]:
    key = company.strip().lower()
    return [job for job in jobs if str(job.get("company") or "").strip().lower() == key]


def jobs_recent_count(jobs: list[dict[str, Any]], *, days: int = 7) -> int:
    now = datetime.now().timestamp()
    horizon_seconds = max(1, int(days)) * 86400
    total = 0
    for job in jobs:
        scraped_at = job.get("scraped_at")
        if isinstance(scraped_at, datetime):
            age = now - scraped_at.timestamp()
            if age <= horizon_seconds:
                total += 1
    return total


def source_breakdown(jobs: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for job in jobs:
        source = str(job.get("source") or "unknown").strip().lower() or "unknown"
        counter[source] += 1
    return dict(counter)


def _job_matches(
    job: dict[str, Any],
    *,
    keyword: str,
    company: str,
    title: str,
) -> bool:
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


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
