"""
src/notification_service.py
---------------------------
Build and send strong-match digest notifications.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import os

from src.db.db import get_matchable_jobs
from src.matcher import load_resume_artifacts, rank_jobs_for_resume, MatchResult
from src.notifiers.email_notifier import send_email


DEFAULT_TIMEZONE = os.environ.get("NOTIFY_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"
DEFAULT_TOP_K = int(os.environ.get("NOTIFY_TOP_K", "10") or "10")
DEFAULT_MIN_TOP_APPLICANT = int(os.environ.get("NOTIFY_MIN_TOP_APPLICANT", "85") or "85")
DEFAULT_MIN_RANKING_SCORE = float(os.environ.get("NOTIFY_MIN_RANKING_SCORE", "0.85") or "0.85")
DEFAULT_CANDIDATE_LIMIT = int(os.environ.get("NOTIFY_CANDIDATE_LIMIT", "5000") or "5000")


def _parse_pref_locations() -> list[str]:
    raw = os.environ.get("NOTIFY_PREFERRED_LOCATIONS", "").strip()
    return [value.strip() for value in raw.split(",") if value.strip()] if raw else []


def _is_today_in_timezone(value: datetime, *, timezone_name: str) -> bool:
    tz = ZoneInfo(timezone_name)
    local_day = datetime.now(tz).date()
    as_local = value.astimezone(tz) if value.tzinfo else value.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    return as_local.date() == local_day


def _job_matches_keyword(job: dict, keyword: str) -> bool:
    key = str(keyword or "").strip().lower()
    if not key:
        return True
    haystack = " ".join(
        [
            str(job.get("job_title", "") or ""),
            str(job.get("company", "") or ""),
            str(job.get("location", "") or ""),
            str(job.get("description", "") or ""),
            str(job.get("processed_summary", "") or ""),
            str(job.get("processed_experience_text", "") or ""),
        ]
    ).lower()
    terms = [term for term in key.split() if term]
    return bool(terms) and all(term in haystack for term in terms)


def collect_strong_matches_today(
    *,
    resume_json_path: str = "output/structured_resume.json",
    resume_embeddings_path: str = "output/structured_resume.embeddings.npz",
    timezone_name: str = DEFAULT_TIMEZONE,
    top_k: int = DEFAULT_TOP_K,
    min_top_applicant: int = DEFAULT_MIN_TOP_APPLICANT,
    min_ranking_score: float = DEFAULT_MIN_RANKING_SCORE,
    source: str | None = None,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    job_keyword: str | None = None,
) -> list[MatchResult]:
    """Compute today's strongest matches based on ranking and top-applicant thresholds."""
    structured_resume, profile_embedding = load_resume_artifacts(
        resume_json_path,
        resume_embeddings_path,
    )

    jobs = get_matchable_jobs(source=source, limit=max(1, candidate_limit))
    jobs_today = [
        job for job in jobs
        if isinstance(job.get("scraped_at"), datetime)
        and _is_today_in_timezone(job["scraped_at"], timezone_name=timezone_name)
    ]

    if job_keyword and str(job_keyword).strip():
        jobs_today = [job for job in jobs_today if _job_matches_keyword(job, job_keyword)]

    if not jobs_today:
        return []

    ranked = rank_jobs_for_resume(
        structured_resume=structured_resume,
        resume_embedding=profile_embedding,
        job_rows=jobs_today,
        top_k=max(10, len(jobs_today)),
        min_score=0.0,
        preferred_locations=_parse_pref_locations(),
    )

    strong = [
        match for match in ranked
        if match.top_applicant_score >= int(min_top_applicant)
        and match.ranking_score >= float(min_ranking_score)
    ]
    strong.sort(key=lambda item: (item.ranking_score, item.top_applicant_score, item.score), reverse=True)
    return strong[: max(1, top_k)]


def build_daily_email_body(matches: list[MatchResult], *, timezone_name: str) -> str:
    tz = ZoneInfo(timezone_name)
    now = datetime.now(tz)

    if not matches:
        return (
            f"JobFinder daily digest ({now.strftime('%Y-%m-%d %H:%M %Z')})\n\n"
            "No strong matches found today for the configured thresholds."
        )

    lines: list[str] = [
        f"JobFinder daily digest ({now.strftime('%Y-%m-%d %H:%M %Z')})",
        "",
        f"Top {len(matches)} jobs today where you are a strong candidate:",
        "",
    ]

    for idx, match in enumerate(matches, start=1):
        lines.append(
            f"{idx}. {match.job_title} @ {match.company} [{match.source}]"
        )
        lines.append(
            "   "
            f"rank={match.ranking_score:.3f} "
            f"top_applicant={match.top_applicant_score}/100 "
            f"final={match.score:.3f}"
        )
        lines.append(
            "   "
            f"salary={match.salary or 'n/a'} "
            f"seniority={match.seniority or 'unknown'} "
            f"location={match.location or 'n/a'}"
        )
        if match.top_applicant_reasons:
            lines.append("   reasons: " + " | ".join(match.top_applicant_reasons))
        lines.append(f"   {match.url}")
        lines.append("")

    return "\n".join(lines).strip()


def send_daily_email_digest(
    *,
    resume_json_path: str = "output/structured_resume.json",
    resume_embeddings_path: str = "output/structured_resume.embeddings.npz",
    timezone_name: str = DEFAULT_TIMEZONE,
    top_k: int = DEFAULT_TOP_K,
    min_top_applicant: int = DEFAULT_MIN_TOP_APPLICANT,
    min_ranking_score: float = DEFAULT_MIN_RANKING_SCORE,
    source: str | None = None,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    dry_run: bool = False,
    job_keyword: str | None = None,
) -> dict:
    """Compute and send today's strong-match email digest."""
    matches = collect_strong_matches_today(
        resume_json_path=resume_json_path,
        resume_embeddings_path=resume_embeddings_path,
        timezone_name=timezone_name,
        top_k=top_k,
        min_top_applicant=min_top_applicant,
        min_ranking_score=min_ranking_score,
        source=source,
        candidate_limit=candidate_limit,
        job_keyword=job_keyword,
    )

    body = build_daily_email_body(matches, timezone_name=timezone_name)
    keyword_suffix = f' for "{job_keyword}"' if job_keyword and str(job_keyword).strip() else ""
    subject = f"JobFinder Daily: {len(matches)} strong matches today{keyword_suffix}"

    if not dry_run:
        send_email(subject=subject, body=body)

    return {
        "sent": not dry_run,
        "subject": subject,
        "timezone": timezone_name,
        "matches_count": len(matches),
        "job_keyword": job_keyword or "",
        "matches": [asdict(match) for match in matches],
        "body": body,
    }
