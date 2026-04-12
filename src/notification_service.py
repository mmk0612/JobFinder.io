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
    """Compute and send today's strong-match email digest with monitoring params."""
    # Step 1: Load all matchable jobs
    all_matchable = get_matchable_jobs(source=source, limit=max(1, candidate_limit))
    total_matchable = len(all_matchable)
    
    # Step 2: Filter to today's jobs
    jobs_today = [
        job for job in all_matchable
        if isinstance(job.get("scraped_at"), datetime)
        and _is_today_in_timezone(job["scraped_at"], timezone_name=timezone_name)
    ]
    total_today = len(jobs_today)
    
    # Step 3: Apply keyword filter if provided
    if job_keyword and str(job_keyword).strip():
        jobs_today_filtered = [job for job in jobs_today if _job_matches_keyword(job, job_keyword)]
    else:
        jobs_today_filtered = jobs_today
    total_keyword_match = len(jobs_today_filtered)
    
    # Step 4: Rank all jobs
    structured_resume, profile_embedding = load_resume_artifacts(
        resume_json_path,
        resume_embeddings_path,
    )
    
    ranked = rank_jobs_for_resume(
        structured_resume=structured_resume,
        resume_embedding=profile_embedding,
        job_rows=jobs_today_filtered,
        top_k=max(10, len(jobs_today_filtered)),
        min_score=0.0,
        preferred_locations=_parse_pref_locations(),
    )
    
    # Step 5: Filter by thresholds
    strong = [
        match for match in ranked
        if match.top_applicant_score >= int(min_top_applicant)
        and match.ranking_score >= float(min_ranking_score)
    ]
    
    # Step 6: Compute filtering metrics
    total_ranked = len(ranked)
    failed_top_applicant = sum(1 for m in ranked if m.top_applicant_score < int(min_top_applicant))
    failed_ranking_score = sum(1 for m in ranked if m.ranking_score < float(min_ranking_score))
    
    top_applicant_dist = [m.top_applicant_score for m in ranked]
    ranking_score_dist = [m.ranking_score for m in ranked]
    
    strong.sort(key=lambda item: (item.ranking_score, item.top_applicant_score, item.score), reverse=True)
    matches = strong[: max(1, top_k)]
    
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
        # Monitoring params for improvement analysis
        "monitoring": {
            "total_matchable_jobs": total_matchable,
            "total_today": total_today,
            "total_keyword_match": total_keyword_match,
            "total_ranked": total_ranked,
            "failed_top_applicant_threshold": failed_top_applicant,
            "failed_ranking_score_threshold": failed_ranking_score,
            "top_applicant_score_distribution": {
                "min": min(top_applicant_dist) if top_applicant_dist else None,
                "max": max(top_applicant_dist) if top_applicant_dist else None,
                "mean": sum(top_applicant_dist) / len(top_applicant_dist) if top_applicant_dist else None,
            },
            "ranking_score_distribution": {
                "min": min(ranking_score_dist) if ranking_score_dist else None,
                "max": max(ranking_score_dist) if ranking_score_dist else None,
                "mean": sum(ranking_score_dist) / len(ranking_score_dist) if ranking_score_dist else None,
            },
            "thresholds": {
                "min_top_applicant": min_top_applicant,
                "min_ranking_score": min_ranking_score,
            },
        },
    }
