"""
src/scrapers/orchestrator.py
----------------------------
Run all scrapers (or a chosen subset), deduplicate by URL,
and bulk-upsert to PostgreSQL.

Usage (from code):
    from src.scrapers.orchestrator import run_all_scrapers
    stats = run_all_scrapers(keywords="python engineer", location="remote")

    # Or a specific source only:
    stats = run_all_scrapers(keywords="backend", location="remote", sources=["hn", "greenhouse"])
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from src.scrapers.models import JobListing

logger = logging.getLogger(__name__)

BACKGROUND_PROCESSING_ENABLED = (
    os.environ.get("JOB_PROCESSING_BACKGROUND", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Registry maps source name → lazy import factory.
# Playwright scrapers are only imported when needed so the process doesn't
# require a browser install just to run HTTP-only scrapers.
_SCRAPER_FACTORIES: dict[str, Callable] = {}


def _register_scrapers() -> None:
    global _SCRAPER_FACTORIES
    if _SCRAPER_FACTORIES:
        return  # already registered

    from src.scrapers.hn_jobs    import HNJobsScraper
    from src.scrapers.greenhouse import GreenhouseScraper
    from src.scrapers.linkedin   import LinkedInScraper

    _SCRAPER_FACTORIES = {
        "hn":         HNJobsScraper,
        "greenhouse": GreenhouseScraper,
        "linkedin":   LinkedInScraper,
    }


ALL_SOURCES = ["hn", "greenhouse", "linkedin"]
HTTP_SOURCES = ["hn", "greenhouse"]
PLAYWRIGHT_SOURCES = ["linkedin"]


def run_all_scrapers(
    keywords: str,
    location: str = "",
    *,
    sources: list[str] | None = None,
    max_results_per_source: int = 50,
    save_to_db: bool = True,
) -> dict:
    """
    Orchestrate multiple scrapers, deduplicate results, and (optionally) save to DB.

    Args:
        keywords:               Search terms.
        location:               City / region / "remote".
        sources:                List of source names to run. Defaults to ALL_SOURCES.
        max_results_per_source: Cap per scraper.
        save_to_db:             If True, upsert results to PostgreSQL.

    Returns:
        Summary dict:
            {
              "total_scraped": int,
              "total_unique": int,
              "db_inserted": int,
              "db_updated": int,
              "by_source": { source: count, … },
              "errors": { source: message, … },
            }
    """
    _register_scrapers()

    active_sources = sources or ALL_SOURCES
    invalid = [s for s in active_sources if s not in _SCRAPER_FACTORIES]
    if invalid:
        raise ValueError(f"Unknown sources: {invalid}. Valid: {list(_SCRAPER_FACTORIES)}")

    all_jobs: list[JobListing] = []
    by_source: dict[str, int] = {}
    errors:    dict[str, str] = {}

    for source in active_sources:
        scraper = _SCRAPER_FACTORIES[source]()
        logger.info("▶  Running scraper: %s", source)
        try:
            jobs = scraper.scrape(keywords, location, max_results=max_results_per_source)
            by_source[source] = len(jobs)
            all_jobs.extend(jobs)
            logger.info("   %s → %d jobs", source, len(jobs))
        except Exception as exc:
            errors[source] = str(exc)
            logger.error("   %s → ERROR: %s", source, exc)

    # Deduplicate by URL (preserve first occurrence = highest-priority source)
    seen: set[str] = set()
    unique_jobs: list[JobListing] = []
    for job in all_jobs:
        if job.url and job.url not in seen:
            seen.add(job.url)
            unique_jobs.append(job)

    total_scraped = len(all_jobs)
    total_unique  = len(unique_jobs)
    db_inserted   = 0
    db_updated    = 0

    if save_to_db and unique_jobs:
        from src.db.db import upsert_jobs
        try:
            result = upsert_jobs(unique_jobs)
            db_inserted = result["inserted"]
            db_updated  = result["updated"]
        except Exception as exc:
            logger.error("DB upsert failed: %s", exc)
            errors["db"] = str(exc)

    processed = 0
    enqueued = 0
    processing_mode = "sync"
    queue_status: dict | None = None
    processing_errors: dict[str, str] = {}
    if save_to_db and unique_jobs and "db" not in errors:
        if BACKGROUND_PROCESSING_ENABLED:
            processing_mode = "background"
            from src.job_processing_queue import enqueue_job_processing, get_queue_status
            try:
                enqueued = enqueue_job_processing(unique_jobs)
                status = get_queue_status()
                queue_status = {
                    "started": status.started,
                    "queued_jobs": status.queued_jobs,
                    "processing_jobs": status.processing_jobs,
                    "done_jobs": status.done_jobs,
                    "failed_jobs": status.failed_jobs,
                    "total_jobs": status.total_jobs,
                }
                if enqueued == 0 and unique_jobs:
                    processing_errors["job_processing_queue"] = (
                        "Background queue is full; jobs were not enqueued."
                    )
            except Exception as exc:
                logger.error("Job enqueue failed: %s", exc)
                processing_errors["job_processing_queue"] = str(exc)
        else:
            from src.job_processor import process_job_listings
            try:
                processing_result = process_job_listings(unique_jobs)
                processed = processing_result["processed"]
                processing_errors = processing_result["errors"]
            except Exception as exc:
                logger.error("Job processing failed: %s", exc)
                processing_errors["job_processing"] = str(exc)

    summary = {
        "total_scraped": total_scraped,
        "total_unique":  total_unique,
        "db_inserted":   db_inserted,
        "db_updated":    db_updated,
        "processed":     processed,
        "processing_mode": processing_mode,
        "enqueued":      enqueued,
        "queue_status":  queue_status,
        "by_source":     by_source,
        "errors":        {**errors, **processing_errors},
    }
    logger.info("Orchestrator done: %s", summary)
    return summary
