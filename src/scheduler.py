"""
src/scheduler.py
----------------
APScheduler-based job scraping scheduler.

Reads configuration from environment variables:

  SCRAPE_KEYWORDS        — comma-separated search terms
                           e.g. "python engineer,backend developer"
  SCRAPE_LOCATION        — location filter  (default: "remote")
  SCRAPE_INTERVAL_HOURS  — how often to run (default: 4)
  SCRAPE_SOURCES         — comma-separated source list (default: all)
                           e.g. "hn,greenhouse,lever"
  SCRAPE_MAX_PER_SOURCE  — max results per scraper per run (default: 50)

Usage:
  python scrape.py --schedule
"""

from __future__ import annotations

import logging
import os

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval   import IntervalTrigger

from src.scrapers.orchestrator import run_all_scrapers, ALL_SOURCES

logger = logging.getLogger(__name__)


def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, "").strip()
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else default


def _scrape_job() -> None:
    """Called by APScheduler on each tick."""
    keywords_list = _env_list("SCRAPE_KEYWORDS", ["software engineer"])
    location      = os.environ.get("SCRAPE_LOCATION", "remote").strip()
    sources       = _env_list("SCRAPE_SOURCES", ALL_SOURCES)
    max_per       = int(os.environ.get("SCRAPE_MAX_PER_SOURCE", "50"))

    for keywords in keywords_list:
        logger.info("Scheduled scrape: keywords=%r location=%r", keywords, location)
        summary = run_all_scrapers(
            keywords=keywords,
            location=location,
            sources=sources,
            max_results_per_source=max_per,
            save_to_db=True,
        )
        logger.info(
            "Done — scraped=%d unique=%d inserted=%d updated=%d errors=%s",
            summary["total_scraped"],
            summary["total_unique"],
            summary["db_inserted"],
            summary["db_updated"],
            summary["errors"] or "none",
        )


def start_scheduler() -> None:
    """Start the blocking APScheduler loop."""
    interval_hours = float(os.environ.get("SCRAPE_INTERVAL_HOURS", "4"))

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        _scrape_job,
        trigger=IntervalTrigger(hours=interval_hours),
        id="scrape_jobs",
        name="Job scraping run",
        replace_existing=True,
        # Also run immediately on start so you don't wait the full interval
        next_run_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )

    logger.info(
        "Scheduler started — running every %.1f hour(s). Press Ctrl+C to stop.",
        interval_hours,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
