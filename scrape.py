"""
scrape.py
---------
CLI entry point for the job scraping pipeline.

Usage examples
--------------
# One-shot: scrape all sources, save to DB
python scrape.py --keywords "python engineer" --location "remote"

# One-shot: HTTP-only sources (no Playwright required)
python scrape.py --keywords "backend engineer" --source hn --source greenhouse

# One-shot: dry run (no DB write, print JSON to stdout)
python scrape.py --keywords "data scientist" --no-db

# Apply the PostgreSQL schema (run once after setting DATABASE_URL)
python scrape.py --apply-schema

# Start the scheduled scraper (runs every N hours, set via SCRAPE_INTERVAL_HOURS)
python scrape.py --schedule
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    from src.scrapers.orchestrator import ALL_SOURCES

    p = argparse.ArgumentParser(
        prog="scrape",
        description="Collect job listings from multiple sources and store in PostgreSQL.",
    )
    p.add_argument(
        "--keywords", "-k",
        default="software engineer",
        help='Search terms (default: "software engineer").',
    )
    p.add_argument(
        "--location", "-l",
        default="remote",
        help='Location filter (default: "remote").',
    )
    p.add_argument(
        "--source", "-s",
        dest="sources",
        action="append",
        metavar="SOURCE",
        choices=ALL_SOURCES,
        help=f"Scrape a specific source. Repeatable. Choices: {ALL_SOURCES}. "
             "Default: all sources.",
    )
    p.add_argument(
        "--max", "-m",
        type=int,
        default=50,
        dest="max_results",
        help="Max results per source (default: 50).",
    )
    p.add_argument(
        "--no-db",
        action="store_true",
        default=False,
        help="Do not write to PostgreSQL — print results as JSON instead.",
    )
    p.add_argument(
        "--apply-schema",
        action="store_true",
        default=False,
        help="Apply src/db/schema.sql to the configured DATABASE_URL and exit.",
    )
    p.add_argument(
        "--schedule",
        action="store_true",
        default=False,
        help="Start the APScheduler loop (runs every SCRAPE_INTERVAL_HOURS hours).",
    )
    return p


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_apply_schema() -> None:
    from src.db.db import apply_schema
    logger.info("Applying schema …")
    apply_schema()
    logger.info("Schema applied successfully.")


def cmd_scrape(
    keywords: str,
    location: str,
    sources: list[str] | None,
    max_results: int,
    save_to_db: bool,
) -> None:
    from src.scrapers.orchestrator import run_all_scrapers

    logger.info("Starting scrape: keywords=%r location=%r sources=%s", keywords, location, sources or "all")

    summary = run_all_scrapers(
        keywords=keywords,
        location=location,
        sources=sources,
        max_results_per_source=max_results,
        save_to_db=save_to_db,
    )

    print("\n── Scrape Summary ──────────────────────────────────")
    print(f"  Total scraped : {summary['total_scraped']}")
    print(f"  Unique jobs   : {summary['total_unique']}")
    if save_to_db:
        print(f"  DB inserted   : {summary['db_inserted']}")
        print(f"  DB updated    : {summary['db_updated']}")
        mode = summary.get("processing_mode", "sync")
        print(f"  Processing    : {mode}")
        if mode == "background":
            print(f"  Enqueued      : {summary.get('enqueued', 0)}")
            queue_status = summary.get("queue_status") or {}
            if queue_status:
                print(
                    "  Queue status  : "
                    f"started={queue_status.get('started')} "
                    f"jobs={queue_status.get('queued_jobs')} "
                    f"processing={queue_status.get('processing_jobs')} "
                    f"failed={queue_status.get('failed_jobs')} "
                    f"total={queue_status.get('total_jobs')}"
                )
        else:
            print(f"  Processed     : {summary.get('processed', 0)}")
    else:
        print("  DB write      : skipped (--no-db)")

    print("\n  By source:")
    for src, count in summary["by_source"].items():
        print(f"    {src:<15} {count}")

    if summary["errors"]:
        print("\n  Errors:")
        for src, msg in summary["errors"].items():
            print(f"    {src:<15} {msg}")

    print("────────────────────────────────────────────────────\n")

    if not save_to_db:
        print(json.dumps(summary, indent=2))


def cmd_schedule() -> None:
    from src.scheduler import start_scheduler
    start_scheduler()


# ── entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    try:
        if args.apply_schema:
            cmd_apply_schema()
            return

        if args.schedule:
            cmd_schedule()
            return

        cmd_scrape(
            keywords   = args.keywords,
            location   = args.location,
            sources    = args.sources,
            max_results= args.max_results,
            save_to_db = not args.no_db,
        )

    except EnvironmentError as exc:
        print(f"\n❌  Config error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        # Cleanly shut down the psycopg connection pool (suppresses thread warnings)
        try:
            from src.db.db import close_pool
            close_pool()
        except Exception:
            pass


if __name__ == "__main__":
    main()
