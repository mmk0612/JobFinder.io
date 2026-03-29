"""
notify.py
---------
CLI entry point for daily strong-match notifications.

Examples
--------
# One-shot send (email)
python notify.py --once

# Preview email body without sending
python notify.py --once --dry-run

# Start daily scheduler
python notify.py --schedule
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger  # noqa: E402

from src.notification_service import send_daily_email_digest, DEFAULT_TIMEZONE  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notify",
        description="Send daily JobFinder strong-match notifications.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run notification job once and exit.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        default=False,
        help="Run scheduler loop for daily digest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute digest but do not send email.",
    )
    parser.add_argument(
        "--timezone",
        default=os.environ.get("NOTIFY_TIMEZONE", DEFAULT_TIMEZONE),
        help="Digest timezone (default from NOTIFY_TIMEZONE or Asia/Kolkata).",
    )
    parser.add_argument(
        "--time",
        default=os.environ.get("NOTIFY_DAILY_TIME", "09:00"),
        help="Daily schedule time in HH:MM (default: 09:00 or NOTIFY_DAILY_TIME).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("NOTIFY_TOP_K", "10") or "10"),
        help="Max jobs in digest (default: 10).",
    )
    parser.add_argument(
        "--min-top-applicant",
        type=int,
        default=int(os.environ.get("NOTIFY_MIN_TOP_APPLICANT", "85") or "85"),
        help="Minimum top applicant score threshold (default: 85).",
    )
    parser.add_argument(
        "--min-ranking-score",
        type=float,
        default=float(os.environ.get("NOTIFY_MIN_RANKING_SCORE", "0.85") or "0.85"),
        help="Minimum ranking score threshold in [0,1] (default: 0.85).",
    )
    parser.add_argument(
        "--resume-json",
        default="output/structured_resume.json",
        help="Path to structured resume JSON.",
    )
    parser.add_argument(
        "--resume-embeddings",
        default="output/structured_resume.embeddings.npz",
        help="Path to resume embeddings NPZ.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source filter (linkedin/hn/greenhouse).",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=int(os.environ.get("NOTIFY_CANDIDATE_LIMIT", "5000") or "5000"),
        help="Max candidate jobs considered before filtering (default: 5000).",
    )
    parser.add_argument(
        "--job-keyword",
        default=None,
        help="Optional keyword phrase filter applied to job title/company/description.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print result as JSON.",
    )
    return parser


def _run_once(args: argparse.Namespace) -> dict:
    return send_daily_email_digest(
        resume_json_path=args.resume_json,
        resume_embeddings_path=args.resume_embeddings,
        timezone_name=args.timezone,
        top_k=max(1, args.top_k),
        min_top_applicant=max(0, min(100, args.min_top_applicant)),
        min_ranking_score=max(0.0, min(1.0, args.min_ranking_score)),
        source=args.source,
        candidate_limit=max(1, args.candidate_limit),
        dry_run=args.dry_run,
        job_keyword=args.job_keyword,
    )


def _parse_hhmm(raw: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except Exception as exc:
        raise ValueError(f"Invalid --time value {raw!r}. Expected HH:MM.") from exc

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid --time value {raw!r}. Hour must be 0-23 and minute 0-59.")
    return hour, minute


def _start_scheduler(args: argparse.Namespace) -> None:
    tz = ZoneInfo(args.timezone)
    hour, minute = _parse_hhmm(args.time)

    scheduler = BlockingScheduler(timezone=tz)

    def _job() -> None:
        result = _run_once(args)
        logger.info(
            "Notification digest complete: sent=%s matches=%s",
            result["sent"],
            result["matches_count"],
        )

    scheduler.add_job(
        _job,
        trigger=CronTrigger(hour=hour, minute=minute, timezone=tz),
        id="daily_notification_digest",
        name="Daily strong-match email digest",
        replace_existing=True,
        next_run_time=datetime.now(tz),
    )

    logger.info(
        "Notification scheduler started — daily at %02d:%02d %s (press Ctrl+C to stop).",
        hour,
        minute,
        args.timezone,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Notification scheduler stopped.")


def main() -> None:
    args = _build_parser().parse_args()

    if not args.once and not args.schedule:
        args.once = True

    try:
        if args.once:
            result = _run_once(args)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("\n── Notification Summary ─────────────────────────")
                print(f"  Sent         : {result['sent']}")
                print(f"  Matches      : {result['matches_count']}")
                print(f"  Subject      : {result['subject']}")
                print(f"  Timezone     : {result['timezone']}")
                print("──────────────────────────────────────────────────\n")
                if args.dry_run:
                    print(result["body"])

        if args.schedule:
            _start_scheduler(args)

    except (EnvironmentError, FileNotFoundError, ValueError) as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
