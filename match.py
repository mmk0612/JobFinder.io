"""
match.py
--------
CLI for Step 5: resume-to-job matching.

Example:
  python match.py \
    --resume-json output/structured_resume.json \
    --resume-embeddings output/structured_resume.embeddings.npz \
    --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.db.db import get_matchable_jobs  # noqa: E402
from src.matcher import (  # noqa: E402
    DEFAULT_WEIGHTS,
    RANKING_WEIGHTS,
    estimate_resume_experience_years,
    load_resume_artifacts,
    rank_jobs_for_resume,
)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="match",
        description="Match a processed resume against processed jobs.",
    )
    parser.add_argument(
        "--resume-json",
        default="output/structured_resume.json",
        help="Path to structured resume JSON (default: output/structured_resume.json)",
    )
    parser.add_argument(
        "--resume-embeddings",
        default="output/structured_resume.embeddings.npz",
        help="Path to resume embeddings NPZ (default: output/structured_resume.embeddings.npz)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top ranked jobs to return (default: 20)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Filter out jobs below this final score in [0,1] (default: 0.0)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source filter (e.g. linkedin, hn, greenhouse).",
    )
    parser.add_argument(
        "--preferred-location",
        dest="preferred_locations",
        action="append",
        default=None,
        help="Preferred location for ranking boost. Repeatable (e.g. --preferred-location remote).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Max number of candidate jobs to score from DB (default: 5000)",
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
        help="Print full results as JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        structured_resume, profile_embedding = load_resume_artifacts(
            args.resume_json,
            args.resume_embeddings,
        )

        jobs = get_matchable_jobs(
            source=args.source,
            limit=max(1, args.limit),
        )
        if args.job_keyword:
            jobs = [job for job in jobs if _job_matches_keyword(job, args.job_keyword)]
        if not jobs:
            print("No matchable jobs found. Run scrape/process pipeline first.")
            return

        matches = rank_jobs_for_resume(
            structured_resume=structured_resume,
            resume_embedding=profile_embedding,
            job_rows=jobs,
            top_k=max(1, args.top_k),
            min_score=max(0.0, min(1.0, args.min_score)),
            weights=DEFAULT_WEIGHTS,
            preferred_locations=args.preferred_locations,
        )

        resume_years = estimate_resume_experience_years(structured_resume)
        if args.json:
            payload = {
                "weights": DEFAULT_WEIGHTS,
                "resume": {
                    "skills": structured_resume.get("skills", []),
                    "estimated_experience_years": resume_years,
                },
                "ranking_weights": RANKING_WEIGHTS,
                "total_candidates": len(jobs),
                "returned": len(matches),
                "matches": [match.to_dict() for match in matches],
            }
            print(json.dumps(payload, indent=2))
            return

        print("\n── Ranking Summary ───────────────────────────────")
        print(f"  Candidate jobs : {len(jobs)}")
        print(f"  Returned       : {len(matches)}")
        print(f"  Resume years   : {resume_years}")
        print(
            "  Match weights  : "
            f"semantic={DEFAULT_WEIGHTS['semantic']} "
            f"skills={DEFAULT_WEIGHTS['skill_overlap']} "
            f"experience={DEFAULT_WEIGHTS['experience_match']}"
        )
        print(
            "  Rank weights   : "
            f"final={RANKING_WEIGHTS['final_score']:.2f} "
            f"salary={RANKING_WEIGHTS['salary']:.2f} "
            f"reputation={RANKING_WEIGHTS['company_reputation']:.2f} "
            f"location={RANKING_WEIGHTS['location_preference']:.2f}"
        )
        if args.preferred_locations:
            print("  Preferences    : " + ", ".join(args.preferred_locations))
        print("────────────────────────────────────────────────────")

        if not matches:
            print("No jobs passed the current filters.")
            return

        print("\nTop jobs where you are the strongest candidate:\n")
        for idx, match in enumerate(matches, start=1):
            print(
                f"{idx:>2}. rank={match.ranking_score:.3f} final={match.score:.3f} "
                f"{match.job_title} @ {match.company} "
                f"[{match.source}]"
            )
            print(
                "    "
                f"semantic={match.semantic_similarity:.3f} "
                f"skills={match.skill_overlap:.3f} "
                f"exp={match.experience_match:.3f}"
            )
            required_years = (
                str(match.experience_required)
                if match.experience_required is not None
                else "unknown"
            )
            tech_preview = ", ".join(match.tech_stack[:5]) if match.tech_stack else "n/a"
            print(
                "    "
                f"top_applicant={match.top_applicant_score}/100 ({match.top_applicant_band}) "
                f"coverage={match.skill_coverage:.2f} "
                f"hiring_signal={match.hiring_aggressiveness:.2f}"
            )
            print(
                "    "
                f"salary_score={match.salary_score:.2f} "
                f"company_rep={match.company_reputation_score:.2f} "
                f"location_pref={match.location_preference_score:.2f}"
            )
            print(
                "    "
                f"required_years={required_years} "
                f"seniority={match.seniority or 'unknown'} "
                f"salary={match.salary or 'n/a'} "
                f"tech_stack={tech_preview}"
            )
            if match.top_applicant_reasons:
                print("    reasons: " + " | ".join(match.top_applicant_reasons))
            print(f"    {match.url}")

    except (EnvironmentError, FileNotFoundError, ValueError) as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
