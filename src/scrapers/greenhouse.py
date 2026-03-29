"""
src/scrapers/greenhouse.py
--------------------------
Scrape job listings via Greenhouse's public JSON API.

  GET https://boards.greenhouse.io/v1/boards/{company_slug}/jobs?content=true

No authentication required — this is publicly accessible for any company
that uses Greenhouse as their ATS.

Company slugs are read from config/companies.json  →  "greenhouse" list.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from src.scrapers.base import BaseScraper
from src.scrapers.models import JobListing

logger = logging.getLogger(__name__)

_API_URL        = "https://boards.greenhouse.io/v1/boards/{slug}/jobs"
_JOB_URL        = "https://boards.greenhouse.io/{slug}/jobs/{job_id}"
_COMPANIES_PATH = Path(__file__).parent.parent.parent / "config" / "companies.json"

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class GreenhouseScraper(BaseScraper):
    source_name = "greenhouse"

    def __init__(self, companies_path: str | Path = _COMPANIES_PATH) -> None:
        super().__init__()
        self._companies: list[str] = self._load_companies(companies_path)

    def scrape(
        self,
        keywords: str,
        location: str = "",
        max_results: int = 50,
    ) -> list[JobListing]:
        logger.info("[greenhouse] scraping %d companies", len(self._companies))
        jobs: list[JobListing] = []
        kw_lower = keywords.lower()

        for slug in self._companies:
            if len(jobs) >= max_results:
                break
            try:
                fetched = self._scrape_company(slug, kw_lower, location, max_results - len(jobs))
                jobs.extend(fetched)
            except Exception as exc:
                logger.warning("[greenhouse] %s failed: %s", slug, exc)
            self.polite_sleep()

        logger.info("[greenhouse] found %d jobs", len(jobs))
        return jobs

    # ── per-company ───────────────────────────────────────────────────────────

    def _scrape_company(
        self,
        slug: str,
        kw_lower: str,
        location: str,
        max_results: int,
    ) -> list[JobListing]:
        url = _API_URL.format(slug=slug)
        resp = self._get(url, params={"content": "true"})
        data = resp.json()

        jobs: list[JobListing] = []
        for item in data.get("jobs", []):
            title = item.get("title", "")
            # Filter by keywords (simple substring match)
            if kw_lower and kw_lower not in title.lower():
                # also check departments / departments
                dept = " ".join(
                    d.get("name", "") for d in item.get("departments", [])
                ).lower()
                if kw_lower not in dept:
                    continue

            job_id       = item.get("id", "")
            job_location = item.get("location", {}).get("name", "") or location
            content      = item.get("content", "")
            description  = _strip_html(content)[:3000]

            jobs.append(JobListing(
                job_title=title,
                company=data.get("company", {}).get("name", slug),
                url=_JOB_URL.format(slug=slug, job_id=job_id),
                source=self.source_name,
                location=job_location,
                description=description,
            ))
            if len(jobs) >= max_results:
                break

        return jobs

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_companies(path: str | Path) -> list[str]:
        path = Path(path)
        if not path.exists():
            logger.warning("[greenhouse] companies.json not found at %s", path)
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("greenhouse", [])


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text).strip()
