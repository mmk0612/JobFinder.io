"""
src/scrapers/hn_jobs.py
-----------------------
Scrape "Who is Hiring?" jobs from Hacker News via the Algolia HN Search API.

Endpoint: https://hn.algolia.com/api/v1/search
  - tag: ask_hn, job
  - query: keywords
  - No authentication required.
  - Clean JSON response, no browser needed.
"""

from __future__ import annotations

import logging
import re

from src.scrapers.base import BaseScraper
from src.scrapers.models import JobListing

logger = logging.getLogger(__name__)

_ALGOLIA_URL = "https://hn.algolia.com/api/v1/search"
_HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"


class HNJobsScraper(BaseScraper):
    source_name = "hn"

    def scrape(
        self,
        keywords: str,
        location: str = "",
        max_results: int = 50,
    ) -> list[JobListing]:
        logger.info("[hn] scraping: keywords=%r location=%r", keywords, location)
        jobs: list[JobListing] = []
        page = 0
        hits_per_page = min(max_results, 50)

        while len(jobs) < max_results:
            params = {
                "query":        keywords,
                "tags":         "job",
                "hitsPerPage":  hits_per_page,
                "page":         page,
            }
            try:
                resp = self._get(_ALGOLIA_URL, params=params)
            except Exception as exc:
                logger.error("[hn] API error: %s", exc)
                break

            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                break

            for hit in hits:
                listing = self._parse_hit(hit, keywords, location)
                if listing:
                    jobs.append(listing)
                if len(jobs) >= max_results:
                    break

            # Algolia paginates; stop when we have enough or exhausted results
            if len(hits) < hits_per_page:
                break
            page += 1
            self.polite_sleep()

        logger.info("[hn] found %d jobs", len(jobs))
        return jobs

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse_hit(self, hit: dict, keywords: str, location: str) -> JobListing | None:
        object_id = hit.get("objectID", "")
        if not object_id:
            return None

        url   = _HN_ITEM_URL.format(object_id)
        text  = hit.get("story_text") or hit.get("comment_text") or ""
        title = hit.get("title") or hit.get("subject") or keywords

        # HN job posts often have company | role | location in the title
        company, job_title, loc = _parse_hn_title(title, text)
        if not location and loc:
            location = loc

        return JobListing(
            job_title=job_title or title,
            company=company or "Unknown",
            url=url,
            source=self.source_name,
            location=location,
            description=_strip_html(text)[:3000],
        )


# ── helpers ───────────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text).strip()


def _parse_hn_title(title: str, body: str) -> tuple[str, str, str]:
    """
    HN job post titles are often in the format:
        "CompanyName | Role | Location (Remote)"
    or just "CompanyName is hiring ..."
    """
    parts = [p.strip() for p in title.split("|")]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""

    # Fallback: try to find "hiring" keyword
    if "hiring" in title.lower():
        company = title.split("hiring")[0].strip().rstrip("is").strip()
        return company, title, ""

    return "", title, ""
