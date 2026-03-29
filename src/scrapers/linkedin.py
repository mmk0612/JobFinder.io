"""
src/scrapers/linkedin.py
------------------------
Scrape job listings from LinkedIn's PUBLIC job search page.

  URL: https://www.linkedin.com/jobs/search/?keywords=…&location=…

⚠️  IMPORTANT — Terms of Service
    LinkedIn prohibits automated scraping in their User Agreement §8.2.
    This implementation uses only the PUBLIC (unauthenticated) job search
    listing page — no login, no personal data accessed.
    Use responsibly. Consider the official LinkedIn Jobs API for production.

This scraper:
  - Does NOT log in.
  - Does NOT access member profiles.
  - Only reads the publicly visible job card list that any browser can view.
  - Applies polite delays to avoid hammering servers.
"""

from __future__ import annotations

import logging
import random
import time
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from src.scrapers.base import BaseScraper, _DELAY_MIN, _DELAY_MAX
from src.scrapers.models import JobListing

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.linkedin.com/jobs/search/"
_BASE        = "https://www.linkedin.com"


class LinkedInScraper(BaseScraper):
    source_name = "linkedin"

    def scrape(
        self,
        keywords: str,
        location: str = "",
        max_results: int = 50,
    ) -> list[JobListing]:
        from playwright.sync_api import sync_playwright
        try:
            from playwright_stealth import Stealth
            _has_stealth = True
        except ImportError:
            _has_stealth = False

        jobs: list[JobListing] = []
        search_locations = [location]
        if location.strip().lower() in {"remote", "any"}:
            search_locations.append("")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/123.0.0.0 Safari/537.36",
                locale="en-US",
                viewport={"width": 1280, "height": 900},
            )
            page = context.new_page()
            if _has_stealth:
                Stealth().apply_stealth_sync(page)

            # LinkedIn public job search paginates via `start` offset (25 per page)
            for location_filter in search_locations:
                if len(jobs) >= max_results:
                    break
                for start in range(0, max_results, 25):
                    if len(jobs) >= max_results:
                        break

                    params = (
                        f"?keywords={quote_plus(keywords)}"
                        f"&location={quote_plus(location_filter)}"
                        f"&start={start}"
                    )
                    try:
                        page.goto(_SEARCH_URL + params, wait_until="domcontentloaded", timeout=30_000)
                        page.wait_for_timeout(2000)
                        time.sleep(random.uniform(_DELAY_MIN + 1, _DELAY_MAX + 1))

                        # Scroll to load lazy content
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        time.sleep(random.uniform(1.0, 2.0))

                        html  = page.content()
                        batch = _parse_linkedin_page(html, self.source_name)
                        if not batch:
                            continue
                        jobs.extend(batch)
                    except Exception as exc:
                        logger.warning("[linkedin] page failed (start=%d): %s", start, exc)
                        continue

            browser.close()

        seen  = set()
        deduped: list[JobListing] = []
        for j in jobs[:max_results]:
            if j.url not in seen:
                seen.add(j.url)
                deduped.append(j)

        logger.info("[linkedin] found %d jobs", len(deduped))
        return deduped


# ── HTML parsing ──────────────────────────────────────────────────────────────

def _parse_linkedin_page(html: str, source: str) -> list[JobListing]:
    soup = BeautifulSoup(html, "html.parser")
    jobs: list[JobListing] = []

    cards = soup.select(
        "div.base-card, li.jobs-search-results__list-item, li[class*='job-search-card'], "
        "div[data-entity-urn*='jobPosting']"
    )
    for card in cards:
        title_el   = card.select_one("h3.base-search-card__title, h3[class*='title'], a.hidden-nested-link")
        company_el = card.select_one("h4.base-search-card__subtitle, a[class*='company'], h4")
        loc_el     = card.select_one("[class*='job-search-card__location'], span[class*='location'], span.job-search-card__location")
        link_el    = card.select_one("a.base-card__full-link, a[href*='/jobs/view/']")

        title   = title_el.get_text(strip=True)   if title_el   else ""
        company = company_el.get_text(strip=True) if company_el else ""
        loc     = loc_el.get_text(strip=True)     if loc_el     else ""
        href    = link_el.get("href", "")         if link_el    else ""
        url     = href.split("?")[0]              if href       else ""

        if not url:
            url = card.find("a", href=True)
            url = url["href"].split("?")[0] if url else ""

        if not title or not url:
            continue
        if not url.startswith("http"):
            url = _BASE + url

        jobs.append(JobListing(
            job_title=title,
            company=company,
            url=url,
            source=source,
            location=loc,
        ))

    return jobs
