"""
src/scrapers/base.py
--------------------
Abstract base class shared by all scrapers.

Provides:
  - polite_sleep()       — random delay between requests
  - _get()               — requests.get with retries + User-Agent rotation
  - scrape()             — abstract; implemented per source
"""

from __future__ import annotations

import abc
import logging
import random
import time
from typing import ClassVar

import requests
from requests import Response

from src.scrapers.models import JobListing

logger = logging.getLogger(__name__)


class ExpectedHTTPStatusError(requests.RequestException):
    """Raised when a scraper wants to ignore a known HTTP status."""

# A small rotation of realistic desktop User-Agents
_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

_DEFAULT_HEADERS: dict[str, str] = {
    "Accept":          "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

# Delay range (seconds) between requests — polite, not aggressive
_DELAY_MIN = 1.5
_DELAY_MAX = 4.0

# Retry config
_MAX_RETRIES   = 3
_RETRY_BACKOFF = 2.0   # multiplied by attempt number


class BaseScraper(abc.ABC):
    """
    Abstract base for all job scrapers.

    Subclasses must implement:
        source_name (class variable)
        scrape(keywords, location, max_results) → list[JobListing]
    """

    source_name: ClassVar[str] = "unknown"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_DEFAULT_HEADERS)

    # ── abstract ──────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def scrape(
        self,
        keywords: str,
        location: str = "",
        max_results: int = 50,
    ) -> list[JobListing]:
        """
        Scrape job listings matching `keywords` and `location`.

        Args:
            keywords:    Search terms (e.g. "python backend engineer").
            location:    City / region / "remote".
            max_results: Maximum number of listings to return.

        Returns:
            List of JobListing objects (may be less than max_results).
        """

    # ── helpers ───────────────────────────────────────────────────────────────

    def polite_sleep(self) -> None:
        """Sleep a random duration to avoid hammering the target server."""
        delay = random.uniform(_DELAY_MIN, _DELAY_MAX)
        logger.debug("[%s] sleeping %.1fs", self.source_name, delay)
        time.sleep(delay)

    def _get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: int = 15,
        retries: int = _MAX_RETRIES,
        retry_backoff: float = _RETRY_BACKOFF,
        ignore_status_codes: set[int] | None = None,
    ) -> Response:
        """
        Perform an HTTP GET with automatic retry + UA rotation.

        Raises requests.HTTPError on non-2xx after all retries.
        """
        merged_headers = {"User-Agent": random.choice(_USER_AGENTS)}
        if headers:
            merged_headers.update(headers)

        last_exc: Exception | None = None
        retries = max(1, retries)
        ignore_status_codes = ignore_status_codes or set()
        for attempt in range(1, retries + 1):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    headers=merged_headers,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                # Fail fast for permanent client errors (e.g. 404 board slug not found).
                # Keep retries for transient statuses like 408/429 and all 5xx responses.
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    status = int(exc.response.status_code)
                    if status in ignore_status_codes:
                        raise ExpectedHTTPStatusError(
                            f"[{self.source_name}] GET {url} returned ignored status {status}"
                        ) from exc
                    retriable_client = {408, 429}
                    if 400 <= status < 500 and status not in retriable_client:
                        logger.warning(
                            "[%s] GET %s failed with non-retriable status %d; skipping retries",
                            self.source_name,
                            url,
                            status,
                        )
                        raise requests.RequestException(
                            f"[{self.source_name}] GET {url} failed with non-retriable status {status}"
                        ) from exc
                logger.warning(
                    "[%s] GET %s failed (attempt %d/%d): %s",
                    self.source_name, url, attempt, retries, exc,
                )
                if attempt < retries:
                    time.sleep(retry_backoff * attempt)

        raise requests.RequestException(
            f"[{self.source_name}] GET {url} failed after {retries} attempts"
        ) from last_exc

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} source={self.source_name!r}>"
