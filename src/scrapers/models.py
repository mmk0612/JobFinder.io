"""
src/scrapers/models.py
----------------------
Data model for a single scraped job listing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class JobListing:
    job_title:           str
    company:             str
    url:                 str
    source:              str                 # 'linkedin' | 'indeed' | 'hn' | …
    location:            str  = ""
    description:         str  = ""
    salary:              str  = ""
    experience_required: str  = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self) -> None:
        # Normalise whitespace in text-heavy fields
        self.job_title   = self.job_title.strip()
        self.company     = self.company.strip()
        self.location    = self.location.strip()
        self.description = " ".join(self.description.split())
        self.url         = self.url.strip()
