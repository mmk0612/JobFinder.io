"""
src/matcher.py
--------------
Resume-to-job matching engine.

Final score:
  0.6 * semantic_similarity
  0.3 * skill_overlap
  0.1 * experience_match
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_WEIGHTS = {
    "semantic": 0.6,
    "skill_overlap": 0.3,
    "experience_match": 0.1,
}

TOP_APPLICANT_USE_LLM = (
    os.environ.get("TOP_APPLICANT_USE_LLM", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
TOP_APPLICANT_LLM_WEIGHT = float(
    os.environ.get("TOP_APPLICANT_LLM_WEIGHT", "0.35") or "0.35"
)
TOP_APPLICANT_LLM_WEIGHT = max(0.0, min(1.0, TOP_APPLICANT_LLM_WEIGHT))

RANKING_WEIGHTS = {
    "final_score": float(os.environ.get("RANKING_WEIGHT_FINAL_SCORE", "0.50") or "0.50"),
    "salary": float(os.environ.get("RANKING_WEIGHT_SALARY", "0.20") or "0.20"),
    "company_reputation": float(os.environ.get("RANKING_WEIGHT_COMPANY_REPUTATION", "0.15") or "0.15"),
    "location_preference": float(os.environ.get("RANKING_WEIGHT_LOCATION_PREFERENCE", "0.15") or "0.15"),
}

USD_TO_INR = float(os.environ.get("SALARY_USD_TO_INR", "83.0") or "83.0")
USD_TO_INR = max(1.0, USD_TO_INR)

_RANKING_WEIGHT_SUM = sum(max(0.0, value) for value in RANKING_WEIGHTS.values())
if _RANKING_WEIGHT_SUM <= 0:
    RANKING_WEIGHTS = {
        "final_score": 0.5,
        "salary": 0.2,
        "company_reputation": 0.15,
        "location_preference": 0.15,
    }
else:
    RANKING_WEIGHTS = {
        key: max(0.0, value) / _RANKING_WEIGHT_SUM
        for key, value in RANKING_WEIGHTS.items()
    }

KNOWN_COMPANY_REPUTATION = {
    "google": 0.98,
    "meta": 0.95,
    "amazon": 0.93,
    "microsoft": 0.95,
    "apple": 0.96,
    "openai": 0.96,
    "netflix": 0.94,
    "airbnb": 0.88,
    "coinbase": 0.84,
    "notion": 0.82,
    "figma": 0.86,
    "robinhood": 0.8,
    "rippling": 0.79,
    "cloudflare": 0.85,
    "discord": 0.83,
    "vercel": 0.81,
    "hashicorp": 0.82,
    "gitlab": 0.8,
}


@dataclass(frozen=True)
class MatchResult:
    score: float
    semantic_similarity: float
    skill_overlap: float
    experience_match: float
    job_title: str
    company: str
    location: str
    source: str
    url: str
    salary: str = ""
    experience_required: int | None = None
    tech_stack: list[str] = field(default_factory=list)
    seniority: str = "unknown"
    skill_coverage: float = 0.0
    hiring_aggressiveness: float = 0.0
    salary_score: float = 0.0
    company_reputation_score: float = 0.0
    location_preference_score: float = 0.0
    ranking_score: float = 0.0
    top_applicant_score: int = 0
    top_applicant_band: str = ""
    top_applicant_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "semantic_similarity": self.semantic_similarity,
            "skill_overlap": self.skill_overlap,
            "experience_match": self.experience_match,
            "job_title": self.job_title,
            "company": self.company,
            "location": self.location,
            "source": self.source,
            "url": self.url,
            "salary": self.salary,
            "experience_required": self.experience_required,
            "tech_stack": self.tech_stack,
            "seniority": self.seniority,
            "skill_coverage": self.skill_coverage,
            "hiring_aggressiveness": self.hiring_aggressiveness,
            "salary_score": self.salary_score,
            "company_reputation_score": self.company_reputation_score,
            "location_preference_score": self.location_preference_score,
            "ranking_score": self.ranking_score,
            "top_applicant_score": self.top_applicant_score,
            "top_applicant_band": self.top_applicant_band,
            "top_applicant_reasons": self.top_applicant_reasons,
        }


def load_resume_artifacts(
    resume_json_path: str | Path,
    resume_embeddings_path: str | Path,
) -> tuple[dict, np.ndarray]:
    """Load structured resume JSON + profile embedding vector."""
    resume_json_path = Path(resume_json_path)
    resume_embeddings_path = Path(resume_embeddings_path)

    with open(resume_json_path, encoding="utf-8") as f:
        structured_resume = json.load(f)

    arr = np.load(resume_embeddings_path)
    if "profile_embedding" not in arr:
        raise ValueError(
            f"Missing `profile_embedding` in {resume_embeddings_path}. "
            "Run `main.py` without --no-embed first."
        )
    profile_embedding = np.asarray(arr["profile_embedding"], dtype=np.float32)
    return structured_resume, profile_embedding


def estimate_resume_experience_years(structured_resume: dict) -> float:
    """
    Estimate total years of experience from resume `experience` entries.

    Uses parsed date ranges and merges overlapping intervals.
    Falls back to 0.0 when dates are unavailable.
    """
    experiences = structured_resume.get("experience", []) or []
    intervals: list[tuple[datetime, datetime]] = []

    for item in experiences:
        start = _parse_resume_date(item.get("start_date", ""))
        end_raw = str(item.get("end_date", "") or "").strip().lower()
        end = datetime.now(timezone.utc) if end_raw in {"present", "current", "now", "ongoing"} else _parse_resume_date(end_raw)
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        intervals.append((start, end))

    if not intervals:
        return 0.0

    intervals.sort(key=lambda pair: pair[0])
    merged: list[tuple[datetime, datetime]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    total_days = sum((end - start).days for start, end in merged)
    return round(max(0.0, total_days / 365.25), 2)


def semantic_similarity(resume_embedding: np.ndarray, job_embedding: np.ndarray) -> float:
    """Cosine similarity mapped from [-1, 1] into [0, 1]."""
    resume_vec = np.asarray(resume_embedding, dtype=np.float32).reshape(1, -1)
    job_vec = np.asarray(job_embedding, dtype=np.float32).reshape(1, -1)
    if resume_vec.shape[1] != job_vec.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: resume={resume_vec.shape[1]} job={job_vec.shape[1]}"
        )
    cos = float(cosine_similarity(resume_vec, job_vec)[0][0])
    mapped = (cos + 1.0) / 2.0
    return round(_clamp01(mapped), 6)


def skill_overlap(resume_skills: list[str], job_skills: list[str]) -> float:
    """Jaccard overlap on normalized/canonical skill sets."""
    resume_set = {s.strip().lower() for s in resume_skills if s and s.strip()}
    job_set = {s.strip().lower() for s in job_skills if s and s.strip()}
    if not resume_set or not job_set:
        return 0.0
    inter = len(resume_set & job_set)
    union = len(resume_set | job_set)
    return round(_clamp01(inter / union if union else 0.0), 6)


def experience_match(resume_years: float, required_years: int | None) -> float:
    """
    Experience alignment in [0,1].

    If job requirement is missing, treat as neutral-pass (1.0).
    """
    if required_years is None or required_years <= 0:
        return 1.0
    if resume_years <= 0:
        return 0.0
    return round(_clamp01(resume_years / float(required_years)), 6)


def score_job(
    *,
    resume_embedding: np.ndarray,
    resume_skills: list[str],
    resume_experience_years: float,
    job_row: dict,
    weights: dict[str, float] | None = None,
) -> MatchResult | None:
    """Compute weighted score for one job row; returns None for invalid rows."""
    weights = weights or DEFAULT_WEIGHTS
    job_embedding = np.asarray(job_row.get("job_embedding", []), dtype=np.float32)
    if job_embedding.size == 0:
        return None

    try:
        semantic = semantic_similarity(resume_embedding, job_embedding)
    except ValueError:
        return None

    overlap = skill_overlap(resume_skills, job_row.get("processed_skills", []) or [])
    exp = experience_match(resume_experience_years, job_row.get("processed_experience_required"))

    score = (
        weights["semantic"] * semantic
        + weights["skill_overlap"] * overlap
        + weights["experience_match"] * exp
    )

    required_years = _coerce_job_required_years(job_row)
    tech_stack = _clean_tokens(job_row.get("processed_tech_stack", []) or [])
    seniority = str(job_row.get("processed_seniority") or "unknown").strip().lower() or "unknown"

    return MatchResult(
        score=round(_clamp01(score), 6),
        semantic_similarity=semantic,
        skill_overlap=overlap,
        experience_match=exp,
        job_title=job_row.get("job_title", ""),
        company=job_row.get("company", ""),
        location=job_row.get("location", ""),
        source=job_row.get("source", ""),
        url=job_row.get("url", ""),
        salary=str(job_row.get("salary", "") or "").strip(),
        experience_required=required_years,
        tech_stack=tech_stack,
        seniority=seniority,
    )


def rank_jobs_for_resume(
    *,
    structured_resume: dict,
    resume_embedding: np.ndarray,
    job_rows: list[dict],
    top_k: int = 20,
    min_score: float = 0.0,
    weights: dict[str, float] | None = None,
    preferred_locations: list[str] | None = None,
) -> list[MatchResult]:
    """Score and rank jobs for the given resume embedding/profile."""
    resume_skills = structured_resume.get("skills", []) or []
    resume_years = estimate_resume_experience_years(structured_resume)
    company_signal = _company_hiring_signal(job_rows)
    normalized_pref_locations = _normalize_locations(
        preferred_locations if preferred_locations is not None else _default_preferred_locations()
    )
    salary_bounds = _salary_bounds(job_rows)

    scored: list[tuple[MatchResult, dict]] = []
    for row in job_rows:
        result = score_job(
            resume_embedding=resume_embedding,
            resume_skills=resume_skills,
            resume_experience_years=resume_years,
            job_row=row,
            weights=weights,
        )
        if result is None:
            continue
        if result.score >= min_score:
            scored.append((result, row))

    enriched: list[MatchResult] = []
    for result, row in scored:
        prediction = _predict_top_applicant(
            resume_skills=resume_skills,
            resume_years=resume_years,
            job_row=row,
            match_result=result,
            company_hiring_signal=company_signal.get((result.company or "").strip().lower(), 0.0),
            structured_resume=structured_resume,
        )

        salary_score = _salary_score_from_row(row, salary_bounds=salary_bounds)
        company_reputation = _company_reputation_score(
            company_name=result.company,
            hiring_aggressiveness=prediction["hiring_aggressiveness"],
        )
        location_preference_score = _location_preference_score(
            job_location=result.location,
            preferred_locations=normalized_pref_locations,
        )
        ranking_score = _compute_ranking_score(
            final_score=result.score,
            salary_score=salary_score,
            company_reputation_score=company_reputation,
            location_preference_score=location_preference_score,
        )

        enriched.append(
            replace(
                result,
                skill_coverage=prediction["skill_coverage"],
                hiring_aggressiveness=prediction["hiring_aggressiveness"],
                salary_score=round(salary_score, 6),
                company_reputation_score=round(company_reputation, 6),
                location_preference_score=round(location_preference_score, 6),
                ranking_score=round(ranking_score, 6),
                top_applicant_score=prediction["score"],
                top_applicant_band=prediction["band"],
                top_applicant_reasons=prediction["reasons"],
            )
        )

    enriched.sort(key=lambda item: (item.ranking_score, item.top_applicant_score, item.score), reverse=True)
    return enriched[: max(1, top_k)]


def _predict_top_applicant(
    *,
    resume_skills: list[str],
    resume_years: float,
    job_row: dict,
    match_result: MatchResult,
    company_hiring_signal: float,
    structured_resume: dict,
) -> dict:
    required_years = _coerce_job_required_years(job_row)
    job_skills = _extract_job_skills(job_row)
    resume_set = set(_clean_tokens(resume_skills))
    skill_coverage = 0.0
    if job_skills:
        skill_coverage = len(resume_set & job_skills) / len(job_skills)

    score = (match_result.score * 100.0) * 0.55
    reasons: list[str] = []

    if required_years is None:
        score += 5
        reasons.append("Experience requirement not strict (neutral-positive).")
    elif required_years <= resume_years + 1:
        score += 14
        reasons.append("Experience requirement aligns (within +1 year).")
    elif required_years <= resume_years + 3:
        score += 7
        reasons.append("Experience is close to requirement.")
    else:
        score -= 8
        reasons.append("Experience requirement is above current profile.")

    if skill_coverage >= 0.80:
        score += 16
        reasons.append("Strong skills coverage (>=80%).")
    elif skill_coverage >= 0.60:
        score += 9
        reasons.append("Good skills coverage (>=60%).")
    elif skill_coverage >= 0.40:
        score += 3
        reasons.append("Partial skills coverage (>=40%).")
    else:
        score -= 6
        reasons.append("Limited direct skills overlap.")

    seniority = str(job_row.get("processed_seniority") or "unknown").strip().lower() or "unknown"
    if _seniority_compatible(resume_years, seniority):
        score += 6
        reasons.append(f"Seniority alignment looks good ({seniority}).")
    elif seniority not in {"", "unknown"}:
        score -= 3
        reasons.append(f"Seniority may be slightly above current profile ({seniority}).")

    location = str(job_row.get("location") or "").strip().lower()
    if "remote" in location or not location:
        score += 3
        reasons.append("Location constraint is flexible.")

    hiring_aggressiveness = _clamp01(company_hiring_signal)
    if hiring_aggressiveness >= 0.7:
        reasons.append("Company appears to be hiring aggressively.")
    score += 8.0 * hiring_aggressiveness

    heuristic_score = int(round(_clamp(score, 0.0, 100.0)))
    final_score = heuristic_score

    if TOP_APPLICANT_USE_LLM:
        llm = _llm_top_applicant_score(
            structured_resume=structured_resume,
            job_row=job_row,
            heuristic_score=heuristic_score,
        )
        if llm is not None:
            final_score = int(round(
                ((1.0 - TOP_APPLICANT_LLM_WEIGHT) * heuristic_score)
                + (TOP_APPLICANT_LLM_WEIGHT * llm)
            ))
            reasons.append("LLM-adjusted competitiveness estimate applied.")

    return {
        "score": int(_clamp(final_score, 0, 100)),
        "band": _band_for_score(final_score),
        "reasons": reasons[:4],
        "skill_coverage": round(_clamp01(skill_coverage), 6),
        "hiring_aggressiveness": round(hiring_aggressiveness, 6),
    }


def _llm_top_applicant_score(
    *,
    structured_resume: dict,
    job_row: dict,
    heuristic_score: int,
) -> int | None:
    try:
        from src.llm_client import call_llm_for_json
    except Exception:
        return None

    resume_summary = {
        "summary": structured_resume.get("summary", ""),
        "skills": structured_resume.get("skills", [])[:30],
        "experience": structured_resume.get("experience", [])[:5],
    }
    job_summary = {
        "job_title": job_row.get("job_title", ""),
        "company": job_row.get("company", ""),
        "location": job_row.get("location", ""),
        "seniority": job_row.get("processed_seniority", "unknown"),
        "experience_required": _coerce_job_required_years(job_row),
        "skills": job_row.get("processed_skills", []),
        "tech_stack": job_row.get("processed_tech_stack", []),
        "summary": job_row.get("processed_summary", ""),
        "description": str(job_row.get("description", ""))[:1500],
    }

    prompt = (
        "Given my resume and the job description, estimate the likelihood I would be a top applicant. "
        "Return only JSON with this schema: {\"top_applicant_score\": <0-100 integer>}.\n\n"
        f"Heuristic baseline score: {heuristic_score}\n"
        f"Resume: {json.dumps(resume_summary, ensure_ascii=False)}\n"
        f"Job: {json.dumps(job_summary, ensure_ascii=False)}"
    )
    try:
        parsed = call_llm_for_json(prompt, temperature=0.1)
        value = int(parsed.get("top_applicant_score"))
        return int(_clamp(value, 0, 100))
    except Exception:
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clean_tokens(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values or []:
        text = str(value or "").strip().lower()
        if text:
            cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def _extract_job_skills(job_row: dict) -> set[str]:
    combined = []
    combined.extend(job_row.get("processed_skills", []) or [])
    combined.extend(job_row.get("processed_tech_stack", []) or [])
    return set(_clean_tokens(combined))


def _coerce_job_required_years(job_row: dict) -> int | None:
    value = job_row.get("processed_experience_required")
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    text = str(value).strip() if value is not None else ""
    if text.isdigit():
        return max(0, int(text))

    fallback = str(job_row.get("experience_required", "") or "")
    match = re.search(r"(\d+)", text or fallback)
    if match:
        return max(0, int(match.group(1)))
    return None


def _company_hiring_signal(job_rows: list[dict]) -> dict[str, float]:
    now = datetime.now(timezone.utc)
    by_company: dict[str, dict[str, int]] = {}

    for row in job_rows:
        company = str(row.get("company", "") or "").strip().lower()
        if not company:
            continue
        bucket = by_company.setdefault(company, {"total": 0, "recent": 0})
        bucket["total"] += 1

        scraped_at = row.get("scraped_at")
        if isinstance(scraped_at, datetime):
            dt = scraped_at if scraped_at.tzinfo else scraped_at.replace(tzinfo=timezone.utc)
            if (now - dt).days <= 14:
                bucket["recent"] += 1

    signal: dict[str, float] = {}
    for company, stats in by_company.items():
        total_component = min(1.0, stats["total"] / 6.0)
        recent_component = min(1.0, stats["recent"] / 4.0)
        signal[company] = _clamp01((0.65 * total_component) + (0.35 * recent_component))
    return signal


def _seniority_compatible(resume_years: float, seniority: str) -> bool:
    level = (seniority or "unknown").strip().lower()
    if level in {"unknown", ""}:
        return True
    if level in {"intern"}:
        return resume_years <= 2.0
    if level in {"junior"}:
        return resume_years <= 3.0
    if level in {"mid"}:
        return 1.0 <= resume_years <= 6.0
    if level in {"senior", "lead"}:
        return resume_years >= 4.0
    if level in {"staff", "principal", "manager", "director"}:
        return resume_years >= 7.0
    return True


def _band_for_score(score: int | float) -> str:
    value = float(score)
    if value >= 80:
        return "high"
    if value >= 60:
        return "medium"
    return "low"


def _compute_ranking_score(
    *,
    final_score: float,
    salary_score: float,
    company_reputation_score: float,
    location_preference_score: float,
) -> float:
    return _clamp01(
        (RANKING_WEIGHTS["final_score"] * _clamp01(final_score))
        + (RANKING_WEIGHTS["salary"] * _clamp01(salary_score))
        + (RANKING_WEIGHTS["company_reputation"] * _clamp01(company_reputation_score))
        + (RANKING_WEIGHTS["location_preference"] * _clamp01(location_preference_score))
    )


def _default_preferred_locations() -> list[str]:
    raw = os.environ.get("MATCH_PREFERRED_LOCATIONS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_locations(values: list[str]) -> list[str]:
    normalized = []
    for value in values:
        text = str(value or "").strip().lower()
        if text:
            normalized.append(text)
    return list(dict.fromkeys(normalized))


def _location_preference_score(*, job_location: str, preferred_locations: list[str]) -> float:
    location = str(job_location or "").strip().lower()
    if not preferred_locations:
        return 0.7 if ("remote" in location or not location) else 0.5

    if "remote" in location and ("remote" in preferred_locations or "any" in preferred_locations):
        return 1.0

    for preferred in preferred_locations:
        if preferred in {"any", "all"}:
            return 0.85
        if preferred and preferred in location:
            return 1.0

    return 0.35


def _salary_bounds(job_rows: list[dict]) -> tuple[float, float] | None:
    values = [
        value
        for value in (
            _salary_midpoint_usd_from_text(str(row.get("salary", "") or ""))
            for row in job_rows
        )
        if value is not None
    ]
    if not values:
        return None
    low, high = min(values), max(values)
    if high <= low:
        return (low, low + 1.0)
    return (low, high)


def _salary_score_from_row(row: dict, *, salary_bounds: tuple[float, float] | None) -> float:
    if salary_bounds is None:
        return 0.5
    value = _salary_midpoint_usd_from_text(str(row.get("salary", "") or ""))
    if value is None:
        return 0.45
    low, high = salary_bounds
    return _clamp01((value - low) / (high - low))


def _salary_midpoint_usd_from_text(raw_salary: str) -> float | None:
    """
    Parse salary text and return midpoint normalized to USD.

    Supports common USD and INR formats, including INR units like
    lakh/lac/lpa and crore/cr.
    """
    text = str(raw_salary or "").strip().lower()
    if not text:
        return None

    is_inr = _is_inr_salary_text(text)
    is_usd = _is_usd_salary_text(text)

    numbers: list[float] = []
    for match in re.finditer(r"(\d+(?:[\.,]\d+)?)\s*(k|m|lpa|lac|lakh|cr|crore)?", text):
        base = float(match.group(1).replace(",", ""))
        suffix = (match.group(2) or "").lower()
        if suffix == "k":
            base *= 1_000.0
        elif suffix == "m":
            base *= 1_000_000.0
        elif suffix in {"lpa", "lac", "lakh"}:
            base *= 100_000.0
        elif suffix in {"cr", "crore"}:
            base *= 10_000_000.0
        elif base < 1_000 and "$" in text:
            base *= 1_000.0
        numbers.append(base)

    if not numbers:
        return None
    midpoint = numbers[0] if len(numbers) == 1 else (min(numbers) + max(numbers)) / 2.0

    # If only INR is detected, convert INR -> USD using configurable FX rate.
    if is_inr and not is_usd:
        return midpoint / USD_TO_INR
    return midpoint


def _is_inr_salary_text(text: str) -> bool:
    return bool(re.search(r"(₹|\binr\b|\brs\.?\b|\brupees?\b|\blpa\b|\blac\b|\blakh\b|\bcr\b|\bcrore\b)", text))


def _is_usd_salary_text(text: str) -> bool:
    return bool(re.search(r"(\$|\busd\b|\bdollars?\b)", text))


def _company_reputation_score(*, company_name: str, hiring_aggressiveness: float) -> float:
    key = str(company_name or "").strip().lower()
    base = KNOWN_COMPANY_REPUTATION.get(key, 0.6)
    return _clamp01((0.8 * base) + (0.2 * _clamp01(hiring_aggressiveness)))


def _parse_resume_date(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    patterns = [
        "%b %Y",   # Jan 2022
        "%B %Y",   # January 2022
        "%m/%Y",   # 01/2022
        "%Y-%m",   # 2022-01
        "%Y",      # 2022
    ]
    for pattern in patterns:
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    # fallback: pull year if present
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        year = int(match.group(0))
        return datetime(year, 1, 1, tzinfo=timezone.utc)
    return None
