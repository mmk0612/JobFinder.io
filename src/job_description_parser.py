"""
src/job_description_parser.py
-----------------------------
Extract structured fields from a raw job description using Gemini.
"""

from __future__ import annotations

import os
import re

from src.llm_client import call_llm_for_json


_SYSTEM_PROMPT = """\
You are a job-description parsing engine.
Return ONLY a valid JSON object, no markdown, no commentary.

Extract this schema from the provided job description text:
{
  "skills": ["Python", "Docker", "FastAPI"],
  "tech_stack": ["AWS", "PostgreSQL", "Kubernetes"],
  "experience_required": 2,
  "experience_text": "2+ years of backend engineering experience",
  "seniority": "mid",
  "summary": "Short 1-2 sentence summary of the role."
}

Rules:
- `skills` should include explicit tools, languages, frameworks, and libraries.
- `tech_stack` should include infra, database, cloud, devops, and platform technologies.
- `experience_required` must be an integer minimum years if inferable, else null.
- `experience_text` should preserve the human-readable requirement if present, else "".
- `seniority` must be one of: intern, junior, mid, senior, staff, principal, lead, manager, director, unknown.
- `summary` should be concise and factual.
- If a field is missing use [] for arrays, "" for strings, and null for `experience_required`.
"""


def extract_job_description(job_text: str) -> dict:
    """Extract structured job fields from a raw job-description string."""
    disable_llm = os.environ.get("JOB_DISABLE_LLM_JOB_EXTRACTION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if disable_llm:
        return _heuristic_extract_job_description(job_text)

    prompt = f"{_SYSTEM_PROMPT}\n\n--- JOB DESCRIPTION START ---\n{job_text}\n--- JOB DESCRIPTION END ---"
    try:
        result = call_llm_for_json(prompt)
    except Exception:
        return _heuristic_extract_job_description(job_text)
    return _validate_structure(result)


def _heuristic_extract_job_description(job_text: str) -> dict:
    text = str(job_text or "")
    lower = text.lower()

    skill_candidates = [
        "python", "java", "javascript", "typescript", "go", "rust", "c++", "sql",
        "react", "node", "fastapi", "django", "flask", "spring", "kotlin", "swift",
    ]
    tech_candidates = [
        "aws", "gcp", "azure", "docker", "kubernetes", "postgresql", "mysql", "mongodb",
        "redis", "airflow", "spark", "hadoop", "terraform", "git", "linux",
    ]

    def _find_terms(candidates: list[str]) -> list[str]:
        found: list[str] = []
        for term in candidates:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, lower):
                found.append(term)
        return found

    years = None
    match = re.search(r"(\d+)\s*\+?\s*(?:years|yrs)\b", lower)
    if match:
        years = max(0, int(match.group(1)))

    seniority = "unknown"
    if re.search(r"\b(intern|internship)\b", lower):
        seniority = "intern"
    elif re.search(r"\b(junior|entry[-\s]?level|associate)\b", lower):
        seniority = "junior"
    elif re.search(r"\b(senior|sr\.?\b)\b", lower):
        seniority = "senior"
    elif re.search(r"\b(staff)\b", lower):
        seniority = "staff"
    elif re.search(r"\b(principal)\b", lower):
        seniority = "principal"
    elif re.search(r"\b(lead|tech lead)\b", lower):
        seniority = "lead"
    elif re.search(r"\b(manager|engineering manager)\b", lower):
        seniority = "manager"
    elif re.search(r"\b(director)\b", lower):
        seniority = "director"
    elif re.search(r"\b(mid|intermediate)\b", lower):
        seniority = "mid"

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    summary = " ".join(sentences[:2]).strip()[:500]

    experience_text = ""
    exp_text_match = re.search(
        r"([^\n\r]{0,120}\b\d+\s*\+?\s*(?:years|yrs)\b[^\n\r]{0,120})",
        text,
        flags=re.IGNORECASE,
    )
    if exp_text_match:
        experience_text = exp_text_match.group(1).strip()

    return _validate_structure({
        "skills": _find_terms(skill_candidates),
        "tech_stack": _find_terms(tech_candidates),
        "experience_required": years,
        "experience_text": experience_text,
        "seniority": seniority,
        "summary": summary,
    })


def _validate_structure(data: dict) -> dict:
    defaults = {
        "skills": [],
        "tech_stack": [],
        "experience_required": None,
        "experience_text": "",
        "seniority": "unknown",
        "summary": "",
    }
    for key, default in defaults.items():
        data.setdefault(key, default)

    data["skills"] = _clean_string_list(data.get("skills", []))
    data["tech_stack"] = _clean_string_list(data.get("tech_stack", []))
    data["experience_text"] = str(data.get("experience_text", "") or "").strip()
    data["summary"] = str(data.get("summary", "") or "").strip()

    seniority = str(data.get("seniority", "unknown") or "unknown").strip().lower()
    allowed = {"intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager", "director", "unknown"}
    data["seniority"] = seniority if seniority in allowed else "unknown"

    exp = data.get("experience_required")
    data["experience_required"] = _coerce_experience_required(exp, data["experience_text"])
    return data


def _clean_string_list(values: list) -> list[str]:
    out: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text:
            out.append(text)
    return list(dict.fromkeys(out))


def _coerce_experience_required(value, experience_text: str) -> int | None:
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    raw = str(value).strip() if value is not None else ""
    if raw.isdigit():
        return max(0, int(raw))

    match = re.search(r"(\d+)", raw or experience_text)
    if match:
        return max(0, int(match.group(1)))
    return None