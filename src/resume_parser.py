"""
resume_parser.py  —  Pass 1
----------------------------
Send raw resume text to Gemini and extract structured sections:
  • skills
  • projects
  • experience
  • education
  • contact  (name, email, phone, links)
  • summary  (optional objective / about section)

Returns a typed Python dict (StructuredResume).
"""

from __future__ import annotations

from src.llm_client import call_llm_for_json

# ── prompt ────────────────────────────────────────────────────────────────────

_PASS1_SYSTEM = """\
You are a professional resume parser.
Your job is to read raw resume text and return ONLY a valid JSON object — \
no markdown, no extra commentary, just JSON.

Extract the following top-level keys.  If a section is absent, use an empty \
list [] or empty string "" for that key.

{
  "contact": {
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "+1-xxx-xxx-xxxx",
    "linkedin": "https://linkedin.com/in/...",
    "github":   "https://github.com/...",
    "portfolio": "https://..."
  },
  "summary": "One-paragraph professional summary or objective.",
  "skills": [
    "React", "Node.js", "Python", "PostgreSQL"
  ],
  "experience": [
    {
      "company":    "Acme Corp",
      "title":      "Software Engineer",
      "location":   "San Francisco, CA",
      "start_date": "Jan 2022",
      "end_date":   "Present",
      "bullets":    ["Built X", "Improved Y by Z%"]
    }
  ],
  "projects": [
    {
      "name":        "MyApp",
      "description": "A full-stack web app for …",
      "tech_stack":  ["React", "FastAPI", "PostgreSQL"],
      "url":         "https://github.com/…"
    }
  ],
  "education": [
    {
      "institution": "State University",
      "degree":      "B.S. Computer Science",
      "gpa":         "3.8",
      "start_date":  "Aug 2018",
      "end_date":    "May 2022"
    }
  ],
  "certifications": [
    "AWS Certified Solutions Architect – Associate"
  ]
}

Rules:
- Output ONLY the JSON — no markdown fences, no extra text.
- Preserve exact technology names from the resume (do NOT normalize yet).
- If a field value is unknown, use "" for strings and [] for arrays.
"""


def extract_sections(resume_text: str) -> dict:
    """
    Pass 1: Extract raw sections from resume text.

    Args:
        resume_text: Plain-text content returned by pdf_extractor.

    Returns:
        A dict with keys: contact, summary, skills, experience,
        projects, education, certifications.
    """
    prompt = f"{_PASS1_SYSTEM}\n\n--- RESUME TEXT START ---\n{resume_text}\n--- RESUME TEXT END ---"
    result = call_llm_for_json(prompt)
    return _validate_structure(result)


# ── validation ────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "contact", "summary", "skills",
    "experience", "projects", "education", "certifications",
}

def _validate_structure(data: dict) -> dict:
    """Ensure all required top-level keys exist, filling gaps with defaults."""
    defaults: dict = {
        "contact":        {},
        "summary":        "",
        "skills":         [],
        "experience":     [],
        "projects":       [],
        "education":      [],
        "certifications": [],
    }
    for key, default in defaults.items():
        data.setdefault(key, default)
    return data


