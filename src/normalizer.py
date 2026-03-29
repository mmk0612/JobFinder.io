"""
normalizer.py  —  Pass 2
--------------------------
Take the structured resume from Pass 1 and normalize skill names so that
job-matching later can do exact/fuzzy lookups reliably.

Examples of normalization:
  C++         → cpp
  ReactJS     → react
  React.js    → react
  NodeJS      → nodejs
  Node.js     → nodejs
  Postgres    → postgresql
  PostgreSQL  → postgresql
  TensorFlow  → tensorflow
  scikit-learn → scikit-learn
  AWS         → aws
  GCP         → gcp
  Javascript  → javascript
  TypeScript  → typescript

Two strategies are applied in order:
  1. A hand-crafted lookup table of well-known aliases (instant, deterministic).
  2. A Gemini LLM pass that handles anything the lookup misses.
"""

from __future__ import annotations

import json
import os

from src.llm_client import call_llm_for_json

# ── static alias table ─────────────────────────────────────────────────────────
# Keys are lowercase input forms; values are the canonical normalized token.

_ALIAS_TABLE: dict[str, str] = {
    # C family
    "c++": "cpp",
    "cplusplus": "cpp",
    # JavaScript ecosystem
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "reactjs": "react",
    "react.js": "react",
    "react js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "angularjs": "angular",
    "angular.js": "angular",
    "nodejs": "nodejs",
    "node.js": "nodejs",
    "node js": "nodejs",
    "nextjs": "nextjs",
    "next.js": "nextjs",
    "expressjs": "express",
    "express.js": "express",
    # Python ecosystem
    "python3": "python",
    "py": "python",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    # Databases
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "mssql": "sqlserver",
    "ms sql": "sqlserver",
    "microsoft sql server": "sqlserver",
    "mongo": "mongodb",
    "mongodb": "mongodb",
    "redis": "redis",
    # Cloud / DevOps
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "microsoft azure": "azure",
    "k8s": "kubernetes",
    "docker compose": "docker-compose",
    # AI / ML
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "huggingface": "huggingface",
    "hugging face": "huggingface",
    "langchain": "langchain",
    # Misc
    "graphql": "graphql",
    "rest api": "rest",
    "restful": "rest",
    "rest apis": "rest",
    "html5": "html",
    "css3": "css",
    "scss": "scss",
    "sass": "sass",
    "tailwindcss": "tailwind",
    "tailwind css": "tailwind",
}


def _static_normalize(skill: str) -> str | None:
    """Return normalized form from alias table, or None if not found."""
    return _ALIAS_TABLE.get(skill.strip().lower())


# ── LLM pass ──────────────────────────────────────────────────────────────────

_PASS2_SYSTEM = """\
You are a skills-normalization engine for a job-matching system.

Given a JSON array of raw skill strings, return a JSON object mapping EACH \
original skill (exact string) to its normalized canonical form.

Normalization rules:
- lowercase only
- remove punctuation variants: "React.js" → "react", "C++" → "cpp"
- collapse common aliases: "ReactJS" → "react", "NodeJS" → "nodejs",
  "Postgres" → "postgresql", "k8s" → "kubernetes"
- keep specificity: "scikit-learn" stays "scikit-learn", not "python"
- if a skill is already normalized, keep it as-is
- do NOT invent new skills; only normalize what is given

Return ONLY a JSON object like:
{
  "C++": "cpp",
  "ReactJS": "react",
  "Postgres": "postgresql",
  "Python": "python"
}
No markdown, no extra text.
"""


def _llm_normalize(skills: list[str]) -> dict[str, str]:
    """Send remaining skills to Gemini for normalization."""
    if not skills:
        return {}
    prompt = f"{_PASS2_SYSTEM}\n\nSkills to normalize:\n{json.dumps(skills, indent=2)}"
    return call_llm_for_json(prompt)


def _build_skill_mapping(raw_skills: list[str], *, use_llm: bool = True) -> dict[str, str]:
    """Build a normalization mapping for a set of raw skills."""
    unique_raw = [skill for skill in dict.fromkeys(raw_skills) if skill and skill.strip()]

    mapping: dict[str, str] = {}
    remaining: list[str] = []
    for skill in unique_raw:
        normalized = _static_normalize(skill)
        if normalized:
            mapping[skill] = normalized
        else:
            remaining.append(skill)

    if remaining and use_llm:
        llm_mapping = _llm_normalize(remaining)
        for skill in remaining:
            mapping[skill] = llm_mapping.get(skill, skill.lower().strip())
    elif remaining:
        for skill in remaining:
            mapping[skill] = skill.lower().strip()

    return mapping


def normalize_skill_list(skills: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Normalize a plain list of skill strings.

    Returns:
        (normalized_unique_skills, mapping)
    """
    mapping = _build_skill_mapping(skills)
    normalized = [mapping.get(skill, skill.lower().strip()) for skill in skills if skill and skill.strip()]
    return list(dict.fromkeys(normalized)), mapping


def normalize_job_description(structured_job: dict) -> dict:
    """
    Normalize extracted job skills and tech stack in-place.

    Adds `_skill_normalization_map` for transparency/debugging.
    """
    raw_skills = list(structured_job.get("skills", []))
    raw_skills.extend(structured_job.get("tech_stack", []))

    disable_llm = os.environ.get("JOB_DISABLE_LLM_SKILL_NORMALIZATION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mapping = _build_skill_mapping(raw_skills, use_llm=not disable_llm)

    structured_job["skills"] = list(dict.fromkeys(
        mapping.get(skill, skill.lower().strip())
        for skill in structured_job.get("skills", [])
        if skill and skill.strip()
    ))
    structured_job["tech_stack"] = list(dict.fromkeys(
        mapping.get(skill, skill.lower().strip())
        for skill in structured_job.get("tech_stack", [])
        if skill and skill.strip()
    ))
    structured_job["_skill_normalization_map"] = mapping
    return structured_job


# ── public API ────────────────────────────────────────────────────────────────

def normalize_skills(structured_resume: dict) -> dict:
    """
    Pass 2: Normalize all skill strings in-place inside structured_resume.

    Also normalizes tech_stack arrays inside projects.

    Args:
        structured_resume: Output dict from resume_parser.extract_sections().

    Returns:
        The same dict with skills (and project tech_stacks) normalized.
    """
    # Collect all unique raw skills
    raw_skills: list[str] = list(structured_resume.get("skills", []))
    for project in structured_resume.get("projects", []):
        raw_skills.extend(project.get("tech_stack", []))

    mapping = _build_skill_mapping(raw_skills)

    # Apply mapping
    structured_resume["skills"] = [
        mapping.get(s, s.lower().strip()) for s in structured_resume.get("skills", [])
    ]
    # Remove duplicates that arise after normalization, preserve order
    structured_resume["skills"] = list(dict.fromkeys(structured_resume["skills"]))

    for project in structured_resume.get("projects", []):
        project["tech_stack"] = list(dict.fromkeys(
            mapping.get(s, s.lower().strip()) for s in project.get("tech_stack", [])
        ))

    # Attach full mapping for transparency / debugging
    structured_resume["_skill_normalization_map"] = mapping

    return structured_resume
