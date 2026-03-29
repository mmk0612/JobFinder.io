"""
embedder.py
-----------
Generate embedding vectors from a structured resume using SentenceTransformers.

Two embeddings are produced:
  1. skills_embedding   — normalized skills list joined as a sentence.
                          Best for fast skill-overlap matching against jobs.
  2. profile_embedding  — richer "resume profile" text (summary + skills +
                          experience titles + project descriptions).
                          Best for semantic similarity against job descriptions.

Both vectors are float32 numpy arrays of shape (embedding_dim,).
"""

from __future__ import annotations

import os
import re
import threading

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ── model registry ─────────────────────────────────────────────────────────────
# All three are strong general-purpose models; bge-large-en gives the best
# retrieval quality but is ~1.3 GB; use all-mpnet-base-v2 (~420 MB) if disk
# space or first-run time matters.

AVAILABLE_MODELS = {
    "bge-large":  "BAAI/bge-large-en",
    "e5-large":   "intfloat/e5-large",
    "mpnet":      "sentence-transformers/all-mpnet-base-v2",
    "minilm":     "sentence-transformers/all-MiniLM-L6-v2",
}
DEFAULT_MODEL = "minilm"
_provider_raw = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
if not _provider_raw:
    _provider_raw = "nvidia" if os.environ.get("NVIDIA_API_KEY", "").strip() else "local"
EMBEDDING_PROVIDER = _provider_raw
NVIDIA_EMBEDDING_MODEL = (
    os.environ.get("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5").strip()
    or "nvidia/nv-embedqa-e5-v5"
)
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip() or "https://integrate.api.nvidia.com/v1"
NVIDIA_EMBEDDING_TIMEOUT_SECONDS = float(
    os.environ.get("NVIDIA_EMBEDDING_TIMEOUT_SECONDS", "30") or "30"
)
NVIDIA_QUERY_INPUT_TYPE = os.environ.get("NVIDIA_QUERY_INPUT_TYPE", "query").strip() or "query"
NVIDIA_PASSAGE_INPUT_TYPE = os.environ.get("NVIDIA_PASSAGE_INPUT_TYPE", "passage").strip() or "passage"
FIXED_EMBEDDING_DIM = max(1, int(os.environ.get("EMBEDDING_DIM", "768") or "768"))
NVIDIA_EMBEDDING_MAX_WORDS = max(
    32,
    int(os.environ.get("NVIDIA_EMBEDDING_MAX_WORDS", "420") or "420"),
)
NVIDIA_EMBEDDING_MAX_CHARS = max(
    256,
    int(os.environ.get("NVIDIA_EMBEDDING_MAX_CHARS", "1800") or "1800"),
)
JOB_EMBEDDING_MAX_SKILLS = max(
    4,
    int(os.environ.get("JOB_EMBEDDING_MAX_SKILLS", "24") or "24"),
)
JOB_EMBEDDING_MAX_TECH_STACK = max(
    4,
    int(os.environ.get("JOB_EMBEDDING_MAX_TECH_STACK", "24") or "24"),
)
JOB_EMBEDDING_MAX_SUMMARY_CHARS = max(
    120,
    int(os.environ.get("JOB_EMBEDDING_MAX_SUMMARY_CHARS", "420") or "420"),
)
JOB_EMBEDDING_MAX_DESCRIPTION_CHARS = max(
    240,
    int(os.environ.get("JOB_EMBEDDING_MAX_DESCRIPTION_CHARS", "1200") or "1200"),
)
JOB_EMBEDDING_MAX_TOTAL_CHARS = max(
    400,
    int(os.environ.get("JOB_EMBEDDING_MAX_TOTAL_CHARS", "1800") or "1800"),
)

# Module-level cache so the model is only loaded once per process
_model_cache: dict[str, SentenceTransformer] = {}
_model_cache_lock = threading.Lock()


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        denom = np.linalg.norm(arr) or 1.0
        return arr / denom
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return arr / denom


def _resize_to_dim(vectors: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] == dim:
            return arr
        if arr.shape[0] > dim:
            return arr[:dim]
        out = np.zeros((dim,), dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out

    if arr.shape[1] == dim:
        return arr
    if arr.shape[1] > dim:
        return arr[:, :dim]

    out = np.zeros((arr.shape[0], dim), dtype=np.float32)
    out[:, : arr.shape[1]] = arr
    return out


def _effective_model_name(base_model: str) -> str:
    return f"{base_model}::dim{FIXED_EMBEDDING_DIM}"


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _clip_text(value: str, max_chars: int) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    last_space = clipped.rfind(" ")
    if last_space >= max_chars // 2:
        clipped = clipped[:last_space]
    return clipped.strip()


def _clean_term_list(values: list[str], max_items: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        term = _clean_text(value)
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max_items:
            break
    return out


def _compress_description(text: str, *, max_chars: int) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    lines = [line.strip() for line in raw.splitlines() if line and line.strip()]
    compact_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned = re.sub(r"https?://\S+", " ", line)
        cleaned = re.sub(r"\S+@\S+", " ", cleaned)
        cleaned = _clean_text(cleaned)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        compact_lines.append(cleaned)

    compact = _clean_text(" ".join(compact_lines))
    return _clip_text(compact, max_chars)


def _truncate_for_nvidia_embedding(text: str) -> str:
    """
    Truncate embedding input to stay under NVIDIA endpoint token limits.

    NVIDIA embed models often enforce a 512-token cap. We apply a conservative
    word-based cap so both batch and single calls remain valid.
    """
    raw = str(text or "")
    words = raw.split()
    trimmed = raw if len(words) <= NVIDIA_EMBEDDING_MAX_WORDS else " ".join(words[:NVIDIA_EMBEDDING_MAX_WORDS])
    if len(trimmed) > NVIDIA_EMBEDDING_MAX_CHARS:
        trimmed = trimmed[:NVIDIA_EMBEDDING_MAX_CHARS]
    return trimmed


def _is_token_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "maximum allowed token size" in text or "input length" in text or "token size" in text


def _use_nvidia_provider() -> bool:
    return EMBEDDING_PROVIDER == "nvidia"


def _get_nvidia_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set; cannot use EMBEDDING_PROVIDER=nvidia")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, max_retries=0)


def _supports_retry_without_input_type(exc: Exception) -> bool:
    text = str(exc).lower()
    # Retry without input_type only when server rejects the field itself.
    return (
        "unknown" in text
        or "unexpected" in text
        or "additional properties" in text
        or "extra fields not permitted" in text
    )


def _embed_text_nvidia(text: str, *, input_type: str | None = None) -> np.ndarray:
    client = _get_nvidia_client()
    safe_text = _truncate_for_nvidia_embedding(text)
    response = None
    for _ in range(3):
        kwargs = {
            "model": NVIDIA_EMBEDDING_MODEL,
            "input": [safe_text],
            "timeout": NVIDIA_EMBEDDING_TIMEOUT_SECONDS,
        }
        if input_type:
            kwargs["extra_body"] = {"input_type": input_type}
        try:
            response = client.embeddings.create(**kwargs)
            break
        except Exception as exc:
            if input_type and _supports_retry_without_input_type(exc):
                response = client.embeddings.create(
                    model=NVIDIA_EMBEDDING_MODEL,
                    input=[safe_text],
                    timeout=NVIDIA_EMBEDDING_TIMEOUT_SECONDS,
                )
                break
            if _is_token_limit_error(exc):
                safe_text = safe_text[: max(128, len(safe_text) // 2)]
                continue
            raise
    if response is None:
        raise RuntimeError("NVIDIA embedding request failed after truncation retries")
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    vector = _resize_to_dim(vector, FIXED_EMBEDDING_DIM)
    return _l2_normalize(vector).astype(np.float32)


def _embed_texts_nvidia(texts: list[str], *, input_type: str | None = None) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    client = _get_nvidia_client()
    safe_texts = [_truncate_for_nvidia_embedding(text) for text in texts]
    response = None
    for _ in range(3):
        kwargs = {
            "model": NVIDIA_EMBEDDING_MODEL,
            "input": safe_texts,
            "timeout": NVIDIA_EMBEDDING_TIMEOUT_SECONDS,
        }
        if input_type:
            kwargs["extra_body"] = {"input_type": input_type}
        try:
            response = client.embeddings.create(**kwargs)
            break
        except Exception as exc:
            if input_type and _supports_retry_without_input_type(exc):
                response = client.embeddings.create(
                    model=NVIDIA_EMBEDDING_MODEL,
                    input=safe_texts,
                    timeout=NVIDIA_EMBEDDING_TIMEOUT_SECONDS,
                )
                break
            if _is_token_limit_error(exc):
                safe_texts = [s[: max(128, len(s) // 2)] for s in safe_texts]
                continue
            raise
    if response is None:
        raise RuntimeError("NVIDIA batch embedding request failed after truncation retries")
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = np.asarray([item.embedding for item in ordered], dtype=np.float32)
    vectors = _resize_to_dim(vectors, FIXED_EMBEDDING_DIM)
    return _l2_normalize(vectors).astype(np.float32)


def load_model(model_key: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load (and cache) a SentenceTransformer model.

    Args:
        model_key: One of "bge-large", "e5-large", "mpnet",
                   or a full HuggingFace model ID.

    Returns:
        A loaded SentenceTransformer instance.
    """
    model_name = AVAILABLE_MODELS.get(model_key, model_key)
    with _model_cache_lock:
        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name)
        return _model_cache[model_name]


# ── text builders ──────────────────────────────────────────────────────────────

def build_skills_text(structured_resume: dict) -> str:
    """
    Create a compact skills sentence.

    Example output:
      "Skills: python react nodejs postgresql aws docker kubernetes"
    """
    skills = structured_resume.get("skills", [])
    if not skills:
        return ""
    return "Skills: " + " ".join(skills)


def build_profile_text(structured_resume: dict) -> str:
    """
    Build a richer resume profile string for semantic embedding.

    Combines: name, summary, skills, experience job titles + companies,
    and project names + descriptions.
    """
    parts: list[str] = []

    # Contact name
    name = structured_resume.get("contact", {}).get("name", "")
    if name:
        parts.append(f"Candidate: {name}")

    # Summary / objective
    summary = structured_resume.get("summary", "")
    if summary:
        parts.append(summary)

    # Skills
    skills_text = build_skills_text(structured_resume)
    if skills_text:
        parts.append(skills_text)

    # Experience: "Title at Company"
    for exp in structured_resume.get("experience", []):
        title   = exp.get("title", "")
        company = exp.get("company", "")
        bullets = exp.get("bullets", [])
        if title or company:
            line = f"Experience: {title} at {company}".strip(" at")
            parts.append(line)
        # Include first 2 bullets for richer context
        for bullet in bullets[:2]:
            if bullet:
                parts.append(f"  - {bullet}")

    # Projects: name + description + tech stack
    for proj in structured_resume.get("projects", []):
        proj_name = proj.get("name", "")
        desc      = proj.get("description", "")
        stack     = proj.get("tech_stack", [])
        proj_line = f"Project: {proj_name}. {desc}"
        if stack:
            proj_line += f" Tech: {', '.join(stack)}"
        parts.append(proj_line.strip())

    return "\n".join(p for p in parts if p)


def build_job_text(processed_job: dict) -> str:
    """
    Build a canonical text representation of a processed job description.

    Combines title, company, location, seniority, experience, skills,
    tech stack, summary, and raw description into one embedding-friendly text.
    """
    parts: list[str] = []

    title = _clean_text(processed_job.get("job_title", ""))
    company = _clean_text(processed_job.get("company", ""))
    if title or company:
        parts.append(f"Role: {title} at {company}".strip())

    location = _clean_text(processed_job.get("location", ""))
    if location:
        parts.append(f"Location: {location}")

    seniority = _clean_text(processed_job.get("processed_seniority", "") or processed_job.get("seniority", ""))
    if seniority:
        parts.append(f"Seniority: {seniority}")

    years = processed_job.get("processed_experience_required")
    if years is None:
        years = processed_job.get("experience_required")
    if years not in (None, ""):
        parts.append(f"Experience required: {years} years")

    skills = _clean_term_list(
        processed_job.get("processed_skills") or processed_job.get("skills", []),
        max_items=JOB_EMBEDDING_MAX_SKILLS,
    )
    if skills:
        parts.append("Skills: " + ", ".join(skills))

    tech_stack = _clean_term_list(
        processed_job.get("processed_tech_stack") or processed_job.get("tech_stack", []),
        max_items=JOB_EMBEDDING_MAX_TECH_STACK,
    )
    if tech_stack:
        parts.append("Tech stack: " + ", ".join(tech_stack))

    summary = _clip_text(
        processed_job.get("processed_summary", "") or processed_job.get("summary", ""),
        JOB_EMBEDDING_MAX_SUMMARY_CHARS,
    )
    if summary:
        parts.append(f"Summary: {summary}")

    description = _compress_description(
        processed_job.get("description", ""),
        max_chars=JOB_EMBEDDING_MAX_DESCRIPTION_CHARS,
    )
    if description:
        parts.append(f"Description: {description}")

    combined = "\n".join(part for part in parts if part)
    return _clip_text(combined, JOB_EMBEDDING_MAX_TOTAL_CHARS)


# ── public API ─────────────────────────────────────────────────────────────────

def generate_embeddings(
    structured_resume: dict,
    *,
    model_key: str = DEFAULT_MODEL,
) -> dict[str, np.ndarray]:
    """
    Generate both embeddings for a structured resume.

    Args:
        structured_resume: Output from resume_parser / normalizer.
        model_key:         Model identifier (default: "bge-large").

    Returns:
        A dict with:
          "skills_embedding"  : np.ndarray shape (dim,)
          "profile_embedding" : np.ndarray shape (dim,)
          "skills_text"       : str  (the text that was embedded)
          "profile_text"      : str  (the text that was embedded)
          "model"             : str  (HuggingFace model name used)
    """
    skills_text  = build_skills_text(structured_resume)
    profile_text = build_profile_text(structured_resume)

    if _use_nvidia_provider():
        vectors = _embed_texts_nvidia(
            [skills_text, profile_text],
            input_type=NVIDIA_QUERY_INPUT_TYPE,
        )
        skills_emb = vectors[0]
        profile_emb = vectors[1]
        model_name = _effective_model_name(NVIDIA_EMBEDDING_MODEL)
    else:
        model = load_model(model_key)
        model_name = _effective_model_name(AVAILABLE_MODELS.get(model_key, model_key))
        # Batch both strings in one encode call to reduce model overhead.
        vectors = model.encode(
            [skills_text, profile_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=2,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors = _l2_normalize(_resize_to_dim(vectors, FIXED_EMBEDDING_DIM))
        skills_emb = vectors[0]
        profile_emb = vectors[1]

    return {
        "skills_embedding":  skills_emb.astype(np.float32),
        "profile_embedding": profile_emb.astype(np.float32),
        "skills_text":       skills_text,
        "profile_text":      profile_text,
        "model":             model_name,
    }


def embed_text(text: str, *, model_key: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed an arbitrary string (e.g. a job description).
    Useful for computing similarity at query time.

    Returns normalized float32 numpy array of shape (dim,).
    """
    if _use_nvidia_provider():
        return _embed_text_nvidia(text, input_type=NVIDIA_QUERY_INPUT_TYPE)
    model = load_model(model_key)
    vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return _l2_normalize(_resize_to_dim(vector, FIXED_EMBEDDING_DIM)).astype(np.float32)


def generate_job_embedding(
    processed_job: dict,
    *,
    model_key: str = DEFAULT_MODEL,
) -> dict[str, np.ndarray | str]:
    """
    Generate a single embedding for a processed job description.

    Returns:
        {
          "job_embedding": np.ndarray,
          "job_text": str,
          "model": str,
        }
    """
    job_text = build_job_text(processed_job)

    if _use_nvidia_provider():
        job_embedding = _embed_text_nvidia(job_text, input_type=NVIDIA_PASSAGE_INPUT_TYPE)
        model_name = _effective_model_name(NVIDIA_EMBEDDING_MODEL)
    else:
        model = load_model(model_key)
        model_name = _effective_model_name(AVAILABLE_MODELS.get(model_key, model_key))
        job_embedding = model.encode(job_text, convert_to_numpy=True, normalize_embeddings=True)
        job_embedding = _l2_normalize(_resize_to_dim(job_embedding, FIXED_EMBEDDING_DIM))

    return {
        "job_embedding": job_embedding.astype(np.float32),
        "job_text": job_text,
        "model": model_name,
    }


def generate_job_embeddings(
    processed_jobs: list[dict],
    *,
    model_key: str = DEFAULT_MODEL,
) -> list[dict[str, np.ndarray | str]]:
    """
    Generate embeddings for multiple processed jobs in one model.encode call.

    This is substantially faster than per-job encode for queue batches.
    """
    if not processed_jobs:
        return []

    job_texts = [build_job_text(job) for job in processed_jobs]

    if _use_nvidia_provider():
        vectors = _embed_texts_nvidia(job_texts, input_type=NVIDIA_PASSAGE_INPUT_TYPE)
        model_name = _effective_model_name(NVIDIA_EMBEDDING_MODEL)
    else:
        model = load_model(model_key)
        model_name = _effective_model_name(AVAILABLE_MODELS.get(model_key, model_key))
        batch_size = max(1, int(os.environ.get("JOB_EMBEDDING_BATCH_SIZE", "32") or "32"))
        vectors = model.encode(
            job_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors = _l2_normalize(_resize_to_dim(vectors, FIXED_EMBEDDING_DIM))

    docs: list[dict[str, np.ndarray | str]] = []
    for idx, text in enumerate(job_texts):
        docs.append(
            {
                "job_embedding": vectors[idx],
                "job_text": text,
                "model": model_name,
            }
        )
    return docs
