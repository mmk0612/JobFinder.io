"""
src/job_processor.py
--------------------
Inline processing pipeline for scraped job descriptions.

Pipeline:
  raw job row
      ↓ Gemini extraction
  structured fields
      ↓ normalization
  canonical skills / tech stack
      ↓ SentenceTransformers
  job embedding
      ↓ PostgreSQL + FAISS
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import numpy as np

from src.db.db import get_jobs_for_indexing, update_processed_job
from src.embedder import AVAILABLE_MODELS, generate_job_embedding, generate_job_embeddings
from src.job_description_parser import extract_job_description
from src.normalizer import normalize_job_description
from src.scrapers.models import JobListing
from src.vector_store import ResumeVectorStore, dim_for_model

logger = logging.getLogger(__name__)

DEFAULT_JOB_INDEX_DIR = "output/job_index"
DEFAULT_JOB_EMBEDDING_MODEL = os.environ.get("JOB_EMBEDDING_MODEL", "minilm").strip() or "minilm"
DEFAULT_JOB_PROCESSING_CONCURRENCY = max(
    1,
    int(os.environ.get("JOB_PROCESSING_CONCURRENCY", "4") or "4"),
)


def process_job_listings(
    jobs: list[JobListing],
    *,
    model_key: str = DEFAULT_JOB_EMBEDDING_MODEL,
    index_dir: str | Path = DEFAULT_JOB_INDEX_DIR,
    concurrency: int = DEFAULT_JOB_PROCESSING_CONCURRENCY,
    rebuild_index: bool = True,
) -> dict:
    """
    Process freshly scraped jobs and rebuild the FAISS job index.

    Returns:
        {
          "processed": int,
          "errors": {url: message}
        }
    """
    result = asyncio.run(
        process_job_listings_async(
            jobs,
            model_key=model_key,
            concurrency=concurrency,
        )
    )

    if rebuild_index and result["processed"]:
        rebuild_job_index(index_dir=index_dir)

    return result


async def process_job_listings_async(
    jobs: list[JobListing],
    *,
    model_key: str = DEFAULT_JOB_EMBEDDING_MODEL,
    concurrency: int = DEFAULT_JOB_PROCESSING_CONCURRENCY,
) -> dict:
    """
    Async/concurrent job processing.

    Each job is processed in a worker thread so blocking SDK/database calls don't
    block the event loop.
    """
    processed = 0
    errors: dict[str, str] = {}
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    prepared: list[tuple[JobListing, dict]] = []

    async def _worker(job: JobListing) -> None:
        async with semaphore:
            try:
                prepared_job = await asyncio.to_thread(_prepare_single_job, job)
                async with lock:
                    prepared.append((job, prepared_job))
            except Exception as exc:
                async with lock:
                    errors[job.url] = str(exc)
                logger.warning("Job processing failed for %s: %s", job.url, exc)

    await asyncio.gather(*(_worker(job) for job in jobs))

    if not prepared:
        return {"processed": 0, "errors": errors}

    batch_docs: list[dict[str, np.ndarray | str]] | None = None
    try:
        batch_docs = await asyncio.to_thread(
            generate_job_embeddings,
            [item[1] for item in prepared],
            model_key=model_key,
        )
    except Exception as exc:
        logger.warning("Batch embedding failed; falling back to per-job embedding: %s", exc)

    persist_semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _persist_one(idx: int, job: JobListing, prepared_job: dict) -> None:
        nonlocal processed
        if job.url in errors:
            return
        async with persist_semaphore:
            try:
                if batch_docs is not None:
                    embedding_doc = batch_docs[idx]
                else:
                    embedding_doc = await asyncio.to_thread(
                        generate_job_embedding,
                        prepared_job,
                        model_key=model_key,
                    )

                await asyncio.to_thread(
                    update_processed_job,
                    job.url,
                    processed_skills=prepared_job["processed_skills"],
                    processed_tech_stack=prepared_job["processed_tech_stack"],
                    processed_experience_required=prepared_job["processed_experience_required"],
                    processed_experience_text=prepared_job["processed_experience_text"],
                    processed_seniority=prepared_job["processed_seniority"],
                    processed_summary=prepared_job["processed_summary"],
                    processed_payload=prepared_job["processed_payload"],
                    job_embedding=embedding_doc["job_embedding"].tolist(),
                    job_embedding_model=str(embedding_doc["model"]),
                    job_embedding_text=str(embedding_doc["job_text"]),
                )
                async with lock:
                    processed += 1
            except Exception as exc:
                async with lock:
                    errors[job.url] = str(exc)
                logger.warning("Job persistence failed for %s: %s", job.url, exc)

    await asyncio.gather(*(
        _persist_one(idx, job, prepared_job)
        for idx, (job, prepared_job) in enumerate(prepared)
    ))

    return {"processed": processed, "errors": errors}


def rebuild_job_index(
    *,
    index_dir: str | Path = DEFAULT_JOB_INDEX_DIR,
) -> int:
    """Rebuild the FAISS job index from processed rows stored in PostgreSQL."""
    rows = get_jobs_for_indexing()
    if not rows:
        model_name = AVAILABLE_MODELS.get(DEFAULT_JOB_EMBEDDING_MODEL, DEFAULT_JOB_EMBEDDING_MODEL)
        store = ResumeVectorStore(dim=dim_for_model(model_name))
        store.save(index_dir)
        return 0

    model_name = rows[0].get("job_embedding_model") or AVAILABLE_MODELS.get(DEFAULT_JOB_EMBEDDING_MODEL, DEFAULT_JOB_EMBEDDING_MODEL)
    dim = dim_for_model(model_name)
    store = ResumeVectorStore(dim=dim)

    vectors: list[np.ndarray] = []
    metadata: list[dict] = []
    for row in rows:
        embedding = np.asarray(row.get("job_embedding", []), dtype=np.float32)
        if embedding.size != dim:
            continue
        vectors.append(embedding)
        metadata.append({
            "type": "job",
            "url": row["url"],
            "job_title": row["job_title"],
            "company": row["company"],
            "source": row["source"],
            "location": row["location"],
            "seniority": row.get("processed_seniority", ""),
            "skills": row.get("processed_skills", []),
            "model": row.get("job_embedding_model", ""),
        })

    if vectors:
        store.add_batch(np.vstack(vectors), metadata)
    store.save(index_dir)
    return len(vectors)


def _prepare_single_job(job: JobListing) -> dict:
    raw_text = _build_job_input_text(job)
    extracted = extract_job_description(raw_text)
    normalized = normalize_job_description(extracted)

    return {
        "job_title": job.job_title,
        "company": job.company,
        "location": job.location,
        "description": job.description,
        "processed_skills": normalized.get("skills", []),
        "processed_tech_stack": normalized.get("tech_stack", []),
        "processed_experience_required": normalized.get("experience_required"),
        "processed_experience_text": normalized.get("experience_text", ""),
        "processed_seniority": normalized.get("seniority", "unknown"),
        "processed_summary": normalized.get("summary", ""),
        "processed_payload": normalized,
    }


def _build_job_input_text(job: JobListing) -> str:
    parts: list[str] = [
        f"Title: {job.job_title}",
        f"Company: {job.company}",
    ]
    if job.location:
        parts.append(f"Location: {job.location}")
    if job.salary:
        parts.append(f"Salary: {job.salary}")
    if job.experience_required:
        parts.append(f"Experience requirement: {job.experience_required}")
    if job.description:
        parts.append(f"Description: {job.description}")
    return "\n".join(parts)