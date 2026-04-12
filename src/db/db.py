"""
src/db/db.py
------------
PostgreSQL access layer using psycopg3 (psycopg[binary]).

All public functions accept / return plain Python dicts or JobListing objects
so the rest of the code never imports psycopg directly.

Connection pool is module-level and re-used across calls within the same
process.  Call `close_pool()` if you need a clean shutdown.
"""

from __future__ import annotations

import atexit
import os
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Generator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from psycopg.types.json import Jsonb

from src.scrapers.models import JobListing


TTL_DURATION = timedelta(days=30)
SALARY_USD_TO_INR = float(os.environ.get("SALARY_USD_TO_INR", "83.0") or "83.0")
SALARY_USD_TO_INR = max(1.0, SALARY_USD_TO_INR)


def _format_inr(value: float) -> str:
    """Format numeric value to INR with Indian grouping commas."""
    amount = int(round(max(0.0, value)))
    text = str(amount)
    if len(text) <= 3:
        return f"INR {text}"
    head = text[:-3]
    tail = text[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.append(head[-2:])
        head = head[:-2]
    if head:
        groups.append(head)
    return f"INR {','.join(reversed(groups))},{tail}"


def _is_inr_salary_text(text: str) -> bool:
    return bool(re.search(r"(₹|\binr\b|\brs\.?\b|\brupees?\b|\blpa\b|\blac\b|\blakh\b|\bcr\b|\bcrore\b)", text))


def _is_usd_salary_text(text: str) -> bool:
    return bool(re.search(r"(\$|\busd\b|\bdollars?\b)", text))


def _salary_values_from_text(raw_salary: str) -> list[float]:
    numbers: list[float] = []
    for match in re.finditer(r"(\d+(?:[\.,]\d+)?)\s*(k|m|lpa|lac|lakh|cr|crore)?", raw_salary):
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
        elif base < 1_000 and "$" in raw_salary:
            base *= 1_000.0
        numbers.append(base)
    return numbers


def normalize_salary_to_inr(raw_salary: str) -> str:
    """
    Normalize salary text to INR-only representation for DB storage.

    Examples:
      "$120k - $150k" -> "INR 99,60,000 - INR 1,24,50,000"
      "12-18 LPA"     -> "INR 12,00,000 - INR 18,00,000"
    """
    text = str(raw_salary or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    values = _salary_values_from_text(lowered)
    if not values:
        return text

    is_inr = _is_inr_salary_text(lowered)
    is_usd = _is_usd_salary_text(lowered)

    if is_usd and not is_inr:
        values = [value * SALARY_USD_TO_INR for value in values]

    if len(values) == 1:
        return _format_inr(values[0])

    low = min(values)
    high = max(values)
    return f"{_format_inr(low)} - {_format_inr(high)}"


# ── connection pool ───────────────────────────────────────────────────────────

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        dsn = os.environ.get("DATABASE_URL", "").strip()
        if not dsn:
            raise EnvironmentError(
                "DATABASE_URL is not set. Add it to your .env file."
            )
        _pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool:
        _pool.close()
        _pool = None


atexit.register(close_pool)


@contextmanager
def _conn() -> Generator[psycopg.Connection, None, None]:
    with _get_pool().connection() as conn:
        yield conn


# ── schema bootstrap ──────────────────────────────────────────────────────────

def apply_schema() -> None:
    """
    Run schema.sql against the connected database.
    Safe to call repeatedly (all statements use IF NOT EXISTS).
    """
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, encoding="utf-8") as f:
        sql = f.read()
    with _conn() as conn:
        conn.execute(sql)
        conn.commit()


# ── write operations ──────────────────────────────────────────────────────────

def upsert_job(job: JobListing) -> bool:
    """
    Insert a job or update it if the URL already exists.

    Returns True if a new row was inserted, False if an existing row
    was updated.
    """
    sql = """
        INSERT INTO jobs (
            job_title, company, location, description,
            salary, experience_required, url,
            source, scraped_at, expires_at, is_active
        ) VALUES (
            %(job_title)s, %(company)s, %(location)s, %(description)s,
            %(salary)s, %(experience_required)s, %(url)s,
            %(source)s, %(scraped_at)s, %(expires_at)s, TRUE
        )
        ON CONFLICT (url) DO UPDATE SET
            job_title           = EXCLUDED.job_title,
            company             = EXCLUDED.company,
            location            = EXCLUDED.location,
            description         = EXCLUDED.description,
            salary              = EXCLUDED.salary,
            experience_required = EXCLUDED.experience_required,
            source              = EXCLUDED.source,
            scraped_at          = EXCLUDED.scraped_at,
            expires_at          = EXCLUDED.expires_at,
            is_active           = TRUE,
            processed_skills              = '[]'::jsonb,
            processed_tech_stack          = '[]'::jsonb,
            processed_experience_required = NULL,
            processed_experience_text     = '',
            processed_seniority           = '',
            processed_summary             = '',
            processed_payload             = '{}'::jsonb,
            job_embedding                 = '[]'::jsonb,
            job_embedding_model           = '',
            job_embedding_text            = '',
            processed_at                  = NULL
        RETURNING (xmax = 0) AS inserted
    """
    scraped_at = datetime.now(timezone.utc)
    payload = job.to_dict()
    payload["salary"] = normalize_salary_to_inr(payload.get("salary", ""))
    params = {
        **payload,
        "scraped_at": scraped_at,
        "expires_at": scraped_at + TTL_DURATION,
    }
    with _conn() as conn:
        row = conn.execute(sql, params).fetchone()
        conn.commit()
    return bool(row["inserted"]) if row else False


def upsert_jobs(jobs: list[JobListing]) -> dict[str, int]:
    """
    Bulk upsert a list of jobs.  Returns {"inserted": n, "updated": m}.
    """
    inserted = updated = 0
    for job in jobs:
        was_new = upsert_job(job)
        if was_new:
            inserted += 1
        else:
            updated += 1
    return {"inserted": inserted, "updated": updated}


def backfill_job_salaries_to_inr(*, limit: int = 10000, offset: int = 0) -> int:
    """
    Convert existing non-empty salary rows in `jobs` table to INR format.

    Returns number of rows updated in this batch.
    """
    sql_select = """
        SELECT id, salary
        FROM jobs
        WHERE COALESCE(salary, '') <> ''
        ORDER BY id ASC
        LIMIT %(limit)s OFFSET %(offset)s
    """
    sql_update = """
        UPDATE jobs
        SET salary = %(salary)s
        WHERE id = %(id)s
    """

    rows_updated = 0
    with _conn() as conn:
        rows = conn.execute(
            sql_select,
            {"limit": max(1, int(limit)), "offset": max(0, int(offset))},
        ).fetchall()

        for row in rows:
            original = str(row.get("salary", "") or "")
            normalized = normalize_salary_to_inr(original)
            if normalized != original:
                conn.execute(sql_update, {"id": int(row["id"]), "salary": normalized})
                rows_updated += 1

        conn.commit()

    return rows_updated


def mark_stale_jobs_inactive(source: str, scraped_before: datetime) -> int:
    """
    Mark jobs from `source` that have not been re-scraped since
    `scraped_before` as inactive (is_active = FALSE).

    Returns the number of rows updated.
    """
    sql = """
        UPDATE jobs
        SET    is_active = FALSE
        WHERE  source = %(source)s
          AND  scraped_at < %(scraped_before)s
          AND  is_active = TRUE
    """
    with _conn() as conn:
        cur = conn.execute(sql, {"source": source, "scraped_before": scraped_before})
        conn.commit()
        return cur.rowcount


def update_processed_job(
    url: str,
    *,
    processed_skills: list[str],
    processed_tech_stack: list[str],
    processed_experience_required: int | None,
    processed_experience_text: str,
    processed_seniority: str,
    processed_summary: str,
    processed_payload: dict,
    job_embedding: list[float],
    job_embedding_model: str,
    job_embedding_text: str,
) -> int:
    """Persist processed fields and embedding for a job row identified by URL."""
    sql = """
        UPDATE jobs
        SET processed_skills              = %(processed_skills)s,
            processed_tech_stack          = %(processed_tech_stack)s,
            processed_experience_required = %(processed_experience_required)s,
            processed_experience_text     = %(processed_experience_text)s,
            processed_seniority           = %(processed_seniority)s,
            processed_summary             = %(processed_summary)s,
            processed_payload             = %(processed_payload)s,
            job_embedding                 = %(job_embedding)s,
            job_embedding_model           = %(job_embedding_model)s,
            job_embedding_text            = %(job_embedding_text)s,
            processed_at                  = NOW()
        WHERE url = %(url)s
    """
    params = {
        "url": url,
        "processed_skills": Jsonb(processed_skills),
        "processed_tech_stack": Jsonb(processed_tech_stack),
        "processed_experience_required": processed_experience_required,
        "processed_experience_text": processed_experience_text,
        "processed_seniority": processed_seniority,
        "processed_summary": processed_summary,
        "processed_payload": Jsonb(processed_payload),
        "job_embedding": Jsonb(job_embedding),
        "job_embedding_model": job_embedding_model,
        "job_embedding_text": job_embedding_text,
    }
    with _conn() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount


def enqueue_jobs_for_processing(
    urls: list[str],
    *,
    max_attempts: int = 3,
) -> int:
    """
    Upsert queue rows so each job URL is in queued state.

    Durable semantics: rows survive process restarts and are retried by workers.
    """
    if not urls:
        return 0

    unique_urls = list(dict.fromkeys(url for url in urls if url))
    sql = """
        INSERT INTO job_processing_queue (
            job_url,
            status,
            attempts,
            max_attempts,
            available_at,
            locked_at,
            worker_id,
            last_error,
            created_at,
            updated_at
        )
        SELECT
            url,
            'queued',
            0,
            %(max_attempts)s,
            NOW(),
            NULL,
            NULL,
            '',
            NOW(),
            NOW()
        FROM jobs
        WHERE url = ANY(%(urls)s)
        ON CONFLICT (job_url) DO UPDATE SET
            status       = 'queued',
            attempts     = 0,
            max_attempts = EXCLUDED.max_attempts,
            available_at = NOW(),
            locked_at    = NULL,
            worker_id    = NULL,
            last_error   = '',
            updated_at   = NOW()
        RETURNING id
    """
    with _conn() as conn:
        rows = conn.execute(sql, {"urls": unique_urls, "max_attempts": max(1, max_attempts)}).fetchall()
        conn.commit()
        return len(rows)


def dequeue_jobs_for_processing(
    *,
    worker_id: str,
    limit: int = 10,
) -> list[dict]:
    """
    Atomically claim queued jobs for a worker using FOR UPDATE SKIP LOCKED.

    Returns queue rows: id, job_url, attempts, max_attempts.
    """
    sql = """
        WITH claimable AS (
            SELECT id
            FROM job_processing_queue
            WHERE status = 'queued'
              AND available_at <= NOW()
            ORDER BY available_at ASC, id ASC
            FOR UPDATE SKIP LOCKED
            LIMIT %(limit)s
        )
        UPDATE job_processing_queue q
        SET status = 'processing',
            locked_at = NOW(),
            worker_id = %(worker_id)s,
            updated_at = NOW()
        FROM claimable c
        WHERE q.id = c.id
        RETURNING q.id, q.job_url, q.attempts, q.max_attempts
    """
    with _conn() as conn:
        rows = conn.execute(sql, {"worker_id": worker_id, "limit": max(1, limit)}).fetchall()
        conn.commit()
        return rows


def mark_job_processing_done(queue_id: int) -> int:
    sql = """
        UPDATE job_processing_queue
        SET status = 'done',
            locked_at = NULL,
            worker_id = NULL,
            last_error = '',
            updated_at = NOW()
        WHERE id = %(queue_id)s
    """
    with _conn() as conn:
        cur = conn.execute(sql, {"queue_id": queue_id})
        conn.commit()
        return cur.rowcount


def mark_job_processing_failed(
    queue_id: int,
    *,
    error_message: str,
    retry_delay_seconds: int = 60,
) -> int:
    """
    Record a failed queue item and either requeue it (with delay)
    or mark it as permanently failed when attempts are exhausted.
    """
    sql = """
        UPDATE job_processing_queue
        SET attempts = attempts + 1,
            status = CASE
                        WHEN attempts + 1 >= max_attempts THEN 'failed'
                        ELSE 'queued'
                     END,
            available_at = CASE
                              WHEN attempts + 1 >= max_attempts THEN NOW()
                              ELSE NOW() + (%(retry_delay_seconds)s::text || ' seconds')::interval
                           END,
            locked_at = NULL,
            worker_id = NULL,
            last_error = %(error_message)s,
            updated_at = NOW()
        WHERE id = %(queue_id)s
    """
    params = {
        "queue_id": queue_id,
        "retry_delay_seconds": max(1, retry_delay_seconds),
        "error_message": error_message[:2000],
    }
    with _conn() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount


def requeue_stale_processing_jobs(*, stale_seconds: int = 300) -> int:
    """
    Move stale `processing` queue items back to `queued`.

    Stale means the row has been in `processing` for at least `stale_seconds`.
    """
    stale_seconds = max(1, int(stale_seconds))
    sql = """
        UPDATE job_processing_queue
        SET status = 'queued',
            available_at = NOW(),
            locked_at = NULL,
            worker_id = NULL,
            updated_at = NOW()
        WHERE status = 'processing'
          AND locked_at IS NOT NULL
          AND locked_at <= NOW() - (%(stale_seconds)s::text || ' seconds')::interval
    """
    with _conn() as conn:
        cur = conn.execute(sql, {"stale_seconds": stale_seconds})
        conn.commit()
        return cur.rowcount


def requeue_processing_jobs_for_worker(*, worker_id: str) -> int:
    """
    Requeue all `processing` rows currently locked by a specific worker.

    Useful during graceful worker shutdown to avoid orphaned processing rows.
    """
    sql = """
        UPDATE job_processing_queue
        SET status = 'queued',
            available_at = NOW(),
            locked_at = NULL,
            worker_id = NULL,
            updated_at = NOW()
        WHERE status = 'processing'
          AND worker_id = %(worker_id)s
    """
    with _conn() as conn:
        cur = conn.execute(sql, {"worker_id": worker_id})
        conn.commit()
        return cur.rowcount


def requeue_all_processing_jobs() -> int:
    """Requeue all rows currently in `processing` state."""
    sql = """
        UPDATE job_processing_queue
        SET status = 'queued',
            available_at = NOW(),
            locked_at = NULL,
            worker_id = NULL,
            updated_at = NOW()
        WHERE status = 'processing'
    """
    with _conn() as conn:
        cur = conn.execute(sql)
        conn.commit()
        return cur.rowcount


def get_job_processing_queue_counts() -> dict[str, int]:
    """Return status-wise queue counts."""
    sql = """
        SELECT status, COUNT(*)::int AS n
        FROM job_processing_queue
        GROUP BY status
    """
    counts = {"queued": 0, "processing": 0, "done": 0, "failed": 0}
    with _conn() as conn:
        rows = conn.execute(sql).fetchall()
    for row in rows:
        counts[row["status"]] = int(row["n"])
    counts["total"] = sum(counts.values())
    return counts


def clear_job_processing_queue(*, include_done: bool = False) -> int:
    """
    Clear queue rows and return number of deleted rows.

    By default, only pending/error rows are removed (queued, processing, failed)
    and completed history rows (`done`) are kept.
    """
    if include_done:
        sql = "DELETE FROM job_processing_queue"
        params: dict[str, object] = {}
    else:
        sql = "DELETE FROM job_processing_queue WHERE status <> %(done)s"
        params = {"done": "done"}

    with _conn() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount


def get_job_by_url(url: str) -> dict | None:
    """Fetch one job row by URL."""
    with _conn() as conn:
        return conn.execute("SELECT * FROM jobs WHERE url = %(url)s", {"url": url}).fetchone()


def create_job_recommendation_request(
    *,
    email: str,
    requested_role: str,
    resume_original_name: str,
    resume_stored_path: str,
) -> int:
    """Create or refresh a queued recommendation request keyed by unique (email, role)."""
    if not requested_role or not str(requested_role).strip():
        raise ValueError("requested_role cannot be empty.")

    normalized_email = email.strip().lower()
    normalized_role = str(requested_role).strip()

    if not normalized_email:
        raise ValueError("email cannot be empty.")

    sql = """
        INSERT INTO job_recommendation_requests (
            email,
            requested_role,
            resume_original_name,
            resume_stored_path,
            status,
            notes,
            created_at,
            updated_at
        ) VALUES (
            %(email)s,
            %(requested_role)s,
            %(resume_original_name)s,
            %(resume_stored_path)s,
            'queued',
            '',
            NOW(),
            NOW()
        )
        ON CONFLICT (email, requested_role) DO UPDATE SET
            resume_original_name = EXCLUDED.resume_original_name,
            resume_stored_path   = EXCLUDED.resume_stored_path,
            status               = 'queued',
            notes                = '',
            updated_at           = NOW()
        RETURNING id
    """
    params = {
        "email": normalized_email,
        "requested_role": normalized_role,
        "resume_original_name": resume_original_name.strip(),
        "resume_stored_path": resume_stored_path.strip(),
    }
    with _conn() as conn:
        row = conn.execute(sql, params).fetchone()
        conn.commit()
    return int(row["id"]) if row else 0


def get_recommendation_requests_by_status(
    *,
    status: str | None = "queued",
    limit: int = 100,
) -> list[dict]:
    """Fetch recommendation requests, optionally filtered by status, oldest first."""
    params: dict[str, object] = {"limit": max(1, int(limit))}
    with _conn() as conn:
        column_rows = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'job_recommendation_requests'
            """
        ).fetchall()
        columns = {str(row["column_name"]) for row in column_rows}

        requested_role_expr = "requested_role" if "requested_role" in columns else "NULL::TEXT AS requested_role"
        requested_roles_expr = "requested_roles" if "requested_roles" in columns else "NULL::JSONB AS requested_roles"

        base_sql = f"""
            SELECT
                id,
                email,
                {requested_role_expr},
                {requested_roles_expr},
                resume_original_name,
                resume_stored_path,
                status,
                notes,
                created_at,
                updated_at
            FROM job_recommendation_requests
        """

        if status is None:
            sql = base_sql + " ORDER BY created_at ASC, id ASC LIMIT %(limit)s"
        else:
            sql = base_sql + " WHERE status = %(status)s ORDER BY created_at ASC, id ASC LIMIT %(limit)s"
            params["status"] = (status or "queued").strip().lower()

        return conn.execute(sql, params).fetchall()


def update_recommendation_request_status(
    *,
    request_id: int,
    status: str,
    notes: str = "",
) -> int:
    """Update recommendation request status and notes."""
    normalized_status = (status or "").strip().lower()
    allowed = {"queued", "processing", "done", "failed"}
    if normalized_status not in allowed:
        raise ValueError(f"Invalid status {status!r}. Expected one of {sorted(allowed)}")

    sql = """
        UPDATE job_recommendation_requests
        SET status = %(status)s,
            notes = %(notes)s,
            updated_at = NOW()
        WHERE id = %(request_id)s
    """
    params = {
        "request_id": int(request_id),
        "status": normalized_status,
        "notes": (notes or "")[:2000],
    }
    with _conn() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount


# ── read operations ───────────────────────────────────────────────────────────

def get_jobs(
    *,
    source: str | None = None,
    active_only: bool = True,
    limit: int = 500,
    offset: int = 0,
) -> list[dict]:
    """
    Fetch jobs with optional source filter.

    Returns a list of plain dicts (column → value).
    """
    conditions = []
    params: dict = {"limit": limit, "offset": offset}

    if active_only:
        conditions.append("is_active = TRUE")
        conditions.append("expires_at > NOW()")
    if source:
        conditions.append("source = %(source)s")
        params["source"] = source

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT * FROM jobs
        {where}
        ORDER BY scraped_at DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """
    with _conn() as conn:
        return conn.execute(sql, params).fetchall()


def get_jobs_needing_processing(
    *,
    source: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[dict]:
    """
    Return active, unexpired jobs that have never been processed or were updated
    after their last processing run.
    """
    conditions = [
        "is_active = TRUE",
        "expires_at > NOW()",
        "(processed_at IS NULL OR processed_at < scraped_at)",
    ]
    params: dict = {"limit": limit, "offset": offset}
    if source:
        conditions.append("source = %(source)s")
        params["source"] = source

    sql = f"""
        SELECT * FROM jobs
        WHERE {' AND '.join(conditions)}
        ORDER BY scraped_at DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """
    with _conn() as conn:
        return conn.execute(sql, params).fetchall()


def get_jobs_for_indexing(*, limit: int = 5000, offset: int = 0) -> list[dict]:
    """Return active processed jobs that have a stored embedding."""
    sql = """
        SELECT * FROM jobs
        WHERE is_active = TRUE
          AND expires_at > NOW()
          AND processed_at IS NOT NULL
          AND jsonb_array_length(job_embedding) > 0
        ORDER BY scraped_at DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """
    with _conn() as conn:
        return conn.execute(sql, {"limit": limit, "offset": offset}).fetchall()


def get_matchable_jobs(
    *,
    source: str | None = None,
    limit: int = 5000,
    offset: int = 0,
) -> list[dict]:
    """
    Return active processed jobs that are ready for resume matching.

    Matchable rows must have a non-empty `job_embedding` and be unexpired.
    """
    conditions = [
        "is_active = TRUE",
        "expires_at > NOW()",
        "processed_at IS NOT NULL",
        "jsonb_array_length(COALESCE(job_embedding, '[]'::jsonb)) > 0",
    ]
    params: dict = {"limit": max(1, int(limit)), "offset": max(0, int(offset))}

    if source:
        conditions.append("source = %(source)s")
        params["source"] = source

    sql = f"""
        SELECT *
        FROM jobs
        WHERE {' AND '.join(conditions)}
        ORDER BY scraped_at DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """
    with _conn() as conn:
        return conn.execute(sql, params).fetchall()


def count_jobs(*, source: str | None = None, active_only: bool = True) -> int:
    """Return total job count matching filters."""
    conditions = []
    params: dict = {}

    if active_only:
        conditions.append("is_active = TRUE")
        conditions.append("expires_at > NOW()")
    if source:
        conditions.append("source = %(source)s")
        params["source"] = source

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    with _conn() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM jobs {where}", params).fetchone()
        return int(row["n"]) if row else 0


def cleanup_expired_jobs() -> dict[str, int]:
    """
    Delete expired jobs (expires_at < NOW()) and vacuum the table to reclaim space.

    Returns {"deleted": n, "vacuumed": bool}.
    """
    with _conn() as conn:
        # Delete expired jobs
        delete_sql = "DELETE FROM jobs WHERE expires_at < NOW()"
        cur = conn.execute(delete_sql)
        deleted_count = cur.rowcount
        conn.commit()

        # Vacuum the table to reclaim space
        conn.set_autocommit(True)
        conn.execute("VACUUM jobs")
        conn.set_autocommit(False)

        return {"deleted": deleted_count, "vacuumed": True}
