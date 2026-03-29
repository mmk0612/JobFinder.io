"""
src/job_processing_queue.py
---------------------------
Durable fire-and-forget queue for job-description processing.

Queue state is persisted in PostgreSQL (`job_processing_queue`) so enqueued
jobs survive process restarts.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
import threading
from dataclasses import dataclass
from uuid import uuid4

from src.db.db import (
    dequeue_jobs_for_processing,
    enqueue_jobs_for_processing,
    get_job_by_url,
    get_job_processing_queue_counts,
    mark_job_processing_done,
    mark_job_processing_failed,
    requeue_processing_jobs_for_worker,
    requeue_stale_processing_jobs,
)
from src.job_processor import process_job_listings
from src.scrapers.models import JobListing

logger = logging.getLogger(__name__)

QUEUE_POLL_INTERVAL_SECONDS = max(
    1,
    int(os.environ.get("JOB_PROCESSING_QUEUE_POLL_SECONDS", "5") or "5"),
)
QUEUE_BATCH_SIZE = max(
    1,
    int(os.environ.get("JOB_PROCESSING_QUEUE_BATCH_SIZE", "10") or "10"),
)
QUEUE_RETRY_DELAY_SECONDS = max(
    1,
    int(os.environ.get("JOB_PROCESSING_RETRY_DELAY_SECONDS", "60") or "60"),
)
QUEUE_MAX_ATTEMPTS = max(
    1,
    int(os.environ.get("JOB_PROCESSING_MAX_ATTEMPTS", "3") or "3"),
)
QUEUE_PROCESSING_CONCURRENCY = max(
    1,
    int(os.environ.get("JOB_PROCESSING_CONCURRENCY", "4") or "4"),
)
QUEUE_CLAIM_LIMIT = max(1, min(QUEUE_BATCH_SIZE, QUEUE_PROCESSING_CONCURRENCY))
QUEUE_STALE_LOCK_SECONDS = max(
    10,
    int(os.environ.get("JOB_PROCESSING_STALE_LOCK_SECONDS", "900") or "900"),
)


@dataclass(frozen=True)
class QueueStatus:
    started: bool
    queued_jobs: int
    processing_jobs: int
    done_jobs: int
    failed_jobs: int
    total_jobs: int


_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()
_worker_id = f"worker-{uuid4().hex[:8]}"


def enqueue_job_processing(jobs: list[JobListing]) -> int:
    """
    Enqueue a batch of jobs for durable background processing.

    Returns number of queue rows created/updated.
    """
    if not jobs:
        return 0

    urls = [job.url for job in jobs if job.url]
    enqueued = enqueue_jobs_for_processing(urls, max_attempts=QUEUE_MAX_ATTEMPTS)
    autostart_worker = os.environ.get("JOB_PROCESSING_AUTOSTART_WORKER", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if autostart_worker:
        _ensure_worker_started()
    return enqueued


def get_queue_status() -> QueueStatus:
    counts = get_job_processing_queue_counts()
    return QueueStatus(
        started=_worker_thread is not None and _worker_thread.is_alive(),
        queued_jobs=counts["queued"],
        processing_jobs=counts["processing"],
        done_jobs=counts["done"],
        failed_jobs=counts["failed"],
        total_jobs=counts["total"],
    )


def start_worker() -> None:
    """Start the durable queue worker thread if it is not already running."""
    _ensure_worker_started()


def wait_until_idle(*, timeout_seconds: int = 1800, poll_seconds: int = 2) -> QueueStatus:
    """
    Block until queue has no queued/processing jobs or timeout is reached.

    Returns the last observed queue status.
    Raises TimeoutError when queue does not drain in time.
    """
    timeout_seconds = max(1, int(timeout_seconds))
    poll_seconds = max(1, int(poll_seconds))
    deadline = time.time() + timeout_seconds

    while True:
        status = get_queue_status()
        if status.queued_jobs == 0 and status.processing_jobs == 0:
            return status
        if time.time() >= deadline:
            raise TimeoutError(
                f"Queue did not drain within {timeout_seconds}s: "
                f"queued={status.queued_jobs}, processing={status.processing_jobs}, "
                f"failed={status.failed_jobs}, done={status.done_jobs}"
            )
        time.sleep(poll_seconds)


def wait_until_idle_with_progress(
    *,
    timeout_seconds: int = 1800,
    poll_seconds: int = 2,
    progress_every_seconds: int = 10,
) -> QueueStatus:
    """
    Wait for queue to drain and print periodic progress updates.

    Prints one status line every `progress_every_seconds` seconds.
    Raises TimeoutError when queue does not drain in time.
    """
    timeout_seconds = max(1, int(timeout_seconds))
    poll_seconds = max(1, int(poll_seconds))
    progress_every_seconds = max(1, int(progress_every_seconds))
    deadline = time.time() + timeout_seconds
    last_print = 0.0

    while True:
        status = get_queue_status()
        now = time.time()
        if now - last_print >= progress_every_seconds:
            print(
                "Queue progress: "
                f"queued={status.queued_jobs} "
                f"processing={status.processing_jobs} "
                f"done={status.done_jobs} "
                f"failed={status.failed_jobs}"
            )
            last_print = now

        if status.queued_jobs == 0 and status.processing_jobs == 0:
            return status

        if now >= deadline:
            raise TimeoutError(
                f"Queue did not drain within {timeout_seconds}s: "
                f"queued={status.queued_jobs}, processing={status.processing_jobs}, "
                f"failed={status.failed_jobs}, done={status.done_jobs}"
            )

        time.sleep(poll_seconds)


def stop_worker() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None:
            return
        _stop_event.set()
        _worker_thread.join(timeout=3)
        reclaimed = requeue_processing_jobs_for_worker(worker_id=_worker_id)
        if reclaimed:
            logger.info("Requeued %s processing job(s) locked by worker %s during stop", reclaimed, _worker_id)
        _worker_thread = None


def _ensure_worker_started() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        reclaimed = requeue_stale_processing_jobs(stale_seconds=QUEUE_STALE_LOCK_SECONDS)
        if reclaimed:
            logger.info("Requeued %s stale processing job(s) before starting worker", reclaimed)
        _stop_event.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            name="job-processing-worker",
            daemon=True,
        )
        _worker_thread.start()


def _worker_loop() -> None:
    while not _stop_event.is_set():
        requeue_stale_processing_jobs(stale_seconds=QUEUE_STALE_LOCK_SECONDS)
        claimed = dequeue_jobs_for_processing(worker_id=_worker_id, limit=QUEUE_CLAIM_LIMIT)
        if not claimed:
            time.sleep(QUEUE_POLL_INTERVAL_SECONDS)
            continue

        jobs_to_process: list[JobListing] = []
        queue_id_by_url: dict[str, int] = {}

        for item in claimed:
            queue_id = int(item["id"])
            url = item["job_url"]
            row = get_job_by_url(url)
            if not row:
                mark_job_processing_failed(
                    queue_id,
                    error_message=f"Job not found for URL: {url}",
                    retry_delay_seconds=QUEUE_RETRY_DELAY_SECONDS,
                )
                continue

            queue_id_by_url[url] = queue_id
            jobs_to_process.append(
                JobListing(
                    job_title=row["job_title"],
                    company=row["company"],
                    url=row["url"],
                    source=row["source"],
                    location=row.get("location") or "",
                    description=row.get("description") or "",
                    salary=row.get("salary") or "",
                    experience_required=row.get("experience_required") or "",
                )
            )

        if not jobs_to_process:
            continue

        try:
            result = process_job_listings(jobs_to_process, rebuild_index=False)
            errors = result.get("errors", {})

            for job in jobs_to_process:
                queue_id = queue_id_by_url.get(job.url)
                if queue_id is None:
                    continue
                if job.url in errors:
                    mark_job_processing_failed(
                        queue_id,
                        error_message=str(errors[job.url]),
                        retry_delay_seconds=QUEUE_RETRY_DELAY_SECONDS,
                    )
                else:
                    mark_job_processing_done(queue_id)

            logger.info(
                "Durable queue batch complete: processed=%s errors=%s",
                result.get("processed", 0),
                len(errors),
            )
        except Exception as exc:
            logger.error("Durable queue worker failed for claimed batch: %s", exc)
            for job in jobs_to_process:
                queue_id = queue_id_by_url.get(job.url)
                if queue_id is None:
                    continue
                mark_job_processing_failed(
                    queue_id,
                    error_message=str(exc),
                    retry_delay_seconds=QUEUE_RETRY_DELAY_SECONDS,
                )


atexit.register(stop_worker)
