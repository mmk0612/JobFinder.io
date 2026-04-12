"""
process_recommendation_requests.py
----------------------------------
Daily workflow runner that processes queued recommendation requests from DB.
For each queued request:
- downloads resume from S3 URI stored in DB,
- runs resume parsing/embedding,
- scrapes + matches per requested role,
- sends notifications to the request email,
- updates request status in DB.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.db.db import (  # noqa: E402
    apply_schema,
    clear_job_processing_queue,
    get_recommendation_requests_by_status,
    update_recommendation_request_status,
)
from src.job_processing_queue import get_queue_status  # noqa: E402
from src.storage.s3_storage import download_s3_uri_to_path, parse_s3_uri  # noqa: E402


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return value or "unknown"


def _run_command(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
) -> None:
    def _tail_text(path: Path, *, max_lines: int = 40) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            return "\n".join(lines[-max_lines:])
        except Exception:
            return ""

    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stdout_path, "w", encoding="utf-8") as out:
            try:
                subprocess.run(cmd, check=True, env=env, stdout=out, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as exc:
                tail = _tail_text(stdout_path)
                message = f"Command failed ({exc.returncode}): {' '.join(cmd)}"
                if tail:
                    message += f"\n--- command output tail ---\n{tail}"
                raise RuntimeError(message) from exc
        return

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed ({exc.returncode}): {' '.join(cmd)}") from exc


def _run_resume_pipeline_with_retries(
    *,
    python_bin: Path,
    local_resume: Path,
    resume_json_path: Path,
) -> None:
    """Run main.py with retries for transient LLM timeout failures."""
    attempts = max(1, int(os.environ.get("RESUME_PIPELINE_MAX_ATTEMPTS", "3") or "3"))
    delay_seconds = max(1, int(os.environ.get("RESUME_PIPELINE_RETRY_DELAY_SECONDS", "10") or "10"))

    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _run_command(
                [
                    str(python_bin),
                    "main.py",
                    "--resume",
                    str(local_resume),
                    "--output",
                    str(resume_json_path),
                ]
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                f"Resume pipeline attempt {attempt}/{attempts} failed; retrying in {delay_seconds}s..."
            )
            time.sleep(delay_seconds)

    assert last_exc is not None
    raise last_exc


def _run_queue_drain() -> None:
    """Drain async job-processing queue and rebuild job index."""
    from src.db.db import requeue_all_processing_jobs
    from src.embedder import DEFAULT_MODEL, EMBEDDING_PROVIDER, FIXED_EMBEDDING_DIM, NVIDIA_EMBEDDING_MODEL
    from src.job_processing_queue import get_queue_status, start_worker, wait_until_idle_with_progress
    from src.job_processor import rebuild_job_index

    model_hint = NVIDIA_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "nvidia" else DEFAULT_MODEL
    print(
        f"Embeddings in queue: provider={EMBEDDING_PROVIDER} "
        f"model_hint={model_hint} dim={FIXED_EMBEDDING_DIM}"
    )

    requeued = requeue_all_processing_jobs()
    if requeued:
        print(f"Queue recovery: requeued stale processing jobs={requeued}")

    start_worker()
    initial = get_queue_status()
    print(
        f"Queue start: queued={initial.queued_jobs} processing={initial.processing_jobs} "
        f"done={initial.done_jobs} failed={initial.failed_jobs}"
    )

    base_timeout_seconds = _base_queue_wait_timeout_seconds()
    timeout_seconds = _adaptive_queue_wait_timeout_seconds(initial)
    if timeout_seconds != base_timeout_seconds:
        print(
            f"Queue wait timeout adjusted to {timeout_seconds}s based on backlog "
            f"(queued={initial.queued_jobs}, processing={initial.processing_jobs})"
        )

    final = wait_until_idle_with_progress(
        timeout_seconds=timeout_seconds,
        poll_seconds=2,
        progress_every_seconds=10,
    )
    print(
        f"Queue done : queued={final.queued_jobs} processing={final.processing_jobs} "
        f"done={final.done_jobs} failed={final.failed_jobs}"
    )

    indexed = rebuild_job_index(index_dir="output/job_index")
    print(f"Job index rebuilt with {indexed} vector(s)")


def _roles_from_row(raw_roles: object, raw_role: object | None = None) -> list[str]:
    roles: list[str] = []

    single = str(raw_role or "").strip()
    if single:
        roles.append(single)

    if isinstance(raw_roles, list):
        values = raw_roles
    elif isinstance(raw_roles, str):
        try:
            parsed = json.loads(raw_roles)
            values = parsed if isinstance(parsed, list) else [raw_roles]
        except Exception:
            values = [raw_roles]
    else:
        values = []

    roles.extend(str(role).strip() for role in values if str(role).strip())
    return list(dict.fromkeys(roles))[:5]


def _base_queue_wait_timeout_seconds() -> int:
    return max(1, int(os.environ.get("QUEUE_WAIT_TIMEOUT_SECONDS", "1800") or "1800"))


def _adaptive_queue_wait_timeout_seconds(queue_status) -> int:
    base_timeout = _base_queue_wait_timeout_seconds()
    queued_bonus = max(0, int(os.environ.get("QUEUE_WAIT_TIMEOUT_PER_QUEUED_JOB_SECONDS", "20") or "20"))
    processing_bonus = max(0, int(os.environ.get("QUEUE_WAIT_TIMEOUT_PER_PROCESSING_JOB_SECONDS", "120") or "120"))
    max_timeout = max(base_timeout, int(os.environ.get("QUEUE_WAIT_TIMEOUT_MAX_SECONDS", "7200") or "7200"))

    adaptive_timeout = base_timeout
    adaptive_timeout += queue_status.queued_jobs * queued_bonus
    adaptive_timeout += queue_status.processing_jobs * processing_bonus
    return min(adaptive_timeout, max_timeout)


def _clear_queue_if_requested() -> None:
    clear_on_start = os.environ.get("QUEUE_CLEAR_ON_START", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not clear_on_start:
        return

    include_done = os.environ.get("QUEUE_CLEAR_INCLUDE_DONE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    deleted = clear_job_processing_queue(include_done=include_done)
    mode = "all rows" if include_done else "queued/processing/failed rows"
    print(f"⚠️  Queue clear requested on startup: deleted {deleted} {mode}.")


def process_requests() -> int:
    apply_schema()
    _clear_queue_if_requested()

    # Check for existing queue backlog
    queue_status = get_queue_status()
    if queue_status.queued_jobs > 0 or queue_status.processing_jobs > 0:
        print(
            f"⚠️  WARNING: Job processing queue has existing backlog:\n"
            f"   queued={queue_status.queued_jobs} processing={queue_status.processing_jobs}\n"
            f"   done={queue_status.done_jobs} failed={queue_status.failed_jobs}\n"
            f"   Consider investigating queue health if backlog persists across runs."
        )

    limit = int(os.environ.get("RECOMMENDATION_REQUEST_BATCH_SIZE", "50") or "50")
    requests = get_recommendation_requests_by_status(status=None, limit=limit)

    if not requests:
        print("No recommendation requests found.")
        return 0

    python_bin = Path(".venv/bin/python")
    if not python_bin.exists():
        raise FileNotFoundError("Missing .venv/bin/python. Ensure workflow created virtual environment.")

    location = os.environ.get("SCRAPE_LOCATION", "remote")
    max_per_source = int(os.environ.get("SCRAPE_MAX_PER_SOURCE", "5") or "5")

    done_count = 0
    failed_count = 0

    for row in requests:
        request_id = int(row["id"])
        email = str(row.get("email") or "").strip().lower()
        resume_s3_uri = str(row.get("resume_stored_path") or "").strip()
        roles = _roles_from_row(row.get("requested_roles"), row.get("requested_role"))

        print(f"\n=== Request {request_id} ===")
        print(f"email={email}")
        print(f"roles={roles}")
        print(f"resume={resume_s3_uri}")

        update_recommendation_request_status(
            request_id=request_id,
            status="processing",
            notes="processing started",
        )

        try:
            if not email:
                raise ValueError("Request missing email")
            if not roles:
                raise ValueError("Request has no requested roles")

            bucket, key = parse_s3_uri(resume_s3_uri)
            ext = Path(key).suffix or ".pdf"

            request_output_dir = Path("output/requests") / f"request_{request_id}"
            request_output_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix=f"req_{request_id}_") as tmpdir:
                local_resume = Path(tmpdir) / f"resume{ext}"
                download_s3_uri_to_path(s3_uri=f"s3://{bucket}/{key}", destination_path=local_resume)

                resume_json_path = request_output_dir / "structured_resume.json"
                _run_resume_pipeline_with_retries(
                    python_bin=python_bin,
                    local_resume=local_resume,
                    resume_json_path=resume_json_path,
                )

                resume_embeddings_path = resume_json_path.with_suffix(".embeddings.npz")

                for role in roles:
                    role_slug = _slugify(role)
                    print(f"Processing role={role}")

                    scrape_env = os.environ.copy()
                    scrape_env["JOB_PROCESSING_BACKGROUND"] = "true"
                    scrape_env["JOB_PROCESSING_AUTOSTART_WORKER"] = "false"
                    _run_command(
                        [
                            str(python_bin),
                            "scrape.py",
                            "--keywords",
                            role,
                            "--location",
                            location,
                            "--max",
                            str(max_per_source),
                        ],
                        env=scrape_env,
                    )

                    _run_queue_drain()

                    _run_command(
                        [
                            str(python_bin),
                            "match.py",
                            "--resume-json",
                            str(resume_json_path),
                            "--resume-embeddings",
                            str(resume_embeddings_path),
                            "--job-keyword",
                            role,
                            "--json",
                        ],
                        stdout_path=request_output_dir / f"latest_match_output_{role_slug}.json",
                    )

                    notify_env = os.environ.copy()
                    notify_env["NOTIFY_EMAIL_TO"] = email
                    _run_command(
                        [
                            str(python_bin),
                            "notify.py",
                            "--once",
                            "--resume-json",
                            str(resume_json_path),
                            "--resume-embeddings",
                            str(resume_embeddings_path),
                            "--job-keyword",
                            role,
                            "--json",
                        ],
                        env=notify_env,
                        stdout_path=request_output_dir / f"latest_notify_output_{role_slug}.json",
                    )

            done_count += 1
            update_recommendation_request_status(
                request_id=request_id,
                status="done",
                notes=f"processed successfully for {len(roles)} role(s)",
            )

        except Exception as exc:
            failed_count += 1
            update_recommendation_request_status(
                request_id=request_id,
                status="failed",
                notes=str(exc),
            )
            print(f"Request {request_id} failed: {exc}")

    print("\n=== Daily request processing summary ===")
    print(f"processed={len(requests)} done={done_count} failed={failed_count}")
    return 0 if failed_count == 0 else 1


def main() -> None:
    raise SystemExit(process_requests())


if __name__ == "__main__":
    main()
