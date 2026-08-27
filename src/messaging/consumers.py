"""
consumers.py
------------
Azure Event Hubs / Kafka consumers for the event-driven JobFinder pipeline.
Decouples the resume intake, analysis, scraping, processing, matching, and notification phases.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("jobfinder.consumers")

# Add workspace to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.db.db import (
    apply_schema,
    update_recommendation_request_status,
)
from src.messaging.kafka_bus import load_kafka_config, publish_json
from src.services.jobfinder_services import (
    analyze_resume_service,
    discover_jobs_service,
)
from src.storage.s3_storage import download_s3_uri_to_path, parse_s3_uri


def _get_consumer(topics: list[str]):
    config = load_kafka_config()
    if not config.enabled:
        raise RuntimeError("Kafka/Event Hubs is not enabled (missing KAFKA_BOOTSTRAP_SERVERS).")

    from kafka import KafkaConsumer
    return KafkaConsumer(
        *topics,
        bootstrap_servers=config.bootstrap_servers,
        client_id=f"{config.client_id}-consumer",
        group_id="jobfinder-workers",
        security_protocol=config.security_protocol,
        sasl_mechanism=config.sasl_mechanism,
        sasl_plain_username=config.sasl_username,
        sasl_plain_password=config.sasl_password,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )


# ── Step 1: Resume Analysis ──────────────────────────────────────────────────

def handle_resume_analysis(payload: dict[str, Any]) -> None:
    request_id = payload["request_id"]
    email = payload["email"]
    resume_path = payload["resume_stored_path"]
    role = payload["requested_role"]

    logger.info(f"[%s] Started resume analysis for %s", request_id, email)
    update_recommendation_request_status(
        request_id=request_id,
        status="processing",
        notes="Started resume analysis...",
    )

    try:
        bucket, key = parse_s3_uri(resume_path)
        ext = Path(key).suffix or ".pdf"

        with tempfile.TemporaryDirectory(prefix=f"req_{request_id}_") as tmpdir:
            local_resume = Path(tmpdir) / f"resume{ext}"
            download_s3_uri_to_path(s3_uri=f"s3://{bucket}/{key}", destination_path=local_resume)
            
            # Import fitz locally to extract text
            import fitz
            doc = fitz.open(local_resume)
            pages = []
            for page in doc:
                pages.append(page.get_text("text").strip())
            doc.close()
            resume_text = "\n".join(pages)

            # Analyze resume
            analysis_result = analyze_resume_service(resume_text=resume_text)
            
            # Save structured resume (in real flow this is cached/written to DB or payload)
            payload["structured_resume"] = analysis_result.get("data", {}).get("structured_resume") or {}
            payload["resume_text"] = resume_text

        logger.info(f"[%s] Resume analysis completed. Dispatching job scrape.", request_id)
        publish_json("job-scrape-requested", payload, key=str(request_id))

    except Exception as exc:
        logger.error(f"[%s] Resume analysis failed: %s", request_id, exc)
        update_recommendation_request_status(
            request_id=request_id,
            status="failed",
            notes=f"Resume analysis failed: {exc}",
        )


# ── Step 2: Job Scrape ────────────────────────────────────────────────────────

def handle_job_scrape(payload: dict[str, Any]) -> None:
    request_id = payload["request_id"]
    role = payload["requested_role"]
    location = payload.get("location") or "remote"
    max_results = int(payload.get("max_results_per_source") or 25)

    logger.info(f"[%s] Triggering job scrape for role: %s", request_id, role)
    update_recommendation_request_status(
        request_id=request_id,
        status="processing",
        notes=f"Scraping jobs for role: {role}...",
    )

    try:
        scrape_result = discover_jobs_service(
            keywords=role,
            location=location,
            max_results_per_source=max_results,
            save_to_db=True,
        )
        logger.info(f"[%s] Job scrape completed: %s", request_id, scrape_result["summary"])
        
        # Dispatch to job-processing-requested for embeddings
        publish_json("job-processing-requested", payload, key=str(request_id))

    except Exception as exc:
        logger.error(f"[%s] Job scrape failed: %s", request_id, exc)
        update_recommendation_request_status(
            request_id=request_id,
            status="failed",
            notes=f"Job scrape failed: {exc}",
        )


# ── Step 3: Job Processing (Embeddings) ───────────────────────────────────────

def handle_job_processing(payload: dict[str, Any]) -> None:
    request_id = payload["request_id"]
    logger.info(f"[%s] Job processing (embeddings generation) requested", request_id)
    update_recommendation_request_status(
        request_id=request_id,
        status="processing",
        notes="Generating embeddings for newly scraped jobs...",
    )

    try:
        # Import queue dynamically to start worker and run idle wait
        from src.job_processing_queue import start_worker, wait_until_idle_with_progress
        from src.job_processor import rebuild_job_index

        start_worker()
        wait_until_idle_with_progress(timeout_seconds=900, poll_seconds=2)
        
        indexed = rebuild_job_index(index_dir="output/job_index")
        logger.info(f"[%s] Job index rebuilt with %s vectors", request_id, indexed)

        publish_json("job-matching-requested", payload, key=str(request_id))

    except Exception as exc:
        logger.error(f"[%s] Job processing failed: %s", request_id, exc)
        update_recommendation_request_status(
            request_id=request_id,
            status="failed",
            notes=f"Job processing failed: {exc}",
        )


# ── Step 4: Matching ──────────────────────────────────────────────────────────

def handle_job_matching(payload: dict[str, Any]) -> None:
    request_id = payload["request_id"]
    role = payload["requested_role"]
    logger.info(f"[%s] Executing resume-to-job matching for %s", request_id, role)
    update_recommendation_request_status(
        request_id=request_id,
        status="processing",
        notes="Ranking matched job listings...",
    )

    try:
        # Call match.py pipeline logic directly
        import subprocess
        # Create temp folder for matching artifacts
        out_dir = Path("output/requests") / f"request_{request_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        resume_json = out_dir / "structured_resume.json"
        resume_json.write_text(json.dumps(payload.get("structured_resume") or {}), encoding="utf-8")

        # Generate resume embeddings
        python_bin = sys.executable
        subprocess.run(
            [python_bin, "main.py", "--resume", str(resume_json), "--output", str(resume_json)],
            check=True,
        )

        # Match jobs
        subprocess.run(
            [
                python_bin,
                "match.py",
                "--resume-json",
                str(resume_json),
                "--resume-embeddings",
                str(resume_json.with_suffix(".embeddings.npz")),
                "--job-keyword",
                role,
                "--json",
            ],
            check=True,
        )

        publish_json("recommendation-requested", payload, key=str(request_id))

    except Exception as exc:
        logger.error(f"[%s] Matching failed: %s", request_id, exc)
        update_recommendation_request_status(
            request_id=request_id,
            status="failed",
            notes=f"Job matching failed: {exc}",
        )


# ── Step 5: Recommendations (Notifications) ───────────────────────────────────

def handle_recommendation(payload: dict[str, Any]) -> None:
    request_id = payload["request_id"]
    email = payload["email"]
    role = payload["requested_role"]

    logger.info(f"[%s] Sending recommendation digest to %s", request_id, email)
    update_recommendation_request_status(
        request_id=request_id,
        status="processing",
        notes="Sending email recommendations...",
    )

    try:
        out_dir = Path("output/requests") / f"request_{request_id}"
        resume_json = out_dir / "structured_resume.json"
        resume_embeddings = resume_json.with_suffix(".embeddings.npz")

        # Run notify.py logic
        import subprocess
        env = os.environ.copy()
        env["NOTIFY_EMAIL_TO"] = email
        subprocess.run(
            [
                sys.executable,
                "notify.py",
                "--once",
                "--resume-json",
                str(resume_json),
                "--resume-embeddings",
                str(resume_embeddings),
                "--job-keyword",
                role,
                "--json",
            ],
            env=env,
            check=True,
        )

        logger.info(f"[%s] Recommendation digest sent successfully", request_id)
        update_recommendation_request_status(
            request_id=request_id,
            status="done",
            notes="Processed successfully. Digest sent via email.",
        )

    except Exception as exc:
        logger.error(f"[%s] Notification failed: %s", request_id, exc)
        update_recommendation_request_status(
            request_id=request_id,
            status="failed",
            notes=f"Notification failed: {exc}",
        )


# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def run_consumer(topic: str) -> None:
    logger.info(f"Starting consumer daemon for topic: %s", topic)
    apply_schema()
    consumer = _get_consumer([topic])

    handlers = {
        "resume-analysis-requested": handle_resume_analysis,
        "job-scrape-requested": handle_job_scrape,
        "job-processing-requested": handle_job_processing,
        "job-matching-requested": handle_job_matching,
        "recommendation-requested": handle_recommendation,
    }

    handler = handlers.get(topic)
    if not handler:
        raise ValueError(f"No handler defined for topic: {topic}")

    for message in consumer:
        payload = message.value
        try:
            handler(payload)
        except Exception as exc:
            logger.error("Unexpected error in handler: %s", exc)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python consumers.py <topic-name>")
        sys.exit(1)

    target_topic = sys.argv[1]
    run_consumer(target_topic)
