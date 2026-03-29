#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN=".venv/bin/python"
RESUME_PATH="${1:-resume/new_resume_1.pdf}"
LOCATION="${LOCATION:-remote}"
MAX_PER_SOURCE="${MAX_PER_SOURCE:-5}"

KEYWORD_SET=(
  "Full stack engineer"
  "Software Engineer 1"
  "Ai engineer"
  "backend engineer"
)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "❌ Missing virtualenv python at $PYTHON_BIN"
  echo "Create it first, then install requirements."
  exit 1
fi

if [[ ! -f "$RESUME_PATH" ]]; then
  echo "❌ Resume not found: $RESUME_PATH"
  exit 1
fi

echo "▶ 1/4 Resume parse + embeddings"
rm -rf output/resume_index
"$PYTHON_BIN" main.py \
  --resume "$RESUME_PATH" \
  --output output/structured_resume.json

echo "▶ 2/4 Scrape + enrich jobs"
rm -rf output/job_index

run_queue_drain() {
  "$PYTHON_BIN" -c '
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

from src.job_processing_queue import start_worker, wait_until_idle_with_progress, get_queue_status
from src.job_processor import rebuild_job_index
from src.db.db import requeue_all_processing_jobs
from src.embedder import EMBEDDING_PROVIDER, NVIDIA_EMBEDDING_MODEL, DEFAULT_MODEL, FIXED_EMBEDDING_DIM

timeout_seconds = int(os.environ.get("QUEUE_WAIT_TIMEOUT_SECONDS", "1800") or "1800")
model_hint = NVIDIA_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "nvidia" else DEFAULT_MODEL
print(f"Embeddings in queue: provider={EMBEDDING_PROVIDER} model_hint={model_hint} dim={FIXED_EMBEDDING_DIM}")
requeued = requeue_all_processing_jobs()
if requeued:
  print(f"Queue recovery: requeued stale processing jobs={requeued}")
start_worker()
initial = get_queue_status()
print(
  f"Queue start: queued={initial.queued_jobs} processing={initial.processing_jobs} "
  f"done={initial.done_jobs} failed={initial.failed_jobs}"
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
'
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

for keyword in "${KEYWORD_SET[@]}"; do
  keyword_slug="$(slugify "$keyword")"
  echo "▶ 2/4 Scrape + enrich jobs [${keyword}]"
  JOB_PROCESSING_BACKGROUND=true JOB_PROCESSING_AUTOSTART_WORKER=false "$PYTHON_BIN" scrape.py \
    --keywords "$keyword" \
    --location "$LOCATION" \
    --max "$MAX_PER_SOURCE"

  echo "▶ 2b/4 Wait for async queue to drain [${keyword}]"
  run_queue_drain

  echo "▶ 3/4 Match [${keyword}]"
  "$PYTHON_BIN" match.py \
    --resume-json output/structured_resume.json \
    --resume-embeddings output/structured_resume.embeddings.npz \
    --job-keyword "$keyword" \
    --json > "output/latest_match_output_${keyword_slug}.json"

  echo "▶ 4/4 Notify send [${keyword}]"
  "$PYTHON_BIN" notify.py --once --top-k 20 --min-top-applicant 0 --min-ranking-score 0 --job-keyword "$keyword" --json > "output/latest_notify_output_${keyword_slug}.json"
done

echo "✅ Dry run complete"
echo "- Match outputs: output/latest_match_output_<keyword>.json"
echo "- Notify outputs: output/latest_notify_output_<keyword>.json"
echo "- Keywords: ${KEYWORD_SET[*]}"
echo "- Jobs per source: ${MAX_PER_SOURCE}"
echo "- Processing concurrency: ${JOB_PROCESSING_CONCURRENCY:-from .env/code default}"
echo "- LLM request timeout (s): ${LLM_REQUEST_TIMEOUT_SECONDS:-from .env/code default}"
echo "- LLM total timeout (s): ${LLM_TOTAL_TIMEOUT_SECONDS:-from .env/code default}"
echo "- LLM min interval (s): ${LLM_MIN_REQUEST_INTERVAL_SECONDS:-from .env/code default}"
echo "- LLM max retries: ${LLM_MAX_RETRIES:-from .env/code default}"
echo "- Queue wait timeout (s): ${QUEUE_WAIT_TIMEOUT_SECONDS:-1800}"
