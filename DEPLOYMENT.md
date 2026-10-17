# Deployment Guide

## FastAPI frontend

The frontend now runs as a FastAPI application in [frontend_app.py](frontend_app.py).

### Local development

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Configure environment variables:
   - `DATABASE_URL`
   - `AWS_REGION`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_S3_BUCKET`
   - optional Kafka variables if you want async dispatch
3. Start the service:
   - `uvicorn frontend_app:app --reload --host 0.0.0.0 --port 8000`

### Docker deployment

1. Build the image:
   - `docker build -t jobfinder-fe .`
2. Run the container:
   - `docker run --rm -p 8000:8000 --env-file .env jobfinder-fe`

### Production notes

- Put the FastAPI app behind a reverse proxy such as Nginx, ALB, or an ingress controller.
- Use a process manager or container orchestrator for the FastAPI app, Kafka-backed job worker, and batch processors.
- Keep PostgreSQL as the source of truth for queued request state.
- Configure `KAFKA_BOOTSTRAP_SERVERS` to enable Kafka dispatch for the background pipeline.
- For Azure Event Hubs Kafka compatibility, set `KAFKA_SECURITY_PROTOCOL=SASL_SSL`, `KAFKA_SASL_MECHANISM=PLAIN`, `KAFKA_SASL_USERNAME=$ConnectionString`, and store the connection string in `KAFKA_SASL_PASSWORD`.

### Suggested service split

- `frontend_app.py` for the API/UI surface.
- `process_recommendation_requests.py` for batch recommendation processing.
- `src/services/jobfinder_services.py` for recommendation/business logic.
- `scrape.py` for scraping and job ingestion.
- `src/messaging/kafka_bus.py` and `src/job_processing_queue.py` for Kafka-backed job dispatch.

## Always-on Kafka worker

Vercel only runs the FastAPI request handler; it does not keep the Kafka consumer alive. Run the consumer as a separate, persistent service on a VM or container host. This repository includes `docker-compose.worker.yml` for that service.

1. Check out this repository on the worker host.
2. Create an untracked `.env.worker` file with the worker's database and S3 settings, Kafka connection settings, and any LLM, embedding, scraper, and email settings needed by the stages you enable. Use the same Kafka bootstrap server, SASL mechanism, username, and password configured in Vercel.
3. Start the worker with `docker compose -f docker-compose.worker.yml up -d --build`.
4. Check its state and logs with `docker compose -f docker-compose.worker.yml ps` and `docker compose -f docker-compose.worker.yml logs -f jobfinder-worker`.

The container restarts after process or host restarts. Kafka offsets are committed after each handled event, so a process interruption does not commit later prefetched events. The worker and Vercel must both be able to reach the Kafka broker and PostgreSQL. Keep `.env.worker` on the host only; `.dockerignore` prevents local environment files and credentials from being copied into the image.

The worker image installs the official CPU-only PyTorch wheel because it is intended for CPU hosts and should not download CUDA runtime libraries.
