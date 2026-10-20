# JobFinder.io

## Frontend Intake (FastAPI)

This project now includes a FastAPI service where users can:

- select up to 5 target roles from a fixed list,
- upload their latest resume (PDF),
- provide an email ID for recommendations.

Submissions are stored in PostgreSQL as queued requests for your scheduled backend pipeline.
When Kafka is configured, the queue also publishes wake-up events so the worker can drain work without tight polling loops.

The FastAPI app exposes:

- a landing page with a candidate intake form,
- a JSON orchestrator endpoint,
- a lightweight orchestrator studio form,
- resume-analysis and agent discovery endpoints.

### Run locally

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Ensure environment is configured:
   - `DATABASE_URL` must be set
   - `AWS_REGION` defaults to `us-east-1` (override if needed)
   - `AWS_ACCESS_KEY_ID` must be set
   - `AWS_SECRET_ACCESS_KEY` must be set
   - `AWS_S3_BUCKET` defaults to `job-finder-resume-s3-bucket` (override if needed)
3. Start the API service:
   - `uvicorn frontend_app:app --reload`

### Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker and production deployment steps.

### Data capture details

- Uploaded resumes are stored in AWS S3 (path saved as `s3://bucket/key`).
- Intake records are written to `job_recommendation_requests` table.
- DB schema is auto-applied on app startup.

### AWS S3 Free Tier quick setup

1. Create an S3 bucket:
   - Keep it private (do not enable public access).
2. Create an IAM user for this app:
   - Generate access key and secret key.
3. Attach minimal permissions (bucket-level):
   - `s3:PutObject`
   - `s3:GetObject`
   - `s3:ListBucket`
4. Add these environment variables:
   - `AWS_REGION`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_S3_BUCKET`
5. Optional for non-AWS S3-compatible endpoints:
   - `AWS_S3_ENDPOINT_URL`

### Daily workflow (DB + Kafka)

The GitHub Action processes queued rows from `job_recommendation_requests` and can use Kafka as the dispatch bus for job-processing wake-ups.
Each queued request uses:

- `email` for notification destination,
- `requested_roles` for role keywords,
- `resume_stored_path` (S3 URI) for resume download.

For each request, the workflow marks status progression in DB:

- `queued` -> `processing` -> `done` (or `failed` with error note).

Kafka is optional and driven by `KAFKA_BOOTSTRAP_SERVERS`.
When set, the job-processing worker publishes and consumes wake-up events on `job-processing-requested` while PostgreSQL remains the durable state store.
For Azure Event Hubs Kafka compatibility, set `KAFKA_SECURITY_PROTOCOL=SASL_SSL`, `KAFKA_SASL_MECHANISM=PLAIN`, `KAFKA_SASL_USERNAME=$ConnectionString`, and store the Event Hubs connection string in `KAFKA_SASL_PASSWORD`.

Required GitHub secrets for request processing:

- `DATABASE_URL`
- `NVIDIA_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `NOTIFY_EMAIL_SMTP_HOST`
- `NOTIFY_EMAIL_SMTP_PORT`
- `NOTIFY_EMAIL_SMTP_SSL`
- `NOTIFY_EMAIL_USE_TLS`
- `NOTIFY_EMAIL_SMTP_USER`
- `NOTIFY_EMAIL_SMTP_PASSWORD`
- `NOTIFY_EMAIL_FROM`

Optional GitHub secrets:

- `AWS_REGION`
- `AWS_S3_BUCKET`
- `AWS_S3_ENDPOINT_URL`
- `KAFKA_BOOTSTRAP_SERVERS`
- `KAFKA_SECURITY_PROTOCOL`
- `KAFKA_CLIENT_ID`

## Microservice Runtime

The app now exposes direct service endpoints instead of an agent/orchestrator layer.

Available services include:

- `resume_analysis`
- `job_discovery`
- `company_research`
- `application_tracker`
- `resume_tailoring`
- `ats_optimization`
- `interview_prep`
- `career_coach`
- `recommendation_plan`

Service calls are implemented in [src/services/jobfinder_services.py](src/services/jobfinder_services.py) and exposed through FastAPI in [frontend_app.py](frontend_app.py).
