-- ─────────────────────────────────────────────────────────────────────────────
-- jobFinder  ·  PostgreSQL schema
-- Apply once:  psql $DATABASE_URL -f src/db/schema.sql
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS jobs (
    id                  BIGSERIAL PRIMARY KEY,

    -- Core job fields (match the scraped JobListing dataclass)
    job_title           TEXT        NOT NULL,
    company             TEXT        NOT NULL,
    location            TEXT        NOT NULL DEFAULT '',
    description         TEXT        NOT NULL DEFAULT '',
    salary              TEXT        NOT NULL DEFAULT '',
    experience_required TEXT        NOT NULL DEFAULT '',
    url                 TEXT        NOT NULL,

    -- Metadata
    source              TEXT        NOT NULL,          -- 'linkedin' | 'indeed' | 'hn' | …
    scraped_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days'),
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,

    -- Processed job-description fields (Step 4)
    processed_skills                JSONB       NOT NULL DEFAULT '[]'::jsonb,
    processed_tech_stack            JSONB       NOT NULL DEFAULT '[]'::jsonb,
    processed_experience_required   INTEGER,
    processed_experience_text       TEXT        NOT NULL DEFAULT '',
    processed_seniority             TEXT        NOT NULL DEFAULT '',
    processed_summary               TEXT        NOT NULL DEFAULT '',
    processed_payload               JSONB       NOT NULL DEFAULT '{}'::jsonb,
    job_embedding                   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    job_embedding_model             TEXT        NOT NULL DEFAULT '',
    job_embedding_text              TEXT        NOT NULL DEFAULT '',
    processed_at                    TIMESTAMPTZ,

    -- Deduplication key: same URL = same job posting
    CONSTRAINT jobs_url_unique UNIQUE (url)
);

-- Migration-safe TTL support for pre-existing databases
ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

ALTER TABLE jobs
    ALTER COLUMN expires_at SET DEFAULT (NOW() + INTERVAL '30 days');

UPDATE jobs
SET    expires_at = scraped_at + INTERVAL '30 days'
WHERE  expires_at IS NULL OR expires_at < scraped_at + INTERVAL '30 days';

ALTER TABLE jobs
    ALTER COLUMN expires_at SET NOT NULL;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS processed_skills JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS processed_tech_stack JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS processed_experience_required INTEGER;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS processed_experience_text TEXT NOT NULL DEFAULT '';

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS processed_seniority TEXT NOT NULL DEFAULT '';

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS processed_summary TEXT NOT NULL DEFAULT '';

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS processed_payload JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS job_embedding JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS job_embedding_model TEXT NOT NULL DEFAULT '';

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS job_embedding_text TEXT NOT NULL DEFAULT '';

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ;

-- Fast lookups by source + recency
CREATE INDEX IF NOT EXISTS idx_jobs_source       ON jobs (source);
CREATE INDEX IF NOT EXISTS idx_jobs_scraped_at   ON jobs (scraped_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_expires_at   ON jobs (expires_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_is_active    ON jobs (is_active);
CREATE INDEX IF NOT EXISTS idx_jobs_processed_at ON jobs (processed_at DESC);

-- Full-text search on title + description (optional but useful later)
CREATE INDEX IF NOT EXISTS idx_jobs_fts ON jobs
    USING gin(to_tsvector('english', job_title || ' ' || description));


-- Durable processing queue (Step 4 background processing)
CREATE TABLE IF NOT EXISTS job_processing_queue (
    id             BIGSERIAL PRIMARY KEY,
    job_url        TEXT        NOT NULL,
    status         TEXT        NOT NULL DEFAULT 'queued', -- queued | processing | done | failed
    attempts       INTEGER     NOT NULL DEFAULT 0,
    max_attempts   INTEGER     NOT NULL DEFAULT 3,
    available_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_at      TIMESTAMPTZ,
    worker_id      TEXT,
    last_error     TEXT        NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_job_processing_queue_job_url
        FOREIGN KEY (job_url)
        REFERENCES jobs(url)
        ON DELETE CASCADE,
    CONSTRAINT chk_job_processing_queue_status
        CHECK (status IN ('queued', 'processing', 'done', 'failed'))
);

-- One active queue record per job URL; enqueue can re-queue this row.
CREATE UNIQUE INDEX IF NOT EXISTS uq_job_processing_queue_job_url
    ON job_processing_queue (job_url);

CREATE INDEX IF NOT EXISTS idx_job_processing_queue_status_available
    ON job_processing_queue (status, available_at);

CREATE INDEX IF NOT EXISTS idx_job_processing_queue_worker
    ON job_processing_queue (worker_id, locked_at DESC);


-- Frontend intake requests (roles + resume + email)
CREATE TABLE IF NOT EXISTS job_recommendation_requests (
    id                     BIGSERIAL PRIMARY KEY,
    email                  TEXT        NOT NULL,
    requested_role         TEXT        NOT NULL,
    resume_original_name   TEXT        NOT NULL,
    resume_stored_path     TEXT        NOT NULL,
    status                 TEXT        NOT NULL DEFAULT 'queued', -- queued | processing | done | failed
    notes                  TEXT        NOT NULL DEFAULT '',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_recommendation_requests_status
        CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    CONSTRAINT chk_recommendation_requests_role_not_empty
        CHECK (requested_role <> '' AND requested_role IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_recommendation_requests_status_created
    ON job_recommendation_requests (status, created_at DESC);

-- Migration-safe normalization for unique (email, role) constraint.
ALTER TABLE job_recommendation_requests
    ADD COLUMN IF NOT EXISTS requested_role TEXT;

UPDATE job_recommendation_requests
SET email = lower(trim(email)),
    requested_role = COALESCE(requested_role, 'default'),
    updated_at = NOW()
WHERE email <> lower(trim(email)) OR requested_role IS NULL OR requested_role = '';

CREATE UNIQUE INDEX IF NOT EXISTS uq_recommendation_requests_email_role
    ON job_recommendation_requests (email, requested_role);
