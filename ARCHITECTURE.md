# JobFinder Architecture

## One-page system diagram

```mermaid
flowchart LR
  %% Inputs
  A1[Resume PDF] --> B1[main.py]
  A2[Scrape CLI / Scheduler] --> B2[scrape.py / scheduler.py]
  A3[Notify CLI / Scheduler] --> B3[notify.py]

  %% Resume pipeline
  subgraph R[Resume Pipeline]
    B1 --> R1[pdf_extractor.py\nPyMuPDF text extraction]
    R1 --> R2[resume_parser.py\nLLM Pass 1: structured JSON]
    R2 --> R3[normalizer.py\nPass 2: canonical skills]
    R3 --> R4[embedder.py\nprofile + skills embeddings]
    R4 --> R5[Artifacts\nstructured_resume.json + .npz]
    R4 --> R6[vector_store.py\nFAISS resume_index]
  end

  %% Scraping pipeline
  subgraph S[Job Ingestion Pipeline]
    B2 --> S1[orchestrator.py\nsource fan-out + URL dedupe]
    S1 --> S2[hn_jobs.py]
    S1 --> S3[greenhouse.py]
    S1 --> S4[linkedin.py]
    S2 --> S5[(PostgreSQL jobs)]
    S3 --> S5
    S4 --> S5
  end

  %% Processing pipeline
  subgraph P[Job Enrichment Pipeline]
    S5 --> P0{Processing mode}
    P0 -->|sync| P1[job_processor.py]
    P0 -->|background| P2[job_processing_queue.py]
    P2 --> P1

    P1 --> P3[job_description_parser.py\nLLM extract skills/seniority/exp]
    P3 --> P4[normalizer.py\ncanonical skills + tech]
    P4 --> P5[embedder.py\njob embedding]
    P5 --> S5
    P5 --> P6[Rebuild FAISS\noutput/job_index]
  end

  %% Matching
  subgraph M[Matching & Ranking Engine]
    M1[match.py] --> M2[load resume artifacts]
    M2 --> M3[matcher.py\nfinal_score + top_applicant + ranking_score]
    S5 --> M3
    M3 --> M4[Top 20 strongest-candidate jobs]
  end

  %% Notifications
  subgraph N[Notification System]
    B3 --> N1[notification_service.py\ncollect strong matches today]
    N1 --> M3
    N1 --> N2{Channel}
    N2 -->|email| N3[notifiers/email_notifier.py\nSMTP digest]
  end

  %% Shared service
  G[llm_client.py\nNVIDIA OpenAI-compatible wrapper]
  G -. used by .-> R2
  G -. used by .-> R3
  G -. used by .-> P3
  G -. optional top-applicant refinement .-> M3

  %% Storage details
  D1[(schema.sql\njobs + job_processing_queue)] --- S5
```

## Runtime flow (concise)

1. **Resume side**: PDF → structured resume JSON → normalized skills → resume embeddings + FAISS resume index.
2. **Jobs side**: scrapers ingest raw jobs into PostgreSQL (dedup by URL).
3. **Enrichment side**: raw job descriptions are parsed, normalized, embedded, and stored back; FAISS job index is rebuilt.
4. **Ranking side**: matcher computes `final_score`, predicts `top_applicant_score`, then ranks with multi-factor `ranking_score` (final score + salary + company reputation + location preference).
5. **Notification side**: daily digest filters today's matches by thresholds (top-applicant + ranking score) and sends top 10 via email.

## Scoring stack

- **Match score (`final_score`)**
  - semantic similarity
  - skill overlap
  - experience match
- **Top applicant prediction (`0-100`)**
  - experience alignment (`required <= resume + 1` boost)
  - skill coverage boost (`>=80%` boost)
  - seniority/location compatibility
  - company hiring aggressiveness signal
  - optional LLM refinement
- **Ranking score (`0-1`)**
  - weighted blend of `final_score`, salary score, company reputation score, and location preference score

## Key boundaries

- **LLM boundary**: all model calls are centralized in `src/llm_client.py`.
- **Persistence boundary**: DB operations live in `src/db/db.py`.
- **Retrieval boundary**: vector indexing/search is isolated in `src/vector_store.py`.
- **Orchestration boundary**: source fan-out and execution policy are in `src/scrapers/orchestrator.py`.
- **Notification boundary**: digest assembly and delivery live in `src/notification_service.py` + `src/notifiers/`.
