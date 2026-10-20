from __future__ import annotations

import html
import json
import re
from datetime import datetime
from urllib.parse import urlsplit

import fitz
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.db.db import (
    apply_schema,
    create_job_recommendation_request,
    get_job_discovery_results,
    list_application_records,
)
from src.services.jobfinder_services import (
    analyze_resume_service,
    application_tracker_service,
    ats_optimization_service,
    career_coach_service,
    discover_jobs_service,
    recommend_service_plan,
    tailor_resume_service,
    interview_prep_service,
)
import time
from uuid import uuid4
from src.storage.s3_storage import upload_resume_bytes
from src.messaging.kafka_bus import (
    TOPIC_COMPANY_RESEARCH_REQUESTED,
    TOPIC_RECOMMENDATION_REQUESTED,
    TOPIC_RESUME_ANALYSIS_REQUESTED,
    TOPIC_JOB_SCRAPE_REQUESTED,
    kafka_enabled,
    publish_json,
)

load_dotenv()

app = FastAPI(title="JobFinder.io", version="3.0.0")

ROLE_OPTIONS = [
    "AI Engineer",
    "Machine Learning Engineer",
    "Backend Engineer",
    "Full Stack Engineer",
    "Software Engineer",
    "Data Engineer",
    "Data Scientist",
    "DevOps Engineer",
    "Site Reliability Engineer",
    "Cloud Engineer",
    "Platform Engineer",
    "Security Engineer",
]

SERVICE_NAMES = [
    "resume_analysis",
    "job_discovery",
    "company_research",
    "application_tracker",
    "resume_tailoring",
    "ats_optimization",
    "interview_prep",
    "career_coach",
    "recommendation_plan",
]

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


@app.on_event("startup")
def _startup() -> None:
    try:
        apply_schema()
    except Exception as exc:
        print(f"Warning: database schema bootstrap skipped: {exc}")


def _is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(email.strip()))


def _parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON must be an object (dictionary).")
    return parsed


def _extract_pdf_text_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            pages.append(text.strip())
        return "\n".join(page for page in pages if page)
    finally:
        doc.close()


def _pretty(payload: dict) -> str:
    return html.escape(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _render_company_research(result: dict) -> str:
    research = (result.get("data") or {}).get("company_research") or {}
    brief = research.get("research_brief") or {}
    if not isinstance(brief, dict):
        brief = {}
    local = research.get("local_hiring_data") or {}

    def esc(value: object) -> str:
        return html.escape(str(value or ""))

    def bullet_section(title: str, values: object) -> str:
        items = values if isinstance(values, list) else []
        if not items:
            return ""
        rendered = "".join(f"<li>{esc(item)}</li>" for item in items if not isinstance(item, dict))
        if not rendered:
            return ""
        return f'<section class="card"><h2>{esc(title)}</h2><ul>{rendered}</ul></section>'

    def interview_section() -> str:
        items = brief.get("interview_priorities") or []
        if not isinstance(items, list) or not items:
            return ""
        cards = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cards.append(
                '<article class="card"><h3>' + esc(item.get("theme")) + '</h3>'
                + f'<p>{esc(item.get("why_it_matters"))}</p>'
                + f'<p><strong>Practice:</strong> {esc(item.get("sample_question"))}</p>'
                + f'<p><strong>Prepare:</strong> {esc(item.get("story_to_prepare"))}</p></article>'
            )
        return f'<section class="card"><h2>Interview preparation</h2><div class="grid">{"".join(cards)}</div></section>' if cards else ""

    def plan_section() -> str:
        items = brief.get("application_plan") or []
        if not isinstance(items, list) or not items:
            return ""
        cards = []
        for item in items:
            if not isinstance(item, dict):
                continue
            actions = item.get("actions") if isinstance(item.get("actions"), list) else []
            actions_html = "".join(f"<li>{esc(action)}</li>" for action in actions)
            cards.append(
                '<article class="card"><h3>' + esc(item.get("phase")) + '</h3>'
                + f'<ul>{actions_html}</ul><p><strong>Deliverable:</strong> {esc(item.get("deliverable"))}</p></article>'
            )
        return f'<section class="card"><h2>Application plan</h2><div class="grid">{"".join(cards)}</div></section>' if cards else ""

    def evidence_section() -> str:
        evidence = research.get("current_evidence") or []
        observations = brief.get("current_observations") or []
        cards = []
        for item in evidence if isinstance(evidence, list) else []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            link = f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(item.get("title"))} ↗</a>' if url.startswith("https://") else esc(item.get("title"))
            cards.append(f'<li><strong>{esc(item.get("id"))}</strong> {link}<br><small>{esc(item.get("publisher"))} · {esc(item.get("published"))}</small></li>')
        observation_html = "".join(f'<li>{esc(item.get("observation"))} <small>({esc(", ".join(item.get("evidence_ids", [])) if isinstance(item.get("evidence_ids"), list) else "")})</small><br>{esc(item.get("candidate_relevance"))}</li>' for item in observations if isinstance(item, dict)) if isinstance(observations, list) else ""
        if not cards and not observation_html:
            return '<section class="card"><h2>Current coverage</h2><p>No dated coverage could be retrieved for this run. Verify current priorities from the official sources below.</p></section>'
        sources = f'<h3>Retrieved coverage (secondary sources)</h3><ul>{"".join(cards)}</ul>' if cards else ""
        notes = f'<h3>What it may mean for applicants</h3><ul>{observation_html}</ul>' if observation_html else ""
        return f'<section class="card"><h2>Recent coverage</h2><p>Coverage search results are leads; confirm time-sensitive claims with official company sources.</p>{sources}{notes}</section>'

    source_links = research.get("research_links") or []
    links_html = "".join(
        f'<a class="pill" href="{esc(item.get("url"))}" target="_blank" rel="noopener noreferrer">{esc(item.get("label"))} ↗</a>'
        for item in source_links if isinstance(item, dict) and str(item.get("url", "")).startswith("https://")
    ) if isinstance(source_links, list) else ""
    local_count = local.get("open_roles_count", 0)
    local_signal = local.get("signal", "unknown")
    warning = (
        '<div class="status">The AI research brief could not be generated. The plan below is a starter checklist; verify company details from official sources.</div>'
        if research.get("research_status") == "limited" else ""
    )

    return f'''
    <div class="card company-report">
      <p><strong>Company Research · {esc(research.get("research_date"))}</strong></p>
      <h2>{esc(research.get("company"))}</h2>
      {f'<p><strong>Target role:</strong> {esc(research.get("target_role"))}</p>' if research.get("target_role") else ''}
      {f'<p><strong>Preferred location:</strong> {esc(research.get("target_location"))}</p>' if research.get("target_location") else ''}
      <p>{esc(brief.get("company_overview") or research.get("research_basis"))}</p>
      <div class="status">JobFinder local data: {esc(local_count)} matching saved roles · hiring signal: {esc(local_signal)}. {esc(local.get("coverage_note"))}</div>
      {warning}
    </div>
    <div class="grid" style="margin-top:18px">
      {bullet_section("Business areas", brief.get("business_areas"))}
      {bullet_section("Strategic context", brief.get("strategic_context"))}
      {evidence_section()}
      {bullet_section("What this means for your target role", brief.get("role_relevance"))}
      {bullet_section("How to position your application", brief.get("candidate_positioning"))}
      {interview_section()}
      {plan_section()}
      {bullet_section("Risks and unknowns", brief.get("risks_and_unknowns"))}
      {bullet_section("Verify before applying", brief.get("facts_to_verify"))}
    </div>
    <div class="card" style="margin-top:18px"><h2>Research links</h2><p>Use these searches to check current details against official company sources.</p><div class="links">{links_html}</div></div>
    '''


def _page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    :root {{
      --bg-1: #07111f;
      --bg-2: #112036;
      --card: rgba(255, 255, 255, 0.08);
      --card-strong: rgba(255, 255, 255, 0.12);
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #22d3ee;
      --accent-2: #fb7185;
      --border: rgba(226, 232, 240, 0.16);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(1200px 500px at 8% -10%, rgba(34,211,238,0.22), transparent 70%),
        radial-gradient(1000px 450px at 95% -5%, rgba(251,113,133,0.18), transparent 65%),
        linear-gradient(135deg, var(--bg-1), var(--bg-2));
      font-family: 'IBM Plex Mono', monospace;
    }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }}
    .hero {{ margin-bottom: 20px; }}
    h1,h2,h3 {{ font-family: 'Space Grotesk', sans-serif; margin: 0 0 10px; }}
    h1 {{ font-size: clamp(2rem, 5vw, 3.4rem); }}
    h2 {{ font-size: 1.2rem; }}
    p {{ color: var(--muted); line-height: 1.6; margin-top: 0; }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 18px;
      backdrop-filter: blur(6px);
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.16);
    }}
    .card strong {{ color: #fff; }}
    label {{ display: block; margin: 0 0 8px; color: #fff; font-size: 0.9rem; }}
    input, select, textarea {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.55);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }}
    textarea {{ min-height: 160px; resize: vertical; }}
    .field {{ margin-bottom: 14px; }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border: 0;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #020617;
      font: 700 0.95rem 'Space Grotesk', sans-serif;
      padding: 12px 18px;
      cursor: pointer;
      text-decoration: none;
    }}
    .status {{
      border-radius: 16px;
      padding: 14px 16px;
      margin: 18px 0;
      border: 1px solid var(--border);
      background: var(--card-strong);
    }}
    .status.ok {{ border-color: rgba(34,211,238,0.35); }}
    .status.error {{ border-color: rgba(251,113,133,0.45); }}
    pre {{
      white-space: pre-wrap;
      overflow-x: auto;
      margin: 0;
      padding: 12px;
      border-radius: 14px;
      background: rgba(2, 6, 23, 0.55);
      color: #dbeafe;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .pill {{ padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.08); color: var(--text); text-decoration: none; }}
    code {{ color: #7dd3fc; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>JobFinder.io</h1>
      <p>FastAPI front end, service-oriented backend, Kafka for async dispatch, PostgreSQL for state, and S3 for resumes/artifacts.</p>
      <div class="links">
        <a class="pill" href="/healthz">Health</a>
        <a class="pill" href="/api/services">Services</a>
        <a class="pill" href="/docs">API Docs</a>
      </div>
    </div>
    {body}
  </div>
</body>
</html>"""


def _render_home(message: str | None = None, error: str | None = None, result: dict | None = None) -> HTMLResponse:
    status_block = ""
    if message:
        status_block = f'<div class="status ok"><strong>Success</strong><div>{html.escape(message)}</div></div>'
    elif error:
        status_block = f'<div class="status error"><strong>Error</strong><div>{html.escape(error)}</div></div>'

    result_block = ""
    if result is not None:
        if isinstance((result.get("data") or {}).get("company_research"), dict):
            result_block = _render_company_research(result)
        else:
            result_block = f'<div class="card"><h2>Result</h2><pre>{_pretty(result)}</pre></div>'

    body = f"""
    {status_block}
    <div class="grid">
      <div class="card">
        <h2>Candidate Intake</h2>
        <p>Upload a PDF resume and queue a recommendation request.</p>
        <form action="/intake" method="post" enctype="multipart/form-data">
          <div class="field"><label>Email</label><input name="email" type="email" placeholder="name@example.com" required /></div>
          <div class="field"><label>Target role</label><select name="role" required>{''.join(f'<option value="{html.escape(role)}">{html.escape(role)}</option>' for role in ROLE_OPTIONS)}</select></div>
          <div class="field"><label>Resume PDF</label><input name="resume" type="file" accept="application/pdf" required /></div>
          <button class="button" type="submit">Queue request</button>
        </form>
      </div>

      <div class="card">
        <h2>Recommendation Plan</h2>
        <p>Send a JSON workflow context to the service layer.</p>
        <form action="/recommendations/run" method="post">
          <div class="field"><label>Context JSON</label><textarea name="context_json">{html.escape(json.dumps({
        "intent": "auto",
        "resume_text": "...",
        "keywords": "software engineer",
        "location": "remote",
        "max_results_per_source": 20,
        "save_to_db": False,
    }, indent=2))}</textarea></div>
          <button class="button" type="submit">Run plan</button>
        </form>
      </div>

      <div class="card">
        <h2>Resume Service</h2>
        <form action="/resume/analyze" method="post" enctype="multipart/form-data">
          <div class="field"><label>Resume PDF</label><input name="resume" type="file" accept="application/pdf" required /></div>
          <button class="button" type="submit">Analyze resume</button>
        </form>
      </div>

      <div class="card">
        <h2>Job Service</h2>
        <form action="/jobs/discover" method="post">
          <div class="field"><label>Keywords</label><input name="keywords" placeholder="software engineer" required /></div>
          <div class="field"><label>Location</label><input name="location" placeholder="City, region, country, or remote" /><small>Only include roles whose listed location matches. Results must have a source posting or update date within the last 30 days.</small></div>
          <button class="button" type="submit">Discover jobs</button>
        </form>
      </div>

      <div class="card">
        <h2 id="company-service">Company Service</h2>
        <form action="/company/research" method="post">
          <div class="field"><label>Company</label><input name="company" placeholder="Google" required /></div>
          <div class="field"><label>Target role</label><input name="target_role" placeholder="Software Engineer" required /></div>
          <div class="field"><label>Preferred location</label><input name="target_location" placeholder="India, Seattle, or remote" /></div>
          <button class="button" type="submit">Research company</button>
        </form>
      </div>

    </div>
    {result_block}
    """
    return HTMLResponse(_page_shell("JobFinder.io", body))


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return _render_home()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/services")
def services() -> dict[str, object]:
    return {"services": SERVICE_NAMES}


@app.post("/api/intake")
async def api_intake(
    email: str = Form(...),
    role: str = Form(...),
    resume: UploadFile = File(...),
) -> dict[str, object]:
    email = email.strip().lower()
    role = role.strip()
    if not email or not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if role not in ROLE_OPTIONS:
        raise HTTPException(status_code=400, detail="Please choose a valid target role.")

    content = await resume.read()
    if not content:
        raise HTTPException(status_code=400, detail="Resume file is empty.")

    try:
        stored_resume_path = upload_resume_bytes(
            original_filename=resume.filename or "resume.pdf",
            content_bytes=content,
            content_type=resume.content_type or "application/pdf",
        )
        request_id = create_job_recommendation_request(
            email=email,
            requested_role=role,
            resume_original_name=resume.filename or "resume.pdf",
            resume_stored_path=stored_resume_path,
        )
        
        if not _dispatch_kafka_event(
            TOPIC_RESUME_ANALYSIS_REQUESTED,
            {
                "request_id": request_id,
                "email": email,
                "resume_stored_path": stored_resume_path,
                "requested_role": role,
                "created_at": time.time(),
            },
            key=str(request_id),
        ):
            update_recommendation_request_status(
                request_id=request_id,
                status="failed",
                notes="Kafka is not configured; this workflow requires a running background worker.",
            )
            raise HTTPException(
                status_code=503,
                detail="Background processing is unavailable because Kafka is not configured.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "queued", "request_id": request_id, "email": email, "role": role, "resume_path": stored_resume_path}


from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from src.db.db import (
    apply_schema,
    create_job_recommendation_request,
    get_recommendation_request_by_id,
    log_pipeline_event,
    update_recommendation_request_status,
)


def _dispatch_kafka_event(topic: str, payload: dict, *, key: str | None = None) -> bool:
    """Publish one event and expose dispatch failures in the request status."""
    if not kafka_enabled():
        return False

    request_id = payload.get("request_id")
    try:
        if not publish_json(topic, payload, key=key):
            raise RuntimeError("Kafka is configured but the event was not published.")
    except Exception as exc:
        if request_id is not None and str(request_id).isdigit():
            update_recommendation_request_status(
                request_id=int(request_id),
                status="failed",
                notes=f"Could not dispatch background task: {exc}",
            )
        if request_id is not None:
            try:
                log_pipeline_event(
                    request_id=request_id,
                    topic=topic,
                    event_type="dispatch_failed",
                    payload={"error": str(exc)},
                )
            except Exception:
                pass
        raise

    if request_id is not None:
        try:
            log_pipeline_event(
                request_id=request_id,
                topic=topic,
                event_type="dispatched",
                payload={"request_id": request_id, "status": "dispatched"},
            )
        except Exception as exc:
            print(f"Warning: could not record Kafka dispatch for request {request_id}: {exc}")
    return True


@app.get("/status/{req_id}", response_class=HTMLResponse)
def status_page(req_id: str) -> HTMLResponse:
    row = None
    if req_id.isdigit():
        row = get_recommendation_request_by_id(int(req_id))

    status_text = row["status"] if row else "queued (processing in background)"
    notes_text = row.get("notes", "") if row else "Task dispatched to Event Hubs / Kafka pipeline."
    analysis = row.get("analysis_result") if row else None
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            analysis = None
    is_company_research = bool(row and row.get("resume_original_name") == "__company_research__")
    is_recommendation_plan = bool(
        row
        and row.get("resume_original_name") == "context.json"
        and row.get("resume_stored_path") == "inline_context"
    )
    if is_company_research and isinstance(analysis, dict):
        analysis_block = _render_company_research(analysis)
    elif isinstance(analysis, dict):
        result_title = "Recommendation Plan" if is_recommendation_plan else "Resume Analysis"
        analysis_block = f'<div class="card" style="margin-top:18px"><h2>{result_title}</h2><p>{html.escape(str(analysis.get("summary") or "Analysis complete."))}</p><pre>{_pretty(analysis.get("data") or analysis)}</pre></div>'
    elif status_text in {"queued", "processing"}:
        if is_company_research:
            pending_text = "Company research is running. Refresh this page for the result."
        elif is_recommendation_plan:
            pending_text = "Your recommendation plan is running. Refresh this page for the result."
        else:
            pending_text = "Your resume analysis will appear here when processing completes."
        analysis_block = f'<div class="status">{pending_text}</div>'
    else:
        analysis_block = ""

    discovery_jobs: list[dict] = []
    if req_id.isdigit():
        try:
            discovery_jobs = get_job_discovery_results(int(req_id), limit=100)
        except Exception as exc:
            print(f"Warning: could not load discovery results for request {req_id}: {exc}")

    job_cards = []
    for job in discovery_jobs:
        title = html.escape(str(job.get("job_title") or "Untitled role"))
        company = html.escape(str(job.get("company") or "Company not listed"))
        location = html.escape(str(job.get("location") or "Location not listed"))
        source = html.escape(str(job.get("source") or "Job board").upper())
        posted_at = job.get("posted_at")
        posted_line = f'<span>Listed/updated {html.escape(posted_at.strftime("%b %d, %Y"))}</span>' if isinstance(posted_at, datetime) else ""
        salary = str(job.get("salary") or "").strip()
        description = str(job.get("processed_summary") or job.get("description") or "").strip()
        description = html.escape(description[:320] + ("…" if len(description) > 320 else ""))
        url = str(job.get("url") or "").strip()
        parsed_url = urlsplit(url)
        apply_link = (
            f'<a class="button" href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">View job ↗</a>'
            if parsed_url.scheme in {"http", "https"} and parsed_url.netloc
            else ""
        )
        salary_line = f'<span>{html.escape(salary)}</span>' if salary else ""
        description_line = f'<p>{description}</p>' if description else ""
        job_cards.append(
            f'<article class="card"><h3>{title}</h3><p><strong>{company}</strong></p>'
            f'<p class="job-meta">{location} · {source} {posted_line} {salary_line}</p>{description_line}{apply_link}</article>'
        )

    discovery_block = (
        '<section class="card" style="margin-top:18px"><h2>Discovered Jobs '
        f'({len(job_cards)})</h2><div class="grid" style="margin-top:14px">{"".join(job_cards)}</div></section>'
        if job_cards
        else (
            '<section class="card" style="margin-top:18px"><h2>Discovered Jobs (0 current)</h2>'
            '<p>No listings met the freshness and location filters. JobFinder only displays jobs with a verified posting or update date within the last 30 days and a matching location. Try a broader role or location.</p></section>'
            if row and row.get("email") == "discovery-request@example.com"
            else ""
        )
    )

    body = f"""
    <div class="card">
      <h2>Task Status</h2>
      <p><strong>Request ID:</strong> <code>{html.escape(req_id)}</code></p>
      <p><strong>Current Status:</strong> <span class="pill">{html.escape(str(status_text))}</span></p>
      {f'<p><strong>Notes:</strong> {html.escape(str(notes_text))}</p>' if notes_text else ''}
      {analysis_block}
      {discovery_block}
      <div style="margin-top: 20px;">
        <a class="button" href="/status/{html.escape(req_id)}">🔄 Refresh Status</a>
        <a class="pill" href="/">← Back to Home</a>
      </div>
    </div>
    """
    return HTMLResponse(_page_shell(f"Status - Request #{req_id}", body))


@app.post("/intake")
async def intake_html(email: str = Form(...), role: str = Form(...), resume: UploadFile = File(...)):
    try:
        payload = await api_intake(email=email, role=role, resume=resume)
        return RedirectResponse(url=f"/status/{payload['request_id']}", status_code=303)
    except HTTPException as exc:
        return _render_home(error=str(exc.detail))


@app.post("/api/resume/analyze")
async def api_resume_analyze(
    resume: UploadFile = File(...),
) -> dict[str, object]:
    content = await resume.read()
    if not content:
        raise HTTPException(status_code=400, detail="Resume file is empty.")

    try:
        stored_resume_path = upload_resume_bytes(
            original_filename=resume.filename or "resume.pdf",
            content_bytes=content,
            content_type=resume.content_type or "application/pdf",
        )
        request_id = create_job_recommendation_request(
            email="analysis-request@example.com",
            requested_role="General Analysis",
            resume_original_name=resume.filename or "resume.pdf",
            resume_stored_path=stored_resume_path,
        )
        payload = {
            "request_id": request_id,
            "email": "analysis-request@example.com",
            "resume_stored_path": stored_resume_path,
            "requested_role": "General Analysis",
            "created_at": time.time(),
        }

        if _dispatch_kafka_event(TOPIC_RESUME_ANALYSIS_REQUESTED, payload, key=str(request_id)):
            return {
                "status": "queued",
                "summary": "Resume analysis task queued asynchronously via Event Hubs.",
                "data": payload,
            }

        try:
            resume_text = _extract_pdf_text_bytes(content)
            result = analyze_resume_service(resume_text=resume_text)
        except Exception as exc:
            update_recommendation_request_status(
                request_id=request_id,
                status="failed",
                notes=f"Resume analysis failed: {exc}",
            )
            raise
        update_recommendation_request_status(
            request_id=request_id,
            status="done",
            notes=result.get("summary", "Resume analysis completed."),
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/resume/analyze")
async def resume_analyze_html(resume: UploadFile = File(...)):
    try:
        result = await api_resume_analyze(resume=resume)
        req_id = result.get("data", {}).get("request_id") or f"anlz_{int(time.time())}"
        return RedirectResponse(url=f"/status/{req_id}", status_code=303)
    except HTTPException as exc:
        return _render_home(error=str(exc.detail))


@app.post("/api/jobs/discover")
async def api_jobs_discover(
    keywords: str = Form(...),
    location: str = Form("remote"),
    source: str | None = Form(None),
    max_results_per_source: int = Form(25),
    save_to_db: bool = Form(True),
) -> dict[str, object]:
    request_id = create_job_recommendation_request(
        email="discovery-request@example.com",
        requested_role=keywords,
        resume_original_name="none",
        resume_stored_path="none",
    )
    payload = {
        "request_id": request_id,
        "requested_role": keywords,
        "location": location or "remote",
        "sources": [source] if source else None,
        "max_results_per_source": max(1, int(max_results_per_source)),
        "save_to_db": bool(save_to_db),
        "is_standalone": True,
        "created_at": time.time(),
    }

    if _dispatch_kafka_event(TOPIC_JOB_SCRAPE_REQUESTED, payload, key=str(request_id)):
        return {
            "status": "queued",
            "message": f"Job discovery task queued asynchronously for '{keywords}'.",
            "data": payload,
        }

    from src.scrapers.orchestrator import HTTP_SOURCES
    sources = [source] if source else HTTP_SOURCES
    try:
        result = discover_jobs_service(
            keywords=keywords,
            location=location or "remote",
            sources=sources,
            max_results_per_source=max(1, int(max_results_per_source)),
            save_to_db=bool(save_to_db),
            request_id=request_id,
        )
    except Exception as exc:
        update_recommendation_request_status(
            request_id=request_id,
            status="failed",
            notes=f"Job discovery failed: {exc}",
        )
        raise
    update_recommendation_request_status(
        request_id=request_id,
        status="done",
        notes=result.get("summary", "Job discovery completed."),
    )
    return result


@app.post("/jobs/discover")
async def jobs_discover_html(
    keywords: str = Form(...),
    location: str = Form("remote"),
    source: str | None = Form(None),
    max_results_per_source: int = Form(25),
    save_to_db: bool = Form(True),
):
    try:
        result = await api_jobs_discover(
            keywords=keywords,
            location=location,
            source=source,
            max_results_per_source=max_results_per_source,
            save_to_db=save_to_db,
        )
        req_id = result.get("data", {}).get("request_id") or f"disc_{int(time.time())}"
        return RedirectResponse(url=f"/status/{req_id}", status_code=303)
    except HTTPException as exc:
        return _render_home(error=str(exc.detail))


@app.post("/api/company/research")
async def api_company_research(
    company: str = Form(...),
    target_role: str | None = Form(None),
    target_location: str | None = Form(None),
    target_job_url: str | None = Form(None),
    source: str | None = Form(None),
    use_llm: bool = Form(True),
) -> dict[str, object]:
    company = company.strip()
    if not company:
        raise HTTPException(status_code=400, detail="Company name is required.")

    request_role = (target_role or "").strip() or "Company research"
    request_id = create_job_recommendation_request(
        email=f"company-research-{uuid4().hex}@jobfinder.invalid",
        requested_role=request_role,
        resume_original_name="__company_research__",
        resume_stored_path="",
    )
    update_recommendation_request_status(
        request_id=request_id,
        status="queued",
        notes=f"Company research for {company} is queued.",
    )
    payload = {
        "request_id": request_id,
        "company": company,
        "target_role": target_role,
        "target_location": target_location,
        "target_job_url": target_job_url,
        "source": source,
        "use_llm": bool(use_llm),
        "created_at": time.time(),
    }
    if not _dispatch_kafka_event(TOPIC_COMPANY_RESEARCH_REQUESTED, payload, key=str(request_id)):
        update_recommendation_request_status(
            request_id=request_id,
            status="failed",
            notes="Company research could not be queued because Kafka is not configured.",
        )
        raise HTTPException(
            status_code=503,
            detail="Company research is unavailable because the background worker is not configured.",
        )

    return {
        "status": "queued",
        "summary": f"Company research for {company} has been queued.",
        "request_id": request_id,
        "status_url": f"/status/{request_id}",
    }


@app.get("/company/research")
def company_research_form() -> RedirectResponse:
    """The research endpoint consumes form data; direct links should open the form."""
    return RedirectResponse(url="/#company-service", status_code=303)


@app.post("/company/research", response_class=HTMLResponse)
async def company_research_html(
    company: str = Form(...),
    target_role: str = Form(...),
    target_location: str | None = Form(None),
    target_job_url: str | None = Form(None),
    source: str | None = Form(None),
    use_llm: bool = Form(True),
) -> HTMLResponse:
    try:
        result = await api_company_research(company=company, target_role=target_role, target_location=target_location, target_job_url=target_job_url, source=source, use_llm=use_llm)
        return RedirectResponse(url=result["status_url"], status_code=303)
    except HTTPException as exc:
        return _render_home(error=str(exc.detail))
    except Exception as exc:
        return _render_home(error=f"Company research failed: {str(exc)[:240]}")


@app.post("/api/applications/track")
async def api_applications_track(payload: dict) -> dict[str, object]:
    return application_tracker_service(payload)


@app.post("/applications/track", response_class=HTMLResponse)
async def applications_track_html(
    email: str = Form(...),
    application_action: str = Form("list"),
    target_job_url: str | None = Form(None),
    application_status: str | None = Form(None),
    application_notes: str | None = Form(None),
    follow_up_due_at: str | None = Form(None),
    application_record_id: int | None = Form(None),
) -> HTMLResponse:
    payload = {
        "email": email,
        "application_action": application_action,
        "target_job_url": target_job_url,
        "application_status": application_status,
        "application_notes": application_notes,
        "follow_up_due_at": follow_up_due_at,
        "application_record_id": application_record_id,
    }
    try:
        result = application_tracker_service(payload)
        return _render_home(message=result["summary"], result=result)
    except Exception as exc:
        return _render_home(error=str(exc))


@app.get("/api/applications")
def api_list_applications(email: str | None = None, status: str | None = None, limit: int = 50) -> dict[str, object]:
    records = list_application_records(email=email, status=status, limit=max(1, int(limit)))
    return {"status": "completed", "data": {"application_records": records}}


@app.post("/api/resume/tailor")
async def api_resume_tailor(payload: dict) -> dict[str, object]:
    structured_resume = payload.get("structured_resume")
    if not isinstance(structured_resume, dict):
        raise HTTPException(status_code=400, detail="structured_resume is required.")
    try:
        return tailor_resume_service(
            structured_resume=structured_resume,
            target_job_url=payload.get("target_job_url"),
            job_keyword=payload.get("job_keyword"),
            company=payload.get("company"),
            job_title=payload.get("job_title"),
            use_llm=bool(payload.get("use_llm", False)),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ats/optimize")
async def api_ats_optimize(payload: dict) -> dict[str, object]:
    structured_resume = payload.get("structured_resume")
    if not isinstance(structured_resume, dict):
        raise HTTPException(status_code=400, detail="structured_resume is required.")
    try:
        return ats_optimization_service(
            structured_resume=structured_resume,
            target_job_url=payload.get("target_job_url"),
            job_keyword=payload.get("job_keyword"),
            company=payload.get("company"),
            job_title=payload.get("job_title"),
            use_llm=bool(payload.get("use_llm", False)),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/interview/prep")
async def api_interview_prep(payload: dict) -> dict[str, object]:
    structured_resume = payload.get("structured_resume")
    if not isinstance(structured_resume, dict):
        raise HTTPException(status_code=400, detail="structured_resume is required.")
    try:
        return interview_prep_service(
            structured_resume=structured_resume,
            target_job_url=payload.get("target_job_url"),
            job_keyword=payload.get("job_keyword"),
            company=payload.get("company"),
            job_title=payload.get("job_title"),
            use_llm=bool(payload.get("use_llm", False)),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/coach")
async def api_career_coach(payload: dict) -> dict[str, object]:
    structured_resume = payload.get("structured_resume")
    if not isinstance(structured_resume, dict):
        raise HTTPException(status_code=400, detail="structured_resume is required.")
    try:
        return career_coach_service(
            structured_resume=structured_resume,
            source=payload.get("source"),
            use_llm=bool(payload.get("use_llm", False)),
            market_sample_limit=int(payload.get("market_sample_limit", 400)),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/recommendations/run")
async def api_recommendations_run(payload: dict) -> dict[str, object]:
    try:
        email = str(payload.get("email") or "plan-user@example.com").strip().lower()
        role = str(payload.get("keywords") or "Software Engineer").strip()
        request_id = create_job_recommendation_request(
            email=email,
            requested_role=role,
            resume_original_name="context.json",
            resume_stored_path="inline_context",
        )
        kafka_payload = {
            "request_id": request_id,
            "email": email,
            "requested_role": role,
            "location": str(payload.get("location") or "remote").strip(),
            "context": payload,
            "is_standalone": True,
            "created_at": time.time(),
        }
        if _dispatch_kafka_event(
            TOPIC_RECOMMENDATION_REQUESTED,
            kafka_payload,
            key=str(request_id),
        ):
            return {
                "status": "queued",
                "summary": "Recommendation plan queued asynchronously via Event Hubs.",
                "data": kafka_payload,
            }
        try:
            result = recommend_service_plan(payload)
        except Exception as exc:
            update_recommendation_request_status(
                request_id=request_id,
                status="failed",
                notes=f"Recommendation plan failed: {exc}",
            )
            raise
        update_recommendation_request_status(
            request_id=request_id,
            status="done",
            notes=f"Recommendation plan {result.get('status', 'completed')}.",
            analysis_result=result,
        )
        result["request_id"] = request_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/recommendations/run")
async def recommendations_run_html(context_json: str = Form(...)):
    try:
        payload = _parse_json_object(context_json)
        result = await api_recommendations_run(payload)
        req_id = result.get("request_id") or result.get("data", {}).get("request_id") or f"plan_{int(time.time())}"
        return RedirectResponse(url=f"/status/{req_id}", status_code=303)
    except (ValueError, json.JSONDecodeError) as exc:
        return _render_home(error=f"Invalid context JSON: {exc}")
    except Exception as exc:
        return _render_home(error=f"Recommendation plan failed: {exc}")


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):  # type: ignore[override]
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def main() -> None:
    uvicorn.run("frontend_app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
