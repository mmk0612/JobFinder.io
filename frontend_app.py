from __future__ import annotations

import html
import json
import re

import fitz
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.db.db import apply_schema, create_job_recommendation_request
from src.services.jobfinder_services import (
    analyze_resume_service,
    application_tracker_service,
    ats_optimization_service,
    career_coach_service,
    discover_jobs_service,
    recommend_service_plan,
    research_company_service,
    tailor_resume_service,
    interview_prep_service,
)
import time
from src.storage.s3_storage import upload_resume_bytes
from src.messaging.kafka_bus import kafka_enabled, publish_json

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
    apply_schema()


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
          <div class="field"><label>Location</label><input name="location" placeholder="remote" /></div>
          <button class="button" type="submit">Discover jobs</button>
        </form>
      </div>

      <div class="card">
        <h2>Company Service</h2>
        <form action="/company/research" method="post">
          <div class="field"><label>Company</label><input name="company" placeholder="Google" required /></div>
          <button class="button" type="submit">Research company</button>
        </form>
      </div>

      <div class="card">
        <h2>Application Service</h2>
        <form action="/applications/track" method="post">
          <div class="field"><label>Email</label><input name="email" type="email" placeholder="name@example.com" required /></div>
          <div class="field"><label>Action</label><select name="application_action"><option value="list">list</option><option value="track">track</option><option value="update">update</option></select></div>
          <div class="field"><label>Target job URL</label><input name="target_job_url" placeholder="https://..." /></div>
          <div class="field"><label>Status</label><input name="application_status" placeholder="saved" /></div>
          <button class="button" type="submit">Update application</button>
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
        
        # Publish request to the start of the event-driven pipeline
        if kafka_enabled():
            publish_json(
                "resume-analysis-requested",
                {
                    "request_id": request_id,
                    "email": email,
                    "resume_stored_path": stored_resume_path,
                    "requested_role": role,
                    "created_at": time.time(),
                },
                key=str(request_id),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "queued", "request_id": request_id, "email": email, "role": role, "resume_path": stored_resume_path}


from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from src.db.db import (
    apply_schema,
    create_job_recommendation_request,
    get_pipeline_event_logs,
    get_recommendation_request_by_id,
)


@app.get("/status/{req_id}", response_class=HTMLResponse)
def status_page(req_id: str) -> HTMLResponse:
    row = None
    if req_id.isdigit():
        row = get_recommendation_request_by_id(int(req_id))

    logs = get_pipeline_event_logs(req_id)

    status_text = row["status"] if row else "queued (processing in background)"
    notes_text = row.get("notes", "") if row else "Task dispatched to Event Hubs / Kafka pipeline."
    details = {
        "request": row if row else {"request_id": req_id, "message": "Task queued asynchronously via Event Hubs."},
        "pipeline_event_logs": logs,
    }

    body = f"""
    <div class="card">
      <h2>Task Status</h2>
      <p><strong>Request ID:</strong> <code>{html.escape(req_id)}</code></p>
      <p><strong>Current Status:</strong> <span class="pill">{html.escape(str(status_text))}</span></p>
      {f'<p><strong>Notes:</strong> {html.escape(str(notes_text))}</p>' if notes_text else ''}
      <div style="margin-top: 18px;">
        <h3>Event History ({len(logs)} event(s) logged in PostgreSQL)</h3>
        <pre>{_pretty(details)}</pre>
      </div>
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

        if kafka_enabled():
            publish_json("resume-analysis-requested", payload, key=str(request_id))
            return {
                "status": "queued",
                "summary": "Resume analysis task queued asynchronously via Event Hubs.",
                "data": payload,
            }

        resume_text = _extract_pdf_text_bytes(content)
        return analyze_resume_service(resume_text=resume_text)
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
    payload = {
        "request_id": f"disc_{int(time.time())}",
        "requested_role": keywords,
        "location": location or "remote",
        "sources": [source] if source else None,
        "max_results_per_source": max(1, int(max_results_per_source)),
        "save_to_db": bool(save_to_db),
        "created_at": time.time(),
    }

    if kafka_enabled():
        publish_json("job-scrape-requested", payload, key=keywords)
        return {
            "status": "queued",
            "message": f"Job discovery task queued asynchronously for '{keywords}'.",
            "data": payload,
        }

    from src.scrapers.orchestrator import HTTP_SOURCES
    sources = [source] if source else HTTP_SOURCES
    return discover_jobs_service(
        keywords=keywords,
        location=location or "remote",
        sources=sources,
        max_results_per_source=max(1, int(max_results_per_source)),
        save_to_db=bool(save_to_db),
    )


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
    target_job_url: str | None = Form(None),
    source: str | None = Form(None),
    use_llm: bool = Form(False),
) -> dict[str, object]:
    return research_company_service(
        company=company,
        target_job_url=target_job_url,
        source=source,
        use_llm=bool(use_llm),
    )


@app.post("/company/research", response_class=HTMLResponse)
async def company_research_html(
    company: str = Form(...),
    target_job_url: str | None = Form(None),
    source: str | None = Form(None),
    use_llm: bool = Form(False),
) -> HTMLResponse:
    try:
        result = await api_company_research(company=company, target_job_url=target_job_url, source=source, use_llm=use_llm)
        return _render_home(message=result["summary"], result=result)
    except HTTPException as exc:
        return _render_home(error=str(exc.detail))


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
        if kafka_enabled():
            req_id = f"plan_{int(time.time())}"
            kafka_payload = {
                "request_id": req_id,
                "email": str(payload.get("email") or "plan-user@example.com").strip().lower(),
                "requested_role": str(payload.get("keywords") or "Software Engineer").strip(),
                "location": str(payload.get("location") or "remote").strip(),
                "context": payload,
                "created_at": time.time(),
            }
            publish_json("recommendation-requested", kafka_payload, key=req_id)
            return {
                "status": "queued",
                "summary": "Recommendation plan queued asynchronously via Event Hubs.",
                "data": kafka_payload,
            }
        return recommend_service_plan(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/recommendations/run")
async def recommendations_run_html(context_json: str = Form(...)):
    try:
        payload = _parse_json_object(context_json)
        result = await api_recommendations_run(payload)
        req_id = result.get("data", {}).get("request_id") or f"plan_{int(time.time())}"
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
