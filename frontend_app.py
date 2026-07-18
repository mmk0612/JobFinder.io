from __future__ import annotations

import json
import re

import streamlit as st
from dotenv import load_dotenv

from src.db.db import apply_schema, create_job_recommendation_request
from src.agent_orchestrator import run_orchestrator
from src.agents.registry import ALL_AGENT_NAMES
from src.storage.s3_storage import upload_resume_bytes

load_dotenv()

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

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


@st.cache_resource
def _init_storage() -> None:
    """Ensure DB tables are available for intake requests."""
    apply_schema()


def _is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(email.strip()))


def _parse_json_object(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON must be an object (dictionary).")
    return parsed


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

          :root {
            --bg-1: #0f172a;
            --bg-2: #1e293b;
            --card: rgba(255, 255, 255, 0.09);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --accent: #22d3ee;
            --accent-2: #fb7185;
          }

          [data-testid='stAppViewContainer'] {
            background:
              radial-gradient(1200px 500px at 8% -10%, rgba(34,211,238,0.22), transparent 70%),
              radial-gradient(1000px 450px at 95% -5%, rgba(251,113,133,0.18), transparent 65%),
              linear-gradient(135deg, var(--bg-1), var(--bg-2));
          }

          .main .block-container {
            padding-top: 2.2rem;
            max-width: 900px;
          }

          h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
            letter-spacing: 0.3px;
          }

          p, div, label, span {
            font-family: 'IBM Plex Mono', monospace;
            color: var(--text);
          }

          .intake-shell {
            background: var(--card);
            border: 1px solid rgba(226, 232, 240, 0.15);
            border-radius: 20px;
            padding: 1.1rem 1rem 1.2rem 1rem;
            backdrop-filter: blur(4px);
            animation: rise 500ms ease-out;
          }

          @keyframes rise {
            from { transform: translateY(12px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
          }

          .caption-strip {
            color: var(--muted);
            border-left: 3px solid var(--accent);
            padding-left: 0.75rem;
            margin-bottom: 1rem;
          }

          .stButton>button {
            font-family: 'Space Grotesk', sans-serif;
            border-radius: 999px;
            border: 1px solid rgba(226,232,240,0.3);
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            color: #020617;
            font-weight: 700;
            transition: transform 120ms ease;
          }

          .stButton>button:hover {
            transform: translateY(-1px);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard() -> None:
    st.title("JobFinder Candidate Intake")
    st.markdown(
        "<p class='caption-strip'>Select your target role, upload your latest resume, and we will queue your request for the scheduled recommendation pipeline.</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=False):
        st.markdown("<div class='intake-shell'>", unsafe_allow_html=True)

        with st.form("candidate_intake_form", clear_on_submit=True):
            email = st.text_input("Email ID", placeholder="name@example.com")
            role = st.selectbox(
                "Target role for recommendations",
                options=ROLE_OPTIONS,
                help="Select the role you want recommendations for.",
            )
            resume = st.file_uploader(
                "Latest resume (PDF)",
                type=["pdf"],
                accept_multiple_files=False,
            )
            submitted = st.form_submit_button("Queue recommendation request")

        st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return

    email = email.strip()
    if not email or not _is_valid_email(email):
        st.error("Please enter a valid email address.")
        return

    if not role:
        st.error("Please select a role.")
        return

    if resume is None:
        st.error("Please upload your latest resume PDF.")
        return

    try:
        stored_resume_path = upload_resume_bytes(
            original_filename=resume.name,
            content_bytes=bytes(resume.getbuffer()),
            content_type=resume.type or "application/pdf",
        )
        
        # Create request for the selected role
        request_id = create_job_recommendation_request(
            email=email,
            requested_role=role,
            resume_original_name=resume.name,
            resume_stored_path=stored_resume_path,
        )
        
        st.success(
            f"✓ Request queued successfully for {role}. Request ID: {request_id}. "
            "The scheduler will pick this up in the next run and email your recommendations."
        )
    except Exception as exc:
        st.error(f"Could not queue request: {exc}")


def _execute_and_display_result(context: dict) -> None:
    with st.spinner("Running orchestrator..."):
        try:
            result = run_orchestrator(context)
        except Exception as exc:
            st.error(f"Orchestrator run failed: {exc}")
            return

    st.success(f"Run completed with status: {result.get('status', 'unknown')}")
    st.subheader("Execution Summary")
    st.json(
        {
            "execution_plan": result.get("execution_plan", []),
            "available_agents": result.get("available_agents", []),
            "next_actions": result.get("next_actions", []),
        }
    )

    run_info = result.get("run", {})
    steps = run_info.get("steps", [])
    if steps:
        st.subheader("Workflow Steps")
        st.dataframe(steps, use_container_width=True)

    outputs = result.get("outputs", {})
    if outputs:
        st.subheader("Agent Outputs")
        for agent_name, payload in outputs.items():
            with st.expander(agent_name, expanded=False):
                st.json(payload)


def _render_workflow_monitor() -> None:
    st.header("Orchestrator Agent Studio")
    st.caption("Run orchestrated agent plans with live context and inspect outputs.")

    intents = [
        "auto",
        "bootstrap",
        "full_assistant",
        "analyze_resume",
        "discover_jobs",
        "tailor_resume",
        "optimize_ats",
        "research_company",
        "track_application",
        "prepare_interview",
        "career_coaching",
    ]
    application_actions = ["list", "track", "update"]
    application_statuses = ["saved", "applied", "interviewing", "offered", "rejected", "archived"]
    sources = ["", "hn", "greenhouse", "linkedin"]

    with st.form("orchestrator_run_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            intent = st.selectbox("Intent", options=intents, index=0)
            fail_fast = st.checkbox("Fail fast", value=True)
            use_llm = st.checkbox("Enable LLM augmentation where supported", value=False)
        with col2:
            source = st.selectbox("Source filter (optional)", options=sources, index=0)
            max_results_per_source = st.number_input(
                "Max results per source",
                min_value=1,
                max_value=200,
                value=25,
                step=1,
            )
            save_to_db = st.checkbox("Save scraped jobs to DB", value=True)

        custom_plan = st.multiselect(
            "Custom execution plan (optional, overrides intent routing)",
            options=ALL_AGENT_NAMES,
            default=[],
        )
        keywords = st.text_input("Keywords (for job_collection)", placeholder="software engineer")
        target_roles_raw = st.text_input(
            "Target roles (comma-separated, optional)",
            placeholder="Backend Engineer, AI Engineer",
        )
        job_keyword = st.text_input("Job keyword (optional target job resolver)")
        target_job_url = st.text_input("Target job URL (optional)")
        company = st.text_input("Company (optional)")
        job_title = st.text_input("Job title (optional)")
        email = st.text_input("Email (optional, required for application tracking)")

        st.markdown("**Application tracker controls**")
        app_col1, app_col2, app_col3 = st.columns(3)
        with app_col1:
            application_action = st.selectbox("application_action", options=application_actions, index=0)
        with app_col2:
            application_status = st.selectbox("application_status", options=application_statuses, index=0)
        with app_col3:
            application_record_id_raw = st.text_input("application_record_id (for update)", placeholder="123")
        follow_up_due_at = st.text_input("follow_up_due_at (ISO, optional)", placeholder="2026-07-25T10:00:00")
        application_notes = st.text_area("application_notes (optional)", height=80)

        st.markdown("**Resume input**")
        resume_text = st.text_area("resume_text (optional)", height=140)
        structured_resume_raw = st.text_area(
            "structured_resume JSON (optional)",
            placeholder='{"skills":["python","sql"],"experience":[],"projects":[]}',
            height=160,
        )

        submitted = st.form_submit_button("Run orchestrator")

    if not submitted:
        return

    try:
        structured_resume = _parse_json_object(structured_resume_raw)
    except Exception as exc:
        st.error(f"Invalid structured_resume JSON: {exc}")
        return

    target_roles = [role.strip() for role in target_roles_raw.split(",") if role.strip()]
    context: dict = {
        "intent": intent,
        "fail_fast": fail_fast,
        "use_llm": use_llm,
        "max_results_per_source": int(max_results_per_source),
        "save_to_db": bool(save_to_db),
        "application_action": application_action,
        "application_status": application_status,
    }
    if custom_plan:
        context["execution_plan"] = custom_plan
    if source:
        context["source"] = source
    if keywords.strip():
        context["keywords"] = keywords.strip()
    if target_roles:
        context["target_roles"] = target_roles
    if job_keyword.strip():
        context["job_keyword"] = job_keyword.strip()
    if target_job_url.strip():
        context["target_job_url"] = target_job_url.strip()
    if company.strip():
        context["company"] = company.strip()
    if job_title.strip():
        context["job_title"] = job_title.strip()
    if email.strip():
        context["email"] = email.strip().lower()
    if follow_up_due_at.strip():
        context["follow_up_due_at"] = follow_up_due_at.strip()
    if application_notes.strip():
        context["application_notes"] = application_notes.strip()
    if application_record_id_raw.strip():
        try:
            context["application_record_id"] = int(application_record_id_raw.strip())
        except ValueError:
            st.error("application_record_id must be an integer.")
            return
    if resume_text.strip():
        context["resume_text"] = resume_text.strip()
    if structured_resume is not None:
        context["structured_resume"] = structured_resume

    _execute_and_display_result(context)


def _render_job_discovery() -> None:
    st.title("Job Discovery")
    st.markdown("Discover personalized job recommendations and search across platforms.")
    
    sources = ["", "hn", "greenhouse", "linkedin"]
    
    with st.form("job_discovery_form"):
        keywords = st.text_input("Keywords", placeholder="software engineer")
        target_roles = st.text_input("Target Roles (comma-separated)", placeholder="Backend Engineer, AI Engineer")
        source = st.selectbox("Source filter (optional)", options=sources, index=0)
        max_results = st.number_input("Max results per source", min_value=1, max_value=200, value=25)
        save_to_db = st.checkbox("Save scraped jobs to DB", value=True)
        
        submitted = st.form_submit_button("Discover Jobs")
        
    if submitted:
        roles_list = [role.strip() for role in target_roles.split(",") if role.strip()]
        context = {
            "intent": "discover_jobs",
            "keywords": keywords.strip(),
            "target_roles": roles_list,
            "source": source,
            "max_results_per_source": int(max_results),
            "save_to_db": bool(save_to_db)
        }
        _execute_and_display_result(context)

def _render_company_research() -> None:
    st.title("Company Research")
    st.markdown("Deep dive into company culture, financials, and recent news.")
    
    with st.form("company_research_form"):
        company = st.text_input("Company Name", placeholder="Google")
        submitted = st.form_submit_button("Research Company")
        
    if submitted:
        context = {
            "intent": "research_company",
            "company": company.strip()
        }
        _execute_and_display_result(context)

def _render_applications() -> None:
    st.title("Applications")
    st.markdown("Track and manage your job applications pipeline.")
    
    actions = ["list", "track", "update"]
    statuses = ["saved", "applied", "interviewing", "offered", "rejected", "archived"]
    
    with st.form("applications_form"):
        action = st.selectbox("Action", options=actions)
        email = st.text_input("Email", placeholder="name@example.com")
        
        col1, col2 = st.columns(2)
        with col1:
            status = st.selectbox("Status", options=statuses)
            record_id = st.text_input("Record ID (for update)", placeholder="123")
        with col2:
            target_job_url = st.text_input("Target Job URL")
            follow_up = st.text_input("Follow-up Date (ISO)", placeholder="2026-07-25T10:00:00")
            
        notes = st.text_area("Notes")
        
        submitted = st.form_submit_button("Submit Application Action")
        
    if submitted:
        context = {
            "intent": "track_application",
            "application_action": action,
            "email": email.strip(),
            "application_status": status,
            "target_job_url": target_job_url.strip(),
            "follow_up_due_at": follow_up.strip(),
            "application_notes": notes.strip()
        }
        if record_id.strip():
            try:
                context["application_record_id"] = int(record_id.strip())
            except ValueError:
                st.error("Record ID must be an integer.")
                return
        _execute_and_display_result(context)

def _render_resume_analyzer() -> None:
    st.title("Resume Analyzer")
    st.markdown("Upload your resume text for AI-powered feedback and scoring.")
    
    with st.form("resume_analyzer_form"):
        resume_text = st.text_area("Resume Text", height=200)
        submitted = st.form_submit_button("Analyze Resume")
        
    if submitted:
        context = {
            "intent": "analyze_resume",
            "resume_text": resume_text.strip()
        }
        _execute_and_display_result(context)

def _render_resume_tailor() -> None:
    st.title("Resume Tailor")
    st.markdown("Tailor your resume to specific job descriptions.")
    
    with st.form("resume_tailor_form"):
        resume_text = st.text_area("Base Resume Text", height=200)
        col1, col2 = st.columns(2)
        with col1:
            target_job_url = st.text_input("Target Job URL")
        with col2:
            job_title = st.text_input("Job Title")
            
        submitted = st.form_submit_button("Tailor Resume")
        
    if submitted:
        context = {
            "intent": "tailor_resume",
            "resume_text": resume_text.strip(),
            "target_job_url": target_job_url.strip(),
            "job_title": job_title.strip()
        }
        _execute_and_display_result(context)

def _render_ats_optimizer() -> None:
    st.title("ATS Optimizer")
    st.markdown("Ensure your resume passes ATS keyword and formatting checks.")
    
    with st.form("ats_optimizer_form"):
        resume_text = st.text_area("Resume Text", height=200)
        job_keyword = st.text_input("Job Keywords")
        submitted = st.form_submit_button("Optimize for ATS")
        
    if submitted:
        context = {
            "intent": "optimize_ats",
            "resume_text": resume_text.strip(),
            "job_keyword": job_keyword.strip()
        }
        _execute_and_display_result(context)

def _render_interview_prep() -> None:
    st.title("Interview Prep")
    st.markdown("Practice technical and behavioral questions with an AI interviewer.")
    
    with st.form("interview_prep_form"):
        target_roles = st.text_input("Target Roles (comma-separated)")
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Company")
        with col2:
            job_title = st.text_input("Job Title")
            
        submitted = st.form_submit_button("Generate Interview Prep")
        
    if submitted:
        roles_list = [role.strip() for role in target_roles.split(",") if role.strip()]
        context = {
            "intent": "prepare_interview",
            "target_roles": roles_list,
            "company": company.strip(),
            "job_title": job_title.strip()
        }
        _execute_and_display_result(context)

def _render_career_coach() -> None:
    st.title("Career Coach")
    st.markdown("Get personalized career advice and strategic planning.")
    
    with st.form("career_coach_form"):
        resume_text = st.text_area("Resume Text", height=200)
        target_roles = st.text_input("Target Roles (comma-separated)")
        submitted = st.form_submit_button("Get Coaching Advice")
        
    if submitted:
        roles_list = [role.strip() for role in target_roles.split(",") if role.strip()]
        context = {
            "intent": "career_coaching",
            "resume_text": resume_text.strip(),
            "target_roles": roles_list
        }
        _execute_and_display_result(context)

def main() -> None:
    st.set_page_config(
        page_title="JobFinder.io",
        page_icon="🎯",
        layout="wide",
    )
    _init_storage()
    _inject_styles()

    st.sidebar.title("Navigation")
    
    pages = {
        "🏠 Dashboard": ["Overview"],
        "🎯 Job Search": ["Job Discovery", "Company Research", "Applications"],
        "📄 Resume Lab": ["Resume Analyzer", "Resume Tailor", "ATS Optimizer"],
        "🚀 Interview Center": ["Interview Prep", "Career Coach"],
        "⚙️ Workflow Monitor": ["Agent Studio"]
    }

    section = st.sidebar.selectbox("Section", list(pages.keys()))
    page = st.sidebar.radio("Page", pages[section])

    if section == "🏠 Dashboard" and page == "Overview":
        _render_dashboard()
    elif section == "🎯 Job Search":
        if page == "Job Discovery":
            _render_job_discovery()
        elif page == "Company Research":
            _render_company_research()
        elif page == "Applications":
            _render_applications()
    elif section == "📄 Resume Lab":
        if page == "Resume Analyzer":
            _render_resume_analyzer()
        elif page == "Resume Tailor":
            _render_resume_tailor()
        elif page == "ATS Optimizer":
            _render_ats_optimizer()
    elif section == "🚀 Interview Center":
        if page == "Interview Prep":
            _render_interview_prep()
        elif page == "Career Coach":
            _render_career_coach()
    elif section == "⚙️ Workflow Monitor" and page == "Agent Studio":
        _render_workflow_monitor()


if __name__ == "__main__":
    main()
