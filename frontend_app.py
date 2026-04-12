from __future__ import annotations

import re

import streamlit as st
from dotenv import load_dotenv

from src.db.db import apply_schema, create_job_recommendation_request
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


def main() -> None:
    st.set_page_config(
        page_title="JobFinder Intake",
        page_icon="JF",
        layout="centered",
    )
    _init_storage()
    _inject_styles()

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


if __name__ == "__main__":
    main()
