"""
pdf_extractor.py
----------------
Extract raw text from a resume PDF using PyMuPDF (fitz).
Preserves page order; strips excessive whitespace.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Open a PDF and return all text as a single cleaned string.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A single string containing all text from every page,
        with pages separated by a form-feed character (\\f).

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError:        If the file is not a readable PDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    pages: list[str] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    for page in doc:
        raw = page.get_text("text")          # plain text, layout-ordered
        cleaned = _clean_page_text(raw)
        if cleaned:
            pages.append(cleaned)

    doc.close()

    if not pages:
        raise ValueError("No text found in PDF. The file may be image-only.")

    return "\f".join(pages)   # '\f' = form-feed, standard page separator


# ── helpers ─────────────────────────────────────────────────────────────────

def _clean_page_text(text: str) -> str:
    """Remove junk whitespace while keeping meaningful line breaks."""
    # Collapse runs of spaces/tabs (but not newlines) to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
