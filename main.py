"""
main.py
-------
Entry point for the Job-Agent resume pipeline.

Flow:
  resume.pdf
      ↓  PyMuPDF
  resume_text
      ↓  Gemini (Pass 1)
  structured sections
      ↓  Gemini (Pass 2)
  normalized JSON
      ↓  SentenceTransformers
  embeddings (skills + profile)
      ↓  FAISS
  output/resume_index/

Usage:
  python main.py --resume resume/my_resume.pdf
  python main.py --resume resume/my_resume.pdf --output output/result.json
  python main.py --resume resume/my_resume.pdf --no-normalize
  python main.py --resume resume/my_resume.pdf --no-embed
  python main.py --resume resume/my_resume.pdf --model mpnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file before importing anything that reads env vars
load_dotenv()

from src.pdf_extractor  import extract_text_from_pdf   # noqa: E402
from src.resume_parser  import extract_sections        # noqa: E402
from src.normalizer     import normalize_skills        # noqa: E402
from src.embedder       import generate_embeddings, AVAILABLE_MODELS, EMBEDDING_PROVIDER  # noqa: E402
from src.vector_store   import ResumeVectorStore, dim_for_model       # noqa: E402


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    normalize: bool = True,
    embed: bool = True,
    model_key: str = "bge-large",
    index_dir: str | Path = "output/resume_index",
    verbose: bool = True,
) -> dict:
    """
    Full resume → structured JSON + embedding pipeline.

    Returns the final structured resume dict.
    """
    _log(f"📄  Reading PDF: {pdf_path}", verbose)
    resume_text = extract_text_from_pdf(pdf_path)
    _log(f"    Extracted {len(resume_text):,} characters across "
         f"{resume_text.count(chr(12)) + 1} page(s).", verbose)

    _log("\n🤖  Pass 1 — Extracting sections …", verbose)
    structured = extract_sections(resume_text)
    _log(f"    Skills found   : {len(structured.get('skills', []))}", verbose)
    _log(f"    Experience items: {len(structured.get('experience', []))}", verbose)
    _log(f"    Projects found : {len(structured.get('projects', []))}", verbose)
    _log(f"    Education items: {len(structured.get('education', []))}", verbose)

    if normalize:
        _log("\n🔧  Pass 2 — Normalizing skills …", verbose)
        structured = normalize_skills(structured)
        mapping = structured.get("_skill_normalization_map", {})
        changed = {k: v for k, v in mapping.items() if k.lower().strip() != v}
        if changed:
            _log(f"    Normalized {len(changed)} skill(s):", verbose)
            for raw, norm in changed.items():
                _log(f"      {raw!r:25s} → {norm!r}", verbose)
        else:
            _log("    All skills already canonical.", verbose)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    _log(f"\n✅  Structured JSON saved to: {output_path}", verbose)

    if embed:
        _log(
            f"\n🧠  Generating embeddings (provider: {EMBEDDING_PROVIDER}, requested_model: {model_key}) …",
            verbose,
        )
        embeddings = generate_embeddings(structured, model_key=model_key)

        model_full_name = embeddings["model"]
        dim = dim_for_model(model_full_name)
        _log(f"    Model       : {model_full_name}", verbose)
        _log(f"    Dimension   : {dim}", verbose)
        _log(f"    Skills text : {embeddings['skills_text'][:80]}…", verbose)

        # Save raw embedding vectors alongside the JSON
        emb_path = output_path.with_suffix(".embeddings.npz")
        import numpy as np
        np.savez(
            emb_path,
            skills_embedding=embeddings["skills_embedding"],
            profile_embedding=embeddings["profile_embedding"],
        )
        _log(f"    Embeddings  : {emb_path}", verbose)

        # Store in FAISS index
        contact  = structured.get("contact", {})
        store    = ResumeVectorStore.load_or_create(index_dir, dim=dim)
        store.add(
            vector=embeddings["profile_embedding"],
            metadata={
                "type":       "resume",
                "name":       contact.get("name", ""),
                "email":      contact.get("email", ""),
                "source":     str(pdf_path),
                "skills":     structured.get("skills", []),
                "model":      model_full_name,
                "skills_text":  embeddings["skills_text"],
                "profile_text": embeddings["profile_text"],
            },
        )
        store.save(index_dir)
        _log(f"    FAISS index : {index_dir}  ({len(store)} vector(s) total)", verbose)

    return structured


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jobfinder",
        description="Parse a resume PDF into structured JSON using Gemini.",
    )
    p.add_argument(
        "--resume", "-r",
        required=True,
        metavar="PATH",
        help="Path to the resume PDF (e.g. resume/my_resume.pdf)",
    )
    p.add_argument(
        "--output", "-o",
        default="output/structured_resume.json",
        metavar="PATH",
        help="Where to save the JSON output (default: output/structured_resume.json)",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Skip Pass 2 skill normalization.",
    )
    p.add_argument(
        "--no-embed",
        action="store_true",
        default=False,
        help="Skip embedding generation and FAISS indexing.",
    )
    p.add_argument(
        "--model", "-m",
        default="minilm",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Embedding model to use (default: minilm).",
    )
    p.add_argument(
        "--index-dir",
        default="output/resume_index",
        metavar="DIR",
        help="Directory for the FAISS index (default: output/resume_index).",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress progress output.",
    )
    return p


def _log(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            pdf_path=args.resume,
            output_path=args.output,
            normalize=not args.no_normalize,
            embed=not args.no_embed,
            model_key=args.model,
            index_dir=args.index_dir,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError, EnvironmentError) as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n❌  Gemini error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
