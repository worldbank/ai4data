"""
pdf_extractor.py
================
Extract Markdown content from PDFs using pymupdf4llm.

Supports local files and URLs. By default extracts only the first N pages
to keep content size manageable for LLM metadata extraction.

Usage:
    from pdf_extractor import extract_pdf_to_markdown

    md = extract_pdf_to_markdown("document.pdf", max_pages=5)
    md_full = extract_pdf_to_markdown("document.pdf", max_pages=0)
"""

from __future__ import annotations

import os
import tempfile
import pymupdf4llm
import requests


def extract_pdf_to_markdown(
    fname_or_url: str,
    *,
    max_pages: int = 5,
) -> str:
    """Extract PDF content to Markdown using pymupdf4llm.

    Args:
        fname_or_url: Local file path or URL to PDF.
        max_pages: Maximum number of pages to extract (1-based). Default 5.
            Use 0 to extract all pages.

    Returns:
        Markdown string with extracted content.

    Example:
        >>> md = extract_pdf_to_markdown("report.pdf", max_pages=3)
        >>> len(md) > 0
        True
    """
    path = _resolve_path(fname_or_url)
    is_temp = path != fname_or_url

    try:
        if max_pages == 0:
            pages = None  # all pages
        else:
            # pymupdf4llm uses 0-based page indices
            pages = list(range(max_pages))

        md_text = pymupdf4llm.to_markdown(path, pages=pages)

        if isinstance(md_text, list):
            # page_chunks=True returns list of dicts; concatenate "text" fields
            return "\n\n".join(
                chunk.get("text", "") for chunk in md_text if isinstance(chunk, dict)
            )

        return md_text or ""
    finally:
        if is_temp and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _resolve_path(fname_or_url: str) -> str:
    """Resolve input to a local file path. Downloads URLs to a temp file."""
    if fname_or_url.startswith("http://") or fname_or_url.startswith("https://"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            r = requests.get(fname_or_url, stream=True, timeout=60)
            r.raise_for_status()
            for chunk in r.iter_content(8192):
                tmp.write(chunk)
            tmp.flush()
            path = tmp.name
        return path
    return fname_or_url
