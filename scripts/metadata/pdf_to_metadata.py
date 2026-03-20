"""
pdf_to_metadata.py
==================
Pipeline: PDF -> pymupdf4llm (Markdown) -> LLM (litellm) -> draft metadata JSON.

Extracts document content from PDFs and generates draft metadata in the
Document Metadata Schema format.

Usage:
    uv run python scripts/metadata/pdf_to_metadata.py --input=report.pdf --output=document.json
    uv run python scripts/metadata/pdf_to_metadata.py --url=https://.../doc.pdf --output=document.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

# Ensure scripts/metadata is on path for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pdf_extractor import extract_pdf_to_markdown
from llm_metadata_extractor import extract_metadata_from_markdown


def _derive_idno(source: str) -> str:
    """Derive idno from input path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        path = urlparse(source).path
        stem = Path(path).stem if path else "document"
    else:
        stem = Path(source).stem
    return stem or "document"


def main(
    input: str | None = None,
    url: str | None = None,
    output: str | None = None,
    max_pages: int = 5,
    model: str = "gpt-4o-mini",
    idno: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
    azure_scope: str | None = None,
    no_pydantic: bool = False,
    max_content_tokens: int = 100_000,
) -> None:
    """Extract metadata from a PDF and write draft schema JSON.

    Either --input (local path) or --url must be provided.
    """
    source = input or url
    if not source:
        print("Error: provide --input=path/to/file.pdf or --url=https://...")
        sys.exit(1)

    if not output:
        output = f"document_{_derive_idno(source)}.json"

    # Derive idno from filename if not provided
    if not idno:
        idno = _derive_idno(source)

    print(f"Extracting Markdown from {source} (max_pages={max_pages or 'all'})...")
    markdown = extract_pdf_to_markdown(source, max_pages=max_pages)

    use_pydantic = not no_pydantic
    if not markdown.strip():
        print("Warning: no content extracted from PDF. Writing empty schema.")
        schema = extract_metadata_from_markdown(
            "",
            idno=idno,
            model=model,
            api_base=api_base,
            api_version=api_version,
            azure_scope=azure_scope,
            use_pydantic=use_pydantic,
            max_content_tokens=max_content_tokens,
        )
    else:
        print(f"Extracting metadata with {model}...")
        schema = extract_metadata_from_markdown(
            markdown,
            idno=idno,
            model=model,
            api_base=api_base,
            api_version=api_version,
            azure_scope=azure_scope,
            use_pydantic=use_pydantic,
            max_content_tokens=max_content_tokens,
        )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract draft metadata from PDF using pymupdf4llm and LLM."
    )
    parser.add_argument("--input", help="Path to local PDF file")
    parser.add_argument("--url", help="URL to PDF file")
    parser.add_argument(
        "--output", help="Output JSON path (default: document_<stem>.json)"
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=5,
        help="Max pages to extract (default: 5). Use 0 for all pages.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="litellm model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--idno", help="Document identifier (default: derived from filename)"
    )
    parser.add_argument("--api_base", help="Custom API base URL for litellm")
    parser.add_argument(
        "--api_version",
        help="API version (for Azure OpenAI, e.g. 2024-02-15-preview)",
    )
    parser.add_argument(
        "--azure_scope",
        help="Override AZURE_SCOPE for client credentials (used exactly as provided)",
    )
    parser.add_argument(
        "--no-pydantic",
        action="store_true",
        help="Use json_object instead of Pydantic schema (fallback if model rejects schema)",
    )
    parser.add_argument(
        "--max_content_tokens",
        type=int,
        default=100_000,
        help="Max tokens of document content to send (default: 100000)",
    )
    args = parser.parse_args()

    main(
        input=args.input,
        url=args.url,
        output=args.output,
        max_pages=args.max_pages,
        model=args.model,
        idno=args.idno,
        api_base=args.api_base,
        api_version=args.api_version,
        azure_scope=args.azure_scope,
        no_pydantic=args.no_pydantic,
        max_content_tokens=args.max_content_tokens,
    )
