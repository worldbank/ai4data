"""Document parsing utilities using PyMuPDF."""

import json
import os
import re
import tempfile
from typing import Dict, List, Union

import fitz  # PyMuPDF
import requests


class DocumentParser:
    """Parse PDF documents from files or URLs."""

    @staticmethod
    def is_references_page(text: str, check_lines: int = 5) -> bool:
        """Check if a page appears to be in the references/appendix section.

        Args:
            text: Page text to check
            check_lines: Number of initial lines to check (default: 5)

        Returns:
            True if the page appears to be a references/appendix page

        Example:
            >>> text = "References\n1. Smith et al. (2020)..."
            >>> DocumentParser.is_references_page(text)
            True
        """
        if not text or not text.strip():
            return False

        # Get first few lines
        lines = text.strip().split("\n")[:check_lines]
        first_text = "\n".join(lines).strip()

        # Common patterns for references/appendix sections
        # Match variations like: "References", "REFERENCES", "Bibliography",
        # "Appendix", "Appendix A", "Annex", "Annex 1", etc.
        patterns = [
            r"^\s*references\s*$",
            r"^\s*bibliography\s*$",
            r"^\s*appendix(\s+[a-z0-9]+)?\s*$",
            r"^\s*annex(\s+[a-z0-9]+)?\s*$",
            r"^\s*works\s+cited\s*$",
            r"^\s*literature\s+cited\s*$",
        ]

        for pattern in patterns:
            if re.search(pattern, first_text, re.IGNORECASE | re.MULTILINE):
                return True

        return False

    @staticmethod
    def load_pdf_chunks(
        fname_or_url: str,
        n_pages: int = 1,
        skip_references: bool = False,
        verbose: bool = False,
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """Load a PDF and return chunks with page information.

        Args:
            fname_or_url: Local file path or URL to PDF
            n_pages: Number of pages per chunk
            skip_references: If True, skip pages after references/appendix section
                           is detected (only checks after halfway point)
            verbose: If True, print logging messages when references are detected
                    and pages are skipped

        Returns:
            List of dicts with 'text' and 'pages' keys

        Example:
            >>> chunks = DocumentParser.load_pdf_chunks("document.pdf", n_pages=2)
            >>> for chunk in chunks:
            ...     print(f"Pages {chunk['pages']}: {chunk['text'][:100]}...")
        """

        def _open_pymupdf(path: str) -> List[Dict[str, Union[str, List[int]]]]:
            doc = fitz.open(path)
            total = len(doc)
            chunks = []
            halfway_point = total // 2
            references_started = False
            skipped_count = 0

            if verbose and skip_references:
                print(f"\n📄 Processing PDF: {total} total pages")
                print(f"   Halfway point: page {halfway_point}")
                print(f"   Will check for references after page {halfway_point}\n")

            for start in range(0, total, n_pages):
                end = min(start + n_pages, total)

                # Check if we should skip this chunk
                if skip_references and start >= halfway_point:
                    # Check first page of this chunk for references section
                    if not references_started:
                        first_page_text = doc[start].get_text()
                        if DocumentParser.is_references_page(first_page_text):
                            references_started = True
                            if verbose:
                                # Show first few lines of the detected page
                                preview = first_page_text.strip().split("\n")[:3]
                                preview_text = "\n   ".join(preview)
                                print(f"🔍 REFERENCES DETECTED on page {start}!")
                                print("   First lines:")
                                print(f"   {preview_text}")
                                print(f"   ⏭️  Skipping all pages from {start} to {total-1}\n")

                    # Skip this chunk if references section has started
                    if references_started:
                        skipped_count += end - start
                        continue

                texts = [doc[i].get_text() for i in range(start, end)]
                chunks.append({"text": "\n\n".join(texts), "pages": list(range(start, end))})

            if verbose and skip_references:
                print("✅ Processing complete:")
                print(f"   Pages processed: {total - skipped_count}")
                print(f"   Pages skipped: {skipped_count}")
                print(f"   Total chunks: {len(chunks)}\n")

            doc.close()
            return chunks

        # URL handling: always download to temp file for pymupdf
        if fname_or_url.startswith("http://") or fname_or_url.startswith("https://"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                r = requests.get(fname_or_url, stream=True)
                r.raise_for_status()
                for chunk in r.iter_content(8192):
                    tmp.write(chunk)
                tmp.flush()
                tmp_path = tmp.name
            results = _open_pymupdf(tmp_path)
            os.unlink(tmp_path)
            return results
        else:
            return _open_pymupdf(fname_or_url)

    @staticmethod
    def load_json_data(filepath: str) -> Union[Dict, List]:
        """Load a JSON or JSONL file from the given path.

        Args:
            filepath: Path to JSON or JSONL file

        Returns:
            Dict or list (for JSON) or list of dicts (for JSONL)

        Example:
            >>> data = DocumentParser.load_json_data("data.jsonl")
            >>> print(f"Loaded {len(data)} records")
        """
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline()
            f.seek(0)
            if first_line.strip().startswith("{") and not first_line.strip().endswith("["):
                # Try JSON Lines: each line is a dict
                try:
                    data = [json.loads(line) for line in f if line.strip()]
                    if len(data) == 1:
                        f.seek(0)
                        data = json.load(f)
                    return data
                except Exception:
                    f.seek(0)
                    return json.load(f)
            else:
                return json.load(f)
