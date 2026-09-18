"""Document parsing utilities using PyMuPDF."""

import json
import logging
import os
import re
import tempfile
from typing import Dict, List, Union

import fitz  # PyMuPDF

# pymupdf4llm is an optional dependency — imported lazily where needed
import requests

logger = logging.getLogger(__name__)

# Download safeguards
_DEFAULT_CONNECT_TIMEOUT = 30  # seconds to establish connection
_DEFAULT_READ_TIMEOUT = 300  # seconds to wait between chunks
_DEFAULT_MAX_SIZE_MB = 100  # reject PDFs larger than this


class DocumentParser:
    """Parse PDF documents from files or URLs."""

    @staticmethod
    def _download_temp_pdf(
        url: str,
        max_size_mb: int = _DEFAULT_MAX_SIZE_MB,
        connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: int = _DEFAULT_READ_TIMEOUT,
    ) -> str:
        """Download a PDF from a URL to a temporary file.

        Sends a HEAD request first to check file size. If the server
        reports a Content-Length exceeding *max_size_mb*, raises
        ``ValueError`` suggesting the user download the file locally.
        Both HEAD and GET use explicit timeouts to prevent indefinite
        hangs on unresponsive servers.

        Args:
            url: URL to the PDF
            max_size_mb: Maximum allowed file size in MB (default: 100).
                         Set to 0 to skip the size check.
            connect_timeout: Seconds to wait for a connection (default: 30)
            read_timeout: Seconds to wait between response chunks (default: 300)

        Returns:
            Path to the temporary file

        Raises:
            ValueError: If the file exceeds *max_size_mb*
            requests.ConnectionError: If the server is unreachable
            requests.Timeout: If a timeout is exceeded
        """
        timeout = (connect_timeout, read_timeout)

        # --- Size pre-check via HEAD request ---
        if max_size_mb > 0:
            try:
                head = requests.head(url, timeout=timeout, allow_redirects=True)
                content_length = head.headers.get("Content-Length")
                if content_length is not None:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > max_size_mb:
                        raise ValueError(
                            f"PDF at URL is too large ({size_mb:.1f} MB, "
                            f"limit is {max_size_mb} MB). Download the file "
                            f"locally and pass the file path instead."
                        )
                    logger.debug("PDF size: %.1f MB (limit: %d MB)", size_mb, max_size_mb)
            except requests.RequestException as exc:
                # HEAD failed (some servers block it) — fall through to GET
                logger.debug("HEAD request failed (%s), skipping size check", exc)

        # --- Download with timeout ---
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            r = requests.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            for chunk in r.iter_content(8192):
                tmp.write(chunk)
            tmp.flush()
            return tmp.name

    @staticmethod
    def get_page_count(fname_or_url: str) -> int:
        """Get the total number of pages in a PDF.

        Args:
            fname_or_url: Local file path or URL to PDF

        Returns:
            Total number of pages

        Example:
            >>> count = DocumentParser.get_page_count("document.pdf")
            >>> print(f"Pages: {count}")
        """
        is_url = fname_or_url.startswith(("http://", "https://"))
        path = DocumentParser._download_temp_pdf(fname_or_url) if is_url else fname_or_url

        try:
            doc = fitz.open(path)
            count = len(doc)
            doc.close()
            return count
        finally:
            if is_url and os.path.exists(path):
                os.unlink(path)

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
    def is_english(text: str, threshold: float = 0.05) -> bool:
        """Check if a text is likely English based on common stopwords.

        Args:
            text: Text to check
            threshold: Minimum ratio of English stopwords (default: 0.05)

        Returns:
            True if the ratio of common English words exceeds threshold

        Example:
            >>> DocumentParser.is_english("This is an analysis of the data.")
            True
            >>> DocumentParser.is_english("Esto es un análisis de los datos.")
            False
        """
        if not text or not text.strip():
            return False

        # Common English function words (stopwords)
        stopwords = {
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "is",
            "it",
            "for",
            "not",
            "on",
            "with",
            "he",
            "as",
            "you",
            "do",
            "at",
        }

        # Tokenize (fast whitespace split and lowercase)
        # We also strip basic punctuation to improve matching
        words = re.findall(r"\b[a-z]{2,}\b", text.lower())
        if not words:
            return False

        stopword_count = sum(1 for word in words if word in stopwords)
        ratio = stopword_count / len(words)

        return ratio >= threshold

    @staticmethod
    def load_pdf_chunks(
        fname_or_url: str,
        n_pages: int = 1,
        skip_references: bool = False,
        as_markdown: bool = True,
        verbose: bool = False,
        pages: List[int] = None,
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """Load a PDF and return chunks with page information.

        Args:
            fname_or_url: Local file path or URL to PDF
            n_pages: Number of pages per chunk
            skip_references: If True, skip pages after references/appendix section
                           is detected (only checks after halfway point)
            as_markdown: If True, use pymupdf4llm to extract text as markdown
                        (default: True). If False, use standard text extraction.
            verbose: If True, print logging messages when references are detected
                    and pages are skipped
            pages: Optional list of 0-indexed page numbers to include. If None,
                   processes all pages.

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

            # Determine which pages to process
            target_pages = list(range(total))
            if pages is not None:
                target_pages = [p for p in pages if 0 <= p < total]

            for idx in range(0, len(target_pages), n_pages):
                chunk_page_indices = target_pages[idx : idx + n_pages]
                start = chunk_page_indices[0]

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
                                print("   Skip remaining chunks.\n")

                    if references_started:
                        skipped_count += len(chunk_page_indices)
                        continue

                # Extract text for this range
                if as_markdown:
                    # Use pymupdf4llm for markdown extraction (optional dependency)
                    try:
                        import pymupdf4llm
                    except ImportError as exc:
                        raise ImportError(
                            "pymupdf4llm is required for markdown extraction. "
                            "Install it with: pip install pymupdf4llm"
                        ) from exc
                    chunk_text = pymupdf4llm.to_markdown(doc, pages=chunk_page_indices)
                else:
                    # Use standard fitz text extraction
                    texts = [doc[i].get_text() for i in chunk_page_indices]
                    chunk_text = "\n\n".join(texts)

                chunks.append({"text": chunk_text, "pages": chunk_page_indices})

            if verbose and skip_references:
                print("✅ Processing complete:")
                print(f"   Pages processed: {len(target_pages) - skipped_count}")
                print(f"   Pages skipped: {skipped_count}")
                print(f"   Total chunks: {len(chunks)}\n")

            doc.close()
            return chunks

        # URL handling: always download to temp file for pymupdf
        if fname_or_url.startswith(("http://", "https://")):
            tmp_path = DocumentParser._download_temp_pdf(fname_or_url)
            try:
                results = _open_pymupdf(tmp_path)
            finally:
                os.unlink(tmp_path)
            return results
        else:
            return _open_pymupdf(fname_or_url)

    @staticmethod
    def load_pdf_as_markdown(fname_or_url: str, pages: List[int] = None) -> str:
        """Load a PDF and return it as markdown text using pymupdf4llm.

        Args:
            fname_or_url: Local file path or URL to PDF
            pages: List of page numbers to include (0-indexed)

        Returns:
            Markdown formatted text

        Example:
            >>> md = DocumentParser.load_pdf_as_markdown("document.pdf", pages=[0, 1])
            >>> print(md[:100])
        """
        try:
            import pymupdf4llm
        except ImportError as exc:
            raise ImportError(
                "pymupdf4llm is required for markdown extraction. "
                "Install it with: pip install pymupdf4llm"
            ) from exc

        # URL handling: download to temp file
        if fname_or_url.startswith(("http://", "https://")):
            tmp_path = DocumentParser._download_temp_pdf(fname_or_url)
            try:
                md_text = pymupdf4llm.to_markdown(tmp_path, pages=pages)
            finally:
                os.unlink(tmp_path)
            return md_text
        else:
            return pymupdf4llm.to_markdown(fname_or_url, pages=pages)

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
