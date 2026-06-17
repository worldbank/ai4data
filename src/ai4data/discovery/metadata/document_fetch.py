"""
Download and cache PDF attachments for document-type metadata (used by :class:`DocumentMetadata`).
"""

from __future__ import annotations

import logging

import backoff
import httpx

from ..auth import get_catalog_auth_headers, get_catalog_cookies
from ..paths import get_document_cache_path
from ..ssl import configure_tls_trust_store

configure_tls_trust_store()

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"


class EmptyDownloadError(ValueError):
    """Raised when the catalog responds with an empty or non-PDF body for a document URL.

    Surfaces the IHSN / NADA case where the server returns ``200 OK`` with ``content-length: 0``
    and a ``Refresh: 0;url=.../auth/login/...`` header for documents that require login.
    """


@backoff.on_exception(backoff.expo, httpx.HTTPStatusError, max_tries=3)
def download_pdf(url: str) -> bytes:
    """Download a PDF file from a URL and validate that the body is actually a PDF.

    The IHSN / NADA catalog returns ``200 OK`` with an empty body and a ``Refresh`` header
    pointing at the login page when a document requires authentication. ``raise_for_status``
    does not catch this, so we explicitly reject empty bodies and payloads that don't start
    with the ``%PDF-`` magic header.

    When :envvar:`AI4DATA_METADATA_CATALOG_X_API_KEY` is configured, an ``x-api-key`` header
    is attached for hosts that match the catalog (or are allow-listed via
    :envvar:`AI4DATA_METADATA_CATALOG_X_API_KEY_HOSTS`). When
    :envvar:`AI4DATA_METADATA_CATALOG_COOKIES` is configured, the parsed cookies are
    attached with the same host scoping (e.g. for NADA instances that gate downloads
    behind a logged-in session). See :mod:`ai4data.discovery.auth`.
    """
    assert isinstance(url, str), "The url must be a string"

    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=30,
        headers=get_catalog_auth_headers(url),
        cookies=get_catalog_cookies(url),
    )
    response.raise_for_status()

    content = response.content
    if not content:
        refresh = response.headers.get("refresh", "")
        hint = ""
        if refresh and ("login" in refresh.lower() or "auth" in refresh.lower()):
            hint = f" (server responded with login Refresh header: {refresh!r})"
        raise EmptyDownloadError(f"Empty body from {url}{hint}")

    if not content.startswith(_PDF_MAGIC):
        preview = content[:64]
        raise EmptyDownloadError(
            f"Response from {url} is not a PDF (first 64 bytes: {preview!r})"
        )

    return content


def cache_download_pdf(
    url: str,
    idno: str,
    metadata_type: str,
    force: bool = False,
    resource_id: str | None = None,
):
    """
    Cache the downloaded pdf file.

    A 0-byte file left over from a previous failed download (e.g. an auth-gated URL that
    returned ``200 OK`` with an empty body) is treated as a cache miss and retried, so we
    don't permanently poison the cache.
    """

    fpath = get_document_cache_path(idno, metadata_type, resource_id=resource_id)

    if fpath.exists() and not force:
        try:
            existing_size = fpath.stat().st_size
        except OSError:
            existing_size = 0
        if existing_size > 0:
            return fpath
        logger.warning(
            "Discarding 0-byte cached PDF before re-downloading: %s", fpath
        )
        fpath.unlink(missing_ok=True)

    fpath.parent.mkdir(parents=True, exist_ok=True)

    try:
        pdf = download_pdf(url)
    except Exception as e:
        raise ValueError(f"Error downloading pdf for {idno}: {e}") from e

    try:
        with open(fpath, "wb") as f:
            f.write(pdf)
    except Exception as e:
        fpath.unlink(missing_ok=True)
        raise e

    return fpath
