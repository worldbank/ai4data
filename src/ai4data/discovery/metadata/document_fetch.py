"""
Download and cache PDF attachments for document-type metadata (used by :class:`DocumentMetadata`).
"""

from __future__ import annotations

import backoff
import httpx

from ..paths import get_document_cache_path


@backoff.on_exception(backoff.expo, httpx.HTTPStatusError, max_tries=3)
def download_pdf(url: str):
    """
    Download a pdf file from a url
    """

    assert isinstance(url, str), "The url must be a string"

    response = httpx.get(url, follow_redirects=True)
    response.raise_for_status()

    return response.content


def cache_download_pdf(url: str, idno: str, metadata_type: str, force: bool = False):
    """
    Cache the downloaded pdf file
    """

    fpath = get_document_cache_path(idno, metadata_type)

    if fpath.exists() and not force:
        return fpath

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
