"""HTTP authentication headers and cookies for NADA / IHSN catalog calls and PDF downloads.

The NADA catalog accepts ``x-api-key`` as the authentication header. We attach it from
:attr:`MetadataCatalogConfig.x_api_key` (env ``AI4DATA_METADATA_CATALOG_X_API_KEY``) to:

- catalog endpoints — always (search, JSON metadata), and
- PDF download URLs — only when the URL's host is the configured catalog host (or is
  explicitly allow-listed via ``AI4DATA_METADATA_CATALOG_X_API_KEY_HOSTS``).

Some NADA instances (notably the IHSN training catalog) gate file downloads behind a
logged-in PHP session rather than an API key. For those cases,
:attr:`MetadataCatalogConfig.cookies` (env ``AI4DATA_METADATA_CATALOG_COOKIES``) carries a
``name=value; name=value`` cookie string that is attached with the same host scoping as
the API key.

The host scoping prevents the credential from being sent to third-party hosts that the
catalog may embed as external resources (e.g. a `World Bank <https://worldbank.org>`_
hosted PDF returned in ``external_resources`` for an NADA record).
"""

from __future__ import annotations

from urllib.parse import urlparse

from .config import metadata_catalog

X_API_KEY_HEADER = "x-api-key"


def _split_hosts(value: str | None) -> set[str]:
    if not value:
        return set()
    return {h.strip() for h in value.split(",") if h.strip()}


def allowed_api_key_hosts() -> set[str]:
    """Return the set of hostnames that may receive the ``x-api-key`` header."""
    catalog_host = urlparse(metadata_catalog.url).netloc
    hosts = {catalog_host} if catalog_host else set()
    hosts.update(_split_hosts(metadata_catalog.x_api_key_hosts))
    return hosts


def get_catalog_auth_headers(url: str | None = None) -> dict[str, str]:
    """Build the ``x-api-key`` header dict for an outgoing HTTP call.

    Parameters
    ----------
    url:
        The full URL the request will be sent to. Pass ``None`` for endpoints that always hit
        the catalog (search, JSON metadata) — the header is sent unconditionally when
        configured. For arbitrary URLs (e.g. download URLs harvested from the catalog), the
        header is sent only when the URL's host is allow-listed (catalog host or
        :attr:`MetadataCatalogConfig.x_api_key_hosts`).

    Returns
    -------
    dict[str, str]
        ``{"x-api-key": ...}`` when authorized, otherwise an empty dict.
    """
    api_key = metadata_catalog.x_api_key
    if not api_key:
        return {}
    if url is None:
        return {X_API_KEY_HEADER: api_key}
    target = urlparse(url).netloc
    if not target:
        # Relative URL — caller is responsible for prepending the catalog base, so treat as
        # catalog-local.
        return {X_API_KEY_HEADER: api_key}
    if target in allowed_api_key_hosts():
        return {X_API_KEY_HEADER: api_key}
    return {}


def _parse_cookie_string(raw: str | None) -> dict[str, str]:
    """Parse a ``name=value; name=value`` cookie string into a dict.

    Tolerant of stray whitespace and empty segments. Returns an empty dict for ``None``
    or strings with no ``name=value`` pairs.
    """
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if name:
            out[name] = value.strip()
    return out


def get_catalog_cookies(url: str | None = None) -> dict[str, str]:
    """Build the cookie dict for an outgoing HTTP call to the catalog.

    Mirrors the host-scoping logic of :func:`get_catalog_auth_headers`.

    Parameters
    ----------
    url:
        The full URL the request will be sent to. Pass ``None`` for endpoints that always
        hit the catalog (search, JSON metadata) — cookies are sent unconditionally when
        configured. For arbitrary URLs (e.g. download URLs harvested from the catalog),
        cookies are sent only when the URL's host is allow-listed (catalog host or
        :attr:`MetadataCatalogConfig.x_api_key_hosts`).

    Returns
    -------
    dict[str, str]
        Parsed cookie dict when authorized, otherwise an empty dict.
    """
    raw = metadata_catalog.cookies
    if not raw:
        return {}
    if url is None:
        return _parse_cookie_string(raw)
    target = urlparse(url).netloc
    if not target:
        return _parse_cookie_string(raw)
    if target in allowed_api_key_hosts():
        return _parse_cookie_string(raw)
    return {}
