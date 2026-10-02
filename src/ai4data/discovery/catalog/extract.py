"""IHSN search-metadata-extract API client (bulk studies list + single study fetch)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote

import httpx

from ..auth import get_catalog_auth_headers, get_catalog_cookies
from ..config import METADATA_CATALOG_URL, metadata_catalog
from ..paths import get_metadata_cache_path
from ..ssl import configure_tls_trust_store
from ..type_normalization import normalize_catalog_metadata_type

configure_tls_trust_store()

SUCCESS_STATUSES = {"", "success", "ok"}


class CatalogExtractError(RuntimeError):
    """Raised when the metadata-extract API returns an error payload."""


def extract_base_url() -> str | None:
    """Return the extract API base URL, or ``None`` when extract mode is disabled."""
    path = metadata_catalog.extract_path
    if not path:
        return None
    return f"{metadata_catalog.url.rstrip('/')}/{path.strip('/')}"


def is_extract_mode() -> bool:
    return extract_base_url() is not None


def _get_extract_auth_headers() -> dict[str, str]:
    headers = dict(get_catalog_auth_headers())
    bearer = metadata_catalog.auth_bearer
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def _extract_query_flags(
    *,
    include_admin_metadata: bool | None = None,
    include_metadata: bool | None = None,
) -> dict[str, str]:
    admin = (
        metadata_catalog.extract_include_admin_metadata
        if include_admin_metadata is None
        else include_admin_metadata
    )
    meta = (
        metadata_catalog.extract_include_metadata
        if include_metadata is None
        else include_metadata
    )
    return {
        "include_admin_metadata": "1" if admin else "0",
        "include_metadata": "1" if meta else "0",
    }


def _parse_extract_response(data: dict[str, Any]) -> None:
    status = str(data.get("status") or "").lower()
    if status not in SUCCESS_STATUSES:
        message = data.get("message") or data.get("error") or status
        raise CatalogExtractError(f"API error ({status}): {message}")


def _request_extract(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = httpx.get(
        url,
        params=params,
        headers=headers if headers is not None else _get_extract_auth_headers(),
        cookies=cookies if cookies is not None else get_catalog_cookies(),
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise CatalogExtractError(f"Expected JSON object from {url}, got {type(data).__name__}")
    _parse_extract_response(data)
    return data


def map_catalog_params_to_extract(params: dict[str, Any] | None) -> dict[str, Any]:
    """Map classic ``/api/catalog/search`` params to extract ``/studies`` query params."""
    params = dict(params or {})
    out: dict[str, Any] = {}

    ps = int(params.get("ps", 100))
    page = int(params.get("page", 1))
    out["limit"] = ps
    out["offset"] = (page - 1) * ps

    if catalog_type := params.get("type"):
        out["type"] = catalog_type

    for key in ("source", "sk", "sort_by", "sort_order"):
        if key in params and params[key] not in (None, ""):
            out[key] = params[key]

    return out


def study_idno(study: dict[str, Any]) -> str | None:
    core = study.get("core_fields")
    if isinstance(core, dict):
        idno = core.get("idno")
        if idno:
            return str(idno).strip()
    idno = study.get("idno")
    return str(idno).strip() if idno else None


def study_metadata_type(study: dict[str, Any]) -> str | None:
    """The normalized dataset type of a study.

    NADA's own ``dataset_type`` (``filters`` / ``core_fields``) is authoritative. The record's inner
    ``metadata.type`` is only a fallback: it is unrelated to the dataset type for some schemas (ISO
    geospatial records carry ``type: "dataset"``), so trusting it first misclassified them.
    """
    for section in ("filters", "core_fields"):
        source = study.get(section)
        dataset_type = source.get("dataset_type") if isinstance(source, dict) else None
        if dataset_type:
            return normalize_catalog_metadata_type(str(dataset_type))

    metadata = study.get("metadata")
    mtype = metadata.get("type") if isinstance(metadata, dict) else None
    if mtype:
        return normalize_catalog_metadata_type(str(mtype))

    return None


def study_download_resources(study: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize extract study resources with ``_links.type == \"download\"`` to catalog shape."""
    raw = study.get("resources")
    if not isinstance(raw, list):
        return []

    downloads: list[dict[str, Any]] = []
    for resource in raw:
        if not isinstance(resource, dict):
            continue
        links = resource.get("_links")
        if not isinstance(links, dict) or links.get("type") != "download":
            continue
        url = links.get("download")
        if not url:
            continue

        item: dict[str, Any] = {
            "resource_id": resource.get("resource_id"),
            "url": url,
            "dcformat": resource.get("dcformat"),
            "is_url": str(resource.get("is_url", "0")),
        }
        for key in ("title", "dctype", "resource_type", "filename", "sort_order"):
            if resource.get(key) is not None:
                item[key] = resource[key]
        downloads.append(item)

    return downloads


def study_to_search_row(study: dict[str, Any]) -> dict[str, Any]:
    """Build a catalog-search-compatible row from one extract study payload."""
    core = study.get("core_fields") if isinstance(study.get("core_fields"), dict) else {}
    return {
        "id": core.get("catalog_id"),
        "idno": study_idno(study),
        "type": study_metadata_type(study),
    }


def study_to_catalog_metadata(study: dict[str, Any]) -> dict[str, Any]:
    """Normalize an extract study payload to the shape returned by ``/api/catalog/json/{idno}``."""
    metadata = study.get("metadata")
    if not isinstance(metadata, dict):
        raise CatalogExtractError("Study missing metadata payload (set include_metadata=1)")

    result = dict(metadata)

    if idno := study_idno(study):
        result["idno"] = idno

    if mtype := study_metadata_type(study):
        result["type"] = mtype

    filters = study.get("filters")
    if isinstance(filters, dict):
        result["_extract_filters"] = filters

    core_fields = study.get("core_fields")
    if isinstance(core_fields, dict):
        result["_extract_core_fields"] = core_fields

    downloads = study_download_resources(study)
    if downloads:
        result["external_resources"] = downloads

    return result


def _studies_from_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    studies = data.get("studies")
    if isinstance(studies, list):
        return [s for s in studies if isinstance(s, dict)]

    study = data.get("study")
    if isinstance(study, dict):
        return [study]

    if "metadata" in data or "filters" in data:
        return [data]

    return []


def fetch_extract_page(
    params: dict[str, Any] | None = None,
    *,
    include_admin_metadata: bool | None = None,
    include_metadata: bool | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch one paginated ``/studies`` page."""
    base = base_url or extract_base_url()
    if not base:
        raise CatalogExtractError("Extract path is not configured")

    query = {**_extract_query_flags(
        include_admin_metadata=include_admin_metadata,
        include_metadata=include_metadata,
    )}
    if params:
        query.update(params)

    return _request_extract(
        f"{base.rstrip('/')}/studies",
        params=query,
        headers=headers,
        cookies=cookies,
    )


def fetch_extract_study(
    idno: str,
    *,
    include_admin_metadata: bool | None = None,
    include_metadata: bool | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch a single study by idno."""
    base = base_url or extract_base_url()
    if not base:
        raise CatalogExtractError("Extract path is not configured")

    query = _extract_query_flags(
        include_admin_metadata=include_admin_metadata,
        include_metadata=include_metadata,
    )
    encoded = quote(idno.strip(), safe="")
    return _request_extract(
        f"{base.rstrip('/')}/studies/{encoded}",
        params=query,
        headers=headers,
        cookies=cookies,
    )


def search_metadata_extract(
    params: dict[str, Any] | None = None,
    *,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return catalog-search-compatible ``{rows, found}`` from one extract page."""
    extract_params = map_catalog_params_to_extract(params)
    data = fetch_extract_page(
        extract_params,
        base_url=base_url,
        headers=headers,
        cookies=cookies,
    )
    rows = [study_to_search_row(study) for study in _studies_from_response(data)]
    total = data.get("total")
    found = int(total) if isinstance(total, int) else len(rows)
    return {"rows": rows, "found": found}


def iter_extract_studies(
    params: dict[str, Any] | None = None,
    *,
    max_items: int | None = None,
    include_admin_metadata: bool | None = None,
    include_metadata: bool | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Paginate all studies matching ``params`` (classic search param shape)."""
    if max_items is not None and max_items <= 0:
        return

    base_params = dict(params or {})
    page_size = int(base_params.pop("ps", 100))
    base_params.pop("page", None)

    offset = 0
    seen = 0

    while True:
        page_params = map_catalog_params_to_extract({**base_params, "ps": page_size, "page": 1})
        page_params["offset"] = offset
        page_params["limit"] = page_size

        data = fetch_extract_page(
            page_params,
            include_admin_metadata=include_admin_metadata,
            include_metadata=include_metadata,
            base_url=base_url,
            headers=headers,
            cookies=cookies,
        )
        batch = _studies_from_response(data)
        if not batch:
            break

        for study in batch:
            yield study
            seen += 1
            if max_items is not None and seen >= max_items:
                return

        if not data.get("has_more"):
            break
        offset += page_size


def _variables_from_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    variables = data.get("variables")
    return [v for v in variables if isinstance(v, dict)] if isinstance(variables, list) else []


def fetch_extract_variables_page(
    params: dict[str, Any] | None = None,
    *,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch one paginated ``/variables`` page (the whole catalog, for a full backfill)."""
    base = base_url or extract_base_url()
    if not base:
        raise CatalogExtractError("Extract path is not configured")

    return _request_extract(f"{base.rstrip('/')}/variables", params=params, headers=headers, cookies=cookies)


def fetch_extract_survey_variables(
    idno: str,
    *,
    params: dict[str, Any] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch ONE page of a study's variables (``limit``/``after_uid``/``offset`` in ``params``).

    A study can have far more variables than fit comfortably in one response, so the route is paged; use
    :func:`iter_extract_survey_variables` to get all of them.
    """
    base = base_url or extract_base_url()
    if not base:
        raise CatalogExtractError("Extract path is not configured")

    encoded = quote(idno.strip(), safe="")
    return _request_extract(
        f"{base.rstrip('/')}/variables/{encoded}",
        params=params,
        headers=headers,
        cookies=cookies,
    )


def _iter_keyset_pages(
    fetch: Any,
    *,
    params: dict[str, Any] | None,
    page_size: int,
    max_items: int | None,
    on_page: Callable[[dict[str, Any]], None] | None,
    items_key: str,
    cursor_param: str,
    cursor_key: str,
) -> Iterator[dict[str, Any]]:
    """Walk a keyset-paged extract route: the page's ``cursor_key`` is the next request's ``cursor_param``.

    ``on_page``, if given, is called with every raw response before its items are yielded — for what a page
    carries besides items (``total`` on the first page of a walk, for a progress figure).
    """
    if max_items is not None and max_items <= 0:
        return

    cursor: int | None = None
    seen = 0
    while True:
        page_params = {**(params or {}), "limit": page_size}
        if cursor is not None:
            page_params[cursor_param] = cursor
        data = fetch(page_params)
        if on_page is not None:
            on_page(data)
        raw = data.get(items_key)
        batch = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        if not batch:
            return

        for item in batch:
            yield item
            seen += 1
            if max_items is not None and seen >= max_items:
                return

        cursor = data.get(cursor_key)
        if not data.get("has_more") or cursor is None:
            return


def _iter_variable_pages(
    fetch: Any,
    *,
    params: dict[str, Any] | None,
    page_size: int,
    max_items: int | None,
    on_page: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Walk a keyset-paged variables route: ``next_after_uid`` of one page is ``after_uid`` of the next."""
    return _iter_keyset_pages(
        fetch,
        params=params,
        page_size=page_size,
        max_items=max_items,
        on_page=on_page,
        items_key="variables",
        cursor_param="after_uid",
        cursor_key="next_after_uid",
    )


def iter_extract_survey_variables(
    idno: str,
    *,
    page_size: int = 1000,
    max_items: int | None = None,
    on_page: Callable[[dict[str, Any]], None] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Every variable of one study, fetched page by page (keyset paging, so a page costs the same however deep)."""
    return _iter_variable_pages(
        lambda params: fetch_extract_survey_variables(
            idno, params=params, base_url=base_url, headers=headers, cookies=cookies
        ),
        params=None,
        page_size=page_size,
        max_items=max_items,
        on_page=on_page,
    )


def iter_extract_variables(
    params: dict[str, Any] | None = None,
    *,
    max_items: int | None = None,
    page_size: int = 1000,
    on_page: Callable[[dict[str, Any]], None] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Every variable in the catalog (the ``/variables`` batch route), keyset-paged."""
    return _iter_variable_pages(
        lambda page_params: fetch_extract_variables_page(
            page_params, base_url=base_url, headers=headers, cookies=cookies
        ),
        params=params,
        page_size=page_size,
        max_items=max_items,
        on_page=on_page,
    )


def fetch_extract_citation(
    citation_id: int,
    *,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch one citation document (``metadata``, ``core_fields``, ``filters``) by its NADA id."""
    base = base_url or extract_base_url()
    if not base:
        raise CatalogExtractError("Extract path is not configured")

    data = _request_extract(f"{base.rstrip('/')}/citations/{int(citation_id)}", headers=headers, cookies=cookies)
    citation = data.get("citation")
    if not isinstance(citation, dict):
        raise CatalogExtractError(f"Extract response for citation {citation_id} has no citation")
    return citation


def fetch_extract_citations_page(
    params: dict[str, Any] | None = None,
    *,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch one paginated ``/citations`` page (``limit``/``after_id``/``offset`` in ``params``)."""
    base = base_url or extract_base_url()
    if not base:
        raise CatalogExtractError("Extract path is not configured")

    return _request_extract(f"{base.rstrip('/')}/citations", params=params, headers=headers, cookies=cookies)


def iter_extract_citations(
    params: dict[str, Any] | None = None,
    *,
    max_items: int | None = None,
    page_size: int = 500,
    on_page: Callable[[dict[str, Any]], None] | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Every citation document in the catalog (the ``/citations`` batch route), keyset-paged on the citation id."""
    return _iter_keyset_pages(
        lambda page_params: fetch_extract_citations_page(
            page_params, base_url=base_url, headers=headers, cookies=cookies
        ),
        params=params,
        page_size=page_size,
        max_items=max_items,
        on_page=on_page,
        items_key="citations",
        cursor_param="after_id",
        cursor_key="next_after_id",
    )


def write_metadata_cache(
    metadata: dict[str, Any],
    idno: str,
    metadata_type: str,
    *,
    include_resources: bool = False,
) -> None:
    cache_path = get_metadata_cache_path(idno, metadata_type, include_resources=include_resources)
    try:
        with cache_path.open("w") as cache_file:
            json.dump(metadata, cache_file, indent=2)
    except OSError as e:
        print(f"Failed to write cache file {cache_path}: {e}")


def _metadata_has_resources(metadata: dict[str, Any]) -> bool:
    for key in ("resources", "external_resources", "microdata_resources"):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def fetch_metadata_from_extract(
    idno: str,
    metadata_type: str | None = None,
    *,
    include_resources: bool = False,
) -> dict[str, Any]:
    """Fetch and normalize metadata for one idno from the extract API."""
    data = fetch_extract_study(idno)
    studies = _studies_from_response(data)
    if not studies:
        raise CatalogExtractError(f"No study found for idno {idno!r}")

    metadata = study_to_catalog_metadata(studies[0])

    if metadata_type and metadata.get("type") != metadata_type:
        raise ValueError(
            f"The metadata type {metadata.get('type')} does not match the requested type: {metadata_type}"
        )

    if (
        include_resources
        and not _metadata_has_resources(metadata)
        and metadata_catalog.extract_fallback_catalog_json
    ):
        response = httpx.get(
            f"{METADATA_CATALOG_URL}/api/catalog/json/{idno}",
            params={"include_resources": True},
            headers=get_catalog_auth_headers(),
            cookies=get_catalog_cookies(),
        )
        response.raise_for_status()
        fallback: dict = response.json()
        if fallback.get("type"):
            fallback["type"] = normalize_catalog_metadata_type(fallback["type"])
        metadata = fallback

    return metadata
