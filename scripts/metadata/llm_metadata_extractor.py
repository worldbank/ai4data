"""
llm_metadata_extractor.py
=========================
Extract document metadata from Markdown content using an LLM via litellm.

Produces draft metadata in the Document Metadata Schema format.
See https://worldbank.github.io/schema-guide/chapter05.html

Usage:
    from llm_metadata_extractor import extract_metadata_from_markdown

    schema = extract_metadata_from_markdown(md_text, idno="DOC001")
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable

import litellm
from pydantic import BaseModel, Field


def _truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to at most max_tokens using model-appropriate tokenizer.

    Uses tiktoken for OpenAI/Azure (gpt-*) models; falls back to cl100k_base for others.
    On ImportError or network failure (tiktoken fetch), falls back to ~4 chars/token heuristic.
    """
    if not text or max_tokens <= 0:
        return text
    try:
        import tiktoken

        # Map model to tiktoken encoding
        model_for_encoding = model.split("/", 1)[-1] if "/" in model else model
        if "gpt" in model_for_encoding.lower():
            try:
                enc = tiktoken.encoding_for_model(model_for_encoding)
            except Exception:
                enc = tiktoken.encoding_for_model("gpt-4")
        else:
            enc = tiktoken.get_encoding("cl100k_base")

        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except (ImportError, Exception):
        # Fallback: ~4 chars per token for English (tiktoken unavailable or fetch failed)
        return text[: max_tokens * 4] if len(text) > max_tokens * 4 else text


def _make_azure_ad_token_provider(
    scope_override: str | None = None,
) -> Callable[[], str] | None:
    """Build azure_ad_token_provider from AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID.

    Returns a callable that fetches a token via client credentials flow, or None if
    credentials are not configured.
    """
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    if not all([client_id, client_secret, tenant_id]):
        return None

    scope = scope_override or os.environ.get(
        "AZURE_SCOPE", "https://cognitiveservices.azure.com/.default"
    )

    from azure.identity import ClientSecretCredential

    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )

    def get_token() -> str:
        token = credential.get_token(scope)
        return token.token

    return get_token


# ------------------------------------------------------------------------------
# Pydantic schema for structured output (Document Metadata Schema)
# ------------------------------------------------------------------------------


class Producer(BaseModel):
    name: str = ""
    abbr: str = ""
    affiliation: str = ""
    role: str = ""


class TitleStatement(BaseModel):
    idno: str = ""
    title: str = ""
    sub_title: str = ""
    alternate_title: str = ""
    translated_title: str = ""


class Author(BaseModel):
    first_name: str = ""
    last_name: str = ""
    full_name: str = ""
    affiliation: str = ""


class Identifier(BaseModel):
    type: str = ""
    identifier: str = ""


class RefCountry(BaseModel):
    name: str = ""
    code: str = ""


class GeographicUnit(BaseModel):
    type: str = ""
    name: str = ""


class Language(BaseModel):
    name: str = ""
    code: str = ""


class Keyword(BaseModel):
    keyword: str = ""
    vocabulary: str = "AI-extracted"


class Theme(BaseModel):
    theme: str = ""
    vocabulary: str = "AI-extracted"


class Topic(BaseModel):
    topic: str = ""
    vocabulary: str = "AI-extracted"


class Contact(BaseModel):
    name: str = ""
    affiliation: str = ""
    email: str = ""
    uri: str = ""


class Relation(BaseModel):
    type: str = ""
    identifier: str = ""
    title: str = ""


class Note(BaseModel):
    note: str = ""


class MetadataInformation(BaseModel):
    title: str = ""
    idno: str = ""
    producers: list[Producer] = Field(default_factory=list)
    production_date: str = ""
    version: str = "AI-extracted draft"


class DocumentDescription(BaseModel):
    title_statement: TitleStatement = Field(default_factory=TitleStatement)
    authors: list[Author] = Field(default_factory=list)
    date_created: str = ""
    date_available: str = ""
    date_modified: str = ""
    date_published: str = ""
    identifiers: list[Identifier] = Field(default_factory=list)
    type: str = ""
    abstract: str = ""
    ref_country: list[RefCountry] = Field(default_factory=list)
    geographic_units: list[GeographicUnit] = Field(default_factory=list)
    spatial_coverage: str = ""
    languages: list[Language] = Field(default_factory=list)
    keywords: list[Keyword] = Field(default_factory=list)
    themes: list[Theme] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    url: str | None = None
    volume: str = ""
    number: str = ""
    series: str = ""
    edition: str = ""
    publisher_address: str = ""
    organization: str = ""
    contacts: list[Contact] = Field(default_factory=list)
    usage_terms: str = ""
    security_classification: str = ""
    access_restrictions: str = ""
    relations: list[Relation] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)


class OriginDescription(BaseModel):
    harvest_date: str = ""
    altered: bool = True
    metadata_namespace: str = "AI_EXTRACTED"
    identifier: str = ""


class ProvenanceItem(BaseModel):
    origin_description: OriginDescription = Field(default_factory=OriginDescription)


class DocumentMetadataSchema(BaseModel):
    """Document Metadata Schema for structured LLM output."""

    metadata_information: MetadataInformation = Field(
        default_factory=MetadataInformation
    )
    document_description: DocumentDescription = Field(
        default_factory=DocumentDescription
    )
    provenance: list[ProvenanceItem] = Field(default_factory=list)


_SCHEMA_INSTRUCTIONS = """
You are a metadata extraction assistant for development and research documents. Extract bibliographic metadata from the provided document content and output valid JSON conforming to the Document Metadata Schema (Metadata Standards for Improved Data Discoverability: https://worldbank.github.io/schema-guide/chapter05.html).

Read the FULL document content (not just the first page). Draw metadata from the abstract, introduction, methodology, findings, conclusions, and any explicit keyword lists.

Output a single JSON object with these top-level keys:

1. metadata_information (required):
   - title: string — The title of the metadata document (usually same as document title).
   - idno: string — Unique identifier for the metadata; use the idno provided if given, else derive from title. No blank spaces.
   - producers: list of {name, abbr, affiliation, role} — Who produced the metadata (not the document authors).
   - production_date: string — Date metadata was produced (YYYY-MM-DD).
   - version: string — e.g. "AI-extracted draft"

2. document_description (required) — Field definitions per the schema guide:

   - title_statement: {idno, title, sub_title, alternate_title, translated_title}
     * idno: Unique document identifier (primary ID). No blank spaces.
     * title: The title of the book, report, or paper. Use sentence capitalization.
     * sub_title: Document subtitle when it distinguishes characteristics.
     * alternate_title: Abbreviated or alternate version (e.g. "WDR 2021" for World Development Report).
     * translated_title: Translation of the title if applicable.

   - authors: list of {first_name, last_name, full_name, affiliation}
     List in the same order as in the source. Use full_name when first/last cannot be distinguished or when the author is an organization.

   - date_created: string — Date when the document was PRODUCED (YYYY-MM-DD). Can differ from published date.
   - date_available: string — Date when the document was MADE AVAILABLE. Different from date_published.
   - date_modified: string — Date when the document was last modified.
   - date_published: string — Date when the document was PUBLISHED.

   - identifiers: list of {type, identifier} — Other IDs: DOI, ISBN, ISSN. type examples: "DOI", "ISBN", "ISSN".

   - type: string — Nature of the resource. Use controlled vocabulary: "working paper", "report", "technical report", "book", "article", "conference proceedings", "manual", "other".

   - abstract: string — Extract the abstract as it appears in the document if present; otherwise use the first substantive paragraph as the abstract.

   - ref_country: list of {name, code} — Countries/regions COVERED by the document. Use ISO 3166-1 alpha-3 (e.g. USA, GBR, NIC). For global/non-country-specific documents use {name: "World", code: "WLD"}.

   - geographic_units: list of {type, name} — Geographic units other than countries: province, state, district, department, watershed, region, town.

   - spatial_coverage: string — Free-text qualification of geographic coverage. Complements ref_country and geographic_units (e.g. "Rohingya refugee camps", "Matiguás-Río Blanco district").

   - languages: list of {name, code} — Language(s) of the document. Use ISO 639-2 (e.g. eng, spa, fra).

   - keywords: list of {keyword, vocabulary} — Terms that improve discoverability. Use vocabulary "Author keyword" for explicit author keywords, "AI-extracted" for inferred.

   - themes: list of {theme, vocabulary} — Higher-level policy/goal areas. Use vocabulary "AI-extracted".

   - topics: list of {topic, vocabulary} — Development subjects covered. Use vocabulary "AI-extracted".

   - url: string or null — URL of the document, preferably a permanent link.

   - volume, number, series, edition: strings — Bibliographic elements (journal volume, report number, series name, edition).

   - publisher_address, organization: string — Publisher location; sponsoring organization.

   - contacts: list — Contact for inquiries (can be empty).

   - usage_terms: string — Legal terms or conditions for use/reproduction.

   - security_classification: string — e.g. "public", "internal only", "confidential".

   - access_restrictions: string — Textual description of access restrictions.

   - relations: list of {type, identifier, title} — Related resources. type from Dublin Core: isPartOf, hasPart, hasFormat, references, replaces, etc.

   - notes: list of {note} — Information that does not fit other elements. Use for methodology notes, data source notes, etc.

3. provenance (required):
   - origin_description: {harvest_date, altered: true, metadata_namespace: "AI_EXTRACTED", identifier}

**Keywords (extract liberally):**
- Include author-provided keywords if present
- Add key concepts: interventions (e.g. PES, conditional cash transfers), methods (e.g. impact evaluation, RCT), indicators
- Include sector terms (e.g. silvopastoral, livestock, forestry), program names, and technical jargon
- Use vocabulary "Author keyword" for explicit author keywords, "AI-extracted" for inferred terms
- Aim for 5–15 keywords that support search and discovery

**Topics (development subjects):**
- Extract from the full document body, not just title/abstract
- Use broad development topics when applicable: Climate Change, Poverty, Gender, Environment, Social Protection, Rural Development, Agriculture, Health, Education, Governance, Trade, Energy, Water, Urban Development, etc.
- Include methodology topics (e.g. Impact Evaluation, Survey Methods) when central to the document
- Use vocabulary "AI-extracted"
- Aim for 2–10 topics

**Themes (higher-level policy/goal areas):**
- Map to thematic areas: Environment and Natural Resources, Economic Policy, Social Development, Human Development, Public Sector Management, Private Sector Development, Finance, etc.
- Use vocabulary "AI-extracted"
- Aim for 1–4 themes

Use empty strings or empty arrays only when nothing can be inferred. Use ISO date format (YYYY-MM-DD) for dates. For ref_country, use ISO3 codes (e.g. "USA", "GBR", "WLD" for World) when inferable.
"""


def extract_metadata_from_markdown(
    markdown_content: str,
    *,
    idno: str | None = None,
    model: str = "gpt-4o-mini",
    api_base: str | None = None,
    api_version: str | None = None,
    azure_scope: str | None = None,
    use_pydantic: bool = True,
    max_content_tokens: int = 100_000,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Extract document metadata from Markdown using an LLM.

    Args:
        markdown_content: Markdown text extracted from the document (e.g. via pymupdf4llm).
        idno: Optional document identifier. If not provided, the LLM will derive one.
        model: litellm model string (e.g. "gpt-4o-mini", "azure/gpt-4o-mini", "claude-3-5-sonnet-20241022").
        api_base: Optional API base URL for custom endpoints.
        api_version: Optional API version (used by Azure OpenAI).
        azure_scope: Override AZURE_SCOPE for client credentials (used as-is).
        use_pydantic: If True (default), use Pydantic schema for structured output.
            Set False to fall back to json_object (less strict).
        max_content_tokens: Max tokens of document content to send (default 100k).
            Uses tiktoken for OpenAI/Azure; reserves room for system prompt and response.
        temperature: Temperature for the LLM (default 0.0).
    Returns:
        Document Metadata Schema dict with metadata_information, document_description, provenance.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON.
    """
    if not markdown_content or not markdown_content.strip():
        return _empty_schema(idno=idno or "unknown")

    idno_instruction = ""
    if idno:
        idno_instruction = f' Use idno="{idno}" for metadata_information.idno and title_statement.idno.'

    content_for_llm = _truncate_to_tokens(
        markdown_content, max_tokens=max_content_tokens, model=model
    )

    user_content = f"""Extract metadata from this document content. Read the full text to capture topics, themes, and keywords from the body (methodology, findings, conclusions), not just the title and abstract. Output only valid JSON, no other text.{idno_instruction}

---DOCUMENT CONTENT (Markdown)---
{content_for_llm}
---END---
"""

    messages = [
        {"role": "system", "content": _SCHEMA_INSTRUCTIONS},
        {"role": "user", "content": user_content},
    ]

    response_format: Any = (
        DocumentMetadataSchema if use_pydantic else {"type": "json_object"}
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": temperature,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_version:
        kwargs["api_version"] = api_version

    # Azure AD client credentials: use token provider when AZURE_CLIENT_ID etc. are set
    if model.startswith("azure/") and not os.environ.get("AZURE_API_KEY"):
        token_provider = _make_azure_ad_token_provider(scope_override=azure_scope)
        if token_provider:
            kwargs["azure_ad_token_provider"] = token_provider

    try:
        response = litellm.completion(**kwargs)
    except Exception as e:
        if use_pydantic and "response_format" in str(e).lower():
            # Fallback to json_object if model doesn't support schema
            kwargs["response_format"] = {"type": "json_object"}
            response = litellm.completion(**kwargs)
        else:
            raise

    content = response.choices[0].message.content

    if use_pydantic:
        try:
            parsed = DocumentMetadataSchema.model_validate_json(content)
            schema = parsed.model_dump(mode="json")
        except Exception:
            schema = _parse_json_response(content, idno=idno)
    else:
        schema = _parse_json_response(content, idno=idno)

    # Ensure top-level type/schematype for schema consistency
    schema.setdefault("type", "document")
    schema.setdefault("schematype", "document")

    _ensure_provenance(schema)
    return schema


def _parse_json_response(content: str | None, *, idno: str | None) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not content or not content.strip():
        return _empty_schema(idno=idno or "unknown")

    text = content.strip()

    # Extract JSON from markdown code block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find a JSON object in the text
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e


def _ensure_provenance(schema: dict[str, Any]) -> None:
    """Ensure provenance block indicates AI extraction."""
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    prov = schema.get("provenance")
    if not prov or not isinstance(prov, list):
        schema["provenance"] = [
            {
                "origin_description": {
                    "harvest_date": now,
                    "altered": True,
                    "metadata_namespace": "AI_EXTRACTED",
                    "identifier": schema.get("metadata_information", {}).get(
                        "idno", "unknown"
                    ),
                }
            }
        ]
        return

    od = prov[0].get("origin_description") if prov else None
    if isinstance(od, dict):
        od["altered"] = True
        od["metadata_namespace"] = "AI_EXTRACTED"
        if "harvest_date" not in od or not od["harvest_date"]:
            od["harvest_date"] = now


def _empty_schema(idno: str = "unknown") -> dict[str, Any]:
    """Return minimal valid schema when content is empty."""
    now = datetime.now().strftime("%Y-%m-%d")
    return {
        "type": "document",
        "schematype": "document",
        "metadata_information": {
            "title": "",
            "idno": idno,
            "producers": [
                {
                    "name": "AI Metadata Extractor",
                    "abbr": "AI",
                    "affiliation": "World Bank",
                    "role": "Extraction",
                }
            ],
            "production_date": now,
            "version": "AI-extracted draft (no content)",
        },
        "document_description": {
            "title_statement": {
                "idno": idno,
                "title": "",
                "sub_title": "",
                "alternate_title": "",
                "translated_title": "",
            },
            "authors": [],
            "date_created": "",
            "date_available": "",
            "date_modified": "",
            "date_published": "",
            "identifiers": [],
            "type": "",
            "abstract": "",
            "ref_country": [],
            "geographic_units": [],
            "spatial_coverage": "",
            "languages": [],
            "keywords": [],
            "themes": [],
            "topics": [],
            "url": None,
            "volume": "",
            "number": "",
            "series": "",
            "edition": "",
            "publisher_address": "",
            "organization": "",
            "contacts": [],
            "usage_terms": "",
            "security_classification": "",
            "access_restrictions": "",
            "relations": [],
            "notes": [],
        },
        "provenance": [
            {
                "origin_description": {
                    "harvest_date": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "altered": True,
                    "metadata_namespace": "AI_EXTRACTED",
                    "identifier": idno,
                }
            }
        ],
    }
