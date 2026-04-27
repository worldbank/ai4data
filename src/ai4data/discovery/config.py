"""Configuration for discovery workflows (NADA catalog, metadata Jinja templates)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class MetadataCatalogConfig(BaseSettings):
    """NADA / IHSN catalog API and related URLs."""

    model_config = SettingsConfigDict(env_prefix="AI4DATA_METADATA_CATALOG_", case_sensitive=False, extra="forbid")

    url: str = Field(default="https://data-compass.ihsn.org/index.php")
    thumbnail_url: str = Field(default="https://data-compass.ihsn.org/files/thumbnails/thumbnail-s{db_id}.jpeg")
    x_api_key: str | None = Field(default=None)


class EmbeddingTemplatesConfig(BaseSettings):
    """Default root directory for per-type Jinja2 templates used when building embedding text from metadata."""

    model_config = SettingsConfigDict(env_prefix="AI4DATA_EMBEDDING_", case_sensitive=False, extra="ignore")

    content_templates_path: Path = Field(
        default=Path(__file__).resolve().parent / "metadata" / "templates",
        description="Directory containing subfolders per metadata type (indicator, document, …).",
    )


class DiscoveryDataConfig(BaseSettings):
    """Root directory for catalog caches (metadata_ids, metadata_cache, document_cache, …)."""

    model_config = SettingsConfigDict(env_prefix="AI4DATA_DISCOVERY_", case_sensitive=False, extra="ignore")

    data_path: Path | None = Field(
        default=None,
        description="If set, used as the discovery data root unless init_discovery_paths is called with an explicit path.",
    )


metadata_catalog: MetadataCatalogConfig = MetadataCatalogConfig()
embedding_templates: EmbeddingTemplatesConfig = EmbeddingTemplatesConfig()
discovery_data: DiscoveryDataConfig = DiscoveryDataConfig()

METADATA_CATALOG_URL = metadata_catalog.url
EMBEDDING_CONTENT_TEMPLATES_PATH = embedding_templates.content_templates_path
