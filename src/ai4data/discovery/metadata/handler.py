import json
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document as LangchainDocument

from ..catalog.http import get_metadata_json
from ..catalog.langdoc_id import get_langdoc_uuid  # noqa: F401 — re-exported for callers
from ..paths import get_contextualized_dimensions_path
from ..processors.document import load_pdf
from .document_fetch import cache_download_pdf
from .filters import (
    DocumentFilterFacets,
    GeospatialFilterFacets,
    IndicatorFilterFacets,
    MicrodataFilterFacets,
)
from .templates.render import get_searchpath, render_embedding_content


class Metadata(ABC):
    def __init__(
        self,
        metadata: dict,
        collection_fields: list,
        metadata_type: str = None,
        searchpath: str = None,
    ):
        """
        Initialize the Metadata handler.

        Args:
            metadata (dict): The metadata dictionary.
            collection_fields (list): Fields to be used for multiple collection indexing for a metadata type, (e.g., 'title', 'abstract', 'passages').
            metadata_type (str, optional): Type of metadata (e.g., 'document', 'indicator').
            searchpath (str, optional): Path used for searching or rendering templates. Defaults to None.
        """
        self._metadata = metadata
        self._payload = None
        self.type = metadata_type
        self.searchpath = searchpath
        self.collection_fields = collection_fields

        path = Path(get_searchpath(self.type, searchpath))
        self.available_fields = set(
            i.stem.split("__")[1] for i in path.glob("*.jinja2")
        )

    @property
    def metadata(self) -> dict:
        """
        Retrieve the metadata.

        Returns:
            dict: The metadata.
        """
        return self._metadata

    @property
    def payload(self) -> dict:
        """
        Retrieve the payload.

        Returns:
            dict: The payload.
        """
        if self._payload is None:
            self._payload = self.get_payload()
        return self._payload

    @abstractmethod
    def get_payload(self) -> dict:
        """
        Retrieve the payload for the metadata that will be used for filtering.

        Returns:
            dict: The payload.
        """
        raise NotImplementedError("Subclasses must implement the get_payload method.")

    def build_metadata_langdoc(self, field: str) -> LangchainDocument:
        """
        Build a LangChain document for a specific field of the metadata.

        Args:
            field (str): The field to build the LangChain document for.

        Returns:
            LangchainDocument: A LangChain document.
        """
        try:
            page_content = self.embedding_content(field)
        except Exception as e:
            print(f"Error building metadata langdoc for {field}: {e}")
            return

        if not page_content:
            return None

        return LangchainDocument(
            page_content=page_content, metadata={"qfield": field, **self.payload}
        )

    def embedding_content(self, field) -> str:
        """
        Generate the embedding content from the metadata for a specific field.

        Returns:
            str: The content to be used for embedding.
        """
        assert field in self.collection_fields, (
            f"Field {field} not in collection fields {self.collection_fields}"
        )

        embedding_content = render_embedding_content(
            self.metadata, self.type, field, searchpath=self.searchpath
        )

        return embedding_content.strip()

    @abstractmethod
    def get_langdocs(self) -> list[LangchainDocument]:
        """
        Retrieve the LangChain documents for the metadata.

        This method must be implemented by subclasses to return content that will be indexed based on the metadata type.

        Returns:
            list: A list of LangChain documents.
        """
        raise NotImplementedError("Subclasses must implement the get_langdocs method.")


class IndicatorMetadata(Metadata):
    def __init__(self, **kwargs):
        super().__init__(
            metadata_type="indicator",
            collection_fields=[
                "name",
                "definition",
                "dimensions",
            ],  # , "relevance", "keywords"],
            **kwargs,
        )

    def get_payload(self) -> dict:
        """
        Retrieve the payload for the indicator metadata that will be used for filtering.

        Returns:
            dict: The payload.
        """

        payload = IndicatorFilterFacets.from_metadata(self.metadata)
        return payload.model_dump()

    def get_dimensions_langdocs(self) -> list[LangchainDocument]:
        """
        Retrieve the LangChain documents for the indicator dimensions. This uses the contextual reformulation model to generate a contextual description based on the dimensions.

        The script `scripts/synthetic_data/indicator/generate_contextual_dimensions.py` is used to generate the contextual dimensions.

        The output is stored in the `DATA_DIR/contextual_dimensions` which comes from the output of the script. The raw output is from `scripts/synthetic_data/output/indicator/generate_contextual_dimensions`.

        Returns:
            list: A list of LangChain documents.
        """
        langdocs = []

        raw_path = get_contextualized_dimensions_path(self.metadata["idno"], raw=True)
        if not raw_path.exists():
            return langdocs

        with open(raw_path) as f:
            data = json.load(f)

        langdocs.append(
            LangchainDocument(
                page_content=data["dimension_info"],
                metadata={
                    "qfield": "dimensions",
                    "doc_meta": {"type": "actual"},
                    **self.payload,
                },
            )
        )

        langdocs.append(
            LangchainDocument(
                page_content=data["output"],
                metadata={
                    "qfield": "dimensions",
                    "doc_meta": {"type": "contextual"},
                    **self.payload,
                },
            )
        )

        return langdocs

    def get_langdocs(self) -> list[LangchainDocument]:
        """
        Retrieve the LangChain documents for the indicator metadata.

        Returns:
            list: A list of LangChain documents.
        """

        langdocs = []

        # Add the metadata document
        for field in self.collection_fields:
            if field == "dimensions":
                langdocs.extend(self.get_dimensions_langdocs())
                continue

            if field not in self.available_fields:
                continue

            field_doc = self.build_metadata_langdoc(field)

            if field_doc is not None:
                langdocs.append(field_doc)

        return langdocs


class DocumentMetadata(Metadata):
    """
    Ideas for how we should work with documents:

    - Include the title in generating the embeddings,
        - If the title is retrieved as the most relevant for a query,
        - we do not use it directly as the input to an LLM. Instead, we use the title to retrieve the document.
        - We then use the most relevant section of the document as the input to the LLM together with the title.

    - Contents considered for the embeddings:
        - Title + Abstract
        - Keywords
        - Sections of the document
    """

    def __init__(self, **kwargs):
        super().__init__(
            metadata_type="document",
            collection_fields=[
                "title",
                "sub_title",
                "abstract",
                "passages",
                "keywords",
            ],
            **kwargs,
        )

    def get_payload(self) -> dict:
        """
        Retrieve the payload for the document metadata that will be used for filtering.

        Returns:
            dict: The payload.
        """

        payload = DocumentFilterFacets.from_metadata(self.metadata)
        return payload.model_dump()

    def get_doc_langdocs(self, field: str = "passages") -> list[LangchainDocument]:
        """
        Retrieve the LangChain documents for the document metadata.

        Returns:
            list: A list of LangChain documents.
        """

        # Check if the document url is in the metadata
        url = self.metadata.get("document_description", {}).get("url", None)

        # If external resources are available, prefer that over the document url
        external_resources = self.metadata.get("external_resources", None)
        if external_resources:
            for resource in external_resources:
                if (
                    resource.get("dcformat", None) == "application/pdf"
                    and resource.get("is_url", 1) == 0
                ):
                    url = resource.get("url", None)
                    # We only consider the first pdf url we find
                    # TODO: We should consider all pdf urls and use them all for the document
                    break

        docs = []

        if url:
            # Download the document pdf.
            # We only consider pdfs for now.
            pdf_path = cache_download_pdf(url, self.metadata.get("idno"), self.type)
            docs = load_pdf(pdf_path)

        return [
            LangchainDocument(
                page_content=doc.page_content,
                metadata={"qfield": field, "doc_meta": doc.metadata, **self.payload},
            )
            for doc in docs
        ]

    def get_langdocs(self) -> list[LangchainDocument]:
        """
        Retrieve the LangChain documents for the document metadata.

        Returns:
            list: A list of LangChain documents.
        """
        langdocs = []

        # Add the metadata document
        for field in self.collection_fields:
            if field == "passages":
                continue

            if field not in self.available_fields:
                continue

            field_doc = self.build_metadata_langdoc(field)

            if field_doc is not None:
                langdocs.append(field_doc)

        # Get the contents from the document itself
        doc_contents = self.get_doc_langdocs()
        langdocs.extend(doc_contents)

        return langdocs


class GeospatialMetadata(Metadata):
    def __init__(self, **kwargs):
        super().__init__(
            metadata_type="geospatial",
            collection_fields=["title", "abstract"],
            **kwargs,
        )

    def get_payload(self) -> dict:
        """
        Retrieve the payload for the geospatial metadata that will be used for filtering.

        Returns:
            dict: The payload.
        """

        payload = GeospatialFilterFacets.from_metadata(self.metadata)
        return payload.model_dump()

    def get_langdocs(self) -> list[LangchainDocument]:
        """
        Retrieve the LangChain documents for the geospatial metadata.

        Returns:
            list: A list of LangChain documents.
        """
        langdocs = []

        # Add the metadata document
        for field in self.collection_fields:
            if field not in self.available_fields:
                continue

            field_doc = self.build_metadata_langdoc(field)

            if field_doc is not None:
                langdocs.append(field_doc)

        return langdocs


class MicrodataMetadata(Metadata):
    def __init__(self, **kwargs):
        super().__init__(
            metadata_type="microdata",
            collection_fields=[
                "title",
                "sub_title",
                "abstract",
            ],  # "variable_groups", file_description", "sample_design", "sample_frame"],
            **kwargs,
        )

    def get_payload(self) -> dict:
        """
        Retrieve the payload for the microdata metadata that will be used for filtering.

        Returns:
            dict: The payload.
        """

        payload = MicrodataFilterFacets.from_metadata(self.metadata)
        return payload.model_dump()

    def get_langdocs(self) -> list[LangchainDocument]:
        """
        Retrieve the LangChain documents for the microdata metadata.

        Returns:
            list: A list of LangChain documents.
        """
        langdocs = []

        # Add the metadata document
        for field in self.collection_fields:
            if field not in self.available_fields:
                continue

            field_doc = self.build_metadata_langdoc(field)

            if field_doc is not None:
                langdocs.append(field_doc)

        return langdocs


class MetadataLoader:
    def __init__(
        self,
        idno: str,
        metadata_type: str,
        force: bool = False,
        searchpath: str = None,
        include_resources: bool = False,
    ):
        """
        Initialize the MetadataLoader.

        Args:
            idno (str): The identifier number for fetching metadata.
            metadata_type (str): Type of metadata (e.g., 'document', 'indicator').
            force (bool, optional): If True, forces fetching the metadata from the catalog even if cached. Defaults to False.
            searchpath (str, optional): Path used for searching or rendering templates. Defaults to None.
            include_resources (bool, optional): If True, include the resources in the metadata. Defaults to False.
        """
        self.idno = idno
        self.type = metadata_type
        self.force = force
        self.searchpath = searchpath
        self.include_resources = include_resources
        self.metadata = self.load_metadata()
        self.metadata["idno"] = self.idno

    def load_metadata(self) -> dict:
        """
        Load the metadata using the idno and metadata type.

        Returns:
            dict: The loaded metadata.
        """
        return get_metadata_json(
            self.idno,
            self.type,
            force=self.force,
            include_resources=self.include_resources,
        )

    def get_metadata_handler(self) -> Metadata:
        """
        Get the appropriate Metadata handler class based on the type of metadata.

        Returns:
            Metadata: An instance of the appropriate Metadata subclass.
        """
        if self.type == "indicator":
            return IndicatorMetadata(metadata=self.metadata, searchpath=self.searchpath)
        elif self.type == "document":
            return DocumentMetadata(metadata=self.metadata, searchpath=self.searchpath)
        elif self.type == "geospatial":
            return GeospatialMetadata(
                metadata=self.metadata, searchpath=self.searchpath
            )
        elif self.type == "microdata":
            return MicrodataMetadata(metadata=self.metadata, searchpath=self.searchpath)
        else:
            raise ValueError(f"Type {self.type} not supported")


def get_metadata_langdocs(
    idno: str, metadata_type: str, force: bool = False, searchpath: str = None
) -> list[LangchainDocument]:
    """
    Get the metadata documents for the given idno and metadata type.

    Args:
        idno (str): The identifier number.
        metadata_type (str): The type of metadata (e.g., 'indicator', 'document').
        force (bool, optional): If True, forces fetching the metadata from the catalog even if cached. Defaults to False.
        searchpath (str, optional): Path used for searching or rendering templates. Defaults to None.

    Returns:
        list: A list of LangChain documents.
    """
    loader = MetadataLoader(
        idno=idno, metadata_type=metadata_type, force=force, searchpath=searchpath
    )
    metadata_handler = loader.get_metadata_handler()
    return metadata_handler.get_langdocs()
