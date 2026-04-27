from __future__ import annotations

# This defines the facets for each metadata type. The facets are used to filter the metadata in the search.
from pydantic import BaseModel

from .parsers import DocumentParser, GeospatialParser, IndicatorParser, MicrodataParser
from .utils import create_uuid_from_string


class FilterFacets(BaseModel):
    type: str
    idno: str
    idno_uuid: str | None = None

    created_at: str | None = None
    updated_at: str | None = None

    year_start: int | None = None
    year_end: int | None = None
    years: list[int] | None = None

    geographies: list[str] | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self.idno_uuid = str(create_uuid_from_string(self.idno))

    @staticmethod
    def from_metadata(metadata: dict):
        """
        Create a FilterFacets object from metadata.

        Args:
            metadata (dict): The metadata.
        """
        raise NotImplementedError()


class IndicatorFilterFacets(FilterFacets):
    type: str = "indicator"
    periodicity: str = None
    source: list[str] = None

    @staticmethod
    def from_metadata(metadata: dict) -> IndicatorFilterFacets:
        """
        Create a FilterFacets object from metadata.

        Args:
            metadata (dict): The metadata.
        """
        parser = IndicatorParser()
        idno = parser.parse_idno(metadata)
        periods = parser.parse_periods(metadata, out_format="details")
        geographies = parser.parse_geographies(metadata)
        source = parser.parse_source(metadata)
        periodicity = parser.parse_periodicity(metadata)

        return IndicatorFilterFacets(
            idno=idno,
            year_start=periods.get("year_start"),
            year_end=periods.get("year_end"),
            years=periods.get("years"),
            geographies=geographies,
            source=source,
            periodicity=periodicity,
        )


class DocumentFilterFacets(FilterFacets):
    type: str = "document"
    document_type: str = None
    date_published: str = None
    date_created: str = None
    authors: list[str] = None

    @staticmethod
    def from_metadata(metadata: dict) -> DocumentFilterFacets:
        """
        Create a FilterFacets object from metadata.

        Args:
            metadata (dict): The metadata.
        """
        parser = DocumentParser()
        idno = parser.parse_idno(metadata)
        date_published = parser.parse_date_published(metadata)
        date_created = parser.parse_date_created(metadata)
        authors = parser.parse_authors(metadata)
        document_type = parser.parse_document_type(metadata)

        periods = parser.parse_periods(metadata, out_format="details")
        geographies = parser.parse_geographies(metadata)

        return DocumentFilterFacets(
            idno=idno,
            year_start=periods.get("year_start"),
            year_end=periods.get("year_end"),
            years=periods.get("years"),
            geographies=geographies,
            date_published=date_published,
            date_created=date_created,
            authors=authors,
            document_type=document_type,
        )


class GeospatialFilterFacets(FilterFacets):
    type: str = "geospatial"
    source: list[str] = None

    @staticmethod
    def from_metadata(metadata: dict) -> GeospatialFilterFacets:
        """
        Create a FilterFacets object from metadata.

        Args:
            metadata (dict): The metadata.
        """
        parser = GeospatialParser()
        idno = parser.parse_idno(metadata)
        geographies = parser.parse_geographies(metadata)
        source = parser.parse_source(metadata)
        periods = parser.parse_periods(metadata, out_format="details")

        return GeospatialFilterFacets(
            idno=idno,
            year_start=periods.get("year_start"),
            year_end=periods.get("year_end"),
            years=periods.get("years"),
            geographies=geographies,
            source=source,
        )


class MicrodataFilterFacets(FilterFacets):
    type: str = "microdata"
    source: list[str] | None = None

    @staticmethod
    def from_metadata(metadata: dict) -> MicrodataFilterFacets:
        """
        Create a FilterFacets object from metadata.

        Args:
            metadata (dict): The metadata.
        """

        parser = MicrodataParser()
        idno = parser.parse_idno(metadata)
        geographies = parser.parse_geographies(metadata)
        source = parser.parse_source(metadata)
        periods = parser.parse_periods(metadata, out_format="details")

        return MicrodataFilterFacets(
            idno=idno,
            year_start=periods.get("year_start"),
            year_end=periods.get("year_end"),
            years=periods.get("years"),
            geographies=geographies,
            source=source,
        )


def get_filter_facets(metadata: dict) -> FilterFacets:
    """
    Get the filter facets for the metadata.

    Args:
        metadata (dict): The metadata.

    Returns:
        FilterFacets: The filter facets.
    """
    metadata_type = metadata.get("type")

    if metadata_type == "indicator":
        return IndicatorFilterFacets.from_metadata(metadata)
    elif metadata_type == "document":
        return DocumentFilterFacets.from_metadata(metadata)
    elif metadata_type == "geospatial":
        return GeospatialFilterFacets.from_metadata(metadata)
    elif metadata_type == "microdata":
        return MicrodataFilterFacets.from_metadata(metadata)
    else:
        raise ValueError(f"Invalid metadata type: {metadata_type}")


# filter_facets = dict(
#     indicator=[
#         dict(
#             name="time_coverage",
#             title="Time Coverage",
#             path="series_description.time_periods",
#         ),
#         dict(
#             name="geographic_coverage",
#             title="Geographic Coverage",
#             path="series_description.ref_country"
#         ),
#         dict(
#             name="periodicity",
#             title="Periodicity",
#             path="series_description.periodicity"
#         ),
#         dict(
#             name="authoring_entity",
#             title="Source",
#             path="series_description.authoring_entity"
#         ),
#     ],
#     document=[
#         dict(
#             name="publication_date",
#             title="Publication Date",
#             path="document_description.date_published",
#         ),
#         dict(path="document_description.subject"),
#         dict(path="document_description.country"),
#     ],
#     microdata=[
#         dict(path="study_desc.subject"),
#         dict(path="study_desc.country"),
#     ],
#     geospatial=[
#         dict(path="description.subject"),
#         dict(path="description.country"),
#     ],
# )
