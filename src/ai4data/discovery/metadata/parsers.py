from __future__ import annotations

import pandas as pd

from .utils import get_idno


def date_parse(date: str) -> pd.Timestamp:
    """
    Parse the date string into a pandas Timestamp.
    Prefer using pandas to parse dates as it can handle a wider range of date formats, e.g., "2021-01-01", "2021/01/01", periodicities, etc.

    # ILOSTAT_POP_XWAP_SEX_AGE_NB_Q
    # 1948-Q1
    # 2024-Q2

    Args:
        date (str): The date string.

    Returns:
        pd.Timestamp: The parsed date.
    """
    return pd.to_datetime(date)


class Parser:
    metadata_type: str = None

    def parse_idno(self, metadata: dict) -> str:
        """
        Parse the idno from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed idno.
        """
        assert self.metadata_type, "metadata_type not defined"

        return get_idno(metadata, self.metadata_type)

    def parse_geographies(self, geographies: list[dict], source: str = "name") -> list[str]:
        """
        Parse the geographies formated as `[{"name": <geo_name>, "code": <geo_code>}, ...]` into a list of strings.

        Args:
            geographies (list[dict]): The list of geographies.
            source (str, optional): The source key to use. Defaults to "name".

        Returns:
            list[str]: The parsed geographies.
        """
        geographic_coverage = filter(lambda gc: gc.get(source), geographies)
        geographic_coverage = sorted(set([gc.get(source) for gc in geographic_coverage]))
        geographic_coverage = None if not geographic_coverage else geographic_coverage

        return geographic_coverage

    def parse_periods(self, periods: list[dict], out_format: str = "summary") -> str | dict:
        """
        Parse the periods into a single string.


        Args:
            periods (list[dict]): The list of periods.
            out_format (str, optional): The output format to use, options are "summary" or "details". Defaults to "summary".

        Returns:
            str: The parsed periods.
        """
        assert out_format in ["summary", "details"], f"Invalid out_format: {out_format}"

        coverage = set()

        for period in periods:
            start = period.get("start")
            if start:
                try:
                    start = date_parse(start).strftime("%Y")
                    coverage.add(start)
                except Exception as e:
                    Warning(f"Could not parse start date: {start} - {e}")

            end = period.get("end")
            if end:
                try:
                    end = date_parse(end).strftime("%Y")
                    coverage.add(end)
                except Exception as e:
                    Warning(f"Could not parse end date: {end} - {e}")

        coverage = sorted(coverage)

        if out_format == "summary":
            if len(coverage) == 1:
                return coverage[0]
            elif len(coverage) == 0:
                return None
            else:
                return " - ".join([coverage[0], coverage[-1]])
        else:
            details = dict(
                year_start=coverage[0] if coverage else None,
                year_end=coverage[-1] if coverage else None,
                years=coverage if coverage else None,
            )
            return details

    def parse_doi_from_identifiers(self, identifiers: list[dict[str, str]] | dict[str, str]) -> str | None:
        """
        Parse the DOI from the metadata identifiers.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed DOI.
        """
        doi: str | None = None

        for identifier in identifiers:
            if identifier.get("type", "").lower().strip() == "doi":
                doi = identifier.get("identifier")
                break

        return doi


class IndicatorParser(Parser):
    metadata_type: str = "indicator"

    def get_series(self, metadata: dict) -> dict:
        """
        Get the series from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            dict: The series.
        """
        return metadata.get("series_description", {})

    def parse_source(self, metadata: dict) -> list[str]:
        """
        Parse the source from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            list[str]: The parsed source.
        """
        series: dict = self.get_series(metadata)

        source: list[dict] = series.get("authoring_entity", [])
        source = sorted(s.get("name") for s in filter(lambda s: s.get("name"), source))

        return source

    def parse_geographies(self, metadata: dict) -> list[str]:
        """
        Parse the geographies from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            list[str]: The parsed geographies.
        """
        series: dict = self.get_series(metadata)

        geographies: list[dict] = series.get("ref_country", [])
        geographies = super().parse_geographies(geographies)

        return geographies

    def parse_periods(self, metadata: dict, out_format: str = "summary") -> str | dict:
        """
        Parse the periods from the metadata.

        Args:
            metadata (dict): The metadata.
            out_format (str, optional): The output format to use, options are "summary" or "details". Defaults to "summary

        Returns:
            str | dict: The parsed periods.
        """
        series: dict = self.get_series(metadata)

        periods: list[dict] = series.get("time_periods", [])
        periods = super().parse_periods(periods, out_format=out_format)

        return periods

    def parse_periodicity(self, metadata: dict) -> str:
        """
        Parse the periodicity from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed periodicity.
        """
        series: dict = self.get_series(metadata)

        periodicity: str = series.get("periodicity", None)

        return periodicity

    def parse_doi(self, metadata: dict) -> str | None:
        """
        Parse the DOI from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed DOI.
        """
        series: dict = self.get_series(metadata)

        return series.get("doi")

    def parse_dataset(self, metadata: dict) -> str | None:
        """
        Parse the dataset from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed dataset.
        """
        series: dict = self.get_series(metadata)

        dataset: str | None = series.get("database_id", None)

        return dataset

    def parse_definition(self, metadata: dict) -> str | None:
        """
        Parse the definition from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed definition.
        """
        series: dict = self.get_series(metadata)

        definition: str | None = series.get("definition_long", series.get("definition_short", None))

        return definition

    def parse_dimensions(self, metadata: dict) -> list[str]:
        """
        Parse the dimensions from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            list[str]: The parsed dimensions.
        """
        series: dict = self.get_series(metadata)

        dimensions: list[dict[str, str | list[dict[str, str | int]]]] = series.get("dimensions", [])

        dimensions = [dim.get("label") for dim in dimensions if dim.get("label")]

        return sorted(dimensions)

    def parse_dimensions_for_llm(self, metadata: dict, add_indicator_info: bool = True, limit: int = 10) -> str | None:
        """
        Parse the dimensions from the metadata.

        Args:
            metadata (dict): The metadata.
            add_indicator_info (bool, optional): Whether to include the indicator information [name and definition]. Defaults to True.
            limit (int, optional): The threshold for considering which dimensions to include. Dimensions with code list entries greater than 10 will not be included. Defaults to 10.

        Returns:
            str: The formatted parsed dimensions. If no dimensions are found, return None.
        """
        series: dict = self.get_series(metadata)

        dimensions: list[dict[str, str | list[dict[str, str | int]]]] = series.get("dimensions", [])

        if not dimensions:
            return None

        output = []

        if add_indicator_info:
            if series.get("name"):
                output.append(f"Indicator name: {series.get('name')}")

            definition = self.parse_definition(metadata)
            if definition:
                output.append(f"Definition: {definition}")

        for dim in dimensions:
            if dim.get("code_list") and len(dim.get("code_list")) > limit:
                continue

            dim_data = []
            if dim.get("label"):
                dim_data.append(f"Label: {dim.get('label')}")

            if dim.get("description"):
                dim_data.append(f"Description: {dim.get('description')}")

            code_list = dim.get("code_list", [])
            if code_list:
                dim_data.append("Breakdown:")
            for cl in code_list:
                dim_data.append(f"\t- {cl.get('label')}")

            output.append("\n".join(dim_data))

        return "\n\n".join(output)


class DocumentParser(Parser):
    metadata_type: str = "document"

    def get_document_description(self, metadata: dict) -> dict:
        """
        Get the document description from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            dict: The document description.
        """
        return metadata.get("document_description", {})

    def parse_document_type(self, metadata: dict) -> str:
        """
        Parse the document type from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed document type.
        """
        document_description = self.get_document_description(metadata)

        document_type: str = document_description.get("type", None)

        return document_type

    def parse_date_published(self, metadata: dict) -> str:
        """
        Parse the date published from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed date published.
        """
        document_description = self.get_document_description(metadata)

        date_published: str = document_description.get("date_published", None)

        date_published = date_parse(date_published).strftime("%Y-%m-%d") if date_published else None

        return date_published

    def parse_date_created(self, metadata: dict) -> str:
        """
        Parse the date created from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed date created.
        """
        document_description = self.get_document_description(metadata)

        date_created: str = document_description.get("date_created", None)

        date_created = date_parse(date_created).strftime("%Y-%m-%d") if date_created else None

        return date_created

    def parse_periods(self, metadata: dict, out_format: str = "summary") -> str | dict:
        """
        Extract the date_created from the metadata representing the time coverage given by the date_created field.

        Args:
            metadata (dict): The metadata information.

        Returns:
            str: The time coverage.
        """
        date_created = self.parse_date_created(metadata)

        periods = [dict(start=date_created, end=date_created)] if date_created else []

        return super().parse_periods(periods, out_format)

    def parse_geographies(self, metadata: dict) -> list[str]:
        """
        Extract the geographic coverage from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            list[str]: The geographic coverage.
        """
        document = self.get_document_description(metadata)

        geographies: list[dict] = document.get("ref_country", [])
        geographies = super().parse_geographies(geographies)

        return geographies

    def parse_authors(self, metadata: dict) -> list[str]:
        """
        Extract the unique authors from the list of "authors".

        Args:
            metadata (dict): The metadata information.

        Returns:
            list[str]: The unique authors.
        """
        document = self.get_document_description(metadata)
        authors: list[dict] = document.get("authors", [])
        author_names = []

        for author in authors:
            name: str = ""

            # Check first if the full_name is available
            if author.get("full_name"):
                name = author.get("full_name", "")
            else:
                if author.get("last_name"):
                    name += author.get("last_name")

                if author.get("first_name"):
                    name += "," + author.get("first_name")
                    name = name.lstrip(",")

            name = name.strip()

            if name and name not in author_names:
                author_names.append(name)

        return author_names

    def parse_abstract(self, metadata: dict) -> str | None:
        """
        Extract the abstract from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            str: The abstract.
        """
        document = self.get_document_description(metadata)
        abstract: str | None = document.get("abstract")

        return abstract

    def parse_doi(self, metadata: dict) -> str | None:
        """
        Extract the DOI from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            str: The DOI.
        """
        document = self.get_document_description(metadata)

        identifiers: list[dict[str, str]] = document.get("identifiers", [])

        return super().parse_doi_from_identifiers(identifiers)


class GeospatialParser(Parser):
    metadata_type: str = "geospatial"

    def parse_source(self, metadata: dict) -> list[str]:
        """
        Extract the unique authoring entities from the list of "citedResponsibleParty".

        Args:
            citation (dict): The citation information.

        Returns:
            list[str]: The unique authoring entities.
        """
        description: dict = metadata.get("description", {})
        identification_info: dict = description.get("identificationInfo", {})
        citation: dict = identification_info.get("citation", {})

        # TODO: Check if this is the correct way to extract the authoring entities
        responsible_party: list[dict] = citation.get("citedResponsibleParty", [])
        source = [crp.get("organisationName") for crp in responsible_party]
        source = list(filter(None, source))
        source = None if not source else source

        return source

    def parse_geographies(self, metadata: dict) -> list[str]:
        """
        Extract the geographic coverage from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            list[str]: The geographic coverage.
        """
        # TODO: Check if this is the correct way to extract the geographic coverage

        description: dict = metadata.get("description", {})
        identification_info: dict = description.get("identificationInfo", {})
        extent: dict = identification_info.get("extent", {})
        geographic_coverage: list[dict] = extent.get("geographicElement", [])
        geographies = [
            {"name": gc.get("geographicDescription")} for gc in geographic_coverage if gc.get("geographicDescription")
        ]

        geographies = super().parse_geographies(geographies)

        return geographies

    def parse_periods(self, metadata: dict, out_format: str = "summary") -> str | dict:
        """
        Process the time coverage from the citation.

        Args:
            metadata (dict): The metadata information.

        Returns:
            str: The time coverage.
        """
        # TODO: Check if this is the correct way to extract the time coverage

        description: dict = metadata.get("description", {})
        identification_info: dict = description.get("identificationInfo", {})
        citation: dict = identification_info.get("citation", {})
        date: list[dict] = citation.get("date", [])
        date = [d.get("date") for d in date if d.get("type") == "temporal coverage"]
        date = sorted(filter(None, date))

        periods = [{"start": d} for d in date]

        return super().parse_periods(periods, out_format)

    def parse_doi(self, metadata: dict) -> str | None:
        """
        Extract the DOI from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            str: The DOI.
        """
        description: dict = metadata.get("description", {})
        identification_info: dict = description.get("identificationInfo", {})
        citation: dict = identification_info.get("citation", {})
        identifier: dict[str, str] = citation.get("identifier", {})

        doi: str | None = None

        if identifier.get("code") and identifier.get("authority").lower().strip() == "doi":
            doi = identifier.get("code")

        return doi

    def parse_abstract(self, metadata: dict) -> str | None:
        """
        Extract the abstract from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            str: The abstract.
        """
        description: dict = metadata.get("description", {})
        identification_info: dict = description.get("identificationInfo", {})

        abstract: str | None = identification_info.get("abstract")

        return abstract


class MicrodataParser(Parser):
    metadata_type: str = "microdata"

    def get_study(self, metadata: dict) -> dict:
        """
        Get the microdata study description from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            dict: The microdata study description.
        """
        study: dict = metadata.get("study_description", {})
        if not study:
            study: dict = metadata.get("study_desc", {})

        return study

    def parse_sub_title(self, metadata: dict) -> str | None:
        """
        Parse the sub title from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed sub title.
        """
        study = self.get_study(metadata)
        title_statement: dict = study.get("title_statement", {})

        return title_statement.get("sub_title")

    def parse_periods(self, metadata: dict, out_format: str = "summary") -> str | dict:
        """
        Parse the period from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed period.
        """
        study = self.get_study(metadata)

        # TODO: Does it make sense to take the `time_periods` first, and if not available, take the `coll_dates` as a fallback?
        # TODO: How do we handle if there are multiple periods?

        study_info: dict = study.get("study_info", {})

        # Check time_periods first

        periods: list[dict] = study_info.get("time_periods", None)
        if not periods:
            periods = study_info.get("coll_dates", [])

        return super().parse_periods(periods, out_format)

    def parse_geographies(self, metadata: dict) -> list[str]:
        """
        Extract the geographic coverage from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            list[str]: The geographic coverage.
        """
        study = self.get_study(metadata)
        study_info: dict = study.get("study_info", {})

        geographies: list[dict] = study_info.get("nation", [])

        return super().parse_geographies(geographies)

    def parse_source(self, metadata: dict) -> list[str] | None:
        """
        Extract the unique authoring entities from the list of "authoring_entity".

        Args:
            study (dict): The study information.

        Returns:
            list[str]: The unique authoring entities.
        """
        study = self.get_study(metadata)
        authoring_entities: list[dict] = study.get("authoring_entity", [])
        authoring_entities = filter(lambda ae: ae.get("name"), authoring_entities)
        source = sorted(set([ae.get("name") for ae in authoring_entities]))
        source = None if not source else source

        return source

    def parse_doi(self, metadata: dict) -> str | None:
        """
        Parse the DOI from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed DOI.
        """
        study = self.get_study(metadata)
        title_statement: dict = study.get("title_statement", {})
        identifiers: list[dict] = title_statement.get("identifiers", [])

        return super().parse_doi_from_identifiers(identifiers)

    def parse_abstract(self, metadata: dict) -> str | None:
        """
        Parse the abstract from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed abstract.
        """
        study = self.get_study(metadata)
        study_info: dict = study.get("study_info", {})
        abstract: str | None = study_info.get("abstract")

        return abstract

    def parse_access_policy(self, metadata: dict) -> str | None:
        """
        Parse the access policy from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed access policy.
        """
        access_policy: str | None = metadata.get("access_policy", None)

        return access_policy


class ScriptParser(Parser):
    metadata_type: str = "script"

    def get_project(self, metadata: dict) -> dict:
        """
        Get the script project description from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            dict: The script project description.
        """
        return metadata.get("project_desc", {})

    def parse_source(self, metadata: dict) -> list[str] | None:
        """
        Extract the unique authoring entities from the list of "authoring_entity".

        Args:
            project (dict): The project information.

        Returns:
            list[str]: The unique authoring entities.
        """
        project = self.get_project(metadata)

        authoring_entities: list[dict] = project.get("authoring_entity", [])
        authoring_entities = filter(lambda ae: ae.get("name"), authoring_entities)
        source: list[str] | None = sorted(set([ae.get("name") for ae in authoring_entities]))
        source = None if not source else source

        return source

    def parse_geographies(self, metadata: dict) -> list[str] | None:
        """
        Extract the geographic coverage from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            list[str]: The geographic coverage.
        """
        project = self.get_project(metadata)
        geographies: list[dict] = project.get("geographic_units", [])

        return super().parse_geographies(geographies)

    def parse_periods(self, metadata: dict, out_format: str = "summary") -> str | dict:
        """
        Parse the period from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed period.
        """
        project = self.get_project(metadata)
        production_date = project.get("production_date", [])
        periods = [{"start": date} for date in production_date]

        return super().parse_periods(periods, out_format)

    def parse_language(self, metadata: dict) -> str | None:
        """
        Extract the programming language from the metadata.

        Args:
            metadata (dict): The metadata information.

        Returns:
            str: The programming language.
        """
        project = self.get_project(metadata)

        software: list[dict] = project.get("software", [])
        software = filter(lambda s: s.get("name"), software)
        software = sorted(set([s.get("name") for s in software]))

        software = None if not software else ", ".join(software)

        return software

    def parse_doi(self, metadata: dict) -> str | None:
        """
        Parse the DOI from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed DOI.
        """
        project = self.get_project(metadata)
        title_statement: dict = project.get("title_statement", {})

        identifiers: list[dict] = title_statement.get("identifiers", [])

        return super().parse_doi_from_identifiers(identifiers)

    def parse_abstract(self, metadata: dict) -> str | None:
        """
        Parse the abstract from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed abstract.
        """
        project = self.get_project(metadata)

        abstract: str | None = project.get("abstract")

        return abstract

    def parse_github(self, metadata: dict) -> str | None:
        """
        Parse the github from the metadata.

        Args:
            metadata (dict): The metadata.

        Returns:
            str: The parsed github.
        """
        project = self.get_project(metadata)
        repository: list[dict[str, str]] = project.get("repository_uri", {})

        github: str | None = None

        for repo in repository:
            if repo.get("type", "").lower().strip() == "github":
                github = repo.get("uri")
                break

        return github
