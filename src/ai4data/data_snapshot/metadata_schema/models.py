"""Define the canonical Data Snapshot Metadata Schema v1.4 models."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    WithJsonSchema,
    field_validator,
    model_validator,
)


def _normalize_text(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value).strip()
    return value


NonEmptyText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, pattern=r"\S"),
    BeforeValidator(_normalize_text),
]

# RFC 5646 section 2.1; registry membership remains a separate concern.
_GRANDFATHERED_TAGS = (
    "en-GB-oed",
    "i-ami",
    "i-bnn",
    "i-default",
    "i-enochian",
    "i-hak",
    "i-klingon",
    "i-lux",
    "i-mingo",
    "i-navajo",
    "i-pwn",
    "i-tao",
    "i-tay",
    "i-tsu",
    "sgn-BE-FR",
    "sgn-BE-NL",
    "sgn-CH-DE",
    "art-lojban",
    "cel-gaulish",
    "no-bok",
    "no-nyn",
    "zh-guoyu",
    "zh-hakka",
    "zh-min",
    "zh-min-nan",
    "zh-xiang",
)
_BCP47_PATTERN = re.compile(
    r"^(?:(?:[A-Za-z]{2,3}(?:-[A-Za-z]{3}){0,3}|[A-Za-z]{4}|[A-Za-z]{5,8})"
    r"(?:-[A-Za-z]{4})?(?:-(?:[A-Za-z]{2}|[0-9]{3}))?"
    r"(?:-(?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3}))*"
    r"(?:-[0-9A-WY-Za-wy-z](?:-[A-Za-z0-9]{2,8})+)*"
    r"(?:-[xX](?:-[A-Za-z0-9]{1,8})+)?|[xX](?:-[A-Za-z0-9]{1,8})+|"
    + "|".join(_GRANDFATHERED_TAGS)
    + r")$"
)
_YEAR_PATTERN = re.compile(r"^[0-9]{4}$")
_MONTH_PATTERN = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])$")
_DAY_PATTERN = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])$")
_DATETIME_PATTERN = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]"
    r"(?::[0-5][0-9](?:[.,][0-9]+)?)?(?:Z|[+-][0-9]{2}:[0-5][0-9])$"
)


# RFC 3986 Appendix A. AnyUrl checks IP literals after this lexical check.
_URI_ATOM = r"(?:[A-Za-z0-9._~!$&'()*+,;=-]|%[0-9A-Fa-f]{2})"
_URI_PCHAR = rf"(?:{_URI_ATOM}|[:@])"
_URI_AUTHORITY = rf"(?:{_URI_ATOM}|:)*@"
_URI_PATTERN = re.compile(
    rf"^[A-Za-z][A-Za-z0-9+.-]*:"
    rf"(?://(?:{_URI_AUTHORITY})?(?:{_URI_ATOM}*|\[[A-Za-z0-9:.!$&'()*+,;=_~-]+\])"
    rf"(?::[0-9]*)?(?:/{_URI_PCHAR}*)*"
    rf"|/(?:{_URI_PCHAR}+(?:/{_URI_PCHAR}*)*)?"
    rf"|{_URI_PCHAR}+(?:/{_URI_PCHAR}*)*|)"
    rf"(?:\?(?:{_URI_PCHAR}|[/?])*)?(?:#(?:{_URI_PCHAR}|[/?])*)?$"
)


def _validate_uri(value: Any) -> Any:
    # Check the original spelling before AnyUrl can strip controls or escape spaces.
    if not isinstance(value, (str, AnyUrl)):
        raise ValueError("URI must be a string or a validated URL.")
    if isinstance(value, str) and not _URI_PATTERN.fullmatch(value):
        raise ValueError("URI must use absolute RFC 3986 syntax and valid escapes.")
    return value


AbsoluteURI = Annotated[
    AnyUrl,
    BeforeValidator(_validate_uri),
    Field(json_schema_extra={"pattern": _URI_PATTERN.pattern}),
]


def _populated(*names: str) -> dict[str, Any]:
    return {
        "required": list(names),
        "properties": {name: {"not": {"type": "null"}} for name in names},
    }


def _content_schema(*names: str) -> dict[str, Any]:
    return {
        "anyOf": [_populated(name) for name in names],
        "x-validation-rules": [
            "At least one non-null value is required: " + ", ".join(names) + "."
        ],
    }


def _variable_schema(schema: dict[str, Any]) -> None:
    schema.update(_content_schema("name", "unit", "currency", "statistical_forms"))
    schema["allOf"] = [
        {
            "if": {
                "anyOf": [
                    _populated("analytical_roles"),
                    _populated("axis_roles"),
                    _populated("multi_axis_assignments"),
                ]
            },
            "then": _populated("name"),
        }
    ]
    schema["x-validation-rules"].append(
        "Analytical roles, axis roles, and multi-axis assignments require a "
        "non-null variable name."
    )


def _temporal_schema(schema: dict[str, Any]) -> None:
    schema["anyOf"] = [
        _populated("source_text"),
        _populated("start"),
        _populated("end"),
    ]
    schema["allOf"] = [
        {
            "if": {"anyOf": [_populated("start"), _populated("end")]},
            "then": _populated("relation", "precision"),
            "else": {
                "properties": {
                    name: {"type": "null"} for name in ("relation", "precision")
                }
            },
        }
    ]
    for relation, alternatives in {
        "point": [("start",)],
        "interval": [("start", "end")],
        "open_ended_interval": [("start",), ("end",)],
    }.items():
        branches = []
        for names in alternatives:
            branch = _populated(*names)
            branch["properties"].update(
                {
                    name: {"type": "null"}
                    for name in ("start", "end")
                    if name not in names
                }
            )
            branches.append(branch)
        schema["allOf"].append(
            {
                "if": {
                    "required": ["relation"],
                    "properties": {"relation": {"const": relation}},
                },
                "then": {"anyOf": branches},
            }
        )
    for precision, pattern in {
        "year": _YEAR_PATTERN,
        "month": _MONTH_PATTERN,
        "day": _DAY_PATTERN,
        "datetime": _DATETIME_PATTERN,
    }.items():
        schema["allOf"].append(
            {
                "if": {
                    "required": ["precision"],
                    "properties": {"precision": {"const": precision}},
                },
                "then": {
                    "properties": {
                        name: {"pattern": pattern.pattern} for name in ("start", "end")
                    }
                },
            }
        )
    schema["x-validation-rules"] = [
        "At least one of source_text, start, or end must be non-null.",
        "Normalized bounds require relation and precision; source-only expressions omit both.",
        "point requires start only; interval requires both bounds; open_ended_interval requires exactly one bound.",
        "Bounds must match the declared precision. Python additionally validates calendar dates and chronological ordering.",
    ]


def _standards(*mappings: tuple[str, str]) -> dict[str, object]:
    return {
        "x-standards": [
            {"term": term, "relationship": relationship}
            for term, relationship in mappings
        ]
    }


def _code_list(
    authority: str, identifier: str, uri: str, release: str | None = None
) -> dict[str, object]:
    metadata: dict[str, str] = {
        "authority": authority,
        "identifier": identifier,
        "uri": uri,
    }
    if release is not None:
        metadata["release"] = release
    return {"x-code-list": metadata}


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _remove_exact_duplicates(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        unique: list[Any] = []
        for item in value:
            if item not in unique:
                unique.append(item)
        return unique


class PresentationRole(str, Enum):
    """Identify how a dimension organizes a table.

    `row` means the dimension's categories vary downward through table rows.
    `column` means they vary horizontally across table columns. The physical
    placement of a label in the first column does not by itself make its
    dimension a column dimension.
    """

    ROW = "row"
    COLUMN = "column"


class AnalyticalRole(str, Enum):
    """Identify an explicit role in a statistical analysis.

    `outcome` is the dependent or response variable the analysis seeks to
    explain or estimate. `predictor` is an explanatory or independent variable
    used to explain or predict an outcome. `control` is a covariate included to
    adjust an estimate rather than serving as the primary predictor.
    `instrumental` is a variable explicitly identified as an instrument in an
    instrumental-variable analysis. Assign these roles only when the analytical
    table or its labels, title, notes, or caption support them; do not infer them
    merely from table position.
    """

    OUTCOME = "outcome"
    PREDICTOR = "predictor"
    INSTRUMENTAL = "instrumental"
    CONTROL = "control"


class AxisRole(str, Enum):
    """Identify a variable's ordinary Cartesian axis in a figure.

    `x_axis` means the variable is encoded on the horizontal axis. `y_axis`
    means it is encoded on the vertical axis. Use `MultiAxisAssignment` instead
    when a graph contains distinct axes of the same dimension and the side or
    outward position is needed to distinguish them.
    """

    X_AXIS = "x_axis"
    Y_AXIS = "y_axis"


class AxisDimension(str, Enum):
    """Identify a Cartesian axis dimension."""

    X = "x"
    Y = "y"


class AxisPosition(str, Enum):
    """Identify the side of a plot where an axis appears."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


# Developer note: This application-owned relation vocabulary uses the
# open-start and open-end semantics described by EDTF for its open-ended value.
# Reference: https://www.loc.gov/standards/datetime/
class TemporalRelation(str, Enum):
    """Describe how normalized temporal bounds form an expression.

    `point` represents a single time point and requires `start` only.
    `interval` represents a bounded period and requires `start` and `end`.
    `open_ended_interval` represents a period for which the snapshot
    explicitly supports only one temporal boundary.
    """

    POINT = "point"
    INTERVAL = "interval"
    OPEN_ENDED_INTERVAL = "open_ended_interval"


# Developer note: These application-owned precision levels correspond to the
# ISO 8601 forms selected by the normalization profile.
# Reference: https://www.iso.org/iso-8601-date-and-time-format.html
class TemporalPrecision(str, Enum):
    """Describe the precision of normalized temporal bounds."""

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    DATETIME = "datetime"


# Developer note: This application vocabulary combines SDMX statistical
# operations, DDI summary-statistic types, quantile forms, and forms retained
# from schema evidence. It is not a verbatim copy of either external vocabulary.
# References:
# - SDMX CL_STATISTICAL_OPERATION:
#   https://sdmx.org/wp-content/uploads/CL_STATISTICAL_OPERATION-1.0.docx
# - DDI Summary Statistic Type 2.1:
#   https://rdf-vocabulary.ddialliance.org/ddi-cv/SummaryStatisticType/2.1.0/SummaryStatisticType.html
class StatisticalFormValue(str, Enum):
    """Enumerate approved normalized statistical forms."""

    OBSERVED_VALUE = "observed_value"
    COUNT = "count"
    ARITHMETIC_MEAN = "arithmetic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    WEIGHTED_MEAN = "weighted_mean"
    WEIGHTED_VALUE = "weighted_value"
    MEDIAN = "median"
    SUM = "sum"
    WEIGHTED_SUM = "weighted_sum"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    RANGE = "range"
    INTERQUARTILE_RANGE = "interquartile_range"
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"
    STANDARD_ERROR = "standard_error"
    MEAN_ABSOLUTE_DEVIATION = "mean_absolute_deviation"
    MEDIAN_ABSOLUTE_DEVIATION = "median_absolute_deviation"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    SCORE = "score"
    RANK = "rank"
    INDEX = "index"
    RATE = "rate"
    HARMONIC_MEAN = "harmonic_mean"
    TRIMMED_MEAN = "trimmed_mean"
    MODE = "mode"
    COEFFICIENT_OF_VARIATION = "coefficient_of_variation"
    VALID_CASES = "valid_cases"
    INVALID_CASES = "invalid_cases"
    PERCENTAGE_OF_VALID_CASES = "percentage_of_valid_cases"
    PERCENTAGE_OF_INVALID_CASES = "percentage_of_invalid_cases"
    QUARTILE = "quartile"
    QUINTILE = "quintile"
    DECILE = "decile"
    PERCENTILE = "percentile"
    RATIO = "ratio"
    PROPORTION = "proportion"
    PERCENTAGE = "percentage"
    CONFIDENCE_INTERVAL = "confidence_interval"
    COEFFICIENT = "coefficient"
    P_VALUE = "p_value"
    CHANGE = "change"


# Developer note: This application-owned vocabulary is grounded where possible
# in Vega-Lite marks and extended with visualization types observed in schema
# evidence.
# Reference: https://vega.github.io/vega-lite/docs/mark.html
class VisualizationTypeValue(str, Enum):
    """Enumerate approved normalized visualization types."""

    TABLE = "table"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    AREA_CHART = "area_chart"
    PIE_CHART = "pie_chart"
    DONUT_CHART = "donut_chart"
    SCATTER_PLOT = "scatter_plot"
    BUBBLE_CHART = "bubble_chart"
    DOT_PLOT = "dot_plot"
    HISTOGRAM = "histogram"
    DENSITY_PLOT = "density_plot"
    HEATMAP = "heatmap"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    ERROR_BAR = "error_bar"
    ERROR_BAND = "error_band"
    RADAR_CHART = "radar_chart"
    TREEMAP = "treemap"
    MAP = "map"
    CHOROPLETH_MAP = "choropleth_map"
    SYMBOL_MAP = "symbol_map"
    NETWORK_DIAGRAM = "network_diagram"
    FLOW_DIAGRAM = "flow_diagram"
    TIMELINE = "timeline"
    DIAGRAM = "diagram"
    INFOGRAPHIC = "infographic"
    DASHBOARD = "dashboard"
    COMPOSITE_FIGURE = "composite_figure"


# Developer note: This interval vocabulary aligns regular frequencies with
# SDMX CL_FREQ.
# Reference: https://sdmx.org/sdmx_cdcl/
class ReportingIntervalValue(str, Enum):
    """Enumerate approved normalized reporting intervals."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"


# Developer note: This application-owned vocabulary implements the geographic
# level design informed by DDI geographic levels and ISO 3166-2 subdivisions.
# References:
# - DDI Lifecycle 3.3 GeographicLevel:
#   https://docs.ddialliance.org/DDI-Lifecycle/3.3/
# - ISO 3166:
#   https://www.iso.org/iso-3166-country-codes.html
class GeographicLevelValue(str, Enum):
    """Enumerate approved normalized geographic levels."""

    GLOBAL = "global"
    WORLD_REGION = "world_region"
    COUNTRY = "country"
    ADMINISTRATIVE_AREA_1 = "administrative_area_1"
    ADMINISTRATIVE_AREA_2 = "administrative_area_2"
    ADMINISTRATIVE_AREA_3 = "administrative_area_3"
    LOCALITY = "locality"
    SITE = "site"


class Identifier(_SchemaModel):
    """Represent a verified identifier and its optional authority context.

    Parameters
    ----------
    value : str
        Identifier exactly as assigned in source evidence or trusted metadata.
    scheme : str | None
        Verified identifier scheme, when available.
    issuer : str | None
        Verified issuing agent, when available.
    uri : AnyUrl | None
        Verified authoritative absolute URI for the identifier.
    """

    value: NonEmptyText = Field(
        examples=["P171254", "P178944"],
        description="Identifier exactly as assigned in the snapshot or trusted metadata. Do not infer or manufacture it from its apparent pattern or from model knowledge.",
    )
    scheme: NonEmptyText | None = Field(
        examples=["World Bank project ID"],
        default=None,
        description="Identifier scheme explicitly supplied by the snapshot, trusted metadata, or configured and verified enrichment. Do not infer it from the identifier's apparent pattern or from model knowledge.",
    )
    issuer: NonEmptyText | None = Field(
        examples=["World Bank"],
        default=None,
        description="Issuing agent explicitly supplied by the snapshot, trusted metadata, or configured and verified enrichment. Do not infer it from model knowledge.",
    )
    uri: AbsoluteURI | None = Field(
        examples=["https://example.org/projects/P171254"],
        default=None,
        description="Authoritative absolute URI supplied by the snapshot, trusted metadata, or configured and verified enrichment. Do not construct or infer it from model knowledge.",
    )


class CodedTerm(_SchemaModel):
    """Represent source wording with optional exact vocabulary identifiers.

    Parameters
    ----------
    source_text : str | None
        Exact source-visible text or symbol that expresses the term.
    code : str | None
        Code in the named scheme.
    scheme : str | None
        Code-list or vocabulary identifier.
    uri : AnyUrl | None
        Authoritative URI for the represented concept.
    """

    model_config = ConfigDict(
        json_schema_extra={
            **_content_schema("source_text", "code", "uri"),
            "allOf": [
                {"if": _populated("code"), "then": _populated("scheme")},
                {"if": _populated("scheme"), "then": _populated("code")},
            ],
            "x-validation-rules": [
                "At least one of source_text, code, or uri must be non-null.",
                "Code and scheme must be non-null together.",
            ],
        }
    )

    source_text: NonEmptyText | None = Field(
        examples=["Grant", "District"],
        default=None,
        description="Exact text or symbol visible in the snapshot that explicitly expresses the value represented by this object. Omit it when the value is inferred from visual form, structure, or context rather than transcribed.",
    )
    code: NonEmptyText | None = Field(
        examples=["110"], default=None, description="Code in the named scheme."
    )
    scheme: NonEmptyText | None = Field(
        examples=["IATI Finance Type"],
        default=None,
        description="Code-list or vocabulary identifier.",
    )
    uri: AbsoluteURI | None = Field(
        examples=["https://example.org/concepts/TERM-001"],
        default=None,
        description="Authoritative URI for the represented concept.",
    )

    @model_validator(mode="after")
    def _validate_term(self) -> CodedTerm:
        if not any((self.source_text, self.code, self.uri)):
            raise ValueError("A coded term must contain source text, a code, or a URI.")
        if (self.code is None) != (self.scheme is None):
            raise ValueError("A coded-term code and scheme must be supplied together.")
        return self


class _NormalizedTerm(_SchemaModel):
    """Validate the shared content rule for normalized term models."""

    model_config = ConfigDict(
        json_schema_extra=_content_schema("source_text", "normalized_value")
    )

    source_text: NonEmptyText | None = None
    normalized_value: NonEmptyText | None = None

    @model_validator(mode="after")
    def _validate_term(self) -> _NormalizedTerm:
        if self.source_text is None and self.normalized_value is None:
            raise ValueError("A normalized term must contain source text or a value.")
        return self


class StatisticalFormTerm(_NormalizedTerm):
    """Represent a known or source-only statistical form.

    Parameters
    ----------
    source_text : str | None
        Exact source-visible expression of the statistical form.
    normalized_value : StatisticalFormValue | None
        Approved normalized statistical form.
    """

    source_text: NonEmptyText | None = Field(
        default=None,
        description="Exact text visible in the snapshot that explicitly expresses the mathematical or statistical form of the values, such as count, percent of total, growth rate, confidence interval, index, sum, or mean. When surrounding source wording supports a form, preserve that wording here. A `%` that specifies the values' measurement scale belongs in the variable's `unit`; a `%` that belongs only to an expression such as `95% CI` is part of the statistical-form wording and is not a percent unit.",
        examples=[
            "Count",
            "Percent of total",
            "Growth rate",
            "95% CI",
            "Index",
            "Average",
        ],
    )
    normalized_value: StatisticalFormValue | None = Field(
        examples=[
            "count",
            "percentage",
            "rate",
            "change",
            "confidence_interval",
            "arithmetic_mean",
        ],
        default=None,
        description="Approved mathematical or statistical form of the values. Prefer explicit source terminology. Use `percentage` for a percentage, share, or proportion per hundred; `rate` for a quantity presented as a rate relative to a base, including incidence, growth, interest, or tariff rates; and `change` for a difference, movement, or return. Do not select a form solely because the unit is `%`.",
    )


class VisualizationTypeTerm(_NormalizedTerm):
    """Represent a known or source-only visualization type.

    Parameters
    ----------
    source_text : str | None
        Exact source-visible name of the visualization type.
    normalized_value : VisualizationTypeValue | None
        Approved normalized visualization type.
    """

    source_text: NonEmptyText | None = Field(
        default=None,
        description="Exact text visible in the snapshot that explicitly names the visualization type.",
        examples=[
            "Bar chart",
            "Line chart",
            "Table",
            "Map",
            "Heatmap",
            "Composite figure: line charts and map",
        ],
    )
    normalized_value: VisualizationTypeValue | None = Field(
        examples=["bar_chart", "table", "composite_figure"],
        default=None,
        description="Approved normalized visualization type.",
    )


class ReportingIntervalTerm(_NormalizedTerm):
    """Represent a known or source-only reporting interval.

    Parameters
    ----------
    source_text : str | None
        Exact source-visible expression of the reporting interval.
    normalized_value : ReportingIntervalValue | None
        Approved normalized reporting interval.
    """

    source_text: NonEmptyText | None = Field(
        default=None,
        description="Exact text visible in the snapshot that explicitly states the reporting interval.",
        examples=["Annual", "Monthly", "Quarterly", "Daily"],
    )
    normalized_value: ReportingIntervalValue | None = Field(
        examples=["annual", "monthly", "quarterly"],
        default=None,
        description="Approved normalized reporting interval.",
    )


class GeographicLevelTerm(_NormalizedTerm):
    """Represent a known or source-only geographic reporting level.

    Parameters
    ----------
    source_text : str | None
        Exact source-visible expression of the geographic level.
    normalized_value : GeographicLevelValue | None
        Approved normalized geographic level.
    """

    source_text: NonEmptyText | None = Field(
        default=None,
        description="Exact text visible in the snapshot that explicitly states the geographic level.",
        examples=["Country", "Province", "District", "Facility"],
    )
    normalized_value: GeographicLevelValue | None = Field(
        examples=["country", "administrative_area_1", "site"],
        default=None,
        description="Approved normalized geographic level.",
    )


class EntityReference(_SchemaModel):
    """Represent a named project, organization, source, or component.

    Parameters
    ----------
    name : str
        Source-visible entity name.
    identifiers : list[Identifier] | None
        Assigned identifiers for the entity.
    """

    name: NonEmptyText = Field(
        examples=[
            "World Development Indicators",
            "DHS",
            "UNHCR Registration Data",
            "National Census",
            "Map Design Unit",
        ],
        description="Source-visible entity name.",
    )
    identifiers: list[Identifier] | None = Field(
        examples=[[{"value": "ENTITY-001", "scheme": "https://example.org/entities"}]],
        default=None,
        min_length=1,
        description="Assigned entity identifiers.",
    )


class Attribution(EntityReference):
    """Represent a named agent and its optional explicit attribution role.

    Parameters
    ----------
    name : str
        Source-visible credited-agent name.
    identifiers : list[Identifier] | None
        Assigned identifiers for the agent.
    role : CodedTerm | None
        Open, source-grounded attribution role, when explicit.
    """

    name: NonEmptyText = Field(
        description="Source-visible entity name.",
        examples=["Map Design Unit", "National Statistics Office"],
    )
    role: CodedTerm | None = Field(
        examples=[{"source_text": "Map maker"}, {"source_text": "Producer"}],
        default=None,
        description="Explicit source-grounded agent role.",
    )


class Unit(_SchemaModel):
    """Represent a displayed unit and its optional normalized qualifiers.

    Parameters
    ----------
    source_text : str
        Complete displayed unit or scale expression.
    code : str | None
        Exact UN/CEFACT Recommendation 20 code.
    multiplier_exponent : int | None
        SDMX unit-multiplier exponent.
    """

    source_text: NonEmptyText = Field(
        examples=[
            "%",
            "People",
            "Kilometers",
            "US$ millions",
            "US$000",
            "US$m/SDRm",
        ],
        description="Exact complete text or symbol visible in the snapshot that states the unit or scale in which the variable's quantitative values are expressed, such as `%`, people, kilometers, `US$ millions`, or `US$000`. Preserve the complete measurement expression, excluding the variable name and annotations that do not specify the values' scale. For example, the `%` in `95% CI` states a confidence level and is not a percent unit. When the expression includes one supported applicable currency, repeat it in `currency`; when it combines multiple currencies or no single currency is supported, preserve the expression here and leave `currency` unset. Record a supported magnitude in `multiplier_exponent`. This intentional repetition preserves source wording while keeping qualifiers queryable. A unit does not determine the statistical form: `%` may accompany a percentage, rate, change, coefficient, or no normalized form.",
    )
    code: (
        Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Z0-9]{1,3}$")]
        | None
    ) = Field(
        examples=["P1", "KMT"],
        default=None,
        description="Exact UN/CEFACT Recommendation 20 common code.",
        json_schema_extra=_code_list(
            "UNECE",
            "Recommendation 20",
            "https://unece.org/trade/uncefact/cl-recommendations",
        ),
    )
    multiplier_exponent: Annotated[int, Field(strict=True)] | None = Field(
        examples=[3, 6, 9],
        default=None,
        description="Base-10 SDMX unit-multiplier exponent supported by the displayed scale, such as `3` for thousand or `6` for million.",
        json_schema_extra=_code_list(
            "SDMX",
            "CL_UNIT_MULT",
            "https://registry.sdmx.org/items/codelist.html",
            "1.1",
        ),
    )


class Currency(_SchemaModel):
    """Represent a displayed currency and optional ISO 4217 code.

    Parameters
    ----------
    source_text : str
        Displayed currency expression.
    code : str | None
        Uppercase ISO 4217 alphabetic code.
    """

    source_text: NonEmptyText = Field(
        examples=["US$", "$", "EUR", "JPY"],
        description="Exact currency name, abbreviation, or symbol visible in the snapshot. When one applicable currency is supported, record it even when the same token also appears inside the complete `unit.source_text`; the repetition is intentional. If the unit expression combines multiple currencies or an ambiguous symbol cannot be resolved from snapshot evidence, preserve the complete unit expression and leave `currency` unset rather than forcing one value. Do not include noncurrency scale wording such as thousand or million.",
    )
    code: (
        Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Z]{3}$")] | None
    ) = Field(
        examples=["USD", "EUR", "JPY"],
        default=None,
        description="Uppercase ISO 4217 alphabetic code. Populate only when snapshot evidence deterministically identifies the currency; do not assume `USD` from an ambiguous `$` symbol alone.",
        json_schema_extra=_code_list(
            "ISO", "ISO 4217", "https://www.iso.org/iso-4217-currency-codes.html"
        ),
    )


class Language(_SchemaModel):
    """Represent a snapshot language using source text and/or BCP 47.

    Parameters
    ----------
    source_text : str | None
        Displayed language label, when present.
    tag : str | None
        Canonical BCP 47 language tag.
    """

    model_config = ConfigDict(json_schema_extra=_content_schema("source_text", "tag"))

    source_text: NonEmptyText | None = Field(
        examples=["English", "French", "Arabic"],
        default=None,
        description="Exact text visible in the snapshot that explicitly names the language. Omit it when the language is inferred from the snapshot content.",
    )
    tag: (
        Annotated[
            NonEmptyText,
            WithJsonSchema(
                {"type": "string", "minLength": 1, "pattern": _BCP47_PATTERN.pattern}
            ),
        ]
        | None
    ) = Field(
        examples=["en", "fr", "ar"],
        default=None,
        description="Canonical BCP 47 language tag.",
        json_schema_extra=_code_list(
            "IETF/IANA",
            "BCP 47",
            "https://www.iana.org/assignments/language-subtag-registry/",
        ),
    )

    @field_validator("tag")
    @classmethod
    def _validate_tag(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _BCP47_PATTERN.fullmatch(value):
            raise ValueError("Language tag must use BCP 47 syntax.")
        if value in _GRANDFATHERED_TAGS:
            return value
        parts = value.split("-")
        variants: set[str] = set()
        singletons: set[str] = set()
        in_extension = False
        for part in [] if parts[0].lower() == "x" else parts[1:]:
            lower = part.lower()
            if lower == "x":
                break
            if len(part) == 1:
                if lower in singletons:
                    raise ValueError(
                        "Language tag cannot repeat an extension singleton."
                    )
                singletons.add(lower)
                in_extension = True
            elif not in_extension and (
                len(part) >= 5 or (len(part) == 4 and part[0].isdigit())
            ):
                if lower in variants:
                    raise ValueError("Language tag cannot repeat a variant.")
                variants.add(lower)
        if parts[0].lower() == "x":
            canonical = [part.lower() for part in parts]
            if value != "-".join(canonical):
                raise ValueError("Language tag must use canonical BCP 47 casing.")
            return value
        canonical = [parts[0].lower()]
        extension = False
        for part in parts[1:]:
            if len(part) == 1:
                extension = True
                canonical.append(part.lower())
            elif extension:
                canonical.append(part.lower())
            elif len(part) == 4 and part.isalpha():
                canonical.append(part.title())
            elif (len(part) == 2 and part.isalpha()) or (
                len(part) == 3 and part.isdigit()
            ):
                canonical.append(part.upper())
            else:
                canonical.append(part.lower())
        if value != "-".join(canonical):
            raise ValueError("Language tag must use canonical BCP 47 casing.")
        return value

    @model_validator(mode="after")
    def _validate_language(self) -> Language:
        if self.source_text is None and self.tag is None:
            raise ValueError("A language requires source_text or tag.")
        return self


class MultiAxisAssignment(_SchemaModel):
    """Bind a variable to one distinct axis in a multi-axis graph.

    Parameters
    ----------
    dimension : AxisDimension
        Cartesian dimension of the axis.
    position : AxisPosition
        Side of the plot where the axis appears.
    position_index : int
        One-based order from the plotting area outward on that side.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"dimension": {"const": "x"}},
                        "required": ["dimension"],
                    },
                    "then": {"properties": {"position": {"enum": ["top", "bottom"]}}},
                },
                {
                    "if": {
                        "properties": {"dimension": {"const": "y"}},
                        "required": ["dimension"],
                    },
                    "then": {"properties": {"position": {"enum": ["left", "right"]}}},
                },
            ],
            "x-validation-rules": [
                "x axes use top or bottom; y axes use left or right."
            ],
        }
    )

    dimension: AxisDimension = Field(
        examples=["x", "y"],
        description="Cartesian dimension of the assigned axis.",
    )
    position: AxisPosition = Field(
        examples=["top", "bottom", "left", "right"],
        description="Side of the plot where the assigned axis appears.",
    )
    position_index: PositiveInt = Field(
        examples=[1, 2],
        description="One-based order from the plotting area outward among axes on the same side.",
    )

    @model_validator(mode="after")
    def _validate_position(self) -> MultiAxisAssignment:
        positions = {
            AxisDimension.X: {AxisPosition.TOP, AxisPosition.BOTTOM},
            AxisDimension.Y: {AxisPosition.LEFT, AxisPosition.RIGHT},
        }
        if self.position not in positions[self.dimension]:
            raise ValueError(
                f"{self.dimension.value}-axis position must be one of "
                f"{sorted(position.value for position in positions[self.dimension])}."
            )
        return self


class Variable(_SchemaModel):
    """Represent a measured variable and its applicable qualifiers.

    Parameters
    ----------
    name : str | None
        Variable, indicator, metric, or measured concept, when identifiable.
    unit : Unit | None
        Applicable unit.
    currency : Currency | None
        Applicable currency.
    analytical_roles : list[AnalyticalRole] | None
        Explicit roles in a statistical analysis.
    axis_roles : list[AxisRole] | None
        Ordinary Cartesian axis roles in a figure.
    multi_axis_assignments : list[MultiAxisAssignment] | None
        Explicit assignments to distinct axes in a multi-axis graph.
    statistical_forms : list[StatisticalFormTerm] | None
        Applicable statistical forms.
    """

    model_config = ConfigDict(json_schema_extra=_variable_schema)

    name: NonEmptyText | None = Field(
        examples=["GDP Growth", "Inflation", "Literacy Rate", "Refugee Population"],
        default=None,
        description="The measured quantity, indicator, metric, or outcome represented by the values. Use a variable for what is measured; use a dimension for categories that classify, group, or organize observations. For example, in a table of total loans by client sector, `Total loans` is a variable while `Corporates`, `Households`, and `Other clients` are categories of a client-sector dimension. Do not use `%`, `Value`, `Unknown`, or another unit or placeholder as the name.",
        json_schema_extra=_standards(("https://schema.org/variableMeasured", "close")),
    )
    unit: Unit | None = Field(
        examples=[
            {"source_text": "Percent", "code": "P1"},
            {"source_text": "USD"},
            {"source_text": "People"},
            {"source_text": "Kilometers", "code": "KMT"},
        ],
        default=None,
        description="The complete displayed unit or scale expression for the variable, such as `%`, people, kilometers, `US$ millions`, or `US$000`. Preserve the complete measurement expression, excluding the variable name and annotations that do not specify the values' scale. When one applicable currency is supported, repeat it in `currency`; when the expression combines multiple currencies or no single currency is supported, leave `currency` unset. Record a supported magnitude in `unit.multiplier_exponent`. The unit does not by itself establish whether values are a percentage, rate, change, coefficient, or another statistical form; the `%` in `95% CI`, for example, is not a percent unit.",
        json_schema_extra=_standards(
            ("https://schema.org/unitCode", "related_structural")
        ),
    )
    currency: Currency | None = Field(
        examples=[
            {"source_text": "USD", "code": "USD"},
            {"source_text": "EUR", "code": "EUR"},
            {"source_text": "JPY", "code": "JPY"},
        ],
        default=None,
        description="The currency denomination used for monetary values.",
        json_schema_extra=_standards(("https://schema.org/currency", "close")),
    )
    analytical_roles: list[AnalyticalRole] | None = Field(
        examples=[["outcome"], ["predictor"], ["control"], ["instrumental"]],
        default=None,
        min_length=1,
        description="Explicit roles in a statistical analysis or analytical table. Use `outcome` for the dependent or response variable, `predictor` for an explanatory or independent variable, `control` for an adjustment covariate, and `instrumental` only for a variable explicitly identified as an instrument. Do not derive these roles merely from row, column, or axis position.",
    )
    axis_roles: list[AxisRole] | None = Field(
        examples=[["x_axis"], ["y_axis"]],
        default=None,
        min_length=1,
        description="Ordinary Cartesian axis roles in a figure. Use `x_axis` for a variable encoded on the horizontal axis and `y_axis` for one encoded on the vertical axis. Do not use this field for analytical-table roles or table row and column organization.",
    )
    multi_axis_assignments: list[MultiAxisAssignment] | None = Field(
        examples=[
            [
                {
                    "dimension": "y",
                    "position": "left",
                    "position_index": 1,
                }
            ],
            [
                {
                    "dimension": "y",
                    "position": "right",
                    "position_index": 1,
                }
            ],
        ],
        default=None,
        min_length=1,
        description="Assignments to distinct Cartesian axes in a multi-axis graph. Use this field only when variables are assigned to different axes of the same dimension and the side or outward position distinguishes them. Use `axis_roles` for an ordinary single or shared x- or y-axis. `position_index` is 1 for the axis nearest the plotting area on a given side and increases outward.",
    )
    statistical_forms: list[StatisticalFormTerm] | None = Field(
        examples=[
            [{"source_text": "Count", "normalized_value": "count"}],
            [
                {
                    "source_text": "Percentage",
                    "normalized_value": "percentage",
                }
            ],
            [{"source_text": "Rate", "normalized_value": "rate"}],
            [{"source_text": "Index", "normalized_value": "index"}],
            [{"source_text": "Average"}],
        ],
        default=None,
        min_length=1,
        description="The mathematical or statistical form of the variable's values. Prefer explicit source terminology. Use `percentage` for a percentage, share, or proportion per hundred; `rate` for a quantity presented as a rate relative to a base, including incidence, growth, interest, or tariff rates; and `change` for a difference, movement, or return. When `%` specifies the scale in which the variable's values are expressed, preserve it in `unit`. Do not create a percent unit when the symbol belongs to a confidence level or significance annotation, or merely appears in a title or note without specifying the variable's measurement scale. Do not select a statistical form from `%` alone.",
        json_schema_extra=_standards(("https://schema.org/statType", "close")),
    )

    @model_validator(mode="after")
    def _validate_content(self) -> Variable:
        if (
            self.name is None
            and self.unit is None
            and self.currency is None
            and self.statistical_forms is None
        ):
            raise ValueError(
                "A variable requires a name, unit, currency, or statistical forms."
            )
        if self.name is None and (
            self.analytical_roles is not None
            or self.axis_roles is not None
            or self.multi_axis_assignments is not None
        ):
            raise ValueError(
                "Analytical roles, axis roles, and multi-axis assignments require "
                "a variable name."
            )
        return self


class CategoryGroup(_SchemaModel):
    """Represent one explicit nonrecursive category grouping.

    Parameters
    ----------
    name : str
        Explicit group heading.
    categories : list[CodedTerm]
        Categories directly contained by the group.
    """

    name: NonEmptyText = Field(
        examples=["Violation of the right to liberty"],
        description="Explicit category-group heading.",
    )
    categories: list[CodedTerm] = Field(
        examples=[
            [
                {"source_text": "Arbitrary arrests"},
                {"source_text": "Abductions"},
            ]
        ],
        min_length=1,
        description="Categories directly contained by the group.",
    )


class Dimension(_SchemaModel):
    """Represent a classificatory dimension and its visible organization.

    Parameters
    ----------
    name : str | None
        Dimension name, when identifiable.
    categories : list[CodedTerm] | None
        Ordered ungrouped categories.
    category_groups : list[CategoryGroup] | None
        One level of explicit category groups.
    presentation_roles : list[PresentationRole] | None
        Explicit row and/or column roles.
    """

    model_config = ConfigDict(
        json_schema_extra=_content_schema("name", "categories", "category_groups")
    )

    name: NonEmptyText | None = Field(
        examples=["Country", "Year", "Education Level", "Industry Sector", "Scenario"],
        default=None,
        description="The characteristic used to classify, group, or organize observations rather than the quantity being measured. Examples include year, country, sex, sector, and scenario. Repeated series or table headings that apply the same measure to different groups are normally dimension categories, not separate variables.",
    )
    categories: list[CodedTerm] | None = Field(
        examples=[
            [{"source_text": "Male"}, {"source_text": "Female"}],
            [
                {"source_text": "Agriculture"},
                {"source_text": "Manufacturing"},
                {"source_text": "Services"},
            ],
            [
                {"source_text": "Kenya"},
                {"source_text": "Uganda"},
                {"source_text": "Tanzania"},
            ],
            [
                {"source_text": "Low"},
                {"source_text": "Medium"},
                {"source_text": "High"},
            ],
        ],
        default=None,
        min_length=1,
        description="The explicit labels or values belonging to the dimension, such as Male and Female for a sex dimension or Corporates and Households for a client-sector dimension.",
        json_schema_extra=_standards(
            ("http://www.w3.org/2004/02/skos/core#Concept", "related_structural")
        ),
    )
    category_groups: list[CategoryGroup] | None = Field(
        examples=[
            [
                {
                    "name": "Violation of the right to liberty",
                    "categories": [
                        {"source_text": "Arbitrary arrests"},
                        {"source_text": "Abductions"},
                    ],
                }
            ]
        ],
        default=None,
        min_length=1,
        description="One level of explicit category groups.",
        json_schema_extra=_standards(
            ("http://www.w3.org/2004/02/skos/core#broader", "related_structural")
        ),
    )
    presentation_roles: list[PresentationRole] | None = Field(
        examples=[["row"], ["column"]],
        default=None,
        min_length=1,
        description="How the dimension organizes a table. `row` means its categories vary downward through rows; `column` means they vary horizontally across columns. Assign the role from the direction in which the categories vary, not from the physical cell containing the dimension label. Do not use these roles for figure axes.",
    )

    @model_validator(mode="after")
    def _validate_content(self) -> Dimension:
        if (
            self.name is None
            and self.categories is None
            and self.category_groups is None
        ):
            raise ValueError(
                "A dimension requires a name, categories, or category groups."
            )
        return self


class TemporalExpression(_SchemaModel):
    """Represent source-visible time with optional normalized bounds.

    Parameters
    ----------
    source_text : str | None
        Complete time expression explicitly visible in the snapshot.
    start : str | None
        Normalized starting value.
    end : str | None
        Normalized ending value.
    relation : TemporalRelation | None
        Relationship between the normalized bounds.
    precision : TemporalPrecision | None
        Precision shared by the normalized bounds.
    """

    model_config = ConfigDict(json_schema_extra=_temporal_schema)

    source_text: NonEmptyText | None = Field(
        examples=["2015–2020", "FY2023", "January 2024"],
        default=None,
        description="Exact complete time expression visible in the snapshot. Omit it when normalized bounds are inferred from separate labels or structure rather than transcribed as one expression.",
    )
    start: NonEmptyText | None = Field(
        examples=["2015", "2024-01", "2024-01-01T12:00:00.1Z"],
        default=None,
        description="Normalized start.",
    )
    end: NonEmptyText | None = Field(
        examples=["2020", "2024-03"], default=None, description="Normalized end."
    )
    relation: TemporalRelation | None = Field(
        examples=["interval", "point", "open_ended_interval"],
        default=None,
        description="Relationship between normalized bounds. `point` represents a single represented-data time point and requires `start` only. `interval` represents a bounded period and requires `start` and `end`. `open_ended_interval` represents a period for which the snapshot explicitly supports only one temporal boundary. Use `start` for expressions such as “since 2015” and `end` for expressions such as “through 2020.” Do not use `open_ended_interval` merely because extraction failed to identify the other boundary.",
    )
    precision: TemporalPrecision | None = Field(
        examples=["year", "month", "day", "datetime"],
        default=None,
        description="Precision of normalized bounds.",
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> TemporalExpression:
        bounds = [bound for bound in (self.start, self.end) if bound is not None]
        if self.source_text is None and not bounds:
            raise ValueError("A temporal expression requires source text or a bound.")
        if not bounds:
            if self.relation is not None or self.precision is not None:
                raise ValueError("Temporal relation and precision require a bound.")
            return self
        if self.relation is None or self.precision is None:
            raise ValueError(
                "Normalized temporal bounds require relation and precision."
            )
        if self.relation is TemporalRelation.POINT:
            if self.start is None or self.end is not None:
                raise ValueError("point requires start only.")
        elif self.relation is TemporalRelation.INTERVAL:
            if self.start is None or self.end is None:
                raise ValueError("interval requires start and end.")
        elif len(bounds) != 1:
            raise ValueError("open_ended_interval requires exactly one bound.")
        for bound in bounds:
            _validate_temporal_value(bound, self.precision)
        if self.start is not None and self.end is not None:
            if _temporal_sort_value(self.start, self.precision) > _temporal_sort_value(
                self.end, self.precision
            ):
                raise ValueError("Temporal start cannot be after end.")
        return self


def _validate_temporal_value(value: str, precision: TemporalPrecision) -> None:
    if precision is TemporalPrecision.YEAR and _YEAR_PATTERN.fullmatch(value):
        return
    if precision is TemporalPrecision.MONTH and _MONTH_PATTERN.fullmatch(value):
        return
    if precision is TemporalPrecision.DAY and _DAY_PATTERN.fullmatch(value):
        date.fromisoformat(value)
        return
    if precision is TemporalPrecision.DATETIME and _DATETIME_PATTERN.fullmatch(value):
        parsed, _ = _parse_datetime(value)
        if parsed.tzinfo is not None:
            return
    raise ValueError(f"Temporal value {value!r} does not match {precision.value}.")


def _parse_datetime(value: str) -> tuple[datetime, Decimal]:
    # Parse whole seconds on Python 3.10; compare the original fraction exactly.
    fraction = re.search(r"[.,]([0-9]+)(?=Z|[+-][0-9]{2}:[0-9]{2}$)", value)
    if fraction:
        seconds = Decimal("0." + fraction.group(1))
        value = value[: fraction.start()] + value[fraction.end() :]
    else:
        seconds = Decimal(0)
    return datetime.fromisoformat(value.replace("Z", "+00:00")), seconds


def _temporal_sort_value(
    value: str, precision: TemporalPrecision
) -> str | tuple[datetime, Decimal]:
    if precision is TemporalPrecision.DATETIME:
        return _parse_datetime(value)
    return value


class TemporalCoverage(_SchemaModel):
    """Group represented-data time and reporting interval.

    Parameters
    ----------
    period : TemporalExpression | None
        Represented-data temporal expression.
    reporting_interval : ReportingIntervalTerm | None
        Interval between successive represented-data time points.
    """

    model_config = ConfigDict(
        json_schema_extra=_content_schema("period", "reporting_interval")
    )

    period: TemporalExpression | None = Field(
        examples=[
            {
                "source_text": "2015–2020",
                "start": "2015",
                "end": "2020",
                "relation": "interval",
                "precision": "year",
            },
            {"source_text": "FY2023"},
            {
                "source_text": "January 2024",
                "start": "2024-01",
                "relation": "point",
                "precision": "month",
            },
        ],
        default=None,
        description="The period or date range represented by the data.\n\nThis field describes **when the represented data apply**. It does not describe when the snapshot artifact or parent document was created, prepared, issued, published, revised, or retrieved. When an explicit artifact date appears only as part of a footer or provenance statement, preserve the complete statement in `interpretive_notes` rather than treating the date as `temporal_coverage.period`.",
        json_schema_extra=_standards(
            ("https://schema.org/temporalCoverage", "exact"),
            ("http://purl.org/dc/terms/temporal", "close"),
        ),
    )
    reporting_interval: ReportingIntervalTerm | None = Field(
        examples=[
            {"source_text": "Annual", "normalized_value": "annual"},
            {"source_text": "Monthly", "normalized_value": "monthly"},
            {"source_text": "Quarterly", "normalized_value": "quarterly"},
            {"source_text": "Daily", "normalized_value": "daily"},
        ],
        default=None,
        description="The interval between successive time points represented by the data, such as hourly, daily, monthly, quarterly, or annual. This describes the spacing of the represented data, not the document's publication schedule or the overall period covered.",
    )

    @model_validator(mode="after")
    def _validate_content(self) -> TemporalCoverage:
        if self.period is None and self.reporting_interval is None:
            raise ValueError(
                "Temporal coverage must contain period or reporting_interval."
            )
        return self


class Place(_SchemaModel):
    """Represent a source-grounded place with optional standard identifiers.

    Parameters
    ----------
    source_text : str | None
        Displayed place expression.
    name : str | None
        Preferred place name.
    iso3_code : str | None
        ISO 3166-1 alpha-3 country code.
    subdivision_code : str | None
        ISO 3166-2 subdivision code.
    m49_code : str | None
        UN M49 statistical-area code.
    identifiers : list[Identifier] | None
        Other authoritative identifiers.
    """

    model_config = ConfigDict(
        json_schema_extra=_content_schema(
            "source_text",
            "name",
            "iso3_code",
            "subdivision_code",
            "m49_code",
            "identifiers",
        )
    )

    source_text: NonEmptyText | None = Field(
        examples=["Global", "Kenya", "Sub-Saharan Africa", "Latin America"],
        default=None,
        description="Exact place expression visible in the snapshot. Preserve the displayed wording here; omit it when the place is inferred from structure or context rather than transcribed.",
    )
    name: NonEmptyText | None = Field(
        examples=["Kenya", "Sub-Saharan Africa", "Philippines"],
        default=None,
        description="Preferred normalized name for the place represented by `source_text` or by an authoritative identifier. This may standardize spelling or abbreviations and need not appear verbatim in the snapshot.",
    )
    iso3_code: (
        Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Z]{3}$")] | None
    ) = Field(
        examples=["KEN", "PHL"],
        default=None,
        description="ISO 3166-1 alpha-3 code for a country or area. Do not use World Bank aggregate or region codes.",
        json_schema_extra=_code_list(
            "ISO", "ISO 3166-1", "https://www.iso.org/iso-3166-country-codes.html"
        ),
    )
    subdivision_code: (
        Annotated[
            str, StringConstraints(strict=True, pattern=r"^[A-Z]{2}-[A-Z0-9]{1,3}$")
        ]
        | None
    ) = Field(
        examples=["US-CA"],
        default=None,
        description="ISO 3166-2 subdivision code.",
        json_schema_extra=_code_list(
            "ISO", "ISO 3166-2", "https://www.iso.org/iso-3166-country-codes.html"
        ),
    )
    m49_code: (
        Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9]{3}$")] | None
    ) = Field(
        examples=["002", "202"],
        default=None,
        description="UN M49 statistical-area code.",
        json_schema_extra=_code_list(
            "United Nations",
            "M49",
            "https://unstats.un.org/unsd/methodology/m49/",
        ),
    )
    identifiers: list[Identifier] | None = Field(
        examples=[[{"value": "KEN", "scheme": "ISO 3166-1 alpha-3"}]],
        default=None,
        min_length=1,
        description="Other authoritative identifiers.",
    )

    @model_validator(mode="after")
    def _validate_name(self) -> Place:
        if not any(
            (
                self.source_text,
                self.name,
                self.iso3_code,
                self.subdivision_code,
                self.m49_code,
                self.identifiers,
            )
        ):
            raise ValueError(
                "A place requires a name, source text, code, or identifier."
            )
        return self


class GeographicLocation(Place):
    """Represent an additional named location and its optional role and type.

    Parameters
    ----------
    source_text : str | None
        Displayed place expression.
    name : str | None
        Preferred place name.
    iso3_code : str | None
        ISO 3166-1 alpha-3 country code.
    subdivision_code : str | None
        ISO 3166-2 subdivision code.
    m49_code : str | None
        UN M49 statistical-area code.
    identifiers : list[Identifier] | None
        Other authoritative identifiers.
    role : str | None
        Explicit source-grounded geographic role.
    type : CodedTerm | None
        Physical or administrative type of the named location.
    """

    role: NonEmptyText | None = Field(
        examples=[
            "Country of origin",
            "Host country",
            "Destination",
            "Reporting location",
        ],
        default=None,
        description="The relation of this location to the represented data, such as country of origin, host country, destination, or reporting location. Record only a relation supported by the snapshot; this field does not describe what kind of place it is or the level at which data are reported.",
    )
    type: CodedTerm | None = Field(
        examples=[
            {"source_text": "Refugee camp"},
            {"source_text": "Hospital"},
            {"source_text": "School"},
            {"source_text": "District"},
        ],
        default=None,
        description="The physical or administrative type of a named geographic location represented in the snapshot. This field describes what the location is; use `geographic_coverage.level` for the administrative or spatial level at which the snapshot's data are reported.",
    )


class GeographicCoverage(_SchemaModel):
    """Group overall geographic scope, locations, and reporting level.

    Parameters
    ----------
    scope : Place | None
        Overall geographic coverage or focus.
    locations : list[GeographicLocation] | None
        Additional named locations.
    level : GeographicLevelTerm | None
        Administrative, geographic, or reporting level.
    """

    model_config = ConfigDict(
        json_schema_extra=_content_schema("scope", "locations", "level")
    )

    scope: Place | None = Field(
        examples=[
            {"source_text": "Global", "name": "Global", "m49_code": "001"},
            {"source_text": "Kenya", "name": "Kenya", "iso3_code": "KEN"},
            {
                "source_text": "Sub-Saharan Africa",
                "name": "Sub-Saharan Africa",
                "m49_code": "202",
            },
            {"source_text": "Latin America", "name": "Latin America"},
        ],
        default=None,
        description="The overall geographic coverage or focus of the snapshot. Use `locations` for additional places represented within that coverage. A scope may be a country, region, aggregate, or other area and does not by itself state the reporting level.",
        json_schema_extra=_standards(("https://schema.org/spatialCoverage", "exact")),
    )
    locations: list[GeographicLocation] | None = Field(
        examples=[
            [{"name": "Uganda", "iso3_code": "UGA"}],
            [{"name": "Nairobi"}],
            [{"name": "West Africa", "m49_code": "011"}],
            [{"name": "Burkina Faso", "iso3_code": "BFA"}],
        ],
        default=None,
        min_length=1,
        description="Additional named places represented within the snapshot. Record the overall coverage in `scope`, and use each location's `role` for its relation to the data. Assign identifiers to the place they identify; do not copy a parent area's code onto a child location.",
        json_schema_extra=_standards(
            ("https://schema.org/spatialCoverage", "related_structural")
        ),
    )
    level: GeographicLevelTerm | None = Field(
        examples=[
            {"source_text": "Country", "normalized_value": "country"},
            {"source_text": "Province"},
            {"source_text": "District"},
            {"source_text": "Facility", "normalized_value": "site"},
        ],
        default=None,
        description="The geographic granularity at which observations are reported, such as country, province, district, or site. This describes the level of the data, not the overall coverage area, an individual place, or a location's semantic role.",
    )

    @model_validator(mode="after")
    def _validate_content(self) -> GeographicCoverage:
        if self.scope is None and self.locations is None and self.level is None:
            raise ValueError("Geographic coverage must contain a value.")
        return self


class Provenance(_SchemaModel):
    """Separate derivation sources from credited agents.

    Parameters
    ----------
    sources : list[EntityReference] | None
        Entities from which represented data derive.
    attributions : list[Attribution] | None
        Agents explicitly credited for the snapshot artifact.
    """

    model_config = ConfigDict(
        json_schema_extra=_content_schema("sources", "attributions")
    )

    sources: list[EntityReference] | None = Field(
        examples=[
            [{"name": "World Development Indicators"}],
            [{"name": "DHS"}],
            [{"name": "UNHCR Registration Data"}],
            [{"name": "National Census"}],
        ],
        default=None,
        min_length=1,
        description="Represented-data derivation sources.",
        json_schema_extra=_standards(
            ("http://www.w3.org/ns/prov#wasDerivedFrom", "related_structural"),
            ("http://purl.org/dc/terms/source", "close"),
        ),
    )
    attributions: list[Attribution] | None = Field(
        examples=[[{"name": "Map Design Unit", "role": {"source_text": "Map maker"}}]],
        default=None,
        min_length=1,
        description="Agents explicitly credited for the snapshot artifact; include a role when it is explicit.",
        json_schema_extra=_standards(
            ("http://www.w3.org/ns/prov#wasAttributedTo", "related_structural")
        ),
    )

    @model_validator(mode="after")
    def _validate_content(self) -> Provenance:
        if self.sources is None and self.attributions is None:
            raise ValueError("Provenance must contain a source or attribution.")
        return self


class Project(_SchemaModel):
    """Represent project, program, operation, or initiative context.

    Parameters
    ----------
    name : str | None
        Associated project-context name.
    identifiers : list[Identifier] | None
        Formal project or operation identifiers.
    components : list[EntityReference] | None
        Explicitly identified subordinate components.
    """

    model_config = ConfigDict(
        json_schema_extra=_content_schema("name", "identifiers", "components")
    )

    name: NonEmptyText | None = Field(
        examples=[
            "Niger - COVID-19 Emergency Response Project",
            "Jordan Health Sector Reform Project",
            "Lebanon - Health Resilience Project",
        ],
        default=None,
        description="The project, program, operation, or initiative associated with the snapshot.",
        json_schema_extra=_standards(("https://schema.org/name", "exact")),
    )
    identifiers: list[Identifier] | None = Field(
        examples=[[{"value": "P171254"}], [{"value": "P178944"}]],
        default=None,
        min_length=1,
        description="The formal identifier assigned to the associated project or operation.",
        json_schema_extra=_standards(
            ("https://schema.org/identifier", "standard_broader")
        ),
    )
    components: list[EntityReference] | None = Field(
        examples=[
            [{"name": "Component 3: Project management"}],
            [{"name": "Results Area 1"}],
        ],
        default=None,
        min_length=1,
        description="The project component, workstream, or results area represented by the snapshot.",
        json_schema_extra=_standards(
            ("https://schema.org/hasPart", "related_structural")
        ),
    )

    @model_validator(mode="after")
    def _validate_content(self) -> Project:
        if self.name is None and self.identifiers is None and self.components is None:
            raise ValueError("Project must contain a name, identifier, or component.")
        return self


class Financing(_SchemaModel):
    """Group project-financing measures, funders, and instruments.

    Parameters
    ----------
    measures : list[str] | None
        Financial quantities or funding-related measures.
    funders : list[EntityReference] | None
        Named funding sources.
    instruments : list[CodedTerm] | None
        Financing mechanisms.
    """

    model_config = ConfigDict(
        json_schema_extra=_content_schema("measures", "funders", "instruments")
    )

    measures: list[NonEmptyText] | None = Field(
        examples=[
            ["Project Cost"],
            ["Disbursement"],
            ["Financing Gap"],
            ["Budget Allocation"],
        ],
        default=None,
        min_length=1,
        description="The financial quantity or funding-related measure represented by the snapshot.",
    )
    funders: list[EntityReference] | None = Field(
        examples=[
            [{"name": "IDA"}],
            [{"name": "IBRD"}],
            [{"name": "Government"}],
            [{"name": "European Union"}],
        ],
        default=None,
        min_length=1,
        description="The organization or funding source providing financial support.",
        json_schema_extra=_standards(("https://schema.org/funder", "exact")),
    )
    instruments: list[CodedTerm] | None = Field(
        examples=[
            [{"source_text": "Grant"}],
            [{"source_text": "Loan"}],
            [{"source_text": "Credit"}],
            [{"source_text": "Trust Fund"}],
        ],
        default=None,
        min_length=1,
        description="The financing mechanism associated with the represented activity.",
        json_schema_extra=_standards(
            (
                "https://reference.iatistandard.org/en/iati-standard/203/codelists/financetype/",
                "close",
            )
        ),
    )

    @model_validator(mode="after")
    def _validate_content(self) -> Financing:
        if self.measures is None and self.funders is None and self.instruments is None:
            raise ValueError("Financing must contain a measure, funder, or instrument.")
        return self


class DataSnapshotMetadata(_SchemaModel):
    """Represent canonical semantic metadata for one data snapshot.

    Parameters
    ----------
    title : str | None
        Primary title, caption, or heading.
    document_label : str | None
        Label assigned within the parent source document.
    subject_domains : list[str] | None
        Broad thematic, policy, or sectoral domains.
    subject_summary : str | None
        Concise analytical summary.
    panel_titles : list[str] | None
        Ordered explicit panel titles.
    variables : list[Variable] | None
        Measured concepts and their qualifiers.
    dimensions : list[Dimension] | None
        Classificatory dimensions and visible organization.
    population_group : str | None
        Human population represented by the data.
    visualization_types : list[VisualizationTypeTerm] | None
        Visible visualization forms.
    temporal_coverage : TemporalCoverage | None
        Represented-data time and reporting interval.
    geographic_coverage : GeographicCoverage | None
        Geographic scope, locations, and level.
    comparisons : list[str] | None
        Explicit comparisons or named comparators.
    provenance : Provenance | None
        Derivation sources and artifact attributions.
    languages : list[Language] | None
        Languages used within the snapshot.
    interpretive_notes : list[str] | None
        Complete source-visible interpretive statements.
    project : Project | None
        Associated project or operational context.
    intervention_types : list[str] | None
        Represented interventions or activities.
    financing : Financing | None
        Project-financing context.
    analysis_methods : list[str] | None
        Explicit analytical methods.
    data_collection_methods : list[CodedTerm] | None
        Explicit data-collection methods.
    """

    model_config = ConfigDict(
        extra="forbid",
        title="Data Snapshot Metadata Schema v1.4",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "x-schema-version": "1.4",
            "x-status": "implementation",
            "x-validation-rules": [
                "Python is the canonical validator. JSON Schema enforces exported structural rules; format assertions require a format-aware validator.",
                "Python additionally applies NFC normalization, outer whitespace trimming, and stable exact deduplication.",
                "Python additionally checks URI syntax before URL normalization, language-tag casing and uniqueness, calendar dates, and chronological ordering.",
                "Missing and null values are equivalent. Serialize records with exclude_none=True to omit unavailable values.",
                "External registry membership, source-grounding, and semantic correctness are not validated. Unpinned code-list metadata identifies a syntax authority only.",
            ],
        },
    )

    title: NonEmptyText | None = Field(
        examples=[
            "Inflation Rate by Country",
            "Annual Government Expenditure",
            "Monthly labor income in Afghanistan and remittances from abroad",
            "Table 6: Determinants of illegal land reallocation at village level",
        ],
        default=None,
        description="The primary title, caption, or heading that identifies the data snapshot.",
        json_schema_extra=_standards(
            ("http://purl.org/dc/terms/title", "exact"),
            ("https://schema.org/name", "exact"),
        ),
    )
    document_label: NonEmptyText | None = Field(
        examples=["Figure 3", "Table 4.2", "Annex B", "Exhibit 7"],
        default=None,
        description="An exact visible label assigned to the snapshot within its source document, such as `Table 3`, `Figure 4.2`, or `Annex B`. Do not use a title, caption, page number, filename, or identifier inferred from document position.",
        json_schema_extra=_standards(
            (
                "https://jats.nlm.nih.gov/publishing/tag-library/1.3/element/label.html",
                "close",
            ),
            ("https://schema.org/identifier", "standard_broader"),
        ),
    )
    subject_domains: list[NonEmptyText] | None = Field(
        examples=[
            ["Education"],
            ["Health"],
            ["Macroeconomics"],
            ["Agriculture"],
            ["Forced Displacement"],
        ],
        default=None,
        min_length=1,
        description="The broad thematic, policy, or sectoral domain represented by the snapshot.",
        json_schema_extra=_standards(
            ("http://purl.org/dc/terms/subject", "standard_broader"),
            ("https://schema.org/about", "standard_broader"),
        ),
    )
    subject_summary: NonEmptyText | None = Field(
        examples=[
            "Trends in primary school enrollment",
            "Distribution of humanitarian funding",
            "Comparison of poverty rates across regions",
        ],
        default=None,
        description="A concise summary describing the primary analytical subject or purpose of the snapshot.",
        json_schema_extra=_standards(
            ("https://schema.org/abstract", "close"),
            ("http://purl.org/dc/terms/description", "standard_broader"),
        ),
    )
    panel_titles: list[NonEmptyText] | None = Field(
        examples=[["(A) Poverty Rate"], ["(B) Literacy Rate"], ["Monthly Returns"]],
        default=None,
        min_length=1,
        description="The title or heading of an individual panel within a multi-panel snapshot.\n\nPopulate only when panel titles are explicitly present.",
        json_schema_extra=_standards(
            (
                "https://jats.nlm.nih.gov/publishing/tag-library/1.3/element/fig-group.html",
                "close",
            ),
            ("https://schema.org/hasPart", "related_structural"),
        ),
    )
    variables: list[Variable] | None = Field(
        examples=[
            [{"name": "GDP Growth"}],
            [{"name": "Inflation"}],
            [{"name": "Literacy Rate"}],
            [{"name": "Refugee Population"}],
            [{"name": None, "unit": {"source_text": "%"}}],
            [
                {
                    "name": "Share of households",
                    "unit": {"source_text": "%", "code": "P1"},
                    "statistical_forms": [
                        {
                            "source_text": "Share",
                            "normalized_value": "percentage",
                        }
                    ],
                }
            ],
            [
                {
                    "name": "Budgeted cost",
                    "unit": {"source_text": "US$"},
                    "currency": {"source_text": "US$", "code": "USD"},
                }
            ],
            [
                {
                    "name": "Monthly returns",
                    "unit": {"source_text": "%", "code": "P1"},
                    "statistical_forms": [
                        {
                            "source_text": "% change",
                            "normalized_value": "change",
                        }
                    ],
                }
            ],
            [
                {
                    "name": "pce_residual",
                    "statistical_forms": [
                        {
                            "source_text": "95% CI",
                            "normalized_value": "confidence_interval",
                        }
                    ],
                }
            ],
            [
                {
                    "name": "Tariff rate",
                    "unit": {"source_text": "ad-valorem in %"},
                    "statistical_forms": [
                        {
                            "source_text": "Tariff rate",
                            "normalized_value": "rate",
                        }
                    ],
                }
            ],
            [
                {
                    "name": "Amount",
                    "unit": {
                        "source_text": "US$m/SDRm",
                        "multiplier_exponent": 6,
                    },
                }
            ],
            [
                {
                    "name": "Revenue",
                    "unit": {
                        "source_text": "US$ millions",
                        "multiplier_exponent": 6,
                    },
                    "currency": {"source_text": "US$", "code": "USD"},
                },
                {
                    "name": "Operating cost",
                    "unit": {
                        "source_text": "US$ millions",
                        "multiplier_exponent": 6,
                    },
                    "currency": {"source_text": "US$", "code": "USD"},
                },
            ],
        ],
        default=None,
        min_length=1,
        description="Measured quantities, indicators, metrics, or outcomes and their applicable qualifiers. Use variables for what is measured and dimensions for the characteristics that classify or organize observations. Populate `name` when the measured concept can be identified; otherwise omit it rather than using a unit, `%`, `Value`, `Unknown`, or another placeholder. Retain an unnamed variable only when it has a unit, currency, or statistical form. Preserve each complete displayed unit or scale expression. When one applicable currency is supported, repeat it in `currency`; when the expression combines currencies or no single currency is supported, leave `currency` unset. Record a supported magnitude in `unit.multiplier_exponent`. Analytical roles, axis roles, and multi-axis assignments require a named variable. Repeat shared qualifiers on every variable to which they apply.",
        json_schema_extra=_standards(
            ("https://schema.org/variableMeasured", "close"),
            ("https://ddialliance.org/Specification/DDI-Lifecycle/3.3/", "close"),
        ),
    )
    dimensions: list[Dimension] | None = Field(
        examples=[
            [{"name": "Country"}],
            [{"name": "Year"}],
            [{"name": "Education Level"}],
            [{"name": "Industry Sector"}],
            [{"name": "Scenario"}],
            [{"name": "Country", "presentation_roles": ["row"]}],
            [{"name": "Indicator", "presentation_roles": ["row"]}],
            [{"name": "Sector", "presentation_roles": ["row"]}],
            [{"name": "Year", "presentation_roles": ["column"]}],
            [{"name": "Region", "presentation_roles": ["column"]}],
            [{"name": "Funding Source", "presentation_roles": ["column"]}],
        ],
        default=None,
        min_length=1,
        description="Characteristics that classify, group, or organize observations, together with their categories and table presentation. Use a dimension for year, geography, sex, sector, scenario, and similar groupings; use a variable for the quantity being measured.",
        json_schema_extra=_standards(
            ("https://sdmx.org/", "close"),
            ("https://ddialliance.org/Specification/DDI-Lifecycle/3.3/", "close"),
        ),
    )
    population_group: NonEmptyText | None = Field(
        examples=[
            "Refugees",
            "Children under five",
            "Female respondents",
            "Host communities",
            "Technical education graduates",
        ],
        default=None,
        description="The human population, beneficiary group, or demographic group that is the primary subject of the represented data. This field describes who the data are about. Do not use it for companies, institutions, products, locations, or mixed entity classes, and do not repeat a population category used only to disaggregate observations.",
        json_schema_extra=_standards(
            (
                "https://docs.ddialliance.org/DDI-Lifecycle/3.3/xmlschema/schemas/conceptualcomponent_xsd/elements/Universe.html",
                "close",
            ),
            ("https://schema.org/populationType", "close"),
        ),
    )
    visualization_types: list[VisualizationTypeTerm] | None = Field(
        examples=[
            [{"normalized_value": "bar_chart"}],
            [{"source_text": "Bar Graph", "normalized_value": "bar_chart"}],
            [{"source_text": "Waffle chart"}],
            None,
            [{"source_text": "Bar chart", "normalized_value": "bar_chart"}],
            [{"source_text": "Line chart", "normalized_value": "line_chart"}],
            [{"source_text": "Table", "normalized_value": "table"}],
            [{"source_text": "Map", "normalized_value": "map"}],
            [{"source_text": "Heatmap", "normalized_value": "heatmap"}],
            [
                {"normalized_value": "composite_figure"},
                {"normalized_value": "line_chart"},
                {"normalized_value": "map"},
            ],
            [
                {
                    "source_text": "Composite figure: line charts and map",
                    "normalized_value": "composite_figure",
                }
            ],
        ],
        default=None,
        min_length=1,
        description="Visualization forms used to encode the represented data. Types are not mutually exclusive: for a composite or multi-panel artifact, record `composite_figure` and each identifiable component type, such as `line_chart` and `map`. Use `normalized_value` when a type is inferred from visual form. Use `source_text` only when wording in the snapshot explicitly names the form; do not manufacture source wording from the design. Preserve an explicitly written unfamiliar type in `source_text` without a normalized value. If neither a listed type nor an explicit unfamiliar label is supported, return `null`; do not force a match to the closest vocabulary value. Use `panel_titles` for explicit panel headings.",
        json_schema_extra=_standards(
            ("http://purl.org/dc/terms/type", "standard_broader"),
            ("https://schema.org/additionalType", "standard_broader"),
        ),
    )
    temporal_coverage: TemporalCoverage | None = Field(
        examples=[
            {
                "period": {
                    "source_text": "2015–2020",
                    "start": "2015",
                    "end": "2020",
                    "relation": "interval",
                    "precision": "year",
                }
            },
            {"period": {"source_text": "FY2023"}},
            {
                "period": {
                    "source_text": "January 2024",
                    "start": "2024-01",
                    "relation": "point",
                    "precision": "month",
                }
            },
            {
                "reporting_interval": {
                    "source_text": "Annual",
                    "normalized_value": "annual",
                }
            },
            {
                "reporting_interval": {
                    "source_text": "Monthly",
                    "normalized_value": "monthly",
                }
            },
            {
                "reporting_interval": {
                    "source_text": "Quarterly",
                    "normalized_value": "quarterly",
                }
            },
            {
                "reporting_interval": {
                    "source_text": "Daily",
                    "normalized_value": "daily",
                }
            },
        ],
        default=None,
        description="When the represented data apply and the interval between successive represented-data time points.",
        json_schema_extra=_standards(
            ("https://schema.org/temporalCoverage", "exact"),
            ("http://purl.org/dc/terms/temporal", "close"),
        ),
    )
    geographic_coverage: GeographicCoverage | None = Field(
        examples=[
            {
                "scope": {
                    "source_text": "Global",
                    "name": "Global",
                    "m49_code": "001",
                }
            },
            {
                "scope": {
                    "source_text": "Kenya",
                    "name": "Kenya",
                    "iso3_code": "KEN",
                }
            },
            {
                "scope": {
                    "source_text": "Sub-Saharan Africa",
                    "name": "Sub-Saharan Africa",
                    "m49_code": "202",
                }
            },
            {"scope": {"source_text": "Latin America", "name": "Latin America"}},
            {"locations": [{"name": "Uganda", "iso3_code": "UGA"}]},
            {"locations": [{"name": "Nairobi"}]},
            {"locations": [{"name": "West Africa", "m49_code": "011"}]},
            {"locations": [{"name": "Burkina Faso", "iso3_code": "BFA"}]},
            {"level": {"source_text": "Country", "normalized_value": "country"}},
            {"level": {"source_text": "Province"}},
            {"level": {"source_text": "District"}},
            {"level": {"source_text": "Facility", "normalized_value": "site"}},
        ],
        default=None,
        description="The snapshot's overall geographic scope, additional named locations, and reporting level. Keep these meanings separate: `scope` is the coverage area, `locations` are specific places represented within it, and `level` is the granularity at which observations are reported.",
        json_schema_extra=_standards(
            ("https://schema.org/spatialCoverage", "exact"),
            ("http://purl.org/dc/terms/spatial", "exact"),
        ),
    )
    comparisons: list[NonEmptyText] | None = Field(
        examples=[
            ["Male vs Female"],
            ["Rural vs Urban"],
            ["Baseline vs Endline"],
            ["Treatment vs Control"],
            ["Before vs After"],
            ["Low-income vs Middle-income vs High-income"],
            ["Europe & Central Asia benchmark"],
            ["Sub-Saharan Africa benchmark"],
        ],
        default=None,
        min_length=1,
        description="An explicit comparative relationship or named benchmark, comparator, reference group, cohort, scenario, or entity against which represented data are compared. Record the complete relation when visible, such as `Treatment vs Control`, or a named standalone benchmark. Do not use this field for totals, ordinary cross-tab categories, dimensions, or measures that merely appear together without an explicit comparison.",
    )
    provenance: Provenance | None = Field(
        examples=[
            {"sources": [{"name": "World Development Indicators"}]},
            {"sources": [{"name": "DHS"}]},
            {"sources": [{"name": "UNHCR Registration Data"}]},
            {"sources": [{"name": "National Census"}]},
            {
                "attributions": [
                    {"name": "Map Design Unit", "role": {"source_text": "Map maker"}}
                ],
            },
        ],
        default=None,
        description="The named dataset, survey, publication, organization, or credited agent from which the represented data originate or which is explicitly credited with producing the snapshot artifact.\n\nUse `sources` for represented-data derivation sources and `attributions` for credited agents; include an attribution role when it is explicit. Do not copy the parent document's authors or publisher into this field solely because they are associated with the document; the source or attribution must be explicitly relevant to the snapshot or its represented data.",
    )
    languages: list[Language] | None = Field(
        examples=[
            [{"source_text": "English", "tag": "en"}],
            [{"source_text": "French", "tag": "fr"}],
            [{"source_text": "Arabic", "tag": "ar"}],
        ],
        default=None,
        min_length=1,
        description="The language used within the snapshot.",
        json_schema_extra=_standards(
            ("https://schema.org/inLanguage", "exact"),
            ("http://purl.org/dc/terms/language", "exact"),
        ),
    )
    interpretive_notes: list[NonEmptyText] | None = Field(
        examples=[
            ["Values are provisional."],
            ["Estimates exclude informal employment."],
            ["Data collected using 2022 census boundaries."],
            ["Sample: 1,204 respondents."],
            ["Shaded areas show 95% confidence intervals."],
            ["Prepared by the Map Design Unit, March 2024."],
        ],
        default=None,
        min_length=1,
        description="Explanatory, methodological, uncertainty, or provenance statements explicitly provided within the snapshot that aid interpretation or traceability.\n\nThis field may preserve complete notes containing sample-size statements, explanations of confidence intervals, standard errors or uncertainty bands, and footer statements that include an artifact date or production credit. It retains the statement as text; it does not create separate structured fields for sample size, uncertainty representation, or artifact publication date.\n\nPopulate only when such notes are explicitly present.",
        json_schema_extra=_standards(
            ("http://purl.org/dc/terms/description", "standard_broader"),
            ("https://schema.org/description", "standard_broader"),
        ),
    )
    project: Project | None = Field(
        examples=[
            {"name": "Niger - COVID-19 Emergency Response Project"},
            {"name": "Jordan Health Sector Reform Project"},
            {"name": "Lebanon - Health Resilience Project"},
        ],
        default=None,
        description="Associated project, program, operation, or initiative.",
        json_schema_extra=_standards(
            ("https://schema.org/Project", "standard_narrower"),
            (
                "https://iatistandard.org/en/iati-standard/203/activity-standard/",
                "close",
            ),
        ),
    )
    intervention_types: list[NonEmptyText] | None = Field(
        examples=[
            ["Cash Transfer"],
            ["Vaccination"],
            ["School Construction"],
        ],
        default=None,
        min_length=1,
        description="The intervention, service, policy, or operational activity represented.",
    )
    financing: Financing | None = Field(
        examples=[
            {"measures": ["Project Cost"]},
            {"measures": ["Disbursement"]},
            {"measures": ["Financing Gap"]},
            {"measures": ["Budget Allocation"]},
            {"funders": [{"name": "IDA"}]},
            {"funders": [{"name": "IBRD"}]},
            {"funders": [{"name": "Government"}]},
            {"funders": [{"name": "European Union"}]},
            {"instruments": [{"source_text": "Grant"}]},
            {"instruments": [{"source_text": "Loan"}]},
            {"instruments": [{"source_text": "Credit"}]},
            {"instruments": [{"source_text": "Trust Fund"}]},
        ],
        default=None,
        description="Project-financing measures, funders, and instruments.",
    )
    analysis_methods: list[NonEmptyText] | None = Field(
        examples=[
            ["Difference-in-Differences"],
            ["Regression"],
            ["Tobit model"],
            ["Cost-Benefit Analysis"],
        ],
        default=None,
        min_length=1,
        description="The analytical, statistical, or computational method used to produce the reported results.\n\nPopulate only when explicitly stated.",
        json_schema_extra=_standards(
            ("https://schema.org/measurementTechnique", "related_structural"),
            ("http://www.w3.org/ns/prov#Activity", "related_structural"),
        ),
    )
    data_collection_methods: list[CodedTerm] | None = Field(
        examples=[
            [{"source_text": "Household Survey"}],
            [{"source_text": "Administrative Records"}],
            [{"source_text": "Key Informant Interviews"}],
            [{"source_text": "Census"}],
        ],
        default=None,
        min_length=1,
        description="The method or instrument used to collect the underlying data.\n\nPopulate only when explicitly stated.",
        json_schema_extra=_standards(
            (
                "https://docs.ddialliance.org/DDI-Lifecycle/3.3/model/composite-types/ModeOfCollectionType/",
                "exact",
            ),
            ("https://schema.org/measurementMethod", "close"),
        ),
    )
