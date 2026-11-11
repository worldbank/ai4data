"""Render purpose-built Streamlit editors for metadata review records."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from enum import Enum
from functools import lru_cache
from typing import Any, get_args

import streamlit as st
from pydantic import BaseModel

from ai4data.data_snapshot.metadata_reviewer.storage import metadata_values_equal
from ai4data.data_snapshot.metadata_schema import (
    AnalyticalRole,
    AxisRole,
    AxisDimension,
    AxisPosition,
    DataSnapshotMetadata,
    GeographicLevelValue,
    PresentationRole,
    ReportingIntervalValue,
    StatisticalFormValue,
    TemporalPrecision,
    TemporalRelation,
    VisualizationTypeValue,
)

Record = dict[str, Any]
ItemRenderer = Callable[[Record, Record, str], None]


def render_metadata_form(working: Record, generated: Record) -> None:
    """Render and synchronize the editable metadata form.

    Parameters
    ----------
    working : dict[str, Any]
        Session-owned metadata record mutated with current widget values.
    generated : dict[str, Any]
        Immutable generated metadata used only for difference highlighting.
    """

    section = st.segmented_control(
        "Metadata section",
        ("Overview", "Structure", "Temporal", "Geographic", "Context"),
        default="Overview",
        key=_widget_key("section"),
        label_visibility="collapsed",
        width="stretch",
    )
    if section == "Structure":
        with st.container(
            height=760, border=True, key="metadata_reviewer_editor_structure"
        ):
            _object_list(
                working,
                generated,
                "variables",
                "Variables",
                "Variable",
                _render_variable,
            )
            _object_list(
                working,
                generated,
                "dimensions",
                "Dimensions",
                "Dimension",
                _render_dimension,
            )
            _object_list(
                working,
                generated,
                "visualization_types",
                "Visualization types",
                "Visualization type",
                _normalized_renderer(VisualizationTypeValue),
            )
    elif section == "Temporal":
        with st.container(
            height=760, border=True, key="metadata_reviewer_editor_temporal"
        ):
            _optional_object(
                working,
                generated,
                "temporal_coverage",
                "Temporal coverage",
                _render_temporal_coverage,
            )
    elif section == "Geographic":
        with st.container(
            height=760, border=True, key="metadata_reviewer_editor_geographic"
        ):
            _optional_object(
                working,
                generated,
                "geographic_coverage",
                "Geographic coverage",
                _render_geographic_coverage,
            )
    elif section == "Context":
        with st.container(
            height=760, border=True, key="metadata_reviewer_editor_context"
        ):
            _text_list(working, generated, "comparisons", "Comparisons")
            _optional_object(
                working, generated, "provenance", "Provenance", _render_provenance
            )
            _object_list(
                working,
                generated,
                "languages",
                "Languages",
                "Language",
                _render_language,
            )
            _text_list(
                working,
                generated,
                "interpretive_notes",
                "Interpretive notes",
                multiline=True,
            )
            _optional_object(working, generated, "project", "Project", _render_project)
            _text_list(working, generated, "intervention_types", "Intervention types")
            _optional_object(
                working, generated, "financing", "Financing", _render_financing
            )
            _text_list(working, generated, "analysis_methods", "Analysis methods")
            _object_list(
                working,
                generated,
                "data_collection_methods",
                "Data collection methods",
                "Method",
                _render_coded_term,
            )
    else:
        with st.container(
            height=760, border=True, key="metadata_reviewer_editor_overview"
        ):
            _text(working, generated, "title", "Title")
            _text(working, generated, "document_label", "Document label")
            _text_list(working, generated, "subject_domains", "Subject domains")
            _text(
                working,
                generated,
                "subject_summary",
                "Subject summary",
            )
            _text_list(working, generated, "panel_titles", "Panel titles")
            _text(working, generated, "population_group", "Population group")


def clear_form_widget_state() -> None:
    """Remove all metadata-form widget values from Streamlit session state."""

    for key in list(st.session_state):
        if str(key).startswith("metadata_reviewer_widget:"):
            del st.session_state[key]


def _render_variable(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "name", "Name", path)
    _optional_object(record, generated, "unit", "Unit", _render_unit, path)
    _optional_object(record, generated, "currency", "Currency", _render_currency, path)
    _enum_list(
        record,
        generated,
        "analytical_roles",
        "Analytical roles",
        AnalyticalRole,
        path,
    )
    _enum_list(
        record,
        generated,
        "axis_roles",
        "Axis roles",
        AxisRole,
        path,
    )
    _object_list(
        record,
        generated,
        "multi_axis_assignments",
        "Multi-axis assignments",
        "Multi-axis assignment",
        _render_multi_axis_assignment,
        path,
    )
    _object_list(
        record,
        generated,
        "statistical_forms",
        "Statistical forms",
        "Statistical form",
        _normalized_renderer(StatisticalFormValue),
        path,
    )


def _render_unit(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "source_text", "Source text", path)
    _text(record, generated, "code", "Code", path)
    _integer(
        record,
        generated,
        "multiplier_exponent",
        "Multiplier exponent",
        path,
    )


def _render_currency(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "source_text", "Source text", path)
    _text(record, generated, "code", "ISO 4217 code", path)


def _render_multi_axis_assignment(record: Record, generated: Record, path: str) -> None:
    _enum(record, generated, "dimension", "Dimension", AxisDimension, path)
    _enum(record, generated, "position", "Position", AxisPosition, path)
    _integer(record, generated, "position_index", "Position index", path)


def _render_dimension(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "name", "Name", path)
    _object_list(
        record,
        generated,
        "categories",
        "Categories",
        "Category",
        _render_coded_term,
        path,
    )
    _object_list(
        record,
        generated,
        "category_groups",
        "Category groups",
        "Category group",
        _render_category_group,
        path,
    )
    _enum_list(
        record,
        generated,
        "presentation_roles",
        "Presentation roles",
        PresentationRole,
        path,
    )


def _render_category_group(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "name", "Name", path)
    _object_list(
        record,
        generated,
        "categories",
        "Categories",
        "Category",
        _render_coded_term,
        path,
    )


def _render_coded_term(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "source_text", "Source text", path)
    _text(record, generated, "code", "Code", path)
    _text(record, generated, "scheme", "Scheme", path)
    _text(record, generated, "uri", "URI", path)


def _normalized_renderer(enum_type: type[Enum]) -> ItemRenderer:
    def render(record: Record, generated: Record, path: str) -> None:
        _text(record, generated, "source_text", "Source text", path)
        _enum(
            record,
            generated,
            "normalized_value",
            "Normalized value",
            enum_type,
            path,
        )

    return render


def _render_temporal_coverage(record: Record, generated: Record, path: str) -> None:
    _optional_object(
        record, generated, "period", "Period", _render_temporal_expression, path
    )
    _optional_object(
        record,
        generated,
        "reporting_interval",
        "Reporting interval",
        _normalized_renderer(ReportingIntervalValue),
        path,
    )


def _render_temporal_expression(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "source_text", "Source text", path)
    _text(record, generated, "start", "Start", path)
    _text(record, generated, "end", "End", path)
    _enum(record, generated, "relation", "Relation", TemporalRelation, path)
    _enum(record, generated, "precision", "Precision", TemporalPrecision, path)


def _render_geographic_coverage(record: Record, generated: Record, path: str) -> None:
    _optional_object(record, generated, "scope", "Scope", _render_place, path)
    _object_list(
        record,
        generated,
        "locations",
        "Locations",
        "Location",
        _render_geographic_location,
        path,
    )
    _optional_object(
        record,
        generated,
        "level",
        "Reporting level",
        _normalized_renderer(GeographicLevelValue),
        path,
    )


def _render_place(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "source_text", "Source text", path)
    _text(record, generated, "name", "Name", path)
    _text(record, generated, "iso3_code", "ISO3 code", path)
    _text(record, generated, "subdivision_code", "Subdivision code", path)
    _text(record, generated, "m49_code", "M49 code", path)
    _object_list(
        record,
        generated,
        "identifiers",
        "Identifiers",
        "Identifier",
        _render_identifier,
        path,
    )


def _render_geographic_location(record: Record, generated: Record, path: str) -> None:
    _render_place(record, generated, path)
    _text(record, generated, "role", "Role", path)
    _optional_object(
        record, generated, "type", "Location type", _render_coded_term, path
    )


def _render_identifier(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "value", "Value", path)
    _text(record, generated, "scheme", "Scheme", path)
    _text(record, generated, "issuer", "Issuer", path)
    _text(record, generated, "uri", "URI", path)


def _render_provenance(record: Record, generated: Record, path: str) -> None:
    _object_list(
        record,
        generated,
        "sources",
        "Sources",
        "Source",
        _render_entity_reference,
        path,
    )
    _object_list(
        record,
        generated,
        "attributions",
        "Attributions",
        "Attribution",
        _render_attribution,
        path,
    )


def _render_entity_reference(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "name", "Name", path)
    _object_list(
        record,
        generated,
        "identifiers",
        "Identifiers",
        "Identifier",
        _render_identifier,
        path,
    )


def _render_attribution(record: Record, generated: Record, path: str) -> None:
    _render_entity_reference(record, generated, path)
    _optional_object(record, generated, "role", "Role", _render_coded_term, path)


def _render_language(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "source_text", "Source text", path)
    _text(record, generated, "tag", "BCP 47 tag", path)


def _render_project(record: Record, generated: Record, path: str) -> None:
    _text(record, generated, "name", "Name", path)
    _object_list(
        record,
        generated,
        "identifiers",
        "Identifiers",
        "Identifier",
        _render_identifier,
        path,
    )
    _object_list(
        record,
        generated,
        "components",
        "Components",
        "Component",
        _render_entity_reference,
        path,
    )


def _render_financing(record: Record, generated: Record, path: str) -> None:
    _text_list(record, generated, "measures", "Measures", path=path)
    _object_list(
        record,
        generated,
        "funders",
        "Funders",
        "Funder",
        _render_entity_reference,
        path,
    )
    _object_list(
        record,
        generated,
        "instruments",
        "Instruments",
        "Instrument",
        _render_coded_term,
        path,
    )


def _text(
    record: Record,
    generated: Record,
    field: str,
    label: str,
    path: str = "",
) -> None:
    field_path = _field_path(path, field)
    key = _widget_key(field_path)
    current = st.session_state.get(key, record.get(field) or "")
    _field_header(label, field_path, current or None, generated.get(field))
    value = st.text_input(
        label,
        value=current,
        key=key,
        label_visibility="collapsed",
    )
    _set_optional(record, field, value)


def _integer(
    record: Record,
    generated: Record,
    field: str,
    label: str,
    path: str,
) -> None:
    field_path = _field_path(path, field)
    key = _widget_key(field_path)
    current = st.session_state.get(key, record.get(field))
    text = "" if current is None else str(current)
    try:
        comparison_value = int(text) if text else None
    except ValueError:
        comparison_value = current
    _field_header(label, field_path, comparison_value, generated.get(field))
    value = st.text_input(
        label,
        value=text,
        key=key,
        label_visibility="collapsed",
    )
    if not value:
        record.pop(field, None)
    else:
        try:
            record[field] = int(value)
        except ValueError:
            record[field] = value


def _enum(
    record: Record,
    generated: Record,
    field: str,
    label: str,
    enum_type: type[Enum],
    path: str,
) -> None:
    field_path = _field_path(path, field)
    key = _widget_key(field_path)
    options = [None, *(member.value for member in enum_type)]
    current = st.session_state.get(key, record.get(field))
    if current not in options:
        options.append(current)
    _field_header(label, field_path, current, generated.get(field))
    value = st.selectbox(
        label,
        options,
        index=options.index(current),
        format_func=lambda item: "—" if item is None else str(item),
        key=key,
        label_visibility="collapsed",
    )
    _set_optional(record, field, value)


def _enum_list(
    record: Record,
    generated: Record,
    field: str,
    label: str,
    enum_type: type[Enum],
    path: str,
) -> None:
    field_path = _field_path(path, field)
    key = _widget_key(field_path)
    current = st.session_state.get(key, record.get(field) or [])
    options = [member.value for member in enum_type]
    options.extend(item for item in current if item not in options)
    _field_header(label, field_path, current or None, generated.get(field))
    value = st.multiselect(
        label,
        options,
        default=current,
        key=key,
        label_visibility="collapsed",
    )
    _set_optional(record, field, value or None)


def _text_list(
    record: Record,
    generated: Record,
    field: str,
    label: str,
    path: str = "",
    *,
    multiline: bool = False,
) -> None:
    field_path = _field_path(path, field)
    stored_items = record.get(field) or []
    placeholder = not stored_items
    items = stored_items if stored_items else [""]
    generated_items = generated.get(field) or []
    _field_header(label, field_path, record.get(field), generated.get(field))
    for index, item in enumerate(items):
        item_path = f"{field_path}[{index}]"
        key = _widget_key(item_path)
        current = st.session_state.get(key, item)
        reference = generated_items[index] if index < len(generated_items) else None
        value_column, up_column, down_column, remove_column = st.columns(
            (8, 1, 1, 1), vertical_alignment="bottom"
        )
        widget = value_column.text_area if multiline else value_column.text_input
        value = widget(
            _changed_label(
                f"{label[:-1] if label.endswith('s') else 'Item'} {index + 1}",
                current or None,
                reference,
            ),
            value=current,
            key=key,
        )
        items[index] = value
        if up_column.button(
            "↑",
            key=_widget_key(f"{item_path}:up"),
            disabled=placeholder or index == 0,
        ):
            _move_item(record, field, index, index - 1, field_path)
        if down_column.button(
            "↓",
            key=_widget_key(f"{item_path}:down"),
            disabled=placeholder or index == len(items) - 1,
        ):
            _move_item(record, field, index, index + 1, field_path)
        if remove_column.button(
            "×", key=_widget_key(f"{item_path}:remove"), disabled=placeholder
        ):
            _remove_item(record, field, index, field_path)
    if placeholder:
        _set_optional(record, field, items if items[0] else None)
    else:
        record[field] = items
    if st.button(
        f"Add {label[:-1].lower() if label.endswith('s') else 'item'}",
        key=_widget_key(f"{field_path}:add"),
        disabled=placeholder and not items[0],
    ):
        items.append("")
        record[field] = items
        _reset_list_widgets(field_path)


def _object_list(
    record: Record,
    generated: Record,
    field: str,
    label: str,
    item_label: str,
    renderer: ItemRenderer,
    path: str = "",
) -> None:
    field_path = _field_path(path, field)
    stored_items = record.get(field) or []
    placeholder = not stored_items
    items = stored_items if stored_items else [{}]
    generated_items = generated.get(field) or []
    _field_header(label, field_path, record.get(field), generated.get(field))
    for index, item in enumerate(items):
        reference = generated_items[index] if index < len(generated_items) else {}
        item_path = f"{field_path}[{index}]"
        with st.expander(
            _changed_label(
                f"{item_label} {index + 1}",
                item,
                reference if reference else {} if placeholder else None,
            ),
            expanded=True,
        ):
            up_column, down_column, remove_column, _ = st.columns((1, 1, 1, 7))
            if up_column.button(
                "Move up",
                key=_widget_key(f"{item_path}:up"),
                disabled=placeholder or index == 0,
            ):
                _move_item(record, field, index, index - 1, field_path)
            if down_column.button(
                "Move down",
                key=_widget_key(f"{item_path}:down"),
                disabled=placeholder or index == len(items) - 1,
            ):
                _move_item(record, field, index, index + 1, field_path)
            if remove_column.button(
                "Remove",
                key=_widget_key(f"{item_path}:remove"),
                disabled=placeholder,
            ):
                _remove_item(record, field, index, field_path)
            renderer(item, reference, item_path)
    if placeholder:
        _set_optional(record, field, items if items[0] else None)
    else:
        record[field] = items
    if st.button(
        f"Add {item_label.lower()}",
        key=_widget_key(f"{field_path}:add"),
        disabled=placeholder and not items[0],
    ):
        items.append({})
        record[field] = items
        _reset_list_widgets(field_path)


def _optional_object(
    record: Record,
    generated: Record,
    field: str,
    label: str,
    renderer: ItemRenderer,
    path: str = "",
) -> None:
    field_path = _field_path(path, field)
    existing = record.get(field)
    current = existing if isinstance(existing, dict) else {}
    reference = generated.get(field)
    _field_header(label, field_path, existing, reference)
    with st.container(border=True):
        renderer(current, reference if isinstance(reference, dict) else {}, field_path)
    _set_optional(record, field, current or None)


def _move_item(
    record: Record, field: str, source: int, destination: int, field_path: str
) -> None:
    items = record[field]
    items[source], items[destination] = items[destination], items[source]
    _reset_list_widgets(field_path)


def _remove_item(record: Record, field: str, index: int, field_path: str) -> None:
    items = record[field]
    items.pop(index)
    _set_optional(record, field, items or None)
    _reset_list_widgets(field_path)


def _reset_list_widgets(field_path: str) -> None:
    _clear_widget_prefix(field_path)
    st.rerun()


def _clear_widget_prefix(path: str) -> None:
    prefix = _widget_key(path)
    for key in list(st.session_state):
        if str(key).startswith(prefix):
            del st.session_state[key]


def _set_optional(record: Record, field: str, value: Any) -> None:
    if value is None or value == "" or value == []:
        record.pop(field, None)
    else:
        record[field] = value


def _field_path(path: str, field: str) -> str:
    return f"{path}.{field}" if path else field


def _widget_key(path: str) -> str:
    snapshot_id = st.session_state.get("metadata_reviewer_item_id", "record")
    return f"metadata_reviewer_widget:{snapshot_id}:{path}"


def _field_header(label: str, path: str, value: Any, reference: Any) -> None:
    description, examples = _field_information().get(_schema_path(path), (None, None))
    label_column, generated_column, examples_column = st.columns(
        (8, 1, 1), vertical_alignment="center"
    )
    label_column.markdown(_changed_label(f"**{label}**", value, reference))
    with generated_column.popover(
        "?",
        help="Generated value",
        key=_widget_key(f"{path}:generated"),
    ):
        st.markdown("**Generated value**")
        _render_json_value(reference, "Missing or null")
    with examples_column.popover(
        "ⓘ",
        help="Schema examples",
        key=_widget_key(f"{path}:examples"),
    ):
        st.markdown("**Schema examples**")
        _render_json_value(examples, "No examples supplied")
    if description:
        st.caption(description)


def _render_json_value(value: Any, empty_message: str) -> None:
    if value is None:
        st.caption(empty_message)
    elif isinstance(value, (dict, list)):
        st.json(value, expanded=True)
    else:
        st.code(json.dumps(value, ensure_ascii=False), language="json", wrap_lines=True)


@lru_cache(maxsize=1)
def _field_information() -> dict[str, tuple[str | None, list[Any] | None]]:
    information: dict[str, tuple[str | None, list[Any] | None]] = {}

    def visit(model: type[BaseModel], prefix: str = "") -> None:
        for name, field in model.model_fields.items():
            path = _field_path(prefix, name)
            information[path] = (field.description, field.examples)
            nested_model = _nested_model(field.annotation)
            if nested_model is not None:
                visit(nested_model, path)

    visit(DataSnapshotMetadata)
    return information


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        nested_model = _nested_model(argument)
        if nested_model is not None:
            return nested_model
    return None


def _schema_path(path: str) -> str:
    return re.sub(r"\[[0-9]+\]", "", path)


def _changed_label(label: str, value: Any, reference: Any) -> str:
    if metadata_values_equal(value, reference):
        return label
    return f":orange[{label}]"
