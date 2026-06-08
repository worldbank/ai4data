"""Pydantic schemas for data dictionary augmentation.

This module defines the canonical data structures used throughout the
metadata augmentation pipeline: loading variables, generating variable groups,
and assembling the augmented dictionary output.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ----- DDI group types ----- #

DDI_GROUP_TYPES = (
    "section",
    "multipleResp",
    "grid",
    "display",
    "repetition",
    "subject",
    "version",
    "iteration",
    "analysis",
    "pragmatic",
    "record",
    "file",
    "randomized",
    "other",
)

GroupType = Literal[
    "section",
    "multipleResp",
    "grid",
    "display",
    "repetition",
    "subject",
    "version",
    "iteration",
    "analysis",
    "pragmatic",
    "record",
    "file",
    "randomized",
    "other",
]


# ----- Base ----- #


class StrictBaseModel(BaseModel):
    """Base model with strict validation (no extra fields)."""

    model_config = ConfigDict(extra="forbid", strict=True)


# ----- Helpers ----- #


def make_vgid(label: str, cluster_id: int) -> str:
    """Build a stable VG_* identifier from label slug and cluster_id."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    slug = slug or "GROUP"
    return f"VG_{slug}_{cluster_id:04d}"[:128]


def make_uncategorized_vgid(cluster_id: int) -> str:
    """Build a deterministic vgid for fallback uncategorized groups."""
    return f"VG_UNCATEGORIZED_{cluster_id:04d}"


# ----- Input schema ----- #


class DictionaryVariable(StrictBaseModel):
    """A single variable entry in a data dictionary.

    Parameters
    ----------
    variable_name : str
        The machine-readable variable identifier (e.g., "HV001", "age_hh_head").
    label : str
        Human-readable label describing the variable (e.g., "Cluster number").
    description : str, optional
        Extended description or question text for the variable.
    value_labels : dict, optional
        Mapping of code strings to label strings for categorical variables
        (e.g., {"1": "Male", "2": "Female"}).
    """

    variable_name: str
    label: str
    description: Optional[str] = None
    value_labels: Optional[Dict[str, str]] = None


# ----- LLM response schema ----- #


class VariableGroupCurationResult(StrictBaseModel):
    """Raw LLM output for a single cluster's variable group curation.

    Only fields the LLM is expected to generate. Deterministic DDI fields
    (``vgid``, ``variable_groups``, ``group_type``, formatted ``variables``)
    are assembled by :class:`VariableGroup`.
    """

    label: str
    universe: str = ""
    notes: str = ""
    txt: str
    definition: str
    variables: Annotated[List[str], Field(min_length=1)]

    @model_validator(mode="after")
    def _normalize_variables(self) -> VariableGroupCurationResult:
        cleaned = [v.strip() for v in self.variables if v and v.strip()]
        if not cleaned:
            raise ValueError("variables must contain at least one non-empty name")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("variables must not contain duplicate names")
        object.__setattr__(self, "variables", cleaned)
        return self

    @classmethod
    def from_llm_response(
        cls,
        data: Union[str, dict],
        *,
        candidate_names: Set[str],
    ) -> VariableGroupCurationResult:
        """Parse and validate an LLM response against the candidate variable set."""
        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data

        result = cls.model_validate(parsed)

        unknown = [name for name in result.variables if name not in candidate_names]
        if unknown:
            raise ValueError(
                f"variables not in candidate set: {', '.join(unknown)}"
            )

        return result


class VariableGroupQAResult(StrictBaseModel):
    """LLM output from the self-consistency QA agent."""

    is_self_consistent: bool
    rationale: str = ""


# ----- Output schema ----- #


class VariableGroup(StrictBaseModel):
    """A validated DDI-style variable group."""

    vgid: str
    variables: str
    variable_groups: str = ""
    group_type: GroupType = "subject"
    label: str
    universe: str = ""
    notes: str = ""
    txt: str
    definition: str
    cluster_id: int
    qa_passed: Optional[bool] = None
    qa_rationale: str = ""

    @classmethod
    def from_curation(
        cls,
        curation: VariableGroupCurationResult,
        *,
        cluster_id: int,
        qa: Optional[VariableGroupQAResult] = None,
        qa_error: Optional[str] = None,
    ) -> VariableGroup:
        """Build a full DDI variable group from validated LLM curation output."""
        if qa is not None:
            qa_passed = qa.is_self_consistent
            qa_rationale = qa.rationale
        elif qa_error is not None:
            qa_passed = None
            qa_rationale = qa_error
        else:
            qa_passed = None
            qa_rationale = ""
        return cls(
            vgid=make_vgid(curation.label, cluster_id),
            variables=" ".join(curation.variables),
            variable_groups="",
            group_type="subject",
            label=curation.label,
            universe=curation.universe,
            notes=curation.notes,
            txt=curation.txt,
            definition=curation.definition,
            cluster_id=cluster_id,
            qa_passed=qa_passed,
            qa_rationale=qa_rationale,
        )

    @classmethod
    def uncategorized_fallback(
        cls,
        *,
        cluster_id: int,
        variable_names: List[str],
    ) -> VariableGroup:
        """Build a fallback group when LLM curation fails."""
        return cls(
            vgid=make_uncategorized_vgid(cluster_id),
            variables=" ".join(variable_names),
            variable_groups="",
            group_type="other",
            label="Uncategorized",
            universe="",
            notes="",
            txt="Variable group curation failed for this cluster.",
            definition="Fallback group assigned when curation could not be completed.",
            cluster_id=cluster_id,
        )


class VariableGroupAssignment(StrictBaseModel):
    """Maps a curated variable to its variable group and source cluster."""

    variable_name: str
    vgid: str
    label: str
    cluster_id: int


class AugmentedDictionary(StrictBaseModel):
    """Top-level output of the augmentation pipeline.

    Contains generated variable groups and assignments for curated variables,
    along with optional run metadata (model, timestamp, config).
    """

    dataset_id: Optional[str] = None
    variable_groups: List[VariableGroup]
    variable_assignments: List[VariableGroupAssignment]
    metadata: Optional[Dict[str, Any]] = None
