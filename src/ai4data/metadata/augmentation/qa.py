"""Self-consistency QA prompts for variable group curation.

After the curation LLM produces a proposed variable group (label, txt,
definition, and selected variables), the QA agent assesses whether those
parts are mutually coherent — the second stage of self-consistency prompting.
"""

from __future__ import annotations

import json
from typing import Dict, List

from .prompts import render_variable_list
from .schemas import (
    DictionaryVariable,
    VariableGroupCurationResult,
    VariableGroupQAResult,
)


QA_SYSTEM_PROMPT = """\
You are a metadata quality assurance reviewer for statistical data catalogs.

Your task is to assess whether a proposed DDI-style variable group is
self-consistent. A group is self-consistent when its label, description,
grouping rationale, and selected variables all describe the same subject
matter and support one another.

OUTPUT SCHEMA:
Return exactly one JSON object with this structure:

{
  "is_self_consistent": true,
  "rationale": "string"
}

FIELD RULES:

is_self_consistent:
- Return true only if the label, txt, definition, and selected variables
  are mutually coherent.
- Return false if any part contradicts or is unsupported by the others.

rationale:
- Briefly explain your decision.
- When is_self_consistent is false, state the specific inconsistency
  (e.g., label claims health but variables are geographic identifiers).
- When is_self_consistent is true, a short confirmation is sufficient.

ASSESSMENT CRITERIA:
- Does the label accurately summarize the shared subject of the selected
  variables?
- Does txt describe what the selected variables collectively measure or
  represent, without overclaiming or referring to unrelated concepts?
- Does definition provide a plausible rationale for grouping these
  specific variables?
- Do the selected variable labels support the stated label and description?
- Ignore any candidate variables not included in the proposed group.

Output ONLY valid JSON. Do not output markdown, explanations outside the
JSON, comments, or additional text.
"""

QA_USER_PROMPT_TEMPLATE = """\
# TASK

Assess whether the following proposed variable group is self-consistent.

Review the group metadata and the labels of the selected variables. Decide
whether the label, txt, definition, and variables form a coherent group.

# PROPOSED GROUP

Label: {label}
Universe: {universe}
Notes: {notes}
Description (txt): {txt}
Definition: {definition}
Selected variables: {variables_json}

# SELECTED VARIABLE LABELS

{variable_list}

Return ONLY valid JSON matching the required schema.
"""


def render_qa_user_prompt(
    cluster_variables: List[DictionaryVariable],
    curation: VariableGroupCurationResult,
) -> str:
    """Render the QA user prompt for a proposed variable group."""
    selected_names = set(curation.variables)
    selected = [v for v in cluster_variables if v.variable_name in selected_names]
    return QA_USER_PROMPT_TEMPLATE.format(
        label=curation.label,
        universe=curation.universe or "(empty)",
        notes=curation.notes or "(empty)",
        txt=curation.txt,
        definition=curation.definition,
        variables_json=json.dumps(curation.variables),
        variable_list=render_variable_list(selected),
    )


def get_qa_response_format() -> Dict:
    """Return the JSON schema dict for structured QA responses."""
    schema = VariableGroupQAResult.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "variable_group_qa",
            "strict": True,
            "schema": schema,
        },
    }
