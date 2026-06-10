"""Prompt templates and utilities for data dictionary variable group curation.

The prompts follow an elicitation design: the LLM is asked to produce
structured JSON conforming to the ``VariableGroupCurationResult`` schema, not
free-form text. Deterministic DDI fields (``vgid``, ``group_type``, etc.)
are assembled downstream by the framework.

Design choices:
- ``SYSTEM_PROMPT``: Establishes the assistant role and output constraints.
  Explicitly forbids inventing variable names outside the provided list.
- ``USER_PROMPT_TEMPLATE``: Renders the cluster variable list with a numbered
  format so the LLM can reference variables by name easily.
- JSON schema: Derived from ``VariableGroupCurationResult.model_json_schema()``
  for strict type-safe validation at the API boundary.
"""

from __future__ import annotations

from typing import Dict, List

from .schemas import DictionaryVariable, VariableGroupCurationResult


# ----- Prompt templates ----- #

SYSTEM_PROMPT = """\
You are an expert metadata curator for social science, development, and administrative datasets.

Your task is to create one useful DDI-style variable group from a set of candidate variables.

The input variables are candidate variables identified as semantically related. They may all belong to the final variable group, or only a coherent subset may belong. Your role is to curate a useful variable group for a statistical data catalog.

OUTPUT SCHEMA:
Return exactly one JSON object with this structure:

{
  "label": "string",
  "universe": "string",
  "notes": "string",
  "txt": "string",
  "definition": "string",
  "variables": ["string"]
}

FIELD RULES:

label:
- A concise user-facing name for the variable group.
- Use 2–6 words in Title Case.
- Capture the shared concept or domain of the selected variables.
- Do not mention "cluster" or the grouping method.

universe:
- Describe the population or unit of observation only if clearly inferable from the variable labels.
- Examples: "Households", "Individuals", "Children under five", "Agricultural households".
- If not clearly inferable, return an empty string.

notes:
- Use only for important curation notes, such as excluded candidate variables or ambiguity in group membership.
- If variables are excluded, briefly state which ones and why.
- If no notes are needed, return an empty string.

txt:
- One or two factual sentences describing the subject matter represented by the variable group.
- Write like metadata in a statistical data catalog.
- Explain what the selected variables collectively describe.
- Prefer the common domain or construct over an exhaustive list of individual variables.
- Do not speculate beyond the provided labels.
- Do not refer to embeddings, clustering, or the grouping process.

definition:
- A brief rationale for the grouping.
- Explain why the selected variables form a coherent variable group.
- Ground the rationale in the shared subject matter of the selected variable labels.

variables:
- Include the variable identifiers that belong to the curated group.
- Use variable names EXACTLY as they appear in the input.
- Return them as a JSON array of strings.
- Include all variables that fit the group.
- Exclude variables that do not fit the common concept.
- The group may include all input variables or only a coherent subset.

CURATION RULES:
- Prefer a coherent and interpretable variable group over maximizing coverage.
- If all candidate variables fit the same concept, include all of them.
- If the candidates contain multiple concepts, select the strongest coherent group and exclude weaker or unrelated variables.
- Do not include variables merely because they share similar words.
- Do not invent, modify, or paraphrase variable names.
- Do not mention coding schemes, missing values, frequencies, or technical metadata unless central to the group.
- Output ONLY valid JSON.
- Do not output markdown, explanations, comments, or additional text.
"""

USER_PROMPT_TEMPLATE = """\
# TASK

Create one DDI-style variable group from the following candidate variables.

The variables are candidates for a thematic variable group. They may all belong to the final group, or only a coherent subset may belong.

Your task is to identify the most useful common subject matter, select the variables that belong to that group, and produce metadata using the required schema.

# VARIABLES

{variable_list}

Return ONLY valid JSON matching the required schema.
"""


# ----- Token counting ----- #


def count_tokens_approx(text: str) -> int:
    """Approximate token count: words * 1.3, rounded up.

    This is a fast approximation for budget checks. For exact counts, use a
    model-specific tokenizer (e.g., tiktoken for OpenAI models).

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Approximate token count.
    """
    return int(len(text.split()) * 1.3) + 1


# ----- Variable list rendering ----- #


def render_variable_list(variables: List[DictionaryVariable]) -> str:
    """Render a list of variables as a numbered text block for the user prompt.

    Each line has the format:
      ``N. variable_name: Label text — Optional description``

    Parameters
    ----------
    variables : list of DictionaryVariable
        Variables to render.

    Returns
    -------
    str
        Formatted numbered list.
    """
    lines = []
    for i, v in enumerate(variables, 1):
        line = f"{i}. {v.variable_name}: {v.label}"
        if v.description:
            desc = v.description.strip()
            if desc:
                line += f" — {desc}"
        lines.append(line)
    return "\n".join(lines)


def render_user_prompt(variables: List[DictionaryVariable]) -> str:
    """Render the full user prompt for a cluster.

    Parameters
    ----------
    variables : list of DictionaryVariable
        Cluster variables.

    Returns
    -------
    str
        Formatted user prompt string.
    """
    return USER_PROMPT_TEMPLATE.format(variable_list=render_variable_list(variables))


# ----- Response format schema ----- #


def get_variable_group_response_format() -> Dict:
    """Return the JSON schema dict for structured variable group curation responses.

    This is passed as the ``response_format`` argument to litellm / OpenAI
    structured output APIs. The schema is derived from
    ``VariableGroupCurationResult`` using Pydantic's ``model_json_schema()``,
    ensuring that the LLM response always matches the expected Python type.

    Returns
    -------
    dict
        ``{"type": "json_schema", "json_schema": {...}}`` format.
    """
    schema = VariableGroupCurationResult.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "variable_group_curation",
            "strict": True,
            "schema": schema,
        },
    }


def get_json_object_format() -> Dict:
    """Return a basic JSON object response format for providers without strict schema support.

    Some litellm providers accept ``{"type": "json_object"}`` but not the full
    JSON Schema format. Use this as a fallback when
    ``get_variable_group_response_format`` is not supported.

    Returns
    -------
    dict
    """
    return {"type": "json_object"}
