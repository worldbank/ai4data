"""Assign discovered field profiles to a refined metadata concept inventory."""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# USD per 1M tokens
PRICING = {
    "gpt-5.5": {
        "input_per_1M": 5.00,
        "output_per_1M": 30.00,
    },
    "gpt-5.4": {
        "input_per_1M": 2.50,
        "output_per_1M": 15.00,
    },
    "gpt-5.4-mini": {
        "input_per_1M": 0.75,
        "output_per_1M": 4.50,
    },
}


class Classification(BaseModel):
    """Inferred canonical name from a field profile.

    Attributes
    ----------
    canonical_name : str
        The inferred canonical name.
    confidence : Literal["high", "medium", "low"]
        Confidence value of the LLM for the inference.
    """

    canonical_name: str
    confidence: Literal["high", "medium", "low"]


def make_classification_model(
    canonical_names: tuple[str, ...],
) -> type[BaseModel]:
    """Build a Classification model constrained to known canonical names.

    Parameters
    ----------
    canonical_names : tuple[str, ...]
        Allowed canonical concept names from the ontology.

    Returns
    -------
    type[BaseModel]
        A Pydantic model with ``canonical_name`` constrained to an enum of
        the provided names and ``confidence`` as
        ``Literal["high", "medium", "low"]``.
    """
    all_names = canonical_names
    CanonicalName = Enum("CanonicalName", {name: name for name in all_names})

    class ConstrainedClassification(BaseModel):
        """Inferred canonical name from a field profile (enum-constrained)."""

        canonical_name: CanonicalName
        confidence: Literal["high", "medium", "low"]

    return ConstrainedClassification


def load_prompt(path: str) -> str:
    """Read a UTF-8 prompt file.

    Parameters
    ----------
    path : str
        Prompt path.

    Returns
    -------
    str
        Prompt contents.
    """
    return Path(path).read_text(encoding="utf-8")


def compute_cost(model: str, usage: Any) -> dict:
    """Compute the USD cost of an API call from a usage object.

    Parameters
    ----------
    model : str
        Model name (must exist in ``PRICING``).
    usage : ResponseUsage
        The ``response.usage`` object returned by the OpenAI Responses API.

    Returns
    -------
    dict
        Token counts and cost breakdown in USD.
    """
    pricing = PRICING[model]

    input_cost = (usage.input_tokens / 1e6) * pricing["input_per_1M"]
    output_cost = (usage.output_tokens / 1e6) * pricing["output_per_1M"]

    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def analyze_field_profile(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5.4-mini",
    classification_model: type[BaseModel] = Classification,
) -> object:
    """Send prompts to the OpenAI Responses API.

    Parameters
    ----------
    system_prompt : str
        System-level instructions for the model.
    user_prompt : str
        Rendered user prompt with placeholders filled.
    model : str
        OpenAI model name.
    classification_model : type[BaseModel]
        Pydantic model for structured output parsing. Use
        ``make_classification_model`` to build one constrained to your
        ontology. Defaults to the unconstrained ``Classification``.

    Returns
    -------
    Response
        Parsed Responses API response with ``output_parsed`` populated
        when successful.
    """
    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model=model,
        reasoning={"effort": "medium"},
        text_format=classification_model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    )

    return response


def process_response(response: object, model: str) -> dict:
    """Parse Response object and add usage details.

    Parameters
    ----------
    response : Response
        Response object to parse
    model : str
        OpenAI model name.

    Returns
    -------
    dict
        Always contains ``parsed_output``, ``raw_output``, ``usage``,
        ``cost``, and ``error``. On parse failure, ``parsed_output`` is
        ``None`` and ``error`` describes the issue.
    """
    raw_output = response.output_text.strip()
    usage = response.usage
    cost = compute_cost(model, usage)
    parsed_output = getattr(response, "output_parsed", None)
    error = None

    if parsed_output is None:
        error = "Structured output parse failed or response was incomplete."
        parsed_output_dict = None
    else:
        parsed_output_dict = parsed_output.model_dump(mode="json")

    output_dict = {
        "raw_output": raw_output,
        "usage": usage.model_dump(),
        "cost": cost,
        "status": response.status,
        "incomplete_details": response.incomplete_details,
        "parsed_output": parsed_output_dict,
        "error": error,
    }

    return output_dict
