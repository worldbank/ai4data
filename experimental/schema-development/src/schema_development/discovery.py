"""Discover reusable metadata fields from individual data snapshots."""

import base64
import os
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


class CandidateField(BaseModel):
    """Candidate metadata field discovered from a snapshot.

    Attributes
    ----------
    metadata_field : str
        Reusable metadata field name.
    observed_value : str
        Value observed in the snapshot or document metadata.
    description : str
        Description of what the field represents.
    source_level : Literal["snapshot", "document", "both"]
        Source where the value was observed.
    discovery_value : Literal["high", "medium", "low"]
        Estimated value of the field for discovery use cases.
    reasoning : str
        Explanation of why the field is useful.
    """

    metadata_field: str
    observed_value: str
    description: str
    source_level: Literal["snapshot", "document", "both"]
    discovery_value: Literal["high", "medium", "low"]
    reasoning: str


class SchemaDiscoveryResult(BaseModel):
    """Structured schema discovery result.

    Attributes
    ----------
    fields : list[CandidateField]
        Candidate metadata fields discovered for the snapshot.
    """

    fields: list[CandidateField]


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


def _encode_image_to_data_url(image_path: str) -> str:
    """Read a local image file and return a base64 data URL.

    Parameters
    ----------
    image_path : str
        Path to the image file (PNG, JPEG, etc.).

    Returns
    -------
    str
        A ``data:image/<ext>;base64,...`` URL string.
    """
    path = Path(image_path)
    suffix = path.suffix.lstrip(".").lower()
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    mime = mime_map.get(suffix, f"image/{suffix}")

    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def analyze_snapshot(
    system_prompt: str,
    user_prompt: str,
    image_path: str,
    model: str = "gpt-5.4-mini",
    max_output_tokens: int = 3000,
) -> object:
    """Send a snapshot image to the OpenAI Responses API.

    Parameters
    ----------
    system_prompt : str
        System-level instructions for the model.
    user_prompt : str
        Rendered user prompt with placeholders filled.
    image_path : str
        Path to the snapshot image file.
    model : str
        OpenAI model name.
    max_output_tokens : int
        Maximum tokens for the model response.

    Returns
    -------
    Response
        Parsed Responses API response with ``output_parsed`` populated as a
        ``SchemaDiscoveryResult`` when successful.
    """
    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "medium"},
        text_format=SchemaDiscoveryResult,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": _encode_image_to_data_url(image_path),
                    },
                ],
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
