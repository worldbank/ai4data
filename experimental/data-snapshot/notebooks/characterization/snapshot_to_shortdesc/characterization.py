import base64
from dotenv import load_dotenv
import json
from pathlib import Path
import os
import re
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# USD per 1M tokens
PRICING = {
    "gpt-4o-mini": {
        "input_per_1M": 0.15,
        "output_per_1M": 0.60,
    }
}


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_user_prompt(template: str, metadata: dict) -> str:
    text = template
    for key, value in metadata.items():
        placeholder = f"{{{{{key}}}}}"
        text = text.replace(placeholder, str(value) if value is not None else "unknown")
    return text


def compute_cost(model: str, usage) -> dict:
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


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping a JSON block.

    Parameters
    ----------
    text : str
        Raw LLM output text.

    Returns
    -------
    str
        Text with leading/trailing code fences removed.
    """
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def analyze_snapshot(
    system_prompt: str,
    user_prompt: str,
    image_path: str,
    model: str = "gpt-4o-mini",
    max_output_tokens: int = 300,
) -> dict:
    """Send a snapshot image to the OpenAI Responses API and parse the result.

    Parameters
    ----------
    system_prompt : str
        System-level instructions for the model.
    user_prompt : str
        Rendered user prompt with metadata placeholders filled.
    image_path : str
        Path to the snapshot image file.
    model : str
        OpenAI model name.
    max_output_tokens : int
        Maximum tokens for the model response.

    Returns
    -------
    dict
        Always contains ``parsed_output``, ``raw_output``, ``usage``,
        ``cost``, and ``error``. On parse failure, ``parsed_output`` is
        ``None`` and ``error`` describes the issue.
    """
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
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
        max_output_tokens=max_output_tokens,
    )

    raw_output = response.output_text.strip()
    usage = response.usage
    cost = compute_cost(model, usage)

    cleaned = _strip_code_fences(raw_output)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "parsed_output": None,
            "raw_output": raw_output,
            "usage": usage,
            "cost": cost,
            "error": f"JSON parse error: {e}",
        }

    return {
        "parsed_output": parsed,
        "raw_output": raw_output,
        "usage": usage,
        "cost": cost,
        "error": None,
    }
