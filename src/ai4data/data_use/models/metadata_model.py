"""Passage-grounded metadata for a cited data mention (LFM2.5 fine-tune).

``rafmacalaba/lfm2.5-350M-focus-metadata`` labels a mention that is already
known; it does not detect mentions. The user turn is the same for both tasks,
only the system prompt selects ``attributes`` or ``usage``.
"""

import json
from typing import Any, Dict, List, Optional

import torch

ATTRIBUTES_SYSTEM = (
    "Extract theme tags and source attributes for the cited data mention using only the "
    "passage. Output ONLY JSON with keys tags and attributes. tags is an array from "
    "geospatial, forced_displacement, agriculture, employment, poverty, education, health, "
    "climate. attributes has geography, year, version, producer, acronym (exact passage text "
    "or null) and data_type (survey, census, administrative, microdata, database, indicator, "
    "geospatial, report, or other). Do not infer values absent from the passage."
)
USAGE_SYSTEM = (
    "Describe how the author uses the cited data mention, using only the passage. Output "
    "ONLY JSON with keys role, usage_action, summary. Pick usage_action from these "
    "boundaries: source = the mention is where the quoted figures come from; analyze = the "
    "author computes, estimates, regresses or aggregates values from it; validate = it is "
    "used to cross-check or corroborate another estimate; curate = the author describes "
    "collecting, compiling, archiving or maintaining it; inform = cited for context or trend "
    "without its values being used. role is primary, supporting, or background. summary is "
    "one concise sentence in English, whatever language the passage is in; do not invent "
    "details."
)

_MAX_PROMPT_TOKENS = 2048


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    # LFM2.5 is a thinking model: it may emit <|im_start|> reasoning before the JSON.
    answer = raw.split("<|im_start|>")[-1] if "<|im_start|>" in raw else raw
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(answer[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _prompt(tokenizer, system: str, item: Dict[str, str]) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f'Text: {item["passage"]}\n\nMention: "{item["mention"]}"'},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def predict_metadata(
    tokenizer,
    model,
    system: str,
    items: List[Dict[str, str]],
    max_new_tokens: int = 256,
    batch_size: int = 16,
) -> List[Optional[Dict[str, Any]]]:
    """Run one task greedily over ``items`` (dicts with ``passage`` and ``mention``).

    Prompts are left-padded and generated in batches of ``batch_size`` (sorted by
    length so padding stays small); results come back in input order. Returns one
    parsed JSON dict per item, or ``None`` when the output has no valid JSON.
    """
    prompts = [_prompt(tokenizer, system, it) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    old_side, old_trunc = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side, tokenizer.truncation_side = "left", "left"
    try:
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            enc = tokenizer(
                [prompts[i] for i in idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_MAX_PROMPT_TOKENS,
                add_special_tokens=False,  # the chat template already adds them
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            n_prompt = enc["input_ids"].shape[1]
            for row, i in zip(out, idx):
                raw = tokenizer.decode(row[n_prompt:], skip_special_tokens=True)
                results[i] = _parse_json(raw)
    finally:
        tokenizer.padding_side, tokenizer.truncation_side = old_side, old_trunc
    return results
