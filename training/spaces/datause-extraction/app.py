"""Data-use extraction on a ZeroGPU Space.

Two fine-tuned models run in sequence, both loaded once at startup:

1. ``rafmacalaba/gliner_agents_v3`` (GLiNER) finds data mentions in a passage and
   types each one as NAMED_DATA, DESCRIPTIVE_DATA or VAGUE_DATA.
2. ``rafmacalaba/lfm2.5-350M-focus-metadata`` (LFM2.5) reads each *named* mention in
   the passage chunk it came from and answers two tasks as JSON: source attributes
   (``attributes``) and how the author uses the data (``usage``).

The GPU is attached only while ``_extract`` runs (ZeroGPU).
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import spaces
import torch
from gliner import GLiNER
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ENTITY_ID = "rafmacalaba/gliner_agents_v3"
METADATA_ID = "rafmacalaba/lfm2.5-350M-focus-metadata"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENTITY_LABELS = {"NAMED_DATA": "Named", "DESCRIPTIVE_DATA": "Descriptive", "VAGUE_DATA": "Vague"}
THEME_TAGS = {"geospatial", "forced_displacement", "agriculture", "employment", "poverty", "education", "health", "climate"}
DATA_TYPES = {"survey", "census", "administrative", "microdata", "database", "indicator", "geospatial", "report", "other"}
USAGE_ROLES = {"primary", "supporting", "background"}
USAGE_ACTIONS = {"source", "analyze", "curate", "validate", "inform"}

# The user turn is the same for both tasks; the system prompt selects the task.
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

CHUNK_WORDS, CHUNK_OVERLAP = 200, 50  # GLiNER was fine-tuned on short passages
MAX_WORDS = 4000
MAX_PROMPT_TOKENS = 2048
TABLE_HEADERS = [
    "Mention", "Type", "Score", "Producer", "Year", "Geography", "Acronym",
    "Data type", "Themes", "Role", "Usage", "Summary",
]

# Models go on the GPU at import time; ZeroGPU emulates CUDA outside @spaces.GPU.
entity_model = GLiNER.from_pretrained(ENTITY_ID)
entity_model.to(DEVICE)
entity_model.eval()
tokenizer = AutoTokenizer.from_pretrained(METADATA_ID)
metadata_model = AutoModelForCausalLM.from_pretrained(METADATA_ID, dtype=torch.bfloat16)
metadata_model.to(DEVICE)
metadata_model.eval()


def chunk_text(text: str) -> List[tuple]:
    """Overlapping word windows as (char_offset, chunk_text)."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    chunks, i = [], 0
    while i < len(words):
        j = min(i + CHUNK_WORDS, len(words))
        start, end = words[i][0], words[j - 1][1]
        chunks.append((start, text[start:end]))
        if j == len(words):
            break
        i += CHUNK_WORDS - CHUNK_OVERLAP
    return chunks


def find_mentions(text: str, threshold: float) -> List[Dict[str, Any]]:
    """Run GLiNER over each chunk and keep the best non-overlapping spans."""
    candidates = []
    for offset, chunk in chunk_text(text):
        for ent in entity_model.predict_entities(chunk, list(ENTITY_LABELS), threshold=threshold):
            candidates.append({
                "start": offset + ent["start"],
                "end": offset + ent["end"],
                "label": ent["label"],
                "score": float(ent["score"]),
                "passage": chunk,
            })
    kept = []
    for cand in sorted(candidates, key=lambda c: -c["score"]):
        if all(cand["end"] <= k["start"] or cand["start"] >= k["end"] for k in kept):
            kept.append(cand)
    kept.sort(key=lambda c: c["start"])
    for m in kept:
        m["mention"] = text[m["start"]:m["end"]]
    return kept


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def run_task(system: str, items: List[Dict[str, str]], batch_size: int = 16) -> List[Optional[Dict[str, Any]]]:
    """Greedy generation for one task over (passage, mention) items, in input order."""
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f'Text: {it["passage"]}\n\nMention: "{it["mention"]}"'},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for it in items
    ]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))  # similar lengths per batch
    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    tokenizer.padding_side = tokenizer.truncation_side = "left"
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tokenizer(
            [prompts[i] for i in idx],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            add_special_tokens=False,  # the chat template already adds them
        ).to(metadata_model.device)
        with torch.no_grad():
            out = metadata_model.generate(
                **enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        prompt_len = enc["input_ids"].shape[1]
        for row, i in zip(out, idx):
            results[i] = _parse_json(tokenizer.decode(row[prompt_len:], skip_special_tokens=True))
    return results


def verbatim(value: Any, passage: str) -> str:
    """Attribute values must be passage text; anything else is dropped."""
    value = "" if value is None else str(value).strip()
    return value if value and value.lower() in passage.lower() else ""


def metadata_row(mention: Dict[str, Any], attrs: Optional[Dict], usage: Optional[Dict]) -> List[Any]:
    passage = mention["passage"]
    body = (attrs or {}).get("attributes") or {}
    tags = [t for t in (attrs or {}).get("tags") or [] if t in THEME_TAGS]
    data_type = body.get("data_type")
    usage = usage or {}
    return [
        mention["mention"],
        ENTITY_LABELS[mention["label"]],
        round(mention["score"], 2),
        verbatim(body.get("producer"), passage),
        verbatim(body.get("year"), passage),
        verbatim(body.get("geography"), passage),
        verbatim(body.get("acronym"), passage),
        data_type if data_type in DATA_TYPES else "",
        ", ".join(tags),
        usage.get("role") if usage.get("role") in USAGE_ROLES else "",
        usage.get("usage_action") if usage.get("usage_action") in USAGE_ACTIONS else "",
        str(usage.get("summary") or "").strip(),
    ]


def _duration(text: str, threshold: float) -> int:
    return min(120, 20 + len(text.split()) // 40)


@spaces.GPU(duration=_duration)
def _extract(text: str, threshold: float):
    started = time.perf_counter()
    mentions = find_mentions(text, threshold)

    named = [m for m in mentions if m["label"] == "NAMED_DATA"]
    items = [{"passage": m["passage"], "mention": m["mention"]} for m in named]
    attrs = run_task(ATTRIBUTES_SYSTEM, items) if items else []
    usage = run_task(USAGE_SYSTEM, items) if items else []
    by_start = {m["start"]: (a, u) for m, a, u in zip(named, attrs, usage)}

    rows, segments, pos = [], [], 0
    for m in mentions:
        segments += [(text[pos:m["start"]], None), (m["mention"], ENTITY_LABELS[m["label"]])]
        pos = m["end"]
        a, u = by_start.get(m["start"], (None, None))
        rows.append(metadata_row(m, a, u) if m["label"] == "NAMED_DATA" else
                    [m["mention"], ENTITY_LABELS[m["label"]], round(m["score"], 2)] + [""] * 9)
    segments.append((text[pos:], None))

    counts = {label: sum(m["label"] == key for m in mentions) for key, label in ENTITY_LABELS.items()}
    status = (
        f"**{len(mentions)} mentions** ({counts['Named']} named, {counts['Descriptive']} descriptive, "
        f"{counts['Vague']} vague) in {time.perf_counter() - started:.1f} s."
    )
    return segments, rows, status


def run(text: str, threshold: float):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Paste a passage first.")
    if len(text.split()) > MAX_WORDS:
        raise gr.Error(f"The text is over {MAX_WORDS:,} words. Paste a shorter passage.")
    return _extract(text, threshold)


EXAMPLES = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))
HEADER = (HERE / "header.html").read_text(encoding="utf-8")
CSS = (HERE / "style.css").read_text(encoding="utf-8")

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

with gr.Blocks(title="Data-use extraction — AI for Data", theme=THEME, css=CSS, analytics_enabled=False) as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="card"):
            text = gr.Textbox(
                label="Passage",
                lines=12,
                placeholder="Paste a passage from a report or paper…",
                value=EXAMPLES[0]["text"],
            )
            threshold = gr.Slider(
                0.1, 0.9, value=0.5, step=0.05, label="Mention threshold",
                info="Higher keeps fewer, surer mentions. On the holdout set, 0.5 gave the best F1 and 0.6 the best F0.5.",
            )
            button = gr.Button("Extract data use", variant="primary")
            gr.Examples(
                examples=[[e["text"], 0.5] for e in EXAMPLES],
                example_labels=[e["label"] for e in EXAMPLES],
                inputs=[text, threshold],
                label="Examples",
            )
        with gr.Column(scale=7, elem_classes="card"):
            status = gr.Markdown("Press **Extract data use** to run the two models.")
            highlighted = gr.HighlightedText(
                label="Data mentions",
                combine_adjacent=False,
                show_legend=True,
                color_map={"Named": "#a5b4fc", "Descriptive": "#99f6e4", "Vague": "#fde68a"},
            )
    with gr.Column(elem_classes="card"):
        table = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["str", "str", "number"] + ["str"] * 9,
            label="Attributes and usage of each mention (named mentions only)",
            interactive=False,
            wrap=True,
        )
    button.click(run, [text, threshold], [highlighted, table, status])

if __name__ == "__main__":
    demo.queue().launch()
