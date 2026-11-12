"""Data-use extraction on a ZeroGPU Space.

Two fine-tuned models run in sequence, both loaded once at startup:

1. ``rafmacalaba/gliner_agents_v3`` (GLiNER) finds data mentions in a passage and
   types each one as NAMED_DATA, DESCRIPTIVE_DATA or VAGUE_DATA.
2. ``rafmacalaba/lfm2.5-350M-focus-metadata`` (LFM2.5) reads each *named* mention in
   the passage chunk it came from and answers two tasks as JSON: source attributes
   (``attributes``) and how the author uses the data (``usage``).

The GPU is attached only while ``_extract`` runs (ZeroGPU).
"""

import html
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


def metadata_fields(mention: Dict[str, Any], attrs: Optional[Dict], usage: Optional[Dict]) -> Dict[str, Any]:
    """Validated attributes and usage of one named mention."""
    passage = mention["passage"]
    body = (attrs or {}).get("attributes") or {}
    usage = usage or {}
    data_type = body.get("data_type")
    return {
        "producer": verbatim(body.get("producer"), passage),
        "year": verbatim(body.get("year"), passage),
        "geography": verbatim(body.get("geography"), passage),
        "acronym": verbatim(body.get("acronym"), passage),
        "data_type": data_type if data_type in DATA_TYPES else "",
        "themes": [t for t in (attrs or {}).get("tags") or [] if t in THEME_TAGS],
        "role": usage.get("role") if usage.get("role") in USAGE_ROLES else "",
        "usage_action": usage.get("usage_action") if usage.get("usage_action") in USAGE_ACTIONS else "",
        "summary": str(usage.get("summary") or "").strip(),
    }


def _duration(text: str, threshold: float) -> int:
    return min(120, 20 + len(text.split()) // 40)


@spaces.GPU(duration=_duration)
def _extract(text: str, threshold: float) -> List[Dict[str, Any]]:
    mentions = find_mentions(text, threshold)
    named = [m for m in mentions if m["label"] == "NAMED_DATA"]
    items = [{"passage": m["passage"], "mention": m["mention"]} for m in named]
    attrs = run_task(ATTRIBUTES_SYSTEM, items) if items else []
    usage = run_task(USAGE_SYSTEM, items) if items else []
    for m, a, u in zip(named, attrs, usage):
        m["fields"] = metadata_fields(m, a, u)
    return mentions


# ── rendering: plain HTML that grows with its content, so only the page scrolls ──
def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_passage(text: str, mentions: List[Dict[str, Any]]) -> str:
    parts, pos = [], 0
    for m in mentions:
        kind = ENTITY_LABELS[m["label"]]
        parts.append(esc(text[pos:m["start"]]))
        parts.append(f'<mark class="hl {kind.lower()}" title="{kind} · score {m["score"]:.2f}">{esc(m["mention"])}</mark>')
        pos = m["end"]
    parts.append(esc(text[pos:]))
    return "".join(parts)


def render_card(m: Dict[str, Any]) -> str:
    f = m["fields"]
    facts = [
        ("Producer", f["producer"]), ("Year", f["year"]), ("Geography", f["geography"]),
        ("Acronym", f["acronym"]), ("Data type", f["data_type"]), ("Themes", ", ".join(f["themes"])),
    ]
    rows = "".join(f"<dt>{k}</dt><dd>{esc(v)}</dd>" for k, v in facts if v)
    badges = "".join(
        f'<span class="badge {cls}">{esc(v)}</span>'
        for cls, v in (("role", f["role"]), ("action", f["usage_action"])) if v
    )
    summary = f'<p class="summary">{esc(f["summary"])}</p>' if f["summary"] else ""
    return (
        f'<article class="mcard"><h3>{esc(m["mention"])}</h3>'
        f'<div class="badges">{badges}</div>{summary}'
        f'{"<dl>" + rows + "</dl>" if rows else ""}</article>'
    )


def render_results(text: str, mentions: List[Dict[str, Any]], elapsed: float) -> str:
    if not mentions:
        return '<div class="results"><p class="empty">No data mentions found at this threshold. Try lowering it.</p></div>'
    counts = {label: sum(ENTITY_LABELS[m["label"]] == label for m in mentions) for label in ENTITY_LABELS.values()}
    legend = "".join(f'<span class="legend {k.lower()}">{k} {n}</span>' for k, n in counts.items())
    named = [m for m in mentions if m["label"] == "NAMED_DATA"]
    other = [m for m in mentions if m["label"] != "NAMED_DATA"]
    cards = "".join(render_card(m) for m in named)
    others = ""
    if other:
        chips = "".join(
            f'<span class="chip {ENTITY_LABELS[m["label"]].lower()}">{esc(m["mention"])}</span>' for m in other
        )
        others = (
            '<section class="panel"><h2>Descriptive and vague mentions</h2>'
            '<p class="note">The metadata model reads named mentions only.</p>'
            f'<div class="chips">{chips}</div></section>'
        )
    sections = [
        f'<p class="status"><strong>{len(mentions)} mentions</strong> in {elapsed:.1f} s</p>',
        f'<section class="panel"><h2>Data mentions</h2><div class="legends">{legend}</div>'
        f'<p class="passage">{render_passage(text, mentions)}</p></section>',
    ]
    if named:
        sections.append(f'<section><h2>Named mentions</h2><div class="mgrid">{cards}</div></section>')
    sections.append(others)
    return '<div class="results">' + "".join(sections) + "</div>"


def run(text: str, threshold: float) -> str:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Paste a passage first.")
    if len(text.split()) > MAX_WORDS:
        raise gr.Error(f"The text is over {MAX_WORDS:,} words. Paste a shorter passage.")
    started = time.perf_counter()
    mentions = _extract(text, threshold)
    return render_results(text, mentions, time.perf_counter() - started)


EXAMPLES = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))
HEADER = (HERE / "header.html").read_text(encoding="utf-8")
CSS = (HERE / "style.css").read_text(encoding="utf-8")

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
PLACEHOLDER = '<div class="results"><p class="empty">Press <strong>Extract data use</strong> to run the two models.</p></div>'

with gr.Blocks(title="Data-use extraction — AI for Data", theme=THEME, css=CSS, analytics_enabled=False) as demo:
    gr.HTML(HEADER)
    with gr.Column(elem_classes="card"):
        text = gr.Textbox(
            label="Passage",
            lines=7,
            max_lines=30,  # grows with the text instead of scrolling
            placeholder="Paste a passage from a report or paper…",
            value=EXAMPLES[0]["text"],
        )
        with gr.Row(equal_height=True):
            threshold = gr.Slider(
                0.1, 0.9, value=0.5, step=0.05, label="Mention threshold",
                info="Higher keeps fewer, surer mentions. On the holdout set, 0.5 gave the best F1 and 0.6 the best F0.5.",
                scale=3,
            )
            button = gr.Button("Extract data use", variant="primary", scale=1, min_width=180)
        with gr.Row(elem_classes="examples"):
            gr.HTML('<span class="examples-label">Examples</span>', elem_classes="examples-label-wrap")
            for example in EXAMPLES:
                chip = gr.Button(example["label"], size="sm", variant="secondary", min_width=0)
                chip.click(lambda t=example["text"]: t, None, text)
    results = gr.HTML(PLACEHOLDER)
    button.click(run, [text, threshold], results)

if __name__ == "__main__":
    demo.queue().launch()
