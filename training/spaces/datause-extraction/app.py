"""Data-use extraction on a ZeroGPU Space.

Two fine-tuned models run in sequence, both loaded once at startup:

1. ``rafmacalaba/gliner_agents_v3`` (GLiNER) finds data mentions in a passage and
   types each one as NAMED_DATA, DESCRIPTIVE_DATA or VAGUE_DATA.
2. ``rafmacalaba/lfm2.5-350M-focus-metadata`` (LFM2.5) reads each *named* mention in
   the passage chunk it came from and answers two tasks as JSON: source attributes
   (``attributes``) and how the author uses the data (``usage``).

The page (``index.html`` and ``main.js``) is served by Gradio's Server mode, and it calls
the ``extract`` endpoint through Gradio's queue, so ZeroGPU attaches a GPU only while a
request runs. ``extract`` streams: first the mentions, then, for each named mention, the
text that the metadata model is writing token by token, then the parsed attributes and usage.
"""

import json
import re
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional

import gradio as gr
import spaces
import torch
from fastapi.responses import FileResponse
from gliner import GLiNER
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

HERE = Path(__file__).parent
ENTITY_ID = "rafmacalaba/gliner_agents_v3"
METADATA_ID = "rafmacalaba/lfm2.5-350M-focus-metadata"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENTITY_LABELS = ["NAMED_DATA", "DESCRIPTIVE_DATA", "VAGUE_DATA"]
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
STREAM_EVERY_S = 0.05  # at most this often, so a long passage does not flood the connection

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
        for ent in entity_model.predict_entities(chunk, ENTITY_LABELS, threshold=threshold):
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
        m["text"] = text[m["start"]:m["end"]]
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


def stream_task(system: str, mention: Dict[str, Any], index: int, slot: str) -> Iterator[Dict[str, Any]]:
    """Generate one task for one mention, yielding the text written so far.

    Generation runs in a thread and the streamer hands over the new text as it appears.
    The generator's return value is the parsed JSON, or None when the output has none.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f'Text: {mention["passage"]}\n\nMention: "{mention["text"]}"'},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    tokenizer.truncation_side = "left"
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
        add_special_tokens=False,  # the chat template already adds them
    ).to(metadata_model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate() -> None:
        try:
            with torch.no_grad():
                metadata_model.generate(
                    **enc, streamer=streamer, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except Exception:
            streamer.end()  # release the reader below, then report the error
            raise

    thread = Thread(target=generate)
    thread.start()
    text, last = "", 0.0
    for piece in streamer:
        text += piece
        now = time.perf_counter()
        if now - last >= STREAM_EVERY_S:
            last = now
            yield {"stage": "token", "index": index, "slot": slot, "text": text}
    thread.join()
    yield {"stage": "token", "index": index, "slot": slot, "text": text}  # the throttle may have skipped the tail
    return _parse_json(text)


def verbatim(value: Any, passage: str) -> str:
    """Attribute values must be passage text; anything else is dropped."""
    value = "" if value is None else str(value).strip()
    return value if value and value.lower() in passage.lower() else ""


def attribute_fields(mention: Dict[str, Any], parsed: Optional[Dict]) -> Dict[str, Any]:
    """Validated attributes of one named mention, with empty values left out."""
    passage = mention["passage"]
    body = (parsed or {}).get("attributes") or {}
    data_type = body.get("data_type")
    fields = {
        "acronym": verbatim(body.get("acronym"), passage),
        "producer": verbatim(body.get("producer"), passage),
        "year": verbatim(body.get("year"), passage),
        "geography": verbatim(body.get("geography"), passage),
        "version": verbatim(body.get("version"), passage),
        "data_type": data_type if data_type in DATA_TYPES else "",
        "themes": [t for t in (parsed or {}).get("tags") or [] if t in THEME_TAGS],
    }
    return {k: v for k, v in fields.items() if v}


def usage_fields(parsed: Optional[Dict]) -> Dict[str, Any]:
    """Validated usage of one named mention, with empty values left out."""
    parsed = parsed or {}
    role, action = parsed.get("role"), parsed.get("usage_action")
    fields = {
        "role": role if role in USAGE_ROLES else "",
        "usage_action": action if action in USAGE_ACTIONS else "",
        "summary": str(parsed.get("summary") or "").strip(),
    }
    return {k: v for k, v in fields.items() if v}


def public(mention: Dict[str, Any]) -> Dict[str, Any]:
    """The mention as the page receives it."""
    out = {"text": mention["text"], "label": mention["label"], "score": round(mention["score"], 4)}
    for key in ("attributes", "usage"):
        if mention.get(key):
            out[key] = mention[key]
    return out


def _duration(text: str, threshold: float = 0.6) -> int:
    return min(120, 30 + len(text.split()) // 25)


@spaces.GPU(duration=_duration)
def _extract(text: str, threshold: float = 0.6) -> Iterator[Dict[str, Any]]:
    started = time.perf_counter()
    mentions = find_mentions(text, threshold)
    span_ms = round((time.perf_counter() - started) * 1000)
    yield {"stage": "spans", "mentions": [public(m) for m in mentions], "timings": {"span_ms": span_ms}}

    for index, mention in enumerate(mentions):
        if mention["label"] != "NAMED_DATA":
            continue
        parsed = yield from stream_task(ATTRIBUTES_SYSTEM, mention, index, "attributes")
        mention["attributes"] = attribute_fields(mention, parsed)
        yield {"stage": "slot", "index": index, "slot": "attributes", "value": mention["attributes"]}
        parsed = yield from stream_task(USAGE_SYSTEM, mention, index, "usage")
        mention["usage"] = usage_fields(parsed)
        yield {"stage": "slot", "index": index, "slot": "usage", "value": mention["usage"]}

    total_ms = round((time.perf_counter() - started) * 1000)
    yield {
        "stage": "done",
        "mentions": [public(m) for m in mentions],
        "timings": {"span_ms": span_ms, "attribution_ms": total_ms - span_ms, "total_ms": total_ms},
    }


server = gr.Server()


@server.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@server.get("/main.js")
def main_js() -> FileResponse:
    return FileResponse(HERE / "main.js", media_type="text/javascript")


@server.api(name="extract")
def extract(text: str, threshold: float = 0.6) -> Iterator[Dict[str, Any]]:
    """Find data mentions in a passage and read the attributes and usage of each named one."""
    text = (text or "").strip()
    if not text:
        raise gr.Error("Paste a passage first.")
    if len(text.split()) > MAX_WORDS:
        raise gr.Error(f"The text is over {MAX_WORDS:,} words. Paste a shorter passage.")
    yield from _extract(text, min(max(float(threshold), 0.05), 0.95))


if __name__ == "__main__":
    server.launch()
