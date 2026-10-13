"""Shared code for the session 01 notebooks.

One model client, one corpus, one retriever, one indicator tool, one verifier,
one evaluation loop. The notebooks import from here so that what changes
between them is the capability being added.

Configuration by environment variables:

    WORKSHOP_API_KEY    key for an OpenAI-compatible chat API; without it,
                        recorded outputs from assets/recorded_responses.json
                        are used and labelled as recorded
    WORKSHOP_BASE_URL   base URL of the API (optional)
    WORKSHOP_MODEL      model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
DOCS_DIR = ASSETS / "documents"
INDICATORS_CSV = ASSETS / "indicators.csv"
INDICATOR_META = ASSETS / "indicator_metadata.json"
QUESTIONS_JSON = ASSETS / "questions.json"
RECORDED_JSON = ASSETS / "recorded_responses.json"

MAIN_QUESTION = (
    "How have internet access and economic development evolved in Rwanda and "
    "Kenya? Compare the recent available data, summarize the main patterns, "
    "and cite the sources."
)

# ----------------------------------------------------------------------------
# Model client
# ----------------------------------------------------------------------------


def llm_available() -> bool:
    """True when an API key is configured."""
    return bool(os.environ.get("WORKSHOP_API_KEY"))


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = "call_0"


@dataclass
class Completion:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    recorded: bool = False
    scenario: str | None = None

    def __str__(self) -> str:
        tag = f"[recorded output: {self.scenario}]\n" if self.recorded else ""
        if self.tool_calls:
            calls = "\n".join(f"-> tool call {c.name}({json.dumps(c.arguments)})" for c in self.tool_calls)
            return tag + calls + ("\n" + self.content if self.content else "")
        return tag + (self.content or "")


class LLM:
    """A chat completion client with a recorded fallback.

    `complete(messages, scenario=...)` calls the API when a key is configured.
    Otherwise it returns the recorded output for `scenario`. Recorded outputs
    for scenarios with tool calls are sequences; each call to `complete` with
    the same scenario returns the next item.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.model = model or os.environ.get("WORKSHOP_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.live = llm_available()
        self._client = None
        self._recorded: dict[str, Any] = {}
        self._cursor: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []
        if self.live:
            from openai import OpenAI  # imported here so that the fallback needs no package

            self._client = OpenAI(
                api_key=os.environ["WORKSHOP_API_KEY"],
                base_url=os.environ.get("WORKSHOP_BASE_URL") or None,
            )
        if RECORDED_JSON.exists():
            self._recorded = json.loads(RECORDED_JSON.read_text())

    def mode(self) -> str:
        return f"live ({self.model})" if self.live else "recorded outputs (no WORKSHOP_API_KEY set)"

    def reset(self, scenario: str | None = None) -> None:
        if scenario is None:
            self._cursor.clear()
        else:
            self._cursor.pop(scenario, None)

    def complete(
        self,
        messages: list[dict[str, str]],
        scenario: str,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        self.calls.append({"scenario": scenario, "messages": messages, "tools": [t["function"]["name"] for t in tools] if tools else []})
        if self.live:
            return self._complete_live(messages, tools, response_format, scenario)
        return self._complete_recorded(scenario)

    def _complete_live(self, messages, tools, response_format, scenario: str) -> Completion:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(name=tc.function.name, arguments=args, id=tc.id or f"call_{i}"))
        completion = Completion(content=msg.content, tool_calls=calls)
        self._record(scenario, completion)
        return completion

    def _record(self, scenario: str, completion: Completion) -> None:
        """When WORKSHOP_RECORD names a file, append the live output under its scenario.

        Every scenario is stored as a sequence; `record_responses.py` runs the
        notebooks with a key and this variable set, then collapses one-step
        sequences into single items.
        """
        path = os.environ.get("WORKSHOP_RECORD")
        if not path:
            return
        store: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path) as fh:
                store = json.load(fh)
        item: dict[str, Any] = {"content": completion.content}
        if completion.tool_calls:
            item["tool_calls"] = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in completion.tool_calls]
        store.setdefault(scenario, []).append(item)
        with open(path, "w") as fh:
            json.dump(store, fh, indent=2)

    def _complete_recorded(self, scenario: str) -> Completion:
        if scenario not in self._recorded:
            raise KeyError(
                f"No recorded output for scenario '{scenario}'. Set WORKSHOP_API_KEY to run live, "
                "or add the scenario to assets/recorded_responses.json."
            )
        item = self._recorded[scenario]
        if isinstance(item, list):
            i = self._cursor.get(scenario, 0)
            if i >= len(item):
                raise RuntimeError(f"Recorded sequence for '{scenario}' is exhausted after {len(item)} steps.")
            self._cursor[scenario] = i + 1
            item = item[i]
        calls = [ToolCall(name=c["name"], arguments=c["arguments"], id=c.get("id", f"call_{k}")) for k, c in enumerate(item.get("tool_calls", []))]
        return Completion(content=item.get("content"), tool_calls=calls, recorded=True, scenario=scenario)


def show(completion: Completion | str, width: int = 100) -> None:
    """Print a completion with wrapped lines."""
    text = str(completion)
    for para in text.split("\n"):
        print(textwrap.fill(para, width=width) if len(para) > width else para)


# ----------------------------------------------------------------------------
# Documents, chunks, embeddings, retrieval
# ----------------------------------------------------------------------------


@dataclass
class Document:
    id: str
    title: str
    type: str
    country: str
    date: str
    source: str
    note: str
    text: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    date: str
    country: str
    type: str
    text: str

    def citation(self) -> str:
        return f"[{self.doc_id} · {self.section} · {self.date[:4]}]"


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.DOTALL)
    meta: dict[str, str] = {}
    if not m:
        return meta, raw
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, raw[m.end():]


def load_documents(folder: Path = DOCS_DIR) -> list[Document]:
    docs = []
    for path in sorted(folder.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text())
        docs.append(
            Document(
                id=meta.get("id", path.stem),
                title=meta.get("title", path.stem),
                type=meta.get("type", ""),
                country=meta.get("country", ""),
                date=meta.get("date", ""),
                source=meta.get("source", ""),
                note=meta.get("note", ""),
                text=body.strip(),
            )
        )
    return docs


def chunk_fixed(doc: Document, size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Fixed-size character windows. Section metadata is lost."""
    text = doc.text
    chunks = []
    start = 0
    k = 0
    while start < len(text):
        piece = text[start : start + size]
        chunks.append(Chunk(id=f"{doc.id}#fixed{k}", doc_id=doc.id, title=doc.title, section="(fixed window)", date=doc.date, country=doc.country, type=doc.type, text=piece))
        k += 1
        start += size - overlap
    return chunks


def chunk_sections(doc: Document) -> list[Chunk]:
    """One chunk per Markdown section, with the section heading kept as metadata."""
    parts = re.split(r"^## ", doc.text, flags=re.MULTILINE)
    chunks = []
    for k, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if k == 0:
            section = "intro"
            body = re.sub(r"^# .*\n", "", part).strip()
            if not body:
                continue
        else:
            heading, _, body = part.partition("\n")
            section = heading.strip()
            body = body.strip()
        text = f"{doc.title}\n{section}\n{body}"
        chunks.append(Chunk(id=f"{doc.id}#{k}", doc_id=doc.id, title=doc.title, section=section, date=doc.date, country=doc.country, type=doc.type, text=text))
    return chunks


class Embedder:
    """Sentence embeddings from a local model when available, else TF-IDF.

    Both back ends expose `encode(list[str]) -> ndarray` with unit-norm rows.
    """

    def __init__(self, prefer_local_model: bool = True):
        self.backend = "tfidf"
        self._model = None
        self._vec = None
        if prefer_local_model:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.backend = "sentence-transformers/all-MiniLM-L6-v2"
            except Exception:
                self._model = None

    def fit(self, texts: list[str]) -> None:
        if self._model is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts)

    def encode(self, texts: list[str]):
        import numpy as np

        if self._model is not None:
            return self._model.encode(texts, normalize_embeddings=True)
        if self._vec is None:
            self.fit(texts)
        m = self._vec.transform(texts).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or Embedder()
        self.embedder.fit([c.text for c in chunks])
        self.matrix = self.embedder.encode([c.text for c in chunks])

    def search(self, query: str, k: int = 3, where: Callable[[Chunk], bool] | None = None, recency_weight: float = 0.0) -> list[Hit]:
        """Cosine similarity, optional metadata filter, optional recency bonus."""
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        hits = []
        for c, s in zip(self.chunks, scores):
            if where and not where(c):
                continue
            s = float(s)
            if recency_weight:
                try:
                    year = int(c.date[:4])
                except ValueError:
                    year = 2000
                s += recency_weight * (year - 2000) / 30.0
            hits.append(Hit(chunk=c, score=s))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def format_context(hits: list[Hit]) -> str:
    """Evidence block for the prompt, each passage with its citation."""
    parts = []
    for h in hits:
        parts.append(f"{h.chunk.citation()}\n{h.chunk.text}")
    return "\n\n".join(parts)


def print_hits(hits: list[Hit]) -> None:
    for h in hits:
        print(f"{h.score:6.3f}  {h.chunk.citation()}")


# ----------------------------------------------------------------------------
# Indicator data: offline from the CSV, live from the World Bank API
# ----------------------------------------------------------------------------

COUNTRY_CODES = {"kenya": "KEN", "rwanda": "RWA", "ken": "KEN", "rwa": "RWA"}


def _load_rows() -> list[dict[str, Any]]:
    rows = []
    with INDICATORS_CSV.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({**r, "year": int(r["year"]), "value": float(r["value"]) if r["value"] not in ("", None) else None})
    return rows


_ROWS = _load_rows() if INDICATORS_CSV.exists() else []
INDICATOR_METADATA = json.loads(INDICATOR_META.read_text())["indicators"] if INDICATOR_META.exists() else {}


def list_indicators() -> list[dict[str, str]]:
    return [{"id": k, "name": v["name"]} for k, v in INDICATOR_METADATA.items()]


def get_indicator(indicator: str, country: str, start: int | None = None, end: int | None = None, live: bool = False) -> dict[str, Any]:
    """Observations for one indicator and one country.

    Returns {"indicator", "country", "name", "source", "observations": [{"year", "value", "claim_id"}]}.
    `claim_id` is "<indicator>:<ISO3>:<year>", the key the verifier uses.
    """
    iso = COUNTRY_CODES.get(country.lower(), country.upper())
    if indicator not in INDICATOR_METADATA:
        return {"error": f"unknown indicator '{indicator}'", "available": [i["id"] for i in list_indicators()]}
    meta = INDICATOR_METADATA[indicator]
    if live:
        obs = _fetch_live(indicator, iso, start, end)
    else:
        obs = [
            {"year": r["year"], "value": r["value"]}
            for r in _ROWS
            if r["indicator_id"] == indicator and r["country_iso3"] == iso and (start is None or r["year"] >= start) and (end is None or r["year"] <= end)
        ]
    obs = [o for o in obs if o["value"] is not None]
    for o in obs:
        o["claim_id"] = f"{indicator}:{iso}:{o['year']}"
    return {
        "indicator": indicator,
        "country": iso,
        "name": meta["name"],
        "unit": meta.get("unit") or "",
        "source": f"{meta['source_database']} ({meta['source_organization'].split(',')[0]})",
        "observations": sorted(obs, key=lambda o: o["year"]),
    }


def _fetch_live(indicator: str, iso: str, start: int | None, end: int | None) -> list[dict[str, Any]]:
    import requests

    date = f"{start or 2000}:{end or 2030}"
    url = f"https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}?format=json&per_page=200&date={date}"
    data = requests.get(url, timeout=30).json()
    if len(data) < 2 or not data[1]:
        return []
    return [{"year": int(r["date"]), "value": r["value"]} for r in data[1]]


def latest(result: dict[str, Any]) -> dict[str, Any] | None:
    obs = result.get("observations") or []
    return obs[-1] if obs else None


# Tool schema as the chat API expects it
INDICATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "get_indicator",
        "description": "Observations of a World Development Indicators series for one country. Use it for any statistic; do not quote figures from memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "indicator": {"type": "string", "description": "Indicator id, for example IT.NET.USER.ZS or NY.GDP.PCAP.CD"},
                "country": {"type": "string", "description": "Country name or ISO3 code"},
                "start": {"type": "integer", "description": "First year (optional)"},
                "end": {"type": "integer", "description": "Last year (optional)"},
            },
            "required": ["indicator", "country"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the document collection for passages on a topic. Returns passages with citations. Use it for definitions, context, and explanations; use get_indicator for numbers.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer", "description": "Number of passages (default 3)"}},
            "required": ["query"],
        },
    },
}

# ----------------------------------------------------------------------------
# Structured outputs
# ----------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field, ValidationError

    class Observation(BaseModel):
        indicator: str = Field(description="Indicator id")
        country: str = Field(description="ISO3 code")
        year: int
        value: float
        claim_id: str = Field(description="<indicator>:<ISO3>:<year>")

    class CountrySummary(BaseModel):
        country: str
        internet_latest: Observation
        gdp_per_capita_latest: Observation
        trend: str = Field(description="One or two sentences on the trend since 2010")

    class ComparisonAnswer(BaseModel):
        question: str
        countries: list[CountrySummary]
        comparison: str = Field(description="Two to four sentences comparing the countries")
        sources: list[str]
        caveats: list[str] = Field(default_factory=list)

    HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover
    HAVE_PYDANTIC = False


# ----------------------------------------------------------------------------
# Tool loop
# ----------------------------------------------------------------------------


def run_tool_loop(
    llm: LLM,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    handlers: dict[str, Callable[..., Any]],
    scenario: str,
    max_steps: int = 5,
    verbose: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Call the model, run the tools it asks for, feed results back, until it answers.

    Returns (final_text, trace). The trace lists every tool call with its result,
    so that a reviewer can see what the model did.
    """
    trace: list[dict[str, Any]] = []
    messages = list(messages)
    for step in range(max_steps):
        completion = llm.complete(messages, scenario=scenario, tools=tools)
        if not completion.tool_calls:
            return completion.content or "", trace
        messages.append({"role": "assistant", "content": completion.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in completion.tool_calls
        ]})
        for call in completion.tool_calls:
            handler = handlers.get(call.name)
            if handler is None:
                result: Any = {"error": f"unknown tool {call.name}"}
            else:
                try:
                    result = handler(**call.arguments)
                except Exception as e:  # the model sees the error and can retry
                    result = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"step": step + 1, "tool": call.name, "arguments": call.arguments, "result": result})
            if verbose:
                short = json.dumps(result)[:160]
                print(f"step {step + 1}: {call.name}({json.dumps(call.arguments)}) -> {short}{'…' if len(json.dumps(result)) > 160 else ''}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return "(stopped: step limit reached without a final answer)", trace


# ----------------------------------------------------------------------------
# Verification: claim tags checked against observations
# ----------------------------------------------------------------------------

CLAIM_RE = re.compile(r'<claim\s+id="([^"]+)"(?:\s+policy="([^"]+)")?(?:\s+decimals="(\d+)")?\s*>([^<]+)</claim>')
NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")


@dataclass
class ClaimCheck:
    text: str
    claim_id: str | None
    status: str  # verified | flagged | unmarked
    detail: str


def verify(answer: str, observations: dict[str, float], default_decimals: int = 1) -> list[ClaimCheck]:
    """Check every number in `answer`.

    A number inside a <claim id=...> tag is compared with observations[id]
    under the policy (exact | rounded). A number outside any tag is reported
    as unmarked: it carries no claim, so it receives no mark, whether or not
    it happens to be right. Years inside tags are not expected; four-digit
    numbers that look like years outside tags are ignored.
    """
    checks: list[ClaimCheck] = []
    for m in CLAIM_RE.finditer(answer):
        cid, policy, decimals, inner = m.group(1), m.group(2) or "rounded", m.group(3), m.group(4).strip()
        dec = int(decimals) if decimals else default_decimals
        try:
            shown = float(inner.replace(",", ""))
        except ValueError:
            checks.append(ClaimCheck(inner, cid, "flagged", "not a number"))
            continue
        if cid not in observations:
            checks.append(ClaimCheck(inner, cid, "flagged", "no observation with this id"))
            continue
        truth = observations[cid]
        if policy == "exact":
            ok = math.isclose(shown, truth, abs_tol=1e-9)
        else:
            ok = math.isclose(round(truth, dec), shown, abs_tol=10 ** (-dec) / 2 + 1e-9) or math.isclose(round(truth), shown, abs_tol=1e-9)
        checks.append(ClaimCheck(inner, cid, "verified" if ok else "flagged", f"record holds {truth:g}" + ("" if ok else f"; shown {shown:g}")))
    stripped = CLAIM_RE.sub(" ", answer)
    for m in NUMBER_RE.finditer(stripped):
        token = m.group(0)
        if re.fullmatch(r"(19|20)\d\d", token):
            continue
        checks.append(ClaimCheck(token, None, "unmarked", "no claim tag, so no mark is drawn"))
    return checks


def print_checks(checks: list[ClaimCheck]) -> None:
    counts = {"verified": 0, "flagged": 0, "unmarked": 0}
    for c in checks:
        counts[c.status] += 1
        mark = {"verified": "✓", "flagged": "✗", "unmarked": "·"}[c.status]
        print(f"{mark} {c.status:9s} {c.text:>10s}  {c.claim_id or '':28s} {c.detail}")
    print(f"\n{counts['verified']} verified, {counts['flagged']} flagged, {counts['unmarked']} unmarked")


def strip_claims(answer: str) -> str:
    """The answer as a reader sees it, tags removed."""
    return CLAIM_RE.sub(lambda m: m.group(4).strip(), answer)


def observations_from_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """claim_id -> value, from a list of get_indicator results."""
    out = {}
    for r in results:
        for o in r.get("observations", []):
            out[o["claim_id"]] = o["value"]
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


def load_questions() -> list[dict[str, Any]]:
    return json.loads(QUESTIONS_JSON.read_text())["questions"]


def retrieval_hit(hits: list[Hit], expected_docs: list[str]) -> bool:
    """True when at least one retrieved chunk comes from an expected document."""
    got = {h.chunk.doc_id for h in hits}
    return bool(got & set(expected_docs)) if expected_docs else True


def numbers_present(answer: str, expected: list[dict[str, Any]], tol: float = 0.06) -> tuple[int, int]:
    """How many expected values appear in the answer (within rounding)."""
    text = strip_claims(answer)
    found = 0
    nums = [float(n.replace(",", "")) for n in NUMBER_RE.findall(text)]
    for e in expected:
        v = e["value"]
        if any(abs(n - v) <= max(tol, abs(v) * 0.005) for n in nums):
            found += 1
    return found, len(expected)


def grounded(answer: str, hits: list[Hit]) -> bool:
    """A cheap groundedness check: the answer cites at least one retrieved passage."""
    return any(h.chunk.doc_id in answer for h in hits)


@dataclass
class EvalRow:
    id: str
    category: str
    retrieval: str
    numbers: str
    cited: str
    verification: str
    note: str = ""


def print_table(rows: list[EvalRow]) -> None:
    head = f"{'id':4s} {'category':18s} {'retrieval':10s} {'numbers':8s} {'cited':6s} {'verify':16s} note"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r.id:4s} {r.category:18s} {r.retrieval:10s} {r.numbers:8s} {r.cited:6s} {r.verification:16s} {r.note}")
