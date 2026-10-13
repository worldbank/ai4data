"""Glue for the session 01 notebooks.

The notebooks are built on LangChain (documents, splitting, embeddings, vector
store, prompts, tools, structured output), LangGraph through
`langchain.agents.create_agent` (the agent loop), FastMCP (the tool server and
client), and `pcn-core` (number verification). This module holds what those
libraries do not: the worked-example data, a chat model that replays recorded
outputs when no API key is configured, and a few printing and evaluation
helpers.

Configuration by environment variables:

    WORKSHOP_API_KEY    key for an OpenAI-compatible chat API; without it,
                        recorded outputs from assets/recorded_responses.json
                        are replayed and labelled
    WORKSHOP_BASE_URL   base URL of the API (optional)
    WORKSHOP_MODEL      model name (default: gpt-4o-mini)
    WORKSHOP_RECORD     path of a file to append live outputs to, by scenario
"""

from __future__ import annotations

import csv
import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pcn_core import Claim, compare_by_policy
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Paths and the worked example
# ----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
DOCS_DIR = ASSETS / "documents"
INDICATORS_CSV = ASSETS / "indicators.csv"
INDICATOR_META = ASSETS / "indicator_metadata.json"
QUESTIONS_JSON = ASSETS / "questions.json"
RECORDED_JSON = ASSETS / "recorded_responses.json"
MCP_SERVER = HERE / "indicator_server.py"
DATA360_MCP_URL = "https://maimcpext.worldbank.org/ext/data360/mcp"

MAIN_QUESTION = (
    "How have internet access and economic development evolved in Rwanda and "
    "Kenya? Compare the recent available data, summarize the main patterns, "
    "and cite the sources."
)

# ----------------------------------------------------------------------------
# Scenarios: the name under which an output is recorded or replayed
# ----------------------------------------------------------------------------

_STATE: dict[str, Any] = {"scenario": None, "cursor": {}}


def use(scenario: str) -> str:
    """Name the scenario of the next model call(s). Resets the replay cursor."""
    _STATE["scenario"] = scenario
    _STATE["cursor"][scenario] = 0
    return scenario


def current_scenario() -> str | None:
    return _STATE["scenario"]


# ----------------------------------------------------------------------------
# The chat model: live, or recorded outputs replayed
# ----------------------------------------------------------------------------


def llm_available() -> bool:
    return bool(os.environ.get("WORKSHOP_API_KEY"))


class RecordedChatModel(BaseChatModel):
    """A chat model that replays the recorded output of the current scenario.

    Works wherever a LangChain chat model works: prompts, `bind_tools`,
    `with_structured_output`, and the agent. Scenarios recorded as sequences
    (tool calls, then an answer) are replayed one item per call.
    """

    recorded: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "recorded"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        scenario = current_scenario()
        if scenario is None or scenario not in self.recorded:
            raise KeyError(
                f"No recorded output for scenario {scenario!r}. Call w.use('<scenario>') before the model, "
                "set WORKSHOP_API_KEY to run live, or add the scenario to assets/recorded_responses.json."
            )
        item = self.recorded[scenario]
        if isinstance(item, list):
            i = _STATE["cursor"].get(scenario, 0)
            if i >= len(item):
                raise RuntimeError(f"Recorded sequence for {scenario!r} is exhausted after {len(item)} steps.")
            _STATE["cursor"][scenario] = i + 1
            item = item[i]
        calls = [
            {"name": c["name"], "args": c["arguments"], "id": c.get("id", f"call_{k}")}
            for k, c in enumerate(item.get("tool_calls", []))
        ]
        msg = AIMessage(content=item.get("content") or "", tool_calls=calls, response_metadata={"recorded": scenario})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def parse(message: AIMessage):
            return schema.model_validate_json(message.content)

        return self | RunnableLambda(parse)


class RecordingHandler(BaseCallbackHandler):
    """Appends every live output to the file named by WORKSHOP_RECORD, under its scenario."""

    def on_llm_end(self, response, **kwargs) -> None:
        path = os.environ.get("WORKSHOP_RECORD")
        scenario = current_scenario()
        if not path or scenario is None:
            return
        msg = response.generations[0][0].message
        item: dict[str, Any] = {"content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content)}
        if getattr(msg, "tool_calls", None):
            item["tool_calls"] = [{"id": c["id"], "name": c["name"], "arguments": c["args"]} for c in msg.tool_calls]
        store: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path) as fh:
                store = json.load(fh)
        store.setdefault(scenario, []).append(item)
        with open(path, "w") as fh:
            json.dump(store, fh, indent=2, ensure_ascii=False)


def make_model(temperature: float = 0.0) -> BaseChatModel:
    """ChatOpenAI when a key is configured, otherwise the recorded model."""
    if llm_available():
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("WORKSHOP_MODEL", "gpt-4o-mini"),
            api_key=os.environ["WORKSHOP_API_KEY"],
            base_url=os.environ.get("WORKSHOP_BASE_URL") or None,
            temperature=temperature,
            callbacks=[RecordingHandler()],
        )
    recorded = json.loads(RECORDED_JSON.read_text()) if RECORDED_JSON.exists() else {}
    return RecordedChatModel(recorded=recorded)


def mode(model: BaseChatModel) -> str:
    if isinstance(model, RecordedChatModel):
        return "recorded outputs (no WORKSHOP_API_KEY set)"
    return f"live ({getattr(model, 'model_name', 'model')})"


def show(message: BaseMessage | str, width: int = 100) -> None:
    """Print a message, with the recorded-scenario label when it was replayed."""
    if isinstance(message, BaseMessage):
        label = message.response_metadata.get("recorded") if hasattr(message, "response_metadata") else None
        if label:
            print(f"[recorded output: {label}]")
        text = message.content if isinstance(message.content, str) else json.dumps(message.content)
        if getattr(message, "tool_calls", None):
            for c in message.tool_calls:
                print(f"-> tool call {c['name']}({json.dumps(c['args'])})")
    else:
        text = message
    for para in text.split("\n"):
        print(textwrap.fill(para, width=width) if len(para) > width else para)


# ----------------------------------------------------------------------------
# Documents, chunks, retrieval (LangChain)
# ----------------------------------------------------------------------------

HEADERS = [("#", "title"), ("##", "section")]


def load_documents(folder: Path = DOCS_DIR) -> list[Document]:
    """One LangChain Document per file; the YAML header becomes the metadata."""
    docs = []
    for path in sorted(folder.glob("*.md")):
        post = frontmatter.load(path)
        meta = {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in post.metadata.items()}
        docs.append(Document(page_content=post.content.strip(), metadata=meta))
    return docs


def split_sections(docs: list[Document]) -> list[Document]:
    """One chunk per Markdown section, headings kept, document metadata carried over."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    out = []
    for d in docs:
        for c in splitter.split_text(d.page_content):
            meta = dict(d.metadata)
            meta.update(c.metadata)
            meta.setdefault("section", "intro")
            out.append(Document(page_content=c.page_content, metadata=meta))
    return out


def split_fixed(docs: list[Document], chunk_size: int = 400, chunk_overlap: int = 50) -> list[Document]:
    """Fixed-size windows; the section is unknown."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = splitter.split_documents(docs)
    for c in out:
        c.metadata["section"] = "(fixed window)"
    return out


def make_embeddings():
    """A small sentence-transformer model, downloaded on first use."""
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        import transformers.utils.logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def mcp_server_transport():
    """A FastMCP stdio transport that runs indicator_server.py with this interpreter."""
    import sys

    from fastmcp.client.transports import PythonStdioTransport

    return PythonStdioTransport(MCP_SERVER, python_cmd=sys.executable, env={**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"})


def build_store(chunks: list[Document], embeddings=None) -> InMemoryVectorStore:
    return InMemoryVectorStore.from_documents(chunks, embeddings or make_embeddings())


def citation(doc: Document) -> str:
    m = doc.metadata
    return f"[{m.get('id', '?')} · {m.get('section', 'intro')} · {str(m.get('date', ''))[:4]}]"


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"{citation(d)}\n{d.page_content}" for d in docs)


def print_hits(results) -> None:
    """Print (Document, score) pairs or plain Documents."""
    for r in results:
        doc, score = (r if isinstance(r, tuple) else (r, None))
        print(f"{'' if score is None else f'{score:6.3f}  '}{citation(doc)}")


def make_search_tool(store: InMemoryVectorStore, k: int = 3, since: str | None = None):
    """A LangChain tool that searches the vector store and returns cited passages.

    `since` is a date string such as "2020"; documents dated before it are left out.
    """

    def where(d: Document) -> bool:
        return since is None or str(d.metadata.get("date", "")) >= since

    @tool
    def search_documents(query: str) -> list[dict]:
        """Search the document collection for passages on a topic. Returns passages with
        citations. Use it for definitions, context, and explanations; use get_indicator
        for numbers."""
        results = store.similarity_search_with_score(query, k=k, filter=where)
        return [{"citation": citation(d), "score": round(float(s), 3), "text": d.page_content} for d, s in results]

    return search_documents


# ----------------------------------------------------------------------------
# Indicator data: a LangChain tool over the committed CSV, or the World Bank API
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


def indicator_lookup(indicator: str, country: str, start: int | None = None, end: int | None = None, live: bool = False) -> dict[str, Any]:
    """The function behind the tool. `claim_id` is "<indicator>:<ISO3>:<year>", the key the verifier uses."""
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
        "source": f"{meta['source_database']} ({meta['source_organization'].split(',')[0]})",
        "observations": sorted(obs, key=lambda o: o["year"]),
    }


def _fetch_live(indicator: str, iso: str, start: int | None, end: int | None) -> list[dict[str, Any]]:
    import requests

    url = f"https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}?format=json&per_page=200&date={start or 2000}:{end or 2030}"
    data = requests.get(url, timeout=30).json()
    if len(data) < 2 or not data[1]:
        return []
    return [{"year": int(r["date"]), "value": r["value"]} for r in data[1]]


@tool
def get_indicator(indicator: str, country: str, start: int | None = None, end: int | None = None) -> dict:
    """Observations of a World Development Indicators series for one country. Use it for
    any statistic; do not quote figures from memory. `indicator` is an id such as
    IT.NET.USER.ZS or NY.GDP.PCAP.CD; `country` is a name or ISO3 code; `start` and `end`
    bound the years."""
    return indicator_lookup(indicator, country, start, end)


def latest(result: dict[str, Any]) -> dict[str, Any] | None:
    obs = result.get("observations") or []
    return obs[-1] if obs else None


# ----------------------------------------------------------------------------
# Structured outputs (Pydantic, used through LangChain's with_structured_output)
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Agent helpers (LangGraph through langchain.agents.create_agent)
# ----------------------------------------------------------------------------


def final_text(result: dict[str, Any]) -> str:
    """The content of the last AI message of an agent result."""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return m.content if isinstance(m.content, str) else json.dumps(m.content)
    return "(no final answer)"


def print_trace(result: dict[str, Any], width: int = 150) -> None:
    """Every tool call and result in an agent run, in order."""
    step = 0
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            step += 1
            for c in m.tool_calls:
                print(f"step {step}: {c['name']}({json.dumps(c['args'])})")
        elif isinstance(m, ToolMessage):
            text = m.content if isinstance(m.content, str) else json.dumps(m.content)
            print(f"   -> {text[:width]}{'…' if len(text) > width else ''}")


def tool_results(result: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Parsed results of every call to the tool `name` in an agent run."""
    out = []
    for m in result["messages"]:
        if isinstance(m, ToolMessage) and m.name == name:
            try:
                out.append(json.loads(m.content) if isinstance(m.content, str) else m.content)
            except json.JSONDecodeError:
                pass
    return out


# ----------------------------------------------------------------------------
# Verification with pcn-core: claim tags checked against observations
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

    A number inside a <claim id=...> tag is compared with observations[id] by
    pcn-core under the policy in the tag (rounded with decimals, or exact). A
    number outside any tag is reported as unmarked: it carries no claim, so it
    receives no mark, whether or not it happens to be right. Years are skipped.
    """
    checks: list[ClaimCheck] = []
    for m in CLAIM_RE.finditer(answer):
        cid, policy, decimals, inner = m.group(1), m.group(2) or "rounded", m.group(3), m.group(4).strip()
        if cid not in observations:
            checks.append(ClaimCheck(inner, cid, "flagged", "no observation with this id"))
            continue
        truth = observations[cid]
        pol: dict[str, Any] = {"type": "exact"} if policy == "exact" else {"type": "rounded", "decimals": int(decimals) if decimals else default_decimals}
        ok = compare_by_policy(inner.replace(",", ""), Claim(value=truth), pol)
        checks.append(ClaimCheck(inner, cid, "verified" if ok else "flagged", f"record holds {truth:g}" + ("" if ok else f"; shown {inner}")))
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
    """claim_id -> value, from get_indicator results."""
    out = {}
    for r in results:
        for o in r.get("observations", []):
            out[o["claim_id"]] = o["value"]
    return out


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------


def load_questions() -> list[dict[str, Any]]:
    return json.loads(QUESTIONS_JSON.read_text())["questions"]


def retrieval_hit(docs: list[Document], expected_docs: list[str]) -> bool:
    got = {d.metadata.get("id") for d in docs}
    return bool(got & set(expected_docs)) if expected_docs else True


def numbers_present(answer: str, expected: list[dict[str, Any]], tol: float = 0.06) -> tuple[int, int]:
    text = strip_claims(answer)
    nums = [float(n.replace(",", "")) for n in NUMBER_RE.findall(text)]
    found = sum(1 for e in expected if any(abs(n - e["value"]) <= max(tol, abs(e["value"]) * 0.005) for n in nums))
    return found, len(expected)


def grounded(answer: str, docs: list[Document]) -> bool:
    return any(str(d.metadata.get("id")) in answer for d in docs)


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
