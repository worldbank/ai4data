"""Generates the six session 01 notebooks and, with --execute, runs them to
cache their outputs.

    python build_notebooks.py             # write 01..06 .ipynb
    python build_notebooks.py --execute   # write and execute (recorded mode unless a key is set)

The notebooks are generated from this file so that their text can be reviewed
and edited in one place. Edit here, then regenerate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

SETUP = '''# Setup. In Colab this clones the repository and installs the dependencies; locally it uses the folder as is.
import os, sys, subprocess, pathlib

REPO_URL = "https://github.com/avsolatorio/ai4data.git"   # replace with the published repository when released
BRANCH = "feat/training-series"
SESSION = "training/session-01-building-ai-applications/notebooks"

if not pathlib.Path("workshop.py").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO_URL, "/content/ai4data"], check=True)
    os.chdir(f"/content/ai4data/{SESSION}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai", "pydantic", "scikit-learn", "requests"], check=True)
sys.path.insert(0, os.getcwd())

# Optional: a key for an OpenAI-compatible API. Without one, the notebook uses recorded outputs and says so.
try:
    from google.colab import userdata  # type: ignore
    for name in ("WORKSHOP_API_KEY", "WORKSHOP_BASE_URL", "WORKSHOP_MODEL"):
        if userdata.get(name):
            os.environ[name] = userdata.get(name)
except Exception:
    pass

import workshop as w
llm = w.LLM()
print("Model client:", llm.mode())'''

LINKS = """## Where this connects to the program

- [Practical Guide to Serving Official Statistics to AI Agents](https://worldbank.github.io/ai4data/cookbook/serving-statistics-to-agents/): the design of tools over a statistical catalog, in full.
- [Practical Guide to Evaluation Suites for Statistical AI](https://worldbank.github.io/ai4data/cookbook/evaluation-suites/): how to build and run the tests.
- [Proof-Carrying Numbers](https://worldbank.github.io/ai4data/docs/pcn/): the claim-tag protocol the verifier in this session follows.
- [Data360 MCP server](https://github.com/worldbank/data360-mcp): the tools of notebook 04, exposed to any MCP client over the World Bank's data platform.
- [End-to-end reference implementation](https://worldbank.github.io/ai4data/docs/reference-implementation/): documentation, validation, publication, agent access, and verification in one run."""


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


# =============================================================================
# 01
# =============================================================================
NB01 = [
    md("""
# 01 · From a language model to an application

**Session 01 · Building AI applications for research and data · notebook 1 of 6**

## Objective

See what a model does with a research question on its own, what changes when it is given instructions and context, and why instructions and context are the first two components of an application.

## The application after this notebook

```
question ──┐
            ├──> [ instructions + context + question ] ──> model ──> answer
context ───┘
```

The worked example for the whole session is one question:

> How have internet access and economic development evolved in Rwanda and Kenya? Compare the recent available data, summarize the main patterns, and cite the sources.

## How to run it

Run the cells in order. Without an API key the model outputs are recorded ones, labelled `[recorded output: ...]`. With a key in `WORKSHOP_API_KEY` the same cells call the model live.
"""),
    code(SETUP),
    md("""
## 1. The function the whole session builds on

Three parts go into every call: **instructions** (the system message), **context** (evidence supplied to the model), and the **question**. Keeping them separate is the first design decision of an application, because each is changed by a different person for a different reason.
"""),
    code('''
INSTRUCTIONS_BASIC = "You are a research assistant. Answer the question."

INSTRUCTIONS_GROUNDED = """You are a research assistant for development data.
Answer only from the supplied context. Cite the document id in square brackets after each figure.
If the context does not contain what is needed, say so and give the closest available information."""

INSTRUCTIONS_CLARIFY = INSTRUCTIONS_GROUNDED + """
If the question is ambiguous, say which readings are possible and ask which one is meant before answering."""


def ask(question, instructions=INSTRUCTIONS_BASIC, context=None, scenario="unnamed"):
    """Build the messages from the three parts and call the model."""
    user = question if context is None else f"CONTEXT:\\n{context}\\n\\nQUESTION: {question}"
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": user}]
    return llm.complete(messages, scenario=scenario)


print(w.MAIN_QUESTION)
'''),
    md("""
## 2. Zero context: the model answers from its training data

The baseline. No evidence is supplied, so the model answers from what it absorbed in training.
"""),
    code('''
answer = ask(w.MAIN_QUESTION, scenario="nb01_zero_context")
w.show(answer)
'''),
    md("""
Read the answer against the data. The cell below prints the latest values from the World Development Indicators that the session uses as its reference.
"""),
    code('''
for ind in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"]:
    for country in ["Kenya", "Rwanda"]:
        r = w.get_indicator(ind, country, start=2023)
        print(f"{r['name']:50s} {country:7s}", [(o["year"], round(o["value"], 1)) for o in r["observations"]])
'''),
    md("""
What to notice:

- The figures are approximate, undated, and in places wrong. "Internet penetration above 80 percent" is a subscriptions figure presented as internet use; the indicator for individuals using the internet stood at 35 percent in Kenya in 2024.
- There are no sources a reader could check.
- The model hedges, which is the right behaviour for a model with no evidence, and of no use to a reader who needs a number.

Prompting changes the style of this answer. It cannot give the model data it does not have.
"""),
    md("""
## 3. The same question with context

The four country briefs from the session's document collection are supplied as context. The instructions now require citations and allow the model to say when something is missing.
"""),
    code('''
docs = {d.id: d for d in w.load_documents()}
context = "\\n\\n".join(f"[{d.id}]\\n{d.text}" for d in [docs["kenya-digital"], docs["rwanda-digital"], docs["kenya-economy"], docs["rwanda-economy"]])
print(f"{len(context):,} characters of context, roughly {len(context)//4:,} tokens\\n")

answer = ask(w.MAIN_QUESTION, INSTRUCTIONS_GROUNDED, context, scenario="nb01_with_context")
w.show(answer)
'''),
    md("""
The figures now come from the briefs, each one is cited, and the comparison is specific. The model did none of the finding; the application did, by putting the right text in front of it.

## 4. Handling missing information

An application has to behave well when the evidence does not contain the answer. The instruction "say so and give the closest available information" is what makes the difference.
"""),
    code('''
answer = ask("What share of Kenya's population used the internet in 2025?", INSTRUCTIONS_GROUNDED, docs["kenya-digital"].text, scenario="nb01_missing_info")
w.show(answer)
'''),
    md("""
## 5. An ambiguous question, with and without an instruction

"How connected is Kenya?" can mean internet use, mobile subscriptions, broadband, or something else. Compare what the model does with the basic instructions and with an instruction to ask.
"""),
    code('''
print("Without an instruction to clarify:\\n")
w.show(ask("How connected is Kenya?", INSTRUCTIONS_BASIC, docs["kenya-digital"].text, scenario="nb01_ambiguous_no_instruction"))
print("\\n" + "=" * 100 + "\\nWith the instruction:\\n")
w.show(ask("How connected is Kenya?", INSTRUCTIONS_CLARIFY, docs["kenya-digital"].text, scenario="nb01_ambiguous_with_instruction"))
'''),
    md("""
The first answer picks a reading and runs with it. The second names the readings and asks. For a research application the second is the behaviour to specify, because a confident answer to the wrong question looks the same as a confident answer to the right one.

## 6. What the model received

The last call's messages, as sent. The three parts are visible: system instructions, then the context and the question in the user message.
"""),
    code('''
for m in llm.calls[-1]["messages"]:
    print(f"--- {m['role']} ({len(m['content']):,} characters)")
    print(m["content"][:400] + (" …" if len(m["content"]) > 400 else ""))
    print()
'''),
    md("""
## 7. What context costs

Four short briefs took about a thousand tokens. A real collection has thousands of documents and cannot be pasted into every call: the context window has a limit, cost grows with every token, and the relevant passages are a small part of the whole. The next two notebooks prepare a collection so that the application can select what to supply.

## Key points

- An application separates instructions, context, and the question. Each is changed by a different person for a different reason.
- Instructions shape behaviour: citations, what to do when information is missing, when to ask. They cannot supply data the model does not have.
- Supplied context is what makes the answer specific and checkable. Selecting it is the job of the rest of the application.

## Exercise

Change `INSTRUCTIONS_GROUNDED` so that the answer is a table with one row per country and columns for the indicator, the latest year, the value, and the source. Rerun section 3. With a live key the output changes; in recorded mode the cell returns the recorded answer, so note what you would expect instead.
"""),
    md(LINKS),
]

# =============================================================================
# 02
# =============================================================================
NB02 = [
    md("""
# 02 · Preparing knowledge for an application

**Session 01 · notebook 2 of 6**

## Objective

Turn a collection of documents into something an application can search: documents with metadata, split into chunks that keep their meaning, represented as vectors. See how the choices made here decide what retrieval can find later.

## The application after this notebook

```
documents ──> metadata ──> chunks ──> embeddings ──> index
                                                      │
question ──> [ instructions + retrieved context ] ──> model ──> answer     (notebook 03)
```
"""),
    code(SETUP),
    md("""
## 1. The collection

Eight short documents. Each has a header with an id, a title, a type, a country, a date, and a source. Four are relevant briefs, one is a definition note, one is an outdated summary, and two are distractors. See `assets/README.md` for what each is for.
"""),
    code('''
docs = w.load_documents()
print(f"{'id':20s} {'type':16s} {'date':12s} title")
for d in docs:
    print(f"{d.id:20s} {d.type:16s} {d.date:12s} {d.title}")
'''),
    md("""
## 2. Metadata is part of the knowledge

The outdated summary is the clearest case. Its text reads as current ("the latest available figures are for 2015") and its figures were correct when it was written. The only thing that tells an application to rank it below the 2026 briefs is its date, which lives in the header.
"""),
    code('''
d = next(d for d in docs if d.id == "regional-2016")
print("date:  ", d.date)
print("type:  ", d.type)
print("note:  ", d.note)
print()
print(d.text[:500])
'''),
    md("""
## 3. Chunking: fixed windows against sections

A chunk is the unit that retrieval returns and the model reads. Two strategies on the same brief:

- **Fixed windows** of 400 characters with an overlap. Simple, and blind to the document's structure.
- **Sections**, one chunk per Markdown heading, with the title and the heading kept in the text and the section name in the metadata.
"""),
    code('''
kenya = next(d for d in docs if d.id == "kenya-digital")
fixed = w.chunk_fixed(kenya, size=400, overlap=50)
sections = w.chunk_sections(kenya)
print(f"fixed windows: {len(fixed)} chunks · sections: {len(sections)} chunks\\n")

print("A fixed window (note where it starts and ends):\\n")
print(repr(fixed[1].text))
print("\\nA section chunk:\\n")
print(sections[1].text)
print("\\nmetadata:", sections[1].section, "|", sections[1].date, "|", sections[1].country)
'''),
    md("""
The fixed window starts and ends mid-sentence and carries no heading, so a passage about mobile subscriptions can be retrieved for a question about internet use without anything in the chunk saying which it is. The section chunk carries its title and heading, which both the retriever and the model can use.

## 4. Embeddings

An embedding is a vector that places a text so that texts with similar meaning are close. The `Embedder` uses a small sentence-transformer model when it can be downloaded and falls back to TF-IDF otherwise; both give the same interface. The cell reports which one is in use.
"""),
    code('''
# Optional, for better embeddings in Colab (about 90 MB download):
# %pip install -q sentence-transformers

import numpy as np

chunks = [c for d in docs for c in w.chunk_sections(d)]
emb = w.Embedder()
emb.fit([c.text for c in chunks])
print("embedding backend:", emb.backend)

sample = [c for c in chunks if c.section in ("Internet use", "Mobile subscriptions", "Growth")][:6]
vectors = emb.encode([c.text for c in sample])
query = emb.encode(["share of people using the internet"])[0]
print()
for c, v in sorted(zip(sample, vectors), key=lambda cv: -float(cv[1] @ query)):
    print(f"{float(v @ query):6.3f}  {c.doc_id:16s} {c.section}")
'''),
    md("""
Sections about internet use score highest for a query about internet use, the mobile-subscription sections next, and growth last. With the sentence-transformer backend the separation is wider; with TF-IDF it rests on shared words. The lesson holds for both: the quality of the ranking depends on what the chunk text contains.

## 5. What poor preparation does

The same collection with the headers stripped, every document concatenated into one text, and fixed windows. Search for the latest Kenya figure and compare the top results with the section index.
"""),
    code('''
blob = w.Document(id="blob", title="", type="", country="", date="", source="", note="", text="\\n\\n".join(d.text for d in docs))
poor = w.Retriever(w.chunk_fixed(blob, size=400, overlap=50))
good = w.Retriever(chunks)

q = "What is the latest available figure for internet use in Kenya?"
print("Poorly prepared (one blob, fixed windows, no metadata):")
for h in poor.search(q, k=3):
    print(f"  {h.score:6.3f}  {h.chunk.text[:90]!r}")
print("\\nPrepared (sections with metadata):")
w.print_hits(good.search(q, k=3))
'''),
    md("""
In the poorly prepared index the top results are fragments that cannot be cited and carry no date. In the prepared index each hit names its document, section, and year, so the application can filter and the reader can check.

## 6. Build the index for the next notebook

Notebook 03 rebuilds this index with one line. The retriever holds the chunks, the embedder, and the matrix of vectors.
"""),
    code('''
retriever = w.Retriever(chunks)
print(len(retriever.chunks), "chunks indexed with", retriever.embedder.backend)
'''),
    md("""
## Key points

- Metadata (date, type, country, section) is part of what the application knows. It decides what can be filtered and ranked, and what a citation can say.
- Chunk boundaries decide what retrieval can return. Sections keep meaning and headings; fixed windows cut through both.
- Embeddings place text by meaning; the quality of the index is the quality of the text and metadata it was built from.

## Exercise

Write a ninth document, a short brief on Uganda with figures you look up from the World Development Indicators, with the same header fields. Add it to `assets/documents/`, rerun this notebook, and check where its sections rank for the queries above.
"""),
    md(LINKS),
]

# =============================================================================
# 03
# =============================================================================
NB03 = [
    md("""
# 03 · Retrieval-augmented generation

**Session 01 · notebook 3 of 6**

## Objective

Connect the index to the model: retrieve the passages that match the question and supply them as context. See a retrieval failure, three ways to fix it, and why retrieval and generation are evaluated separately.

## The application after this notebook

```
question ──> retriever ──> top-k passages ──┐
                                             ├──> [ instructions + passages + question ] ──> model ──> answer with citations
question ────────────────────────────────────┘
```
"""),
    code(SETUP),
    code('''
docs = w.load_documents()
chunks = [c for d in docs for c in w.chunk_sections(d)]
retriever = w.Retriever(chunks)
print(len(chunks), "chunks ·", retriever.embedder.backend)

INSTRUCTIONS = """You are a research assistant for development data.
Answer only from the supplied passages. After each figure, cite the passage in the square-bracket form given.
If the passages do not contain what is needed, say so."""


def rag_answer(question, k=4, scenario="unnamed", where=None, recency_weight=0.0):
    hits = retriever.search(question, k=k, where=where, recency_weight=recency_weight)
    user = f"PASSAGES:\\n{w.format_context(hits)}\\n\\nQUESTION: {question}"
    completion = llm.complete([{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}], scenario=scenario)
    return completion, hits
'''),
    md("""
## 1. The worked example through the pipeline

Retrieve, then answer. The hits are printed first so that the evidence the model saw is visible.
"""),
    code('''
answer, hits = rag_answer(w.MAIN_QUESTION, k=5, scenario="nb03_rag_main")
w.print_hits(hits)
print()
w.show(answer)
'''),
    md("""
## 2. A retrieval failure

"What is the latest available figure for internet use in Kenya?" The 2016 summary contains the words "latest available figures" and ranks high. With a word-based backend it ranks first. The model then answers from it, correctly summarizing an outdated passage.
"""),
    code('''
q = "What is the latest available figure for internet use in Kenya?"
answer, hits = rag_answer(q, k=3, scenario="nb03_rag_outdated")
w.print_hits(hits)
print()
w.show(answer)
'''),
    md("""
The answer is faithful to its evidence and wrong for the reader. Nothing in the generation step can fix this; the evidence was selected badly.

## 3. Three fixes, all on the retrieval side

1. **A metadata filter**: exclude documents dated before 2020 for a question about the latest figure.
2. **A recency weight**: add a bonus to the score for newer documents.
3. **Query reformulation**: search for what the passage would say ("internet use Kenya 2024") instead of what the user asked.
"""),
    code('''
print("1. metadata filter (date >= 2020):")
w.print_hits(retriever.search(q, k=3, where=lambda c: c.date >= "2020"))
print("\\n2. recency weight:")
w.print_hits(retriever.search(q, k=3, recency_weight=0.3))
print("\\n3. reformulated query:")
w.print_hits(retriever.search("Kenya internet use 2024 percent of population", k=3))
'''),
    code('''
answer, hits = rag_answer(q, k=3, scenario="nb03_rag_recency", recency_weight=0.3)
w.show(answer)
'''),
    md("""
## 4. Retrieval always returns something

A question outside the collection still gets top-k passages, ranked by whatever is least unlike it. Compare a question the collection can answer with one it cannot.
"""),
    code('''
for q2 in ["What is the population of Rwanda?", "What was inflation in Uganda in 2023?"]:
    print(q2)
    w.print_hits(retriever.search(q2, k=3))
    print()
'''),
    md("""
Both questions get three passages with a score. With the word-based backend the out-of-collection question can even score higher, because it shares words such as "2023" and "percent" with the growth sections. The score alone does not say "no evidence". Two things do: a threshold set from examples on the backend in use, which the evaluation in notebook 06 is for, and instructions that tell the model to say when the passages do not answer the question, which notebook 01 introduced. Applications use both.

## 5. Retrieval and generation are evaluated separately

Retrieval can be tested without calling a model: for each evaluation question, did the top-k passages include a passage from the expected document? The cell runs that test for all eight questions at two values of k.
"""),
    code('''
questions = w.load_questions()
print(f"{'id':4s} {'category':18s} {'k=2':6s} {'k=4':6s} expected documents")
for qd in questions:
    r2 = w.retrieval_hit(retriever.search(qd["question"], k=2), qd["expected_documents"])
    r4 = w.retrieval_hit(retriever.search(qd["question"], k=4), qd["expected_documents"])
    print(f"{qd['id']:4s} {qd['category']:18s} {'hit' if r2 else 'miss':6s} {'hit' if r4 else 'miss':6s} {', '.join(qd['expected_documents'])}")
'''),
    md("""
A miss here is a retrieval problem, fixed by chunking, metadata, ranking, or the query. A wrong answer with a hit is a generation problem, fixed by instructions or a different model. Mixing the two makes both impossible to improve.

## Key points

- Retrieval selects the evidence; the model writes from it. A faithful answer to the wrong evidence is wrong.
- Fixes for retrieval live in metadata, ranking, and query formulation. The model cannot repair a bad selection.
- Retrieval always returns something; a score threshold turns "least unlike" into "no evidence".
- Test retrieval on its own, with expected documents per question.

## Exercise

Change `chunk_sections` to `chunk_fixed` in the first cell and rerun section 5. Count the misses. Then try k=8.
"""),
    md(LINKS),
]

# =============================================================================
# 04
# =============================================================================
NB04 = [
    md("""
# 04 · Connecting the application to data and tools

**Session 01 · notebook 4 of 6**

## Objective

Give the model a tool that returns authoritative statistics, let it call the tool, and make its output a typed object that other software can read. See what happens when the data does not exist.

## The application after this notebook

```
question ──> model ──┬──> tool call: get_indicator(indicator, country, years) ──> data ──┐
                     │<──────────────────────────────────────────────────────────────────┘
                     └──> answer, or a JSON object that matches a schema
```

Passages are for definitions and context. Numbers come from a tool over the data.
"""),
    code(SETUP),
    md("""
## 1. A number from memory and a number from data
"""),
    code('''
answer = llm.complete([{"role": "user", "content": "What was Rwanda's GDP per capita in 2023?"}], scenario="nb04_from_memory")
w.show(answer)
print()
r = w.get_indicator("NY.GDP.PCAP.CD", "Rwanda", start=2023, end=2023)
print("Data tool:", r["name"], r["observations"], "·", r["source"])
'''),
    md("""
The memory answer is close and hedged, with no way for a reader to tell how close. The data answer carries the indicator, the year, the value, the source, and a `claim_id` that notebook 06 will use to verify it.

## 2. The tool and its schema

A tool is a function plus a description the model can read. The description says what the tool is for and what its parameters mean; the model fills the parameters from the question.
"""),
    code('''
import json
print(json.dumps(w.INDICATOR_TOOL, indent=2))
print()
print("available indicators:")
for i in w.list_indicators():
    print(f"  {i['id']:20s} {i['name']}")
'''),
    md("""
## 3. The model calls the tool

`run_tool_loop` sends the question with the tool schema, runs the tool the model asks for, returns the result, and repeats until the model answers in text. The trace shows every call.
"""),
    code('''
INSTRUCTIONS = """You are a research assistant for development data.
Use get_indicator for every statistic; do not quote figures from memory. Name the indicator and the source with each figure.
If the tool returns no observation for the year asked, say so and give the latest available year."""

handlers = {"get_indicator": w.get_indicator}

llm.reset()
final, trace = w.run_tool_loop(
    llm,
    [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": "What was Rwanda's GDP per capita in 2023?"}],
    tools=[w.INDICATOR_TOOL], handlers=handlers, scenario="nb04_tool_call",
)
print()
w.show(final)
'''),
    md("""
## 4. Data that does not exist

The planted problem: a year with no observation. The tool returns an empty list, and the instructions say what to do with it.
"""),
    code('''
llm.reset()
final, trace = w.run_tool_loop(
    llm,
    [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": "What share of Kenya's population used the internet in 2025?"}],
    tools=[w.INDICATOR_TOOL], handlers=handlers, scenario="nb04_missing_year",
)
print()
w.show(final)
print("\\ntool result for 2025:", trace[0]["result"]["observations"])
'''),
    md("""
Compare with notebook 01, where the model answered the same question from a brief. Here the absence is a fact returned by the tool, which the application can test for, instead of a sentence in the model's answer.

## 5. Structured output

An answer that another program will read, store, or display needs a fixed shape. The schema below is a Pydantic model; its JSON schema is sent with the request, and the response is validated on the way back.
"""),
    code('''
print(json.dumps(w.ComparisonAnswer.model_json_schema(), indent=2)[:1200], "…")
'''),
    code('''
schema = {"type": "json_schema", "json_schema": {"name": "comparison", "schema": w.ComparisonAnswer.model_json_schema()}}
completion = llm.complete(
    [
        {"role": "system", "content": "Fill the schema from the data you are given. Use the claim_id values exactly as given."},
        {"role": "user", "content": "DATA:\\n" + json.dumps([w.get_indicator(i, c, start=2024) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]) + "\\n\\nQUESTION: " + w.MAIN_QUESTION},
    ],
    scenario="nb04_structured", response_format=schema,
)
parsed = w.ComparisonAnswer.model_validate_json(completion.content)
for cs in parsed.countries:
    print(cs.country, "internet", cs.internet_latest.year, cs.internet_latest.value, "| GDP pc", cs.gdp_per_capita_latest.year, round(cs.gdp_per_capita_latest.value))
print()
print(parsed.comparison)
print(parsed.sources)
'''),
    md("""
Validation is what makes the schema a contract. The cell below damages the JSON and shows the error that a downstream program would receive instead of a silently wrong value.
"""),
    code('''
from pydantic import ValidationError

damaged = completion.content.replace('"year": 2024', '"year": "twenty twenty-four"', 1)
try:
    w.ComparisonAnswer.model_validate_json(damaged)
except ValidationError as e:
    print(e.errors()[0]["loc"], "->", e.errors()[0]["msg"])
'''),
    md("""
## 6. Live data and MCP

The same tool can read the World Bank API instead of the committed file. The cell is guarded because the session must not depend on the network.

The Model Context Protocol (MCP) is a standard way to expose tools like this one to any client. The program's [Data360 MCP server](https://github.com/worldbank/data360-mcp) exposes search, metadata, and data tools over the World Bank's data platform; a desktop assistant, a chatbot, or this notebook can call them with the same schema idea as `get_indicator`. The slides cover it; the notebooks keep tools local so that they run offline.
"""),
    code('''
try:
    live = w.get_indicator("IT.NET.USER.ZS", "Kenya", start=2022, live=True)
    print("live:", [(o["year"], round(o["value"], 1)) for o in live["observations"]])
except Exception as e:
    print("live call skipped:", type(e).__name__)
'''),
    md("""
## Key points

- Numbers come from a tool over authoritative data; the model's memory is for language, not for statistics.
- A tool is a function with a description the model can read. The model fills the parameters; the application runs the function and returns the result.
- Absence of data is a tool result the application can test for.
- A schema turns the answer into a contract: validated on the way in, readable by other software on the way out.

## Exercise

Add a second tool that returns population (`SP.POP.TOTL`) and ask "How many people in Rwanda used the internet in 2024?". The answer needs two tool calls and a calculation; check whether the model does the arithmetic or asks for a tool to do it.
"""),
    md(LINKS),
]

# =============================================================================
# 05
# =============================================================================
NB05 = [
    md("""
# 05 · Workflows and agents

**Session 01 · notebook 5 of 6**

## Objective

Give the application two tools, document search and indicator data, and compare two ways of deciding which to call: a fixed workflow written in code, and a loop in which the model chooses. See how to bound the loop, inspect its decisions, and handle a failed call.

## The application after this notebook

```
                 ┌──> search_documents(query) ──> passages ──┐
question ──> model                                           ├──> model ──> answer
                 └──> get_indicator(indicator, country) ──> data ──┘
                 (fixed order in code, or chosen by the model, with a step limit and a trace)
```
"""),
    code(SETUP),
    code('''
import json

docs = w.load_documents()
chunks = [c for d in docs for c in w.chunk_sections(d)]
retriever = w.Retriever(chunks)


def search_documents(query, k=3):
    return [{"citation": h.chunk.citation(), "score": round(h.score, 3), "text": h.chunk.text} for h in retriever.search(query, k=k, recency_weight=0.2)]


handlers = {"search_documents": search_documents, "get_indicator": w.get_indicator}
tools = [w.SEARCH_TOOL, w.INDICATOR_TOOL]

INSTRUCTIONS = """You are a research assistant for development data.
Use get_indicator for every statistic and search_documents for definitions, context, and explanations.
Name the indicator and the source with each figure; cite passages in the square-bracket form given.
If the tools do not provide what is needed, say so. Do not quote figures from memory."""
'''),
    md("""
## 1. A fixed workflow

The order of calls is written in code: search the documents, fetch the four series, compose. The model is called once, to write. Every run makes the same calls.
"""),
    code('''
def fixed_workflow(question, scenario="nb05_compose"):
    passages = search_documents(question, k=4)
    series = [w.get_indicator(i, c, start=2010) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]
    user = ("PASSAGES:\\n" + "\\n\\n".join(f"{p['citation']}\\n{p['text']}" for p in passages)
            + "\\n\\nDATA:\\n" + json.dumps(series)
            + f"\\n\\nQUESTION: {question}")
    return llm.complete([{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}], scenario=scenario)


w.show(fixed_workflow(w.MAIN_QUESTION))
'''),
    md("""
## 2. The model chooses the tools

The same tools, offered to the model with their schemas. The loop runs what it asks for, up to a step limit, and records a trace.
"""),
    code('''
llm.reset()
final, trace = w.run_tool_loop(
    llm, [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": w.MAIN_QUESTION}],
    tools=tools, handlers=handlers, scenario="nb05_agent_main", max_steps=4,
)
print()
w.show(final)
print(f"\\n{len(trace)} tool calls in {max(t['step'] for t in trace)} step(s)")
'''),
    md("""
The model chose the same calls the fixed workflow made, in one step. On a different question it would choose differently, which is the point and the risk.

## 3. A question outside the tools

The planted out-of-scope question. A good loop searches, finds nothing usable, and says so.
"""),
    code('''
llm.reset()
final, trace = w.run_tool_loop(
    llm, [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": "How have Kenya's coffee export earnings changed in recent years?"}],
    tools=tools, handlers=handlers, scenario="nb05_agent_offscope", max_steps=3,
)
print()
w.show(final)
'''),
    md("""
## 4. A failed call

The model asks for an indicator id that does not exist. The tool returns an error object instead of raising; the model reads it and retries with the right id. Returning errors as data is what lets the loop recover.
"""),
    code('''
llm.reset()
final, trace = w.run_tool_loop(
    llm, [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": "What share of Rwanda's population used the internet in 2023?"}],
    tools=tools, handlers=handlers, scenario="nb05_agent_error", max_steps=4,
)
print()
w.show(final)
'''),
    md("""
## 5. Bounding the loop

A step limit is the simplest control. With `max_steps=1` the loop returns after the first round of tool calls and says that it stopped. An application decides what to do then: answer with what it has, ask the user, or fail.
"""),
    code('''
llm.reset()
final, trace = w.run_tool_loop(
    llm, [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": w.MAIN_QUESTION}],
    tools=tools, handlers=handlers, scenario="nb05_agent_main", max_steps=1, verbose=False,
)
print(final)
print(len(trace), "tool calls were made before the limit")
'''),
    md("""
## 6. Choosing between the two

| | Fixed workflow | Model-directed loop |
|---|---|---|
| Calls per run | Known in advance | Varies with the question |
| Behaviour on a new kind of question | Same calls, possibly irrelevant | Adapts, possibly wrongly |
| Cost | Predictable | Bounded by the step limit |
| Testing | Test the composer | Test the trace and the answer |
| Use when | The questions have one shape | The questions vary and the tools are few and safe |

The worked example has one shape, so the fixed workflow is the right design for it. The loop earns its place when users ask questions the designer did not anticipate, and only with a step limit, a trace, and tools that fail as data.

## Key points

- A workflow written in code is predictable and testable; prefer it when the task has one shape.
- A model-directed loop adapts to the question; bound it with a step limit, keep a trace, and return tool errors as data.
- Both need the same two things from the tools: good descriptions and results the model can read.

## Exercise

Remove `get_indicator` from `tools` and rerun section 2. Read what the model does without a data tool and whether the instructions stop it from quoting figures from memory. In recorded mode, write down what you expect; with a key, compare.
"""),
    md(LINKS),
]

# =============================================================================
# 06
# =============================================================================
NB06 = [
    md("""
# 06 · Making the application trustworthy

**Session 01 · notebook 6 of 6**

## Objective

Add verification of every number in an answer and an evaluation suite that runs the application over a set of questions with known evidence. Compare two configurations of the application on the same suite.

## The checks

| Check | Question it answers | Where it runs |
|---|---|---|
| Retrieval | Did the application find the right evidence? | Notebook 03, section 5 |
| Grounding | Is the answer written from that evidence? | Citations in the answer |
| Verification | Does each number match its source record? | Claim tags, this notebook |
| Evaluation | Does the application behave across many questions? | The suite, this notebook |

## The application after this notebook

```
question ──> tools ──> answer with <claim> tags ──> verifier ──> answer with marks: verified · flagged · unmarked
                                                   evaluation suite ──> table per question
```
"""),
    code(SETUP),
    code('''
import json

docs = w.load_documents()
chunks = [c for d in docs for c in w.chunk_sections(d)]
retriever = w.Retriever(chunks)


def search_documents(query, k=3):
    return [{"citation": h.chunk.citation(), "score": round(h.score, 3), "text": h.chunk.text} for h in retriever.search(query, k=k, recency_weight=0.2)]


handlers = {"search_documents": search_documents, "get_indicator": w.get_indicator}
tools = [w.SEARCH_TOOL, w.INDICATOR_TOOL]

INSTRUCTIONS_CLAIMS = """You are a research assistant for development data.
Use get_indicator for every statistic and search_documents for definitions and context. Do not quote figures from memory.
Write every number that comes from get_indicator inside a claim tag that names its claim_id, like
<claim id="IT.NET.USER.ZS:KEN:2024" policy="rounded" decimals="1">35.0</claim>.
Name the indicator and the source. If the tools do not provide what is needed, say so.
If the question is ambiguous, ask which reading is meant."""
'''),
    md("""
## 1. Claim tags

The instructions ask the model to wrap each figure in a tag that names the observation it came from. The verifier compares the number shown with the record under a policy (rounded to the stated decimals, or exact). This is the Proof-Carrying Numbers protocol used in the program's chatbot, reduced to one function.
"""),
    code('''
results = [w.get_indicator(i, c, start=2010) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]
observations = w.observations_from_results(results)
print(len(observations), "observations available to the verifier, for example", list(observations)[:2])

completion = llm.complete(
    [{"role": "system", "content": INSTRUCTIONS_CLAIMS}, {"role": "user", "content": "DATA:\\n" + json.dumps(results) + "\\n\\nQUESTION: " + w.MAIN_QUESTION}],
    scenario="nb06_claims_main",
)
print("\\nAs the reader sees it:\\n")
w.show(w.strip_claims(completion.content))
print("\\nVerification:\\n")
w.print_checks(w.verify(completion.content, observations))
'''),
    md("""
Every tagged number matches its record. Years are not claims and are skipped.

## 2. A wrong number and a number without a claim

The planted problem. One tagged figure has been altered, and one figure has no tag because it came from nowhere in the data. The verifier flags the first and marks the second as unmarked: it carries no claim, so it receives no mark, whether or not it happens to be right.
"""),
    code('''
completion = llm.complete(
    [{"role": "system", "content": INSTRUCTIONS_CLAIMS}, {"role": "user", "content": "DATA:\\n" + json.dumps(results) + "\\n\\nQUESTION: " + w.MAIN_QUESTION}],
    scenario="nb06_claims_error",
)
w.show(w.strip_claims(completion.content))
print()
w.print_checks(w.verify(completion.content, observations))
'''),
    md("""
A reader of the chatbot sees a mark beside each number. A flagged number is one the model changed. An unmarked number is one the model did not tie to a record, and the fail-closed rule says it gets no mark.

What the check says nothing about: whether the record is the right one for the question, whether the method behind it is sound, or whether the sentence around the number interprets it correctly. Those stay with the reader.

## 3. Two configurations of the application

- **A, retrieval only**: passages as context, no tools, no tags. The application of notebook 03.
- **B, tools and claims**: search and data tools, claim tags, the verifier. The application of this notebook.
"""),
    code('''
INSTRUCTIONS_RAG = """You are a research assistant for development data.
Answer only from the supplied passages and cite them in the square-bracket form given. If they do not contain what is needed, say so."""


def app_rag(question, scenario):
    hits = retriever.search(question, k=4)
    user = f"PASSAGES:\\n{w.format_context(hits)}\\n\\nQUESTION: {question}"
    completion = llm.complete([{"role": "system", "content": INSTRUCTIONS_RAG}, {"role": "user", "content": user}], scenario=scenario)
    return completion.content or "", hits, {}


def app_tools(question, scenario):
    llm.reset(scenario)
    final, trace = w.run_tool_loop(
        llm, [{"role": "system", "content": INSTRUCTIONS_CLAIMS}, {"role": "user", "content": question}],
        tools=tools, handlers=handlers, scenario=scenario, max_steps=4, verbose=False,
    )
    obs = w.observations_from_results([t["result"] for t in trace if t["tool"] == "get_indicator" and "observations" in t["result"]])
    hits = retriever.search(question, k=4)  # for the retrieval column; the loop may or may not have searched
    return final, hits, obs
'''),
    md("""
## 4. The evaluation suite

Eight questions with expected documents, expected numbers, and expected behaviour, from `assets/questions.json`. For each question and configuration the table reports:

- **retrieval**: did the top-4 passages include an expected document
- **numbers**: how many expected values appear in the answer
- **cited**: does the answer cite a retrieved document or name a source
- **verify**: counts of verified, flagged, and unmarked numbers (configuration B only)
"""),
    code('''
questions = w.load_questions()


def evaluate(app, prefix):
    rows = []
    for qd in questions:
        answer, hits, obs = app(qd["question"], f"{prefix}_{qd['id']}")
        found, total = w.numbers_present(answer, qd["expected_numbers"])
        cited = w.grounded(answer, hits) or "WDI" in answer or "World Development Indicators" in answer
        if obs:
            checks = w.verify(answer, obs)
            v = sum(c.status == "verified" for c in checks); f = sum(c.status == "flagged" for c in checks); u = sum(c.status == "unmarked" for c in checks)
            ver = f"{v}v {f}f {u}u"
        else:
            ver = "n/a" if prefix.endswith("rag") else "no data used"
        rows.append(w.EvalRow(qd["id"], qd["category"], "hit" if w.retrieval_hit(hits, qd["expected_documents"]) else "miss", f"{found}/{total}", "yes" if cited else "no", ver))
    return rows


print("A · retrieval only\\n")
rows_a = evaluate(app_rag, "nb06_rag")
w.print_table(rows_a)
print("\\n\\nB · tools and claims\\n")
rows_b = evaluate(app_tools, "nb06_tools")
w.print_table(rows_b)
'''),
    md("""
## 5. Reading the table

- **q8, outdated evidence.** Configuration A quotes the 2015 figure when the 2016 summary ranks first; B fetches the series and reports 2024. The expected number is found in B and missed in A.
- **q3, ambiguous.** Neither configuration should produce a number. B asks which reading is meant, as its instructions say; A picks one. The table shows 0/0 for both; the behaviour has to be read.
- **q4, missing data.** Both say that 2025 is unavailable; B's statement rests on an empty tool result.
- **q5, out of scope.** Both give no number. A's retrieval "hit" is the coffee note, which is the expected document and contains no figures.
- **verify.** In B every number from the data tool is verified; in A there is nothing to verify against, because no number came from a tool.

A table like this is what distinguishes a demonstration from a measured application. It is rerun after every change to the instructions, the tools, the index, or the model.

## 6. What the suite does not cover

- Whether the application chose the right indicator for the question. A verified number from the wrong series is verified.
- Whether the interpretation around the numbers is right.
- Behaviour on questions nobody wrote down. The suite grows from use: every failure found in use becomes a question.

The program's evaluation cookbook describes how to build suites that cover retrieval, grounding, and verification at scale, and how to gate changes on them.

## Key points

- Verification checks each number against its source record and fails closed: no claim, no mark.
- Evaluation runs the application over questions with known evidence and expected behaviour, and is rerun on every change.
- The two configurations differ most on outdated evidence and on numbers: tools make numbers checkable; retrieval alone does not.

## Exercise

Add a ninth question to `assets/questions.json` with an expected number that is deliberately wrong, rerun the suite, and confirm that the numbers column catches it. Then correct it.
"""),
    md(LINKS),
]

NOTEBOOKS = {
    "01-from-model-to-application.ipynb": NB01,
    "02-preparing-knowledge.ipynb": NB02,
    "03-retrieval-augmented-generation.ipynb": NB03,
    "04-tools-and-structured-outputs.ipynb": NB04,
    "05-workflows-and-agents.ipynb": NB05,
    "06-verification-and-evaluation.ipynb": NB06,
}


def write_all() -> list[Path]:
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        nb.metadata["colab"] = {"provenance": []}
        path = HERE / name
        nbformat.write(nb, path)
        paths.append(path)
    return paths


def execute(paths: list[Path]) -> None:
    from nbclient import NotebookClient

    kernel = os.environ.get("WORKSHOP_KERNEL", "python3")
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name=kernel, resources={"metadata": {"path": str(HERE)}}).execute()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.write(nb, path)
        print("executed", path.name)


if __name__ == "__main__":
    written = write_all()
    print("wrote", len(written), "notebooks")
    if "--execute" in sys.argv:
        execute(written)
