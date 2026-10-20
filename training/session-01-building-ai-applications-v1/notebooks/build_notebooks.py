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

REPO_URL = "https://github.com/worldbank/ai4data.git"   # replace with the published repository when released
BRANCH = "feat/training-series"
SESSION = "training/session-01-building-ai-applications-v1/notebooks"

if not pathlib.Path("workshop.py").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO_URL, "/content/ai4data"], check=True)
    os.chdir(f"/content/ai4data/{SESSION}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, os.getcwd())

# Optional: a key for an OpenAI-compatible API. Without one, the notebook replays recorded outputs and says so.
try:
    from google.colab import userdata  # type: ignore
    for name in ("WORKSHOP_API_KEY", "WORKSHOP_BASE_URL", "WORKSHOP_MODEL"):
        if userdata.get(name):
            os.environ[name] = userdata.get(name)
except Exception:
    pass

import workshop as w
model = w.make_model()
print("Model:", w.mode(model))'''

STACK = """## The stack

| Need | Library |
|---|---|
| Documents, splitting, embeddings, vector store, prompts, tools, structured output | [LangChain](https://python.langchain.com/) |
| The agent loop | [LangGraph](https://langchain-ai.github.io/langgraph/), through `langchain.agents.create_agent` |
| Tool server and client over the Model Context Protocol | [FastMCP](https://gofastmcp.com/) |
| Number verification | [`pcn-core`](https://github.com/worldbank/pcn), the Python core of Proof-Carrying Numbers |
| Output schemas | [Pydantic](https://docs.pydantic.dev/) |

`workshop.py` holds what these do not: the worked-example data, a chat model that replays recorded outputs when no API key is configured, and a few printing and evaluation helpers."""

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

Run the cells in order. Without an API key the model outputs are recorded ones, labelled `[recorded output: ...]`; each cell names its scenario with `w.use(...)` before calling the model. With a key in `WORKSHOP_API_KEY` the same cells call the model live.
"""),
    md(STACK),
    code(SETUP),
    md("""
## 1. A prompt with three parts

Three parts go into every call: **instructions** (the system message), **context** (evidence supplied to the model), and the **question**. LangChain's `ChatPromptTemplate` keeps them as separate slots, and `prompt | model` is the whole application for now. Keeping the parts separate is the first design decision, because each is changed by a different person for a different reason.
"""),
    code('''
from langchain_core.prompts import ChatPromptTemplate

INSTRUCTIONS_BASIC = "You are a research assistant. Answer the question."

INSTRUCTIONS_GROUNDED = """You are a research assistant for development data.
Answer only from the supplied context. Cite the document id in square brackets after each figure.
If the context does not contain what is needed, say so and give the closest available information."""

INSTRUCTIONS_CLARIFY = INSTRUCTIONS_GROUNDED + """
If the question is ambiguous, say which readings are possible and ask which one is meant before answering."""

prompt = ChatPromptTemplate.from_messages([
    ("system", "{instructions}"),
    ("human", "CONTEXT:\\n{context}\\n\\nQUESTION: {question}"),
])
chain = prompt | model

print(w.MAIN_QUESTION)
'''),
    md("""
## 2. Zero context: the model answers from its training data

The baseline. No evidence is supplied, so the model answers from what it absorbed in training.
"""),
    code('''
w.use("nb01_zero_context")
answer = chain.invoke({"instructions": INSTRUCTIONS_BASIC, "context": "(none)", "question": w.MAIN_QUESTION})
w.show(answer)
'''),
    md("""
Read the answer against the data. The cell below prints the latest values from the World Development Indicators that the session uses as its reference.
"""),
    code('''
for ind in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"]:
    for country in ["Kenya", "Rwanda"]:
        r = w.indicator_lookup(ind, country, start=2023)
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
docs = {d.metadata["id"]: d for d in w.load_documents()}
context = "\\n\\n".join(f"[{i}]\\n{docs[i].page_content}" for i in ["kenya-digital", "rwanda-digital", "kenya-economy", "rwanda-economy"])
print(f"{len(context):,} characters of context, roughly {len(context)//4:,} tokens\\n")

w.use("nb01_with_context")
answer = chain.invoke({"instructions": INSTRUCTIONS_GROUNDED, "context": context, "question": w.MAIN_QUESTION})
w.show(answer)
'''),
    md("""
The figures now come from the briefs, each one is cited, and the comparison is specific. The model did none of the finding; the application did, by putting the right text in front of it.

## 4. Handling missing information

An application has to behave well when the evidence does not contain the answer. The instruction "say so and give the closest available information" is what makes the difference.
"""),
    code('''
w.use("nb01_missing_info")
answer = chain.invoke({"instructions": INSTRUCTIONS_GROUNDED, "context": docs["kenya-digital"].page_content,
                       "question": "What share of Kenya's population used the internet in 2025?"})
w.show(answer)
'''),
    md("""
## 5. The ambiguous question under two sets of instructions

"How connected is Kenya?" can mean internet use, mobile subscriptions, broadband, or something else. Compare what the model does with the basic instructions and with an instruction to ask.
"""),
    code('''
print("Without an instruction to clarify:\\n")
w.use("nb01_ambiguous_no_instruction")
w.show(chain.invoke({"instructions": INSTRUCTIONS_BASIC, "context": docs["kenya-digital"].page_content, "question": "How connected is Kenya?"}))
print("\\n" + "=" * 100 + "\\nWith the instruction:\\n")
w.use("nb01_ambiguous_with_instruction")
w.show(chain.invoke({"instructions": INSTRUCTIONS_CLARIFY, "context": docs["kenya-digital"].page_content, "question": "How connected is Kenya?"}))
'''),
    md("""
The first answer picks a reading and runs with it. The second names the readings and asks. For a research application the second is the behaviour to specify, because a confident answer to the wrong question looks the same as a confident answer to the right one.

## 6. What the model received

The messages the prompt produced for the last call. The three parts are visible: system instructions, then the context and the question in the human message.
"""),
    code('''
messages = prompt.format_messages(instructions=INSTRUCTIONS_CLARIFY, context=docs["kenya-digital"].page_content, question="How connected is Kenya?")
for m in messages:
    print(f"--- {m.type} ({len(m.content):,} characters)")
    print(m.content[:400] + (" …" if len(m.content) > 400 else ""))
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

Change `INSTRUCTIONS_GROUNDED` so that the answer is a table with one row per country and columns for the indicator, the latest year, the value, and the source. Rerun section 3. With a live key the output changes; in recorded mode the cell returns the recorded answer, so note the expected result instead.
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

Turn a collection of documents into something an application can search: documents with metadata, split into chunks that keep their meaning, represented as vectors in a vector store. See how the choices made here decide what retrieval can find later.

## The application after this notebook

```
documents ──> metadata ──> chunks ──> embeddings ──> vector store
                                                         │
question ──> [ instructions + retrieved context ] ──> model ──> answer     (notebook 03)
```

The tools are LangChain's: `Document`, `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, and `InMemoryVectorStore`.
"""),
    code(SETUP),
    md("""
## 1. The collection

Eight short documents, each loaded as a LangChain `Document` whose metadata comes from the file's YAML header: id, title, type, country, date, source. Four are relevant briefs, one is a definition note, one is an outdated summary, and two are distractors. See `assets/README.md` for what each is for.
"""),
    code('''
docs = w.load_documents()
print(f"{'id':20s} {'type':16s} {'date':12s} title")
for d in docs:
    m = d.metadata
    print(f"{m['id']:20s} {m['type']:16s} {str(m['date']):12s} {m['title']}")
'''),
    md("""
## 2. Metadata is part of the knowledge

The outdated summary is the clearest case. Its text reads as current ("the latest available figures are for 2015") and its figures were correct when it was written. The only thing that tells an application to rank it below the 2026 briefs is its date, which lives in the metadata.
"""),
    code('''
d = next(d for d in docs if d.metadata["id"] == "regional-2016")
for k in ("date", "type", "note"):
    print(f"{k:6s}", d.metadata[k])
print()
print(d.page_content[:500])
'''),
    md("""
## 3. Chunking: fixed windows against sections

A chunk is the unit that retrieval returns and the model reads. Two LangChain splitters on the same brief:

- `RecursiveCharacterTextSplitter`: windows of 400 characters with an overlap. Simple, and blind to the document's structure.
- `MarkdownHeaderTextSplitter`: one chunk per heading, with the heading kept in the text and the section name added to the metadata.
"""),
    code('''
kenya = [d for d in docs if d.metadata["id"] == "kenya-digital"]
fixed = w.split_fixed(kenya, chunk_size=400, chunk_overlap=50)
sections = w.split_sections(kenya)
print(f"fixed windows: {len(fixed)} chunks · sections: {len(sections)} chunks\\n")

print("A fixed window (note where it starts and ends):\\n")
print(repr(fixed[1].page_content))
print("\\nA section chunk:\\n")
print(sections[1].page_content)
print("\\nmetadata:", sections[1].metadata["section"], "|", sections[1].metadata["date"], "|", sections[1].metadata["country"])
'''),
    md("""
The fixed window starts and ends mid-sentence and carries no heading, so a passage about mobile subscriptions can be retrieved for a question about internet use without anything in the chunk saying which it is. The section chunk carries its title and heading, which both the retriever and the model can use.

## 4. Embeddings

An embedding is a vector that places a text so that texts with similar meaning are close. The session uses a small sentence-transformer model through LangChain's `HuggingFaceEmbeddings`; it downloads about 90 MB on first use. The cell embeds a few sections and a query and ranks the sections by similarity.
"""),
    code('''
import numpy as np

chunks = w.split_sections(docs)
embeddings = w.make_embeddings()

sample = [c for c in chunks if c.metadata["section"] in ("Internet use", "Mobile subscriptions", "Growth")][:6]
vectors = np.array(embeddings.embed_documents([c.page_content for c in sample]))
query = np.array(embeddings.embed_query("share of people using the internet"))
print(f"{len(chunks)} chunks · vectors of {vectors.shape[1]} dimensions\\n")
for c, v in sorted(zip(sample, vectors), key=lambda cv: -float(cv[1] @ query)):
    print(f"{float(v @ query):6.3f}  {c.metadata['id']:16s} {c.metadata['section']}")
'''),
    md("""
Sections about internet use score highest for a query about internet use, the mobile-subscription sections next, and growth last. The ranking depends on what the chunk text contains, which is why the splitter choice in section 3 matters here.

## 5. What poor preparation does

The same collection with the headers stripped, every document concatenated into one text, and fixed windows. Search for the latest Kenya figures and compare the top results with the section index.
"""),
    code('''
from langchain_core.documents import Document

blob = Document(page_content="\\n\\n".join(d.page_content for d in docs), metadata={"id": "blob", "date": "", "section": "(fixed window)"})
poor = w.build_store(w.split_fixed([blob], chunk_size=400, chunk_overlap=50), embeddings)
good = w.build_store(chunks, embeddings)

q = "What are the latest available figures for internet use in Kenya and Rwanda?"
print("Poorly prepared (one blob, fixed windows, no metadata):")
for d, s in poor.similarity_search_with_score(q, k=3):
    print(f"  {s:6.3f}  {d.page_content[:90]!r}")
print("\\nPrepared (sections with metadata):")
w.print_hits(good.similarity_search_with_score(q, k=3))
'''),
    md("""
In the poorly prepared index the top results are fragments that cannot be cited and carry no date. In the prepared index each hit names its document, section, and year, so the application can filter and the reader can check. The 2016 summary ranks high in the prepared index too; its date is what the next notebook uses to deal with it.

## 6. Build the vector store for the next notebook

Notebook 03 rebuilds this store with one line. `InMemoryVectorStore` is enough for a workshop; the same code runs against a persistent store such as Chroma, FAISS, or pgvector by changing the class.
"""),
    code('''
store = w.build_store(chunks, embeddings)
print(len(store.store), "chunks in the vector store")
'''),
    md("""
## Key points

- Metadata (date, type, country, section) is part of what the application knows. It decides what can be filtered and ranked, and what a citation can say.
- Chunk boundaries decide what retrieval can return. Sections keep meaning and headings; fixed windows cut through both.
- Embeddings place text by meaning; the quality of the index is the quality of the text and metadata it was built from.

## Exercise

Write a ninth document, a short brief on Uganda with figures looked up from the World Development Indicators, with the same header fields. Add it to `assets/documents/`, rerun this notebook, and check where its sections rank for the queries above.
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

Connect the vector store to the model: retrieve the passages that match the question and supply them as context. See two retrieval failures, their fixes, and why retrieval and generation are evaluated separately.

## The application after this notebook

```
question ──> retriever ──> top-k passages ──┐
                                             ├──> [ instructions + passages + question ] ──> model ──> answer with citations
question ────────────────────────────────────┘
```

The chain is written in LangChain's expression language: a retrieval step feeding a prompt feeding the model.
"""),
    code(SETUP),
    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

docs = w.load_documents()
chunks = w.split_sections(docs)
store = w.build_store(chunks)
print(len(chunks), "chunks in the store")

INSTRUCTIONS = """You are a research assistant for development data.
Answer only from the supplied passages. After each figure, cite the passage in the square-bracket form given.
If the passages do not contain what is needed, say so."""

prompt = ChatPromptTemplate.from_messages([("system", INSTRUCTIONS), ("human", "PASSAGES:\\n{context}\\n\\nQUESTION: {question}")])


def make_rag(k=4, where=None):
    """A RAG chain: question -> retrieved passages -> prompt -> model."""
    retrieve = RunnableLambda(lambda q: store.similarity_search(q, k=k, filter=where))
    return {"context": retrieve | w.format_context, "question": RunnablePassthrough()} | prompt | model


def hits(question, k=4, where=None):
    return store.similarity_search_with_score(question, k=k, filter=where)
'''),
    md("""
## 1. The worked example through the pipeline

Retrieve, then answer. The hits are printed first so that the evidence the model saw is visible.
"""),
    code('''
w.print_hits(hits(w.MAIN_QUESTION, k=4))
print()
w.use("nb03_rag_main")
w.show(make_rag(k=4).invoke(w.MAIN_QUESTION))
'''),
    md("""
## 2. A retrieval failure: a question with two parts

The question asks about internet access and economic development. One query embeds both, and the internet passages win every slot: the economy briefs are absent from the top four, and the model says that half of the question cannot be answered. The 2016 summary also ranks high, for a reason the next failure makes explicit.

The fix is on the retrieval side: ask the store two questions and merge the passages. (LangChain's `MultiQueryRetriever` does this with a model generating the sub-queries; here the two are written by hand so the mechanism is visible.)
"""),
    code('''
sub_queries = ["internet use in Kenya and Rwanda, recent figures", "GDP per capita and economic growth in Kenya and Rwanda, recent figures"]
merged = []
for q in sub_queries:
    for d in store.similarity_search(q, k=3):
        if d not in merged:
            merged.append(d)
w.print_hits(merged)
print()
w.use("nb03_rag_two_queries")
w.show((prompt | model).invoke({"context": w.format_context(merged), "question": w.MAIN_QUESTION}))
'''),
    md("""
## 3. A retrieval failure: outdated evidence

"What are the latest available figures for internet use in Kenya and Rwanda?" The 2016 summary contains the words "latest available figures" and ranks first. With two passages supplied, the Kenya figure comes from 2016 and the Rwanda figure from 2026, and the model reports both faithfully.
"""),
    code('''
q = "What are the latest available figures for internet use in Kenya and Rwanda?"
w.print_hits(hits(q, k=2))
print()
w.use("nb03_rag_outdated")
w.show(make_rag(k=2).invoke(q))
'''),
    md("""
The answer is faithful to its evidence and wrong for the reader: a 2015 figure presented as the latest. Nothing in the generation step can fix this; the evidence was selected badly.

## 4. The fix is on the retrieval side

Three options:

1. **A metadata filter**: exclude documents dated before 2020 for a question about the latest figures. `InMemoryVectorStore` takes a filter function over the document metadata.
2. **Query reformulation**: search for what the passage would say ("internet use 2024 percent of population"). It helps in many cases and fails in this one, as the cell shows: the 2016 summary still ranks first, because its wording is close to any query about figures.
3. **Re-ranking by date** after retrieval, when the store has no filter.

The filter is the reliable fix here, because the date is metadata the embedding does not see.
"""),
    code('''
since_2020 = lambda d: str(d.metadata.get("date", "")) >= "2020"
print("1. metadata filter (date >= 2020):")
w.print_hits(hits(q, k=2, where=since_2020))
print("\\n2. reformulated query:")
w.print_hits(hits("Kenya and Rwanda internet use 2024, percent of population", k=2))
print()
w.use("nb03_rag_filtered")
w.show(make_rag(k=2, where=since_2020).invoke(q))
'''),
    md("""
## 5. Retrieval always returns something

A question outside the collection still gets top-k passages, ranked by whatever is least unlike it. Compare a question the collection can answer with one it cannot.
"""),
    code('''
for q2 in ["What is the population of Rwanda?", "What was inflation in Uganda in 2023?"]:
    print(q2)
    w.print_hits(hits(q2, k=3))
    print()
'''),
    md("""
Both questions get three passages with a score. The out-of-collection question scores lower, and a threshold could separate the two here, but a threshold has to be set from examples on the embedding model in use, which is part of the evaluation in notebook 06. The second safeguard is the instruction that tells the model to say when the passages do not answer the question, which notebook 01 introduced. Applications use both.

## 6. Retrieval and generation are evaluated separately

Retrieval can be tested without calling a model: for each evaluation question, did the top-k passages include a passage from the expected document? The cell runs that test for all eight questions at two values of k.
"""),
    code('''
questions = w.load_questions()
print(f"{'id':4s} {'category':18s} {'k=2':6s} {'k=4':6s} expected documents")
for qd in questions:
    r2 = w.retrieval_hit(store.similarity_search(qd["question"], k=2), qd["expected_documents"])
    r4 = w.retrieval_hit(store.similarity_search(qd["question"], k=4), qd["expected_documents"])
    print(f"{qd['id']:4s} {qd['category']:18s} {'hit' if r2 else 'miss':6s} {'hit' if r4 else 'miss':6s} {', '.join(qd['expected_documents'])}")
'''),
    md("""
A miss here is a retrieval problem, fixed by chunking, metadata, ranking, or the query. A wrong answer with a hit is a generation problem, fixed by instructions or a different model. Mixing the two makes both impossible to improve.

## Key points

- Retrieval selects the evidence; the model writes from it. A faithful answer to the wrong evidence is wrong.
- Fixes for retrieval live in metadata, ranking, and query formulation. The model cannot repair a bad selection.
- Retrieval always returns something; a threshold set from examples and an instruction to say "no evidence" are the two safeguards.
- Test retrieval on its own, with expected documents per question.

## Exercise

Change `w.split_sections` to `w.split_fixed` in the first cell and rerun section 6. Count the misses. Then try k=8.
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

Give the model a tool that returns authoritative statistics, watch it call the tool, make its output a typed object that other software can read, and expose the same tool over the Model Context Protocol with FastMCP.

## The application after this notebook

```
question ──> model ──┬──> tool call: get_indicator(indicator, country, years) ──> data ──┐
                     │<──────────────────────────────────────────────────────────────────┘
                     └──> answer, or an object that matches a schema

the same tool, served over MCP ──> any MCP client (this notebook, a chatbot, a desktop assistant)
```

Passages are for definitions and context. Numbers come from a tool over the data.
"""),
    code(SETUP),
    md("""
## 1. A number from memory and a number from data
"""),
    code('''
w.use("nb04_from_memory")
w.show(model.invoke("What was Rwanda's GDP per capita in 2023?"))
print()
r = w.indicator_lookup("NY.GDP.PCAP.CD", "Rwanda", start=2023, end=2023)
print("Data:", r["name"], r["observations"], "·", r["source"])
'''),
    md("""
The memory answer is close and hedged, with no way for a reader to tell how close. The data answer carries the indicator, the year, the value, the source, and a `claim_id` that notebook 06 will use to verify it.

## 2. The tool and its schema

`get_indicator` is a Python function decorated with LangChain's `@tool`. The docstring becomes the description the model reads, and the type hints become the parameter schema. The model fills the parameters from the question.
"""),
    code('''
import json

print("name:       ", w.get_indicator.name)
print("description:", w.get_indicator.description)
print(json.dumps(w.get_indicator.args, indent=2))
print()
print("available indicators:")
for i in w.list_indicators():
    print(f"  {i['id']:20s} {i['name']}")
'''),
    md("""
## 3. The model calls the tool, step by step

`bind_tools` tells the model which tools exist. The model answers with a tool call; the application runs the tool, returns the result as a `ToolMessage`, and calls the model again. Notebook 05 lets LangGraph run this loop; here it is written out once so that the mechanics are visible.
"""),
    code('''
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

INSTRUCTIONS = """You are a research assistant for development data.
Use get_indicator for every statistic; do not quote figures from memory. Name the indicator and the source with each figure.
If the tool returns no observation for the year asked, say so and give the latest available year."""

model_with_tools = model.bind_tools([w.get_indicator])
messages = [SystemMessage(content=INSTRUCTIONS), HumanMessage(content="What was Rwanda's GDP per capita in 2023?")]

w.use("nb04_tool_call")
reply = model_with_tools.invoke(messages)
w.show(reply)
for call in reply.tool_calls:
    result = w.get_indicator.invoke(call["args"])
    print("tool result:", json.dumps(result)[:160], "…")
    messages += [reply, ToolMessage(content=json.dumps(result), tool_call_id=call["id"])]
print()
w.show(model_with_tools.invoke(messages))
'''),
    md("""
## 4. Data that does not exist

The planted problem: a year with no observation. The tool returns an empty list, and the instructions say what to do with it.
"""),
    code('''
messages = [SystemMessage(content=INSTRUCTIONS), HumanMessage(content="What share of Kenya's population used the internet in 2025?")]
w.use("nb04_missing_year")
reply = model_with_tools.invoke(messages)
for call in reply.tool_calls:
    result = w.get_indicator.invoke(call["args"])
    print("tool result for 2025:", result["observations"])
    messages += [reply, ToolMessage(content=json.dumps(result), tool_call_id=call["id"])]
print()
w.show(model_with_tools.invoke(messages))
'''),
    md("""
Compare with notebook 01, where the model answered the same question from a brief. Here the absence is a fact returned by the tool, which the application can test for.

## 5. Structured output

An answer that another program will read, store, or display needs a fixed shape. `ComparisonAnswer` is a Pydantic model; `with_structured_output` sends its schema with the request and validates the response on the way back.
"""),
    code('''
print(json.dumps(w.ComparisonAnswer.model_json_schema(), indent=2)[:1200], "…")
'''),
    code('''
data = [w.indicator_lookup(i, c, start=2024) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]
structured = model.with_structured_output(w.ComparisonAnswer)

w.use("nb04_structured")
parsed = structured.invoke([
    SystemMessage(content="Fill the schema from the data you are given. Use the claim_id values exactly as given."),
    HumanMessage(content="DATA:\\n" + json.dumps(data) + "\\n\\nQUESTION: " + w.MAIN_QUESTION),
])
for cs in parsed.countries:
    print(cs.country, "internet", cs.internet_latest.year, cs.internet_latest.value, "| GDP pc", cs.gdp_per_capita_latest.year, round(cs.gdp_per_capita_latest.value))
print()
print(parsed.comparison)
print(parsed.sources)
'''),
    md("""
Validation is what makes the schema a contract. The cell below damages the JSON and shows the error that a downstream program would receive. Without validation it would receive a wrong value and no error.
"""),
    code('''
from pydantic import ValidationError

damaged = parsed.model_dump_json().replace('"year":2024', '"year":"twenty twenty-four"', 1)
try:
    w.ComparisonAnswer.model_validate_json(damaged)
except ValidationError as e:
    print(e.errors()[0]["loc"], "->", e.errors()[0]["msg"])
'''),
    md("""
## 6. The same tool over MCP

The Model Context Protocol (MCP) is a standard way to expose tools to any client. `indicator_server.py` wraps `get_indicator` in a FastMCP server in a dozen lines; the FastMCP client starts it as a subprocess, lists its tools, and calls one. A chatbot, a desktop assistant, or an agent connects the same way.
"""),
    code('''
from fastmcp import Client

async with Client(w.mcp_server_transport()) as client:
    for t in await client.list_tools():
        print(f"{t.name:16s} {t.description[:90]}")
    result = await client.call_tool("get_indicator", {"indicator": "IT.NET.USER.ZS", "country": "Rwanda", "start": 2022})
    print()
    print(result.data)
'''),
    md("""
A LangChain tool can wrap the MCP call, so an agent uses the server like any other tool. The cell defines one and runs an agent over it (the agent itself is the subject of notebook 05).
"""),
    code('''
from langchain_core.tools import tool
from langchain.agents import create_agent


@tool("get_indicator")
async def get_indicator_over_mcp(indicator: str, country: str, start: int | None = None, end: int | None = None) -> dict:
    """Observations of a World Development Indicators series for one country, from the indicator MCP server."""
    async with Client(w.mcp_server_transport()) as client:
        args = {k: v for k, v in dict(indicator=indicator, country=country, start=start, end=end).items() if v is not None}
        return (await client.call_tool("get_indicator", args)).data


agent = create_agent(model, [get_indicator_over_mcp], system_prompt=INSTRUCTIONS)
w.use("nb04_mcp")
result = await agent.ainvoke({"messages": [HumanMessage(content="How has internet use in Rwanda changed since 2022?")]})
w.print_trace(result)
print()
w.show(w.final_text(result))
'''),
    md("""
The program's [Data360 MCP server](https://github.com/worldbank/data360-mcp) exposes search, metadata, and data tools over the World Bank's data platform in the same way, at a public endpoint. The cell lists its tools and runs one search; it is guarded because the session must not depend on the network.
"""),
    code('''
try:
    async with Client(w.DATA360_MCP_URL) as client:
        tools = await client.list_tools()
        print(len(tools), "tools, for example:", [t.name for t in tools][:5])
        found = await client.call_tool("data360_search_indicators", {"query": "individuals using the internet", "limit": 3})
        for ind in found.data["indicators"]:
            print(f"  {ind['idno']:28s} {ind['name']}")
except Exception as e:
    print("Data360 MCP call skipped:", type(e).__name__)
'''),
    md("""
## Key points

- Numbers come from a tool over authoritative data; the model's memory is for language.
- A tool is a function with a description the model can read. The model fills the parameters; the application runs the function and returns the result.
- Absence of data is a tool result the application can test for.
- A schema turns the answer into a contract: validated on the way in, readable by other software on the way out.
- MCP exposes the same tool to any client; FastMCP makes a server out of a function.

## Exercise

Add a `get_population` tool to `indicator_server.py` (series `SP.POP.TOTL`) and ask the agent "How many people in Rwanda used the internet in 2024?". The answer needs two tool calls and a calculation; check whether the model does the arithmetic or asks for a tool to do it.
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

Give the application two tools, document search and indicator data, and compare two ways of deciding which to call: a fixed workflow written as a chain, and a LangGraph agent in which the model chooses. See how to bound the agent, inspect its decisions, and handle a failed call.

## The application after this notebook

```
                 ┌──> search_documents(query) ──> passages ──┐
question ──> model                                           ├──> model ──> answer
                 └──> get_indicator(indicator, country) ──> data ──┘
                 (fixed order in a chain, or chosen by the model, with a recursion limit and a trace)
```
"""),
    code(SETUP),
    code('''
import json
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

docs = w.load_documents()
store = w.build_store(w.split_sections(docs))
search_documents = w.make_search_tool(store, k=3, since="2020")
tools = [search_documents, w.get_indicator]

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
compose = ChatPromptTemplate.from_messages([("system", INSTRUCTIONS), ("human", "PASSAGES:\\n{passages}\\n\\nDATA:\\n{data}\\n\\nQUESTION: {question}")]) | model


def fixed_workflow(question):
    passages = search_documents.invoke({"query": question})
    series = [w.indicator_lookup(i, c, start=2010) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]
    return compose.invoke({
        "passages": "\\n\\n".join(f"{p['citation']}\\n{p['text']}" for p in passages),
        "data": json.dumps(series),
        "question": question,
    })


w.use("nb05_compose")
w.show(fixed_workflow(w.MAIN_QUESTION))
'''),
    md("""
## 2. The model chooses the tools

`create_agent` builds a LangGraph agent: the model is offered the tools with their schemas, the graph runs what it asks for, feeds the results back, and stops when the model answers in text. The trace lists every call and result.
"""),
    code('''
agent = create_agent(model, tools, system_prompt=INSTRUCTIONS)

w.use("nb05_agent_main")
result = agent.invoke({"messages": [HumanMessage(content=w.MAIN_QUESTION)]}, config={"recursion_limit": 10})
w.print_trace(result)
print()
w.show(w.final_text(result))
'''),
    md("""
The model chose the same calls the fixed workflow made, in one step. On a different question it would choose differently, which is the point and the risk.

## 3. A question outside the tools

The planted out-of-scope question. A good agent searches, finds nothing usable, and says so.
"""),
    code('''
w.use("nb05_agent_offscope")
result = agent.invoke({"messages": [HumanMessage(content="How have Kenya's coffee export earnings changed in recent years?")]}, config={"recursion_limit": 10})
w.print_trace(result)
print()
w.show(w.final_text(result))
'''),
    md("""
## 4. A failed call

The model asks for an indicator id that does not exist. The tool returns an error object; the model reads it and retries with the right id. Returning errors as data is what lets the loop recover.
"""),
    code('''
w.use("nb05_agent_error")
result = agent.invoke({"messages": [HumanMessage(content="What share of Rwanda's population used the internet in 2023?")]}, config={"recursion_limit": 10})
w.print_trace(result)
print()
w.show(w.final_text(result))
'''),
    md("""
## 5. Bounding the loop

LangGraph's `recursion_limit` caps the number of graph steps. When a model keeps calling tools, the run stops with `GraphRecursionError`, and the application decides what to do then: answer with what it has, ask the user, or fail. The recorded scenario below is a model that never stops.
"""),
    code('''
from langgraph.errors import GraphRecursionError

w.use("nb05_runaway")
try:
    agent.invoke({"messages": [HumanMessage(content=w.MAIN_QUESTION)]}, config={"recursion_limit": 6})
except GraphRecursionError as e:
    print("stopped:", str(e)[:80])
'''),
    md("""
## 6. Choosing between the two

| | Fixed workflow | Agent |
|---|---|---|
| Calls per run | Known in advance | Varies with the question |
| Behaviour on a new kind of question | Same calls, possibly irrelevant | Adapts, possibly wrongly |
| Cost | Predictable | Bounded by the recursion limit |
| Testing | Test the composer | Test the trace and the answer |
| Use when | The questions have one shape | The questions vary and the tools are few and safe |

The worked example has one shape, so the fixed workflow is the right design for it. The agent earns its place when users ask questions the designer did not anticipate, and only with a recursion limit, a trace, and tools that fail as data.

## Key points

- A workflow written as a chain is predictable and testable; prefer it when the task has one shape.
- An agent adapts to the question; bound it with a recursion limit, keep the trace, and return tool errors as data.
- Both need the same two things from the tools: good descriptions and results the model can read.

## Exercise

Remove `w.get_indicator` from `tools` and rerun section 2. Read what the model does without a data tool and whether the instructions stop it from quoting figures from memory. In recorded mode, write down the expected behaviour; with a key, compare.
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

Add verification of every number in an answer, with the program's `pcn-core` package, and an evaluation suite that runs the application over a set of questions with known evidence. Compare two configurations of the application on the same suite.

## The checks

| Check | Question it answers | Where it runs |
|---|---|---|
| Retrieval | Did the application find the right evidence? | Notebook 03, section 6 |
| Grounding | Is the answer written from that evidence? | Citations in the answer |
| Verification | Does each number match its source record? | Claim tags and `pcn-core`, this notebook |
| Evaluation | Does the application behave across many questions? | The suite, this notebook |

## The application after this notebook

```
question ──> agent with tools ──> answer with <claim> tags ──> pcn-core ──> marks: verified · flagged · unmarked
                                                             evaluation suite ──> table per question
```
"""),
    code(SETUP),
    code('''
import json
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

docs = w.load_documents()
store = w.build_store(w.split_sections(docs))
search_documents = w.make_search_tool(store, k=3, since="2020")
tools = [search_documents, w.get_indicator]

INSTRUCTIONS_CLAIMS = """You are a research assistant for development data.
Use get_indicator for every statistic and search_documents for definitions and context. Do not quote figures from memory.
Write every number that comes from get_indicator inside a claim tag that names its claim_id, like
<claim id="IT.NET.USER.ZS:KEN:2024" policy="rounded" decimals="1">35.0</claim>.
Name the indicator and the source. If the tools do not provide what is needed, say so.
If the question is ambiguous, ask which reading is meant."""
'''),
    md("""
## 1. Claim tags

The instructions ask the model to wrap each figure in a tag that names the observation it came from. `w.verify` finds the tags and hands each number and its record to `pcn_core.compare_by_policy` with the policy from the tag: rounded to the stated decimals, or exact. This is the Proof-Carrying Numbers protocol used in the program's chatbot, with the same comparison code.
"""),
    code('''
results = [w.indicator_lookup(i, c, start=2010) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]
observations = w.observations_from_results(results)
print(len(observations), "observations available to the verifier, for example", list(observations)[:2])

w.use("nb06_claims_main")
reply = model.invoke([SystemMessage(content=INSTRUCTIONS_CLAIMS), HumanMessage(content="DATA:\\n" + json.dumps(results) + "\\n\\nQUESTION: " + w.MAIN_QUESTION)])
print("\\nAs the reader sees it:\\n")
w.show(w.strip_claims(reply.content))
print("\\nVerification:\\n")
w.print_checks(w.verify(reply.content, observations))
'''),
    md("""
Every tagged number matches its record. Years are not claims and are skipped.

## 2. A wrong number and a number without a claim

The planted problem. One tagged figure has been altered, and one figure has no tag because it came from nowhere in the data. The verifier flags the first and marks the second as unmarked: it carries no claim, so it receives no mark, whether or not it happens to be right.
"""),
    code('''
w.use("nb06_claims_error")
reply = model.invoke([SystemMessage(content=INSTRUCTIONS_CLAIMS), HumanMessage(content="DATA:\\n" + json.dumps(results) + "\\n\\nQUESTION: " + w.MAIN_QUESTION)])
w.show(w.strip_claims(reply.content))
print()
w.print_checks(w.verify(reply.content, observations))
'''),
    md("""
A reader of the chatbot sees a mark beside each number. A flagged number is one the model changed. An unmarked number is one the model did not tie to a record, and the fail-closed rule says it gets no mark.

What the check says nothing about: whether the record is the right one for the question, whether the method behind it is sound, or whether the sentence around the number interprets it correctly. Those stay with the reader.

## 3. Two configurations of the application

- **A, retrieval only**: the RAG chain of notebook 03. Passages as context, no tools, no tags.
- **B, tools and claims**: the agent of notebook 05 with the claim instructions, and the verifier.
"""),
    code('''
INSTRUCTIONS_RAG = """You are a research assistant for development data.
Answer only from the supplied passages and cite them in the square-bracket form given. If they do not contain what is needed, say so."""
rag_prompt = ChatPromptTemplate.from_messages([("system", INSTRUCTIONS_RAG), ("human", "PASSAGES:\\n{context}\\n\\nQUESTION: {question}")])
rag_chain = rag_prompt | model
agent = create_agent(model, tools, system_prompt=INSTRUCTIONS_CLAIMS)


def app_rag(question, scenario):
    hits = store.similarity_search(question, k=4)
    w.use(scenario)
    reply = rag_chain.invoke({"context": w.format_context(hits), "question": question})
    return reply.content, hits, {}


def app_tools(question, scenario):
    w.use(scenario)
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config={"recursion_limit": 10})
    obs = w.observations_from_results(w.tool_results(result, "get_indicator"))
    hits = store.similarity_search(question, k=4)  # for the retrieval column; the agent may or may not have searched
    return w.final_text(result), hits, obs
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
w.print_table(evaluate(app_rag, "nb06_rag"))
print("\\n\\nB · tools and claims\\n")
w.print_table(evaluate(app_tools, "nb06_tools"))
'''),
    md("""
## 5. Reading the table

- **q8, outdated evidence.** Configuration A answers from the 2016 summary when it ranks first and reports 2015 figures; B fetches the series and reports 2024. The expected numbers are found in B and missed in A.
- **q3, ambiguous.** Neither configuration should produce a number. B asks which reading is meant, as its instructions say; A picks one. The table shows 0/0 for both; the behaviour has to be read.
- **q4, missing data.** Both say that 2025 is unavailable; B's statement rests on an empty tool result.
- **q5, out of scope.** Both give no number. A's retrieval "hit" is the coffee note, which is the expected document and contains no figures.
- **verify.** In B every number from the data tool is verified; in A there is nothing to verify against, because no number came from a tool.

A table like this is what distinguishes a demonstration from a measured application. It is rerun after every change to the instructions, the tools, the store, or the model. For suites at scale, with a model as a judge for the behaviours a rule cannot check, see the program's evaluation cookbook; libraries such as Ragas and DeepEval provide those judges.

## 6. What the suite does not cover

- Whether the application chose the right indicator for the question. A verified number from the wrong series is verified.
- Whether the interpretation around the numbers is right.
- Behaviour on questions nobody wrote down. The suite grows from use: every failure found in use becomes a question.

## Key points

- Verification checks each number against its source record with `pcn-core` and fails closed: no claim, no mark.
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
        NotebookClient(nb, timeout=900, kernel_name=kernel, resources={"metadata": {"path": str(HERE)}}).execute()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.write(nb, path)
        print("executed", path.name)


if __name__ == "__main__":
    written = write_all()
    print("wrote", len(written), "notebooks")
    if "--execute" in sys.argv:
        execute(written)
