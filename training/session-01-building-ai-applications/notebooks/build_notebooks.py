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

import base64
import re

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

SETUP = '''# Setup. In Colab this clones the repository and installs the dependencies; locally it uses the folder as is.
import os, sys, subprocess, pathlib

REPO_URL = "https://github.com/worldbank/ai4data.git"   # the program's repository
BRANCH = "main"   # the branch the Colab setup clones
SESSION = "training/session-01-building-ai-applications/notebooks"

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

# Notebook 02 never calls the model, so its setup cell does not create or announce one.
SETUP_NO_MODEL = SETUP.split("# Optional: a key")[0] + 'import workshop as w\nprint("Ready.")'

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


def diagram_md(name: str) -> str:
    """The diagram drawn here, at build time, and embedded as an image so that the notebook needs no code to show it."""
    import diagrams

    svg = diagrams.render_svg(diagrams.flow(name))
    return f"![{diagrams.ALT[name]}](data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()})"


def explainer_md(name: str) -> str:
    """An animated SVG explainer, drawn here at build time and embedded as an image."""
    import explainers

    svg = {
        "chunking": lambda: explainers.chunking_svg()[0],
        "embeddings": explainers.embeddings_svg,
        "prompt": explainers.prompt_svg,
        "before_after": explainers.before_after_svg,
        "desk": explainers.desk_svg,
        "metadata": explainers.metadata_svg,
        "toolloop": explainers.toolloop_svg,
    }[name]()
    return f"![{explainers.ALT[name]}](data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()})"


def md(text: str):
    text = re.sub(r"@@DIAGRAM:(\w+)@@", lambda m: diagram_md(m.group(1)), text)
    text = re.sub(r"@@EXPLAINER:(\w+)@@", lambda m: explainer_md(m.group(1)), text)
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


CLAIMS_CELL = '''
def value(ind, country):
    return w.latest(w.indicator_lookup(ind, country, start=2023))["value"]


def growth(country):
    return [o["value"] for o in w.indicator_lookup("NY.GDP.MKTP.KD.ZG", country, start=2015)["observations"]]


rows = [  # (claim, what the model said, what the data says, verdict)
    ("Kenya · internet use", "“above 80 percent… when measured by subscriptions”",
     f"{value('IT.NET.USER.ZS', 'Kenya'):.1f}% of people online; {value('IT.CEL.SETS.P2', 'Kenya'):.1f} mobile subscriptions per 100", "blends two measures"),
    ("Kenya · GDP per capita", "“roughly 2,000 US dollars”", f"{value('NY.GDP.PCAP.CD', 'Kenya'):,.0f} US dollars", "close"),
    ("Kenya · growth", "“around 5 percent a year over the past decade”", f"{sum(growth('Kenya')) / len(growth('Kenya')):.1f} percent a year, 2015 to 2024", "close"),
    ("Rwanda · internet use", "“around 30 percent”", f"{value('IT.NET.USER.ZS', 'Rwanda'):.1f}% of people online", "close"),
    ("Rwanda · GDP per capita", "“close to 1,000 US dollars”", f"{value('NY.GDP.PCAP.CD', 'Rwanda'):,.0f} US dollars", "close"),
    ("Rwanda · growth", "“frequently above 7 percent”", f"above 7 percent in {sum(g > 7 for g in growth('Rwanda'))} of {len(growth('Rwanda'))} years, 2015 to 2024", "close"),
]
w.show_claim_check(rows)
'''

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

@@DIAGRAM:prompt@@

The prompt is the one piece this notebook builds; the rest of the application is added in the notebooks that follow.

The worked example for the whole session is one question:

> How have internet access and economic development evolved in Rwanda and Kenya? Compare the recent available data, summarize the main patterns, and cite the sources.

## What you will see

Four short scenes, each with a result you can read:

1. The model answers from what it learned in training. The answer sounds right, but it names no source, so a reader cannot check it.
2. The same question with four country briefs supplied. The answer is specific and cites its sources.
3. A question the briefs cannot answer, and what the model does about it.
4. A question with two meanings, and what changes when the instructions tell the model to ask.

## How to run it

Run the cells in order. Without an API key the model outputs are recorded ones, labelled `[recorded output: ...]`; each cell names its scenario with `w.use(...)` before calling the model. With a key in `WORKSHOP_API_KEY` the same cells call the model live.
"""),
    md("""
The next cell installs the session's tools and loads its helper module. It takes about a minute the first time, and you do not need to read it.
"""),
    code(SETUP),
    md(STACK),
    md("""
## 1. A prompt with three parts

Three parts go into every call:

- **Instructions**: the system message. It tells the model how to behave: its role, how to cite, what to do when information is missing.
- **Context**: the evidence supplied to the model, such as the documents or data it should answer from.
- **Question**: what the user asks.

LangChain's `ChatPromptTemplate` keeps them as separate slots, and `prompt | model` is the whole application for now. Keeping the parts separate is the first design decision, because each is changed by a different person for a different reason.

@@EXPLAINER:prompt@@

*The animation loops. Its text comes from this notebook: the instructions from the cell below, the context and the answer from section 3, and the question from the worked example. Section 6 shows the same three parts as the model received them.*
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
To judge that answer we need official figures to check it against. They come from the World Bank's World Development Indicators. The model never saw them; we use them only afterwards, to check its answer.

1. The first cell prints the official figures for Kenya and Rwanda: the share of people using the internet, and GDP per capita.
2. The second cell takes six claims from the model's answer and sets each one beside the matching official figure.
"""),
    code('''
rows = []
for ind in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"]:
    fmt = "{:,.0f}" if ind == "NY.GDP.PCAP.CD" else "{:.1f}"
    for country in ["Kenya", "Rwanda"]:
        r = w.indicator_lookup(ind, country, start=2023)
        by_year = {o["year"]: fmt.format(o["value"]) for o in r["observations"]}
        rows.append((r["name"], country, by_year[2023], by_year[2024]))
w.show_table(["Indicator", "Country", "2023", "2024"], rows, source="Source: World Bank, World Development Indicators.")
'''),
    code(CLAIMS_CELL),
    md("""
What to notice:

- Five of the six claims are close to the data, so the answer sounds reliable.
- One sentence blends two measures. "Above 80 percent" fits mobile subscriptions (126.5 per 100 people in 2024), not the share of people online (35.0 percent), and nothing in the answer tells the reader which one is meant.
- No claim names a source or a year, so a reader cannot tell the sound figures from the blended one. The model hedges, which is the right behaviour for a model with no evidence, and of no use to a reader who needs a number they can check.

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
@@EXPLAINER:before_after@@

*The two answers, shortened. They are the recorded outputs of sections 2 and 3.*
"""),
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

The messages the prompt produced for the last call, with each part in the colour it had in section 1: the instructions in the system message, then the context and the question in the human message.
"""),
    code('''
messages = prompt.format_messages(instructions=INSTRUCTIONS_CLARIFY, context=docs["kenya-digital"].page_content, question="How connected is Kenya?")
w.show_prompt(messages)
'''),
    md("""
## 7. What context costs

Four short briefs took about a thousand tokens. A natural reaction at this point:

> "So we can just give the AI all our documents as context!"

It is the first idea most people have, and it does not work, for three reasons:

- **It does not fit.** The context window, the amount of text a model can read in one call, has a limit. A real collection of thousands of documents is far larger.
- **It costs more with every page.** Cost grows with every token, and the whole collection would be sent again with every question.
- **The answer gets lost.** The passages that matter for one question are a small part of the whole, and the rest is noise around them.

The next two notebooks prepare a collection so that the application can select what to supply.

@@EXPLAINER:desk@@

*The token count is the one printed in section 3.*
"""),
    md("""
## Key points

- A prompt has three parts: instructions, context, and the question. Each is set by someone different, so they are kept separate.
- Without context, the model answers from training. Most figures were close, one blended two measures, and none named a source, so a reader could not tell which to trust.
- With context, the answer is specific and cites its sources. The model did none of the finding; the application did, by putting the right text in front of it.
- Instructions shape behaviour: cite sources, say so when information is missing, ask when a question has more than one meaning. They cannot supply data the model does not have.
- Giving the model everything does not work: the context window has a limit, cost grows with every token, and the answer gets lost. The application has to select what to supply, which the next notebooks cover.

## Exercise

1. Read the answers of sections 2 and 3 side by side. For each number, note whether you could check it and where. Then say which answer you would send to a colleague, and why.
2. In the cell of section 1, add a sentence to `INSTRUCTIONS_GROUNDED`, for example "Answer in no more than three sentences." Run that cell again, then run the cell of section 6. The Instructions block now shows your sentence, so you can see exactly what the model would receive. The recorded answers do not react to new instructions; with a live key in `WORKSHOP_API_KEY`, rerun section 3 to see how the answer changes.
"""),
    md(LINKS),
]

# =============================================================================
# 02
# =============================================================================
NB02 = [
    md("""
# 02 · Preparing knowledge for an application

**Session 01 · Building AI applications for research and data · notebook 2 of 6**

## Objective

Notebook 01 ended with a problem: an application cannot hand the model every document it has, so it has to select what to supply. Selecting needs a collection that has been prepared: documents together with information about them, cut into pieces, called chunks, that keep their meaning, and placed where a search can find them by meaning. This notebook builds that collection, and shows how the choices made here decide what retrieval can find later.

## The application after this notebook

@@DIAGRAM:knowledge@@

It shows the pipeline this notebook builds, and where notebook 03 picks it up.

## What you will see

Four short scenes:

1. The collection: eight documents, each with its text and information about it. One of them is ten years older than the rest, and its metadata records that.
2. The same brief cut two ways, by length and by heading.
3. Text turned into numbers that place it on a map of meaning.
4. The same collection prepared badly and prepared well, searched with the same question.

## How to run it

Run the cells in order. This notebook only prepares material for the model that writes the answers, so it does not use that model, and no API key is needed. The model comes back in notebook 03. Everything below is computed when you run the cell. It does use a small embedding model, which turns text into numbers; it downloads about 90 MB the first time and runs inside the notebook.
"""),
    md("""
The next cell installs the session's tools and loads its helper module. It takes about a minute the first time, and you do not need to read it.
"""),
    code(SETUP_NO_MODEL),
    md("""
The tools in this notebook are LangChain's: `Document`, `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, and `InMemoryVectorStore`.
"""),
    md("""
## 1. The collection

The collection is eight short documents. Each file has two parts: a header with information about the document (its id, title, type, country, date, and source), and the text itself. LangChain loads each file as a `Document`: the text is the `page_content`, and the header becomes the `metadata`.

Four are the country briefs the worked example needs, one defines the internet indicator, one is a regional summary from 2016, and two, on coffee and on mobile money, are about other topics and are there to distract. See `assets/README.md` for what each is for.
"""),
    code('''
docs = w.load_documents()
print(f"{'id':20s} {'type':16s} {'date':12s} title")
for d in docs:
    m = d.metadata
    print(f"{m['id']:20s} {m['type']:16s} {str(m['date']):12s} {m['title']}")
'''),
    md("""
## 2. Metadata: what the text cannot say about itself

Metadata is information about a document that is not in its text: when it was written, what kind of document it is, which country it covers, where its numbers come from.

@@EXPLAINER:metadata@@

*The cell below prints this document: some of its metadata (the date, the type, and a note), then its full text.*

The 2016 summary shows why that matters. Read on its own, its text sounds current: it ends with "The latest available figures are for 2015." Its figures were correct when it was written. A reader might notice the year in the title, but an application does not read for that: it can only filter or rank on the date, which is kept as metadata, and which says that the summary was written in 2016, ten years before the briefs.

The date does nothing by itself: an application has to use it. Section 5 shows what is lost when the metadata is dropped, and notebook 03 uses the date to keep the 2016 summary out of an answer about the latest figures.
"""),
    code('''
d = next(d for d in docs if d.metadata["id"] == "regional-2016")
for k in ("date", "type", "note"):
    print(f"{k:6s}", d.metadata[k])
print()
print(d.page_content)
'''),
    md("""
## 3. Chunking: fixed windows against sections

A chunk is the unit that is stored and searched: a search returns whole chunks. Two LangChain splitters on the same brief:

- `RecursiveCharacterTextSplitter`: windows of 400 characters with an overlap. Simple, and blind to the document's structure.
- `MarkdownHeaderTextSplitter`: one chunk per heading, with the heading kept in the text and the section name added to the metadata.

@@EXPLAINER:chunking@@

*The animation loops. It is drawn from the real brief and the two splitters. The cell below prints how many chunks each splitter made (4 windows and 4 sections) and shows two of them: window 2 of the fixed split, and the Mobile subscriptions section.*
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
Window 2 cuts by length, not by meaning. It starts in the middle of the Internet use section, carries no heading of its own, and ends with the heading of the next section, separated from the paragraph it introduces. A passage about internet use is then stored next to a mobile-subscriptions heading, and nothing in the chunk says which section it belongs to. The Mobile subscriptions section carries its title and heading, which both the retriever and the model can use.

## 4. Embeddings

An embedding turns a text into a list of numbers that places it on a map of meaning: texts with similar meaning land close together. The session uses a small embedding model through LangChain's `HuggingFaceEmbeddings`; it downloads about 90 MB on first use. The cell embeds a few sections and a question and ranks the sections by similarity.

@@EXPLAINER:embeddings@@

*The animation loops. The six chunks and their similarity scores are the ones the cell below prints, for the question "share of people using the internet"; the dot positions and the example numbers are illustrative.*
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
Sections about internet use score highest for a question about internet use, the mobile-subscription sections next, and growth last. The ranking depends on what the chunk text contains, which is why the splitter choice in section 3 matters here.

## 5. What poor preparation does

Two versions of the same collection, searched with the same question.

- **Prepared badly:** the metadata dropped, every document joined into one long text, and cut into fixed windows.
- **Prepared well:** what sections 1 to 4 built, with metadata and one chunk per section.

Each version is kept in a **vector store**: a place that holds the chunks together with their numbers, and can find the ones closest to a question. The question needs two figures, Kenya's latest and Rwanda's latest. The cell shows the top three chunks each search returns, and marks which of the two figures turn up in them.
"""),
    code('''
from langchain_core.documents import Document

blob = Document(page_content="\\n\\n".join(d.page_content for d in docs), metadata={"id": "blob", "date": "", "section": "(fixed window)"})
poor = w.build_store(w.split_fixed([blob], chunk_size=400, chunk_overlap=50), embeddings)
good = w.build_store(chunks, embeddings)

q = "What are the latest available figures for internet use in Kenya and Rwanda?"
expected = next(item for item in w.load_questions() if item["id"] == "q8")["expected_numbers"]
countries = {"KEN": "Kenya", "RWA": "Rwanda"}
checks = [f"{countries[e['country']]}, {e['year']}: {e['value']}" for e in expected]


def search(store, title, tone):
    hits = store.similarity_search(q, k=3)
    text = " ".join(h.page_content for h in hits)
    return {"title": title, "tone": tone, "hits": [(w.citation(h), h.page_content) for h in hits],
            "found": [w.numbers_present(text, [e])[0] == 1 for e in expected]}


w.show_search_comparison(q, [search(poor, "Prepared badly", "bad"), search(good, "Prepared well", "good")], checks)
'''),
    code('''
def numbers_found(store):
    found = total = 0
    for item in w.load_questions():
        text = " ".join(h.page_content for h in store.similarity_search(item["question"], k=3))
        for e in item.get("expected_numbers", []):
            found += w.numbers_present(text, [e])[0]
            total += 1
    return found, total


for name, store in (("Prepared badly", poor), ("Prepared well", good)):
    found, total = numbers_found(store)
    print(f"{name}: {found} of {total} needed numbers appear in the top 3 chunks, across the eight evaluation questions")
'''),
    md("""
Prepared badly, the search does not find Kenya's latest figure. Its top three chunks are the 2016 summary, Rwanda's internet paragraph, and a window that starts in the last section of Rwanda's brief and runs on into the title of the next document. Every chunk is labelled `blob`, so a reader cannot tell where any of it came from or how old it is. Prepared well, the same question finds both figures, and each chunk names its document, section, and year. The 2016 summary comes first in both, because a search matches meaning, not age; its date is what the next notebook uses to deal with it.

The second cell counts the same thing across all eight evaluation questions. The gap is smaller there, because for several questions both versions find what they need.

## 6. What a vector store holds

For each chunk, a vector store holds three things: its text, its metadata, and its numbers. The cell opens one record of the prepared collection, the Mobile subscriptions section of the Kenya brief.
"""),
    code('''
store = good   # the prepared collection of section 5
print(len(store.store), "chunks in the vector store. One of them:\\n")
record = next(r for r in store.store.values() if r["metadata"]["id"] == "kenya-digital" and r["metadata"]["section"] == "Mobile subscriptions")
print("text:    ", record["text"][:90].replace("\\n", " ") + " …")
print("metadata:", {k: record["metadata"][k] for k in ("id", "section", "country", "date")})
print("numbers: ", [round(x, 3) for x in record["vector"][:6]], "… (384 in all)")
'''),
    md("""
This store lives in memory and disappears when the notebook closes, which is enough here. A real application keeps its store in a database, so that it lasts and can be shared with others. Notebook 03 builds the same store and searches it to answer questions.
"""),
    md("""
## Key points

- Metadata is information about a document that its text does not say: date, type, country, source. The 2016 summary sounds current; its date, kept as metadata, is how an application can tell that it is old. An application can use the date only if it kept it.
- Chunk boundaries decide what retrieval can return. Fixed windows cut by length and can separate a heading from its paragraph; section chunks keep the heading with its meaning.
- Embeddings place text by meaning, so a question lands near the chunks that answer it. They know nothing about dates.
- A collection prepared badly returns chunks that cannot be cited or dated, and can miss the figures a question needs. Prepared well, each hit names its document, section, and year, and the figures turn up.
- Preparation decides what retrieval can find. The next notebook uses the collection to answer questions, and shows where it still goes wrong.

## Exercise

1. Look at the table in section 1. If someone asked for the latest figures on internet use, which document would you be wary of, and which two are about something else? Say what in the table told you.
2. In the cell of section 3, change `chunk_size=400` to `200` and run it again. How many fixed windows are there now, and where do they cut? Try `800`. The section chunks do not change, because they follow the headings.
"""),
    md(LINKS),
]

# =============================================================================
# 03
# =============================================================================
NB03 = [
    md("""
# 03 · Retrieval-augmented generation

**Session 01 · Building AI applications for research and data · notebook 3 of 6**

## Objective

Notebook 02 built the vector store and left two things open: a search for the latest figures puts the 2016 summary first, and nothing yet uses the date. This notebook puts the store to work. The application searches it for the chunks closest to a question, gives them to the model as context, and the model answers from them. A chunk that the search returns and the model reads is called a passage.

This pattern has a name: **retrieval-augmented generation**, or **RAG**. *Retrieval* is the search of the store, *augmented* means the search results are added to what the model is given, and *generation* is the model writing the answer. RAG has two halves: notebook 02 prepared the collection, and this notebook uses it to answer. The rest of the series builds on it. This notebook builds the second half, then shows three ways it goes wrong, and what fixes each.

## The application after this notebook

@@DIAGRAM:retrieval@@

Retrieval (the search) and the passages it returns are what this notebook adds, and the model's step becomes generation: together, the RAG pattern.

## What you will see

Four short scenes:

1. The question from notebook 01 answered from the store. The answer cites its passages, but half of the question goes unanswered.
2. Why the economy briefs never reached the model, and a fix that breaks the question into its parts and searches once for each.
3. The 2016 summary outranking the 2026 briefs, and the date from notebook 02 removing it.
4. A question the collection cannot answer still gets passages, and how to test a search without calling the model.

## How to run it

Run the cells in order. The model's answers are recorded ones, labelled `[recorded output: ...]`, as in notebook 01. The searches, rankings and tables are computed when you run the cell.
"""),
    md("""
The next cell installs the session's tools and loads its helper module. It takes about a minute the first time, and you do not need to read it.
"""),
    code(SETUP),
    md("""
The next cell builds the same store as notebook 02, in three lines. Each Colab notebook starts with an empty session, so the store from notebook 02 is not there to reuse; the three lines rebuild it. The cell also sets up the application: the instructions, a prompt, and two functions, one that searches the store and asks the model, and one that only searches. `k` is the number of passages a search returns; the model reads only those.
"""),
    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

docs = w.load_documents()
chunks = w.split_sections(docs)
store = w.build_store(chunks)
print(len(chunks), "chunks in the store, the same collection as in notebook 02")

INSTRUCTIONS = """You are a research assistant for development data.
Answer only from the supplied passages. After each figure, cite the passage in the square-bracket form given.
If the passages do not contain what is needed, say so."""

prompt = ChatPromptTemplate.from_messages([("system", INSTRUCTIONS), ("human", "PASSAGES:\\n{context}\\n\\nQUESTION: {question}")])


def make_rag(k=4, where=None):
    """RAG (retrieval-augmented generation): search the store, put the passages in the prompt, ask the model."""
    retrieve = RunnableLambda(lambda q: store.similarity_search(q, k=k, filter=where))
    return {"context": retrieve | w.format_context, "question": RunnablePassthrough()} | prompt | model


def hits(question, k=4, where=None):
    return store.similarity_search_with_score(question, k=k, filter=where)
'''),
    md("""
## 1. The question from notebook 01, answered from the store

The question is the main one from notebook 01. The search returns the four chunks closest to it, and they become the passages the model reads. The cell prints the question, then the passages, so that the evidence the model saw is visible, then the model's answer.
"""),
    code('''
w.show("Question: " + w.MAIN_QUESTION)
print()
w.print_hits(hits(w.MAIN_QUESTION, k=4))
print()
w.use("nb03_rag_main")
w.show(make_rag(k=4).invoke(w.MAIN_QUESTION))
'''),
    md("""
The answer cites its passages, and it is honest: it says the passages hold no figures on GDP or growth, so half of the question cannot be answered. The briefs with those figures are in the store. Why did they not come back?

## 2. Why half the question was lost

The question asks about internet access and economic development. The search compares the whole question with every chunk, and keeps the four closest. The cell ranks the chunks for the question, coloured by topic, with a line after the first four, the ones that go to the model.
"""),
    code('''
topic = {"kenya-digital": "internet", "rwanda-digital": "internet", "regional-2016": "internet", "internet-indicator": "internet",
         "kenya-economy": "economy", "rwanda-economy": "economy", "coffee": "other", "mobile-money": "other"}
legend = {"internet": ("internet use", "#2f5f9e"), "economy": ("economy", "#d9822b"), "other": ("other topics", "#8a94a3")}

ranked = hits(w.MAIN_QUESTION, k=10)
w.show_ranking([(w.citation(d), score, topic[d.metadata["id"]]) for d, score in ranked], k=4, legend=legend, question=w.MAIN_QUESTION)
'''),
    md("""
The top four are three internet chunks, one of them the 2016 summary, and the note on mobile money. The first economy chunk comes fifth, just below the line (0.570 against 0.591), and the one with Kenya's GDP per capita comes ninth. The search does not know that the question has two halves: it ranks chunks by their closeness to the whole question. Raising the number of passages is a poor fix: Kenya's GDP per capita appears only at nine passages, together with the coffee note and others, and every passage costs, as notebook 01 showed.

The fix is on the search side: **break the question down into its parts, search once for each part, and merge the passages.** This question has two parts, internet use and the economy, so it becomes two smaller questions. Libraries can have a model write these smaller questions; here the two are written by hand so that the mechanism is visible. The cell shows the original search and the merged one side by side, and marks which of the four figures the question needs each of them brings back. The fix is not free: two searches of four chunks each give eight different passages, against four before, and the 2016 summary and the mobile-money note come along with them. The model ignores both here, but section 3 shows what happens when it does not.
"""),
    code('''
# the question broken into its two parts: one smaller question for each
sub_queries = ["internet use in Kenya and Rwanda, recent figures", "GDP per capita and economic growth in Kenya and Rwanda, recent figures"]
merged = []
for q in sub_queries:
    for d in store.similarity_search(q, k=4):
        if d not in merged:
            merged.append(d)

one_search = [d for d, _ in hits(w.MAIN_QUESTION, k=4)]
needed = next(item for item in w.load_questions() if item["id"] == "q1")["expected_numbers"]
countries = {"KEN": "Kenya", "RWA": "Rwanda"}
names = {"IT.NET.USER.ZS": "internet use", "NY.GDP.PCAP.CD": "GDP per capita"}
checks = [f"{countries[n['country']]} {names[n['indicator']]}, {n['year']}: {n['value']:,}" for n in needed]


def column(title, tone, found):
    text = " ".join(d.page_content for d in found)
    return {"title": title, "tone": tone, "hits": [(w.citation(d), d.page_content) for d in found],
            "found": [w.numbers_present(text, [n])[0] == 1 for n in needed]}


w.show_search_comparison(w.MAIN_QUESTION, [column("One search for the whole question: 4 passages", "bad", one_search),
                                           column(f"The question broken into two, one search each, merged: {len(merged)} passages", "good", merged)], checks)
print()
w.use("nb03_rag_two_queries")
w.show((prompt | model).invoke({"context": w.format_context(merged), "question": w.MAIN_QUESTION}))
'''),
    md("""
The merged passages bring back all four figures, where the single search brought two, and the answer covers both halves of the question.

## 3. The 2016 summary outranks the briefs

Notebook 02 ended with the 2016 summary ranking first. Here is what that costs. The question is "What are the latest available figures for internet use in Kenya and Rwanda?", and the 2016 summary contains the words "latest available figures". With two passages supplied, the Kenya figure comes from the 2016 summary (a 2015 value) and the Rwanda figure from a 2026 brief (a 2024 value), and the model reports both faithfully.
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

## 4. Three ideas for fixing outdated evidence

The fix is on the search side. Three ideas:

1. **Filter on the date.** Tell the search to ignore documents dated before 2020. Notebook 02 kept each document's date as metadata, for this moment.
2. **Reword the question** to look like the passage it should find, for example "internet use 2024, percent of population". It often helps, but not here.
3. **Sort the results by date** after the search, when the store has no filter. It does the same job as the filter; it is not shown here.

The notebook tries the first two. The first cell compares the same search without and with the date filter, and marks which of the two figures the question needs turn up.
"""),
    code('''
since_2020 = lambda d: str(d.metadata.get("date", "")) >= "2020"
expected = next(item for item in w.load_questions() if item["id"] == "q8")["expected_numbers"]
countries = {"KEN": "Kenya", "RWA": "Rwanda"}
checks = [f"{countries[e['country']]}, {e['year']}: {e['value']}" for e in expected]


def search(title, tone, where=None):
    found = store.similarity_search(q, k=2, filter=where)
    text = " ".join(h.page_content for h in found)
    return {"title": title, "tone": tone, "hits": [(w.citation(h), h.page_content) for h in found],
            "found": [w.numbers_present(text, [e])[0] == 1 for e in expected]}


w.show_search_comparison(q, [search("No filter", "bad"), search("Date filter: 2020 or later", "good", since_2020)], checks)
'''),
    md("""
With the filter, the 2016 summary is gone and both figures come back. The model's answer from the filtered passages:
"""),
    code('''
w.use("nb03_rag_filtered")
w.show(make_rag(k=2, where=since_2020).invoke(q))
'''),
    code('''
w.print_hits(hits("Kenya and Rwanda internet use 2024, percent of population", k=2))
'''),
    md("""
## 5. A question outside the collection

A search always returns the closest chunks, even when none of them is close. Compare a question the collection can answer with one it cannot. The score shows how close a passage is to the question: higher is closer.
"""),
    code('''
for q2 in ["What is the population of Rwanda?", "What was inflation in Uganda in 2023?"]:
    print(q2)
    w.print_hits(hits(q2, k=3))
    print()
'''),
    md("""
Both questions get three passages. The scores are lower for the question outside the collection, and a cut-off score could separate the two here, but a cut-off has to be set from examples on the embedding model in use. The second safeguard is the instruction from notebook 01 that tells the model to say when the passages do not answer the question; notebook 05 shows it at work. Applications use both.

## 6. Testing the search without the model

A search can be tested before any model is involved. This is the retrieval half of RAG on its own, and the model is the generation half. For each of the eight evaluation questions, the cell makes two checks: did the passages include an expected document (one of them is enough), and do they contain the numbers an answer needs? `K` is the number of passages the search returns.
"""),
    code('''
K = 4   # passages returned by the search: try 2, then 8

rows = []
for item in w.load_questions():
    passages = store.similarity_search(item["question"], k=K)
    text = " ".join(p.page_content for p in passages)
    needed = item.get("expected_numbers", [])
    found = sum(w.numbers_present(text, [n])[0] for n in needed)
    rows.append((item["id"], item["category"],
                 "hit" if w.retrieval_hit(passages, item["expected_documents"]) else "miss",
                 f"{found}/{len(needed)}" if needed else "none needed"))
w.show_table(["id", "category", "expected document", "needed numbers"], rows, source=f"The search returns {K} passages per question.")
'''),
    md("""
Question q1, the main question, passes the first check, though only the two internet documents of its four expected ones came back, and fails the second at `K = 4`: documents of the right kind came back, but only two of the four numbers the answer needs. That is a retrieval problem, and no instruction to the model can fix it. When both checks pass and the answer is still wrong, the fault is in the generation: the instructions or the model, which notebook 06 measures. Keeping the two apart is what makes each one possible to improve.
"""),
    md("""
## Key points

- Retrieval-augmented generation (RAG), introduced at the top: search the store for the chunks closest to the question, give them to the model as passages, and have it answer from them, citing each.
- What the search returns decides what the answer can say. For the main question the economy chunks ranked just below the line and never reached the model; breaking the question into its parts and searching once for each fixed it.
- A search matches meaning, not age. The 2016 summary outranked the 2026 briefs, and rewording the question did not help; the date kept as metadata, used as a filter, did.
- A search always returns something, even for a question the collection cannot answer. Two safeguards: a cut-off score set from examples, and an instruction telling the model to say when the passages do not answer.
- A search can be tested without the model: check that the right documents and the needed numbers are among the passages. A miss there cannot be fixed by changing the instructions.

## Exercise

1. Look at the ranking in section 2. Which chunk takes fourth place, and is it a good use of one of the four slots?
2. In the cell of section 6, set `K` to `2`, run it, then set it to `8` and run it again. Which questions lose numbers at 2, and which gain at 8? At 8, does every question find all the numbers it needs?
3. Set `K` to `2` and look at q8, the question from section 3. Why does it pass at 4 and not at 2?
"""),
    md(LINKS),
]

# =============================================================================
# 04
# =============================================================================
NB04 = [
    md("""
# 04 · Tools and structured outputs

**Session 01 · Building AI applications for research and data · notebook 4 of 6**

## Objective

Notebook 03 ended on a limit of search. Even when the right documents come back, the passages may not hold the figures an answer needs: for the main question, the four passages held two of the four numbers. For numbers, the application can skip the search and ask a tool instead, which looks the figure up in the data itself. This notebook gives the model such a tool, watches it use the tool, makes its answer a structured output that other software can read, and then offers the same tool to other programs.

The application so far keeps documents in a store, searches the store, and hands the model the passages it finds (notebook 03). That stays, for definitions and context. This notebook adds a second source of evidence, a tool for exact numbers, and a second kind of answer.

Four terms, in plain words:

- A **tool** is a function that the application can run for the model. The model cannot run it itself: it asks, the application runs it, and the result goes back to the model.
- A **structured output** is an answer returned in a fixed shape, with named fields, instead of free text, so that another program can read it without guessing.
- A **schema** is the description of that shape: the named fields and their types ("year: a whole number") that the answer must fill in.
- **MCP**, the Model Context Protocol, is a standard plug for tools. A tool offered through it can be used by any program that speaks the standard, such as a chatbot or a desktop assistant.

## The application after this notebook

@@DIAGRAM:tools@@

The light boxes are what notebooks 01 to 03 built. The dark boxes are what this notebook adds: the tool call and the data it returns, the structured output, and the same tool served over MCP. Passages are for definitions and context. Numbers come from a tool over the data.

## What you will see

Five short scenes:

1. The four figures that the main question needs: what the search of notebook 03 found, and what the tool returns.
2. The tool as the model sees it, and an animation of the model asking for it, step by step.
3. A year that has no data: the tool says so, and the model passes it on.
4. A structured output: a program that misreads a sentence, then the same answer in a blank form that the model fills in, and what happens when a box is damaged.
5. The same tool offered over MCP and used by an agent, and a call to the program's public server.

## How to run it

Run the cells in order. The model's answers are recorded ones, labelled `[recorded output: ...]`, as in notebook 01. The tool results, tables and checks are computed when you run the cell. The last cell reaches a public server and needs the internet.
"""),
    code(SETUP),
    md("""
## 1. Search and tool, for the same four figures

Section 6 of notebook 03 found that the main question needs four numbers: internet use and GDP per capita, for Kenya and Rwanda, in 2024. The cell rebuilds the store of notebook 03 in three lines, runs the same search for the four closest passages, and checks for each number whether it is in them. Beside it, the tool's answer for the same figure.
"""),
    code('''
docs = w.load_documents()
store = w.build_store(w.split_sections(docs))   # the store of notebooks 02 and 03

needed = next(item for item in w.load_questions() if item["id"] == "q1")["expected_numbers"]
passages = store.similarity_search(w.MAIN_QUESTION, k=4)
text = " ".join(p.page_content for p in passages)

countries = {"KEN": "Kenya", "RWA": "Rwanda"}
figures = {"IT.NET.USER.ZS": ("internet use, % of population", "{:.1f}"), "NY.GDP.PCAP.CD": ("GDP per capita, US dollars", "{:,.0f}")}

rows = []
for n in needed:
    name, fmt = figures[n["indicator"]]
    result = w.indicator_lookup(n["indicator"], countries[n["country"]], start=n["year"], end=n["year"])
    rows.append((countries[n["country"]], name, n["year"],
                 "hit" if w.numbers_present(text, [n])[0] == 1 else "miss",
                 fmt.format(w.latest(result)["value"])))
w.show_table(["country", "figure", "year", "in the 4 passages", "from the tool"], rows, source="The search is the one of notebook 03, section 1.")
'''),
    md("""
The search found two of the four. The tool returned all four, each with its indicator, year and source. A tool does not guess which passage might hold a figure: the application asks for the figure by name.

The model's own memory is a third way to get a number, and the weakest. The cell asks the model for Rwanda's GDP per capita in 2023, then asks the tool.
"""),
    code('''
w.use("nb04_from_memory")
w.show(model.invoke("What was Rwanda's GDP per capita in 2023?"))
print()
r = w.indicator_lookup("NY.GDP.PCAP.CD", "Rwanda", start=2023, end=2023)
w.show_table(["year", "value"], [(o["year"], f"{o['value']:,.1f}") for o in r["observations"]], source=f"{r['name']} · {r['source']}")
'''),
    md("""
The answer from memory is close and hedged, and nothing tells a reader how close. The tool's answer is exact and carries its source. The application can now get an exact number, with its source, instead of hoping that a passage holds it.

## 2. The tool, as the model sees it

`get_indicator` is a Python function decorated with LangChain's `@tool`. Its docstring becomes the description the model reads, and its type hints become the form the model fills in: one field for each parameter. The cell shows both, and the five indicators the data holds.
"""),
    code('''
print("name:       ", w.get_indicator.name)
print("description:", w.get_indicator.description)
print()

meaning = {"indicator": "which series, as an id such as IT.NET.USER.ZS", "country": "a name or a three-letter code",
           "start": "the first year wanted", "end": "the last year wanted"}
kinds = {"string": "text", "integer": "whole number"}
fields = []
for field, spec in w.get_indicator.args.items():
    kind = spec.get("type") or " or ".join(a["type"] for a in spec["anyOf"] if a["type"] != "null")
    fields.append((field, kinds.get(kind, kind), "optional" if "default" in spec else "required", meaning[field]))
w.show_table(["field", "type", "", "meaning"], fields, source="The form the model fills in, from the tool's type hints.")
print()
w.show_table(["indicator id", "name"], [(i["id"], i["name"]) for i in w.list_indicators()], source="The series the tool can look up.")
'''),
    md("""
## 3. The model asks, the application runs the tool

A model does not run tools. It replies with a request that names the tool and fills in the form, the application runs the tool, and the result goes back to the model, which then answers. The animation shows the exchange.

@@EXPLAINER:toolloop@@

`bind_tools` tells the model which tools exist. The cell does the exchange once, written out, and labels the three steps. Notebook 05 lets LangGraph run this loop.
"""),
    code('''
import json
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

INSTRUCTIONS = """You are a research assistant for development data.
Use get_indicator for every statistic; do not quote figures from memory. Name the indicator and the source with each figure.
If the tool returns no observation for the year asked, say so and give the latest available year."""

model_with_tools = model.bind_tools([w.get_indicator])
messages = [SystemMessage(content=INSTRUCTIONS), HumanMessage(content="What was Rwanda's GDP per capita in 2023?")]

w.use("nb04_tool_call")
print("1. The model asks for the tool:")
reply = model_with_tools.invoke(messages)
w.show(reply)
for call in reply.tool_calls:
    result = w.get_indicator.invoke(call["args"])
    print()
    print("2. The application runs it, and the tool returns:")
    w.show_table(["year", "value"], [(o["year"], f"{o['value']:,.1f}") for o in result["observations"]], source=f"{result['name']} · {result['source']}")
    messages += [reply, ToolMessage(content=json.dumps(result), tool_call_id=call["id"])]
print()
print("3. The model writes its answer from the result:")
final = model_with_tools.invoke(messages)
w.show(final)
'''),
    md("""
The application can now use a tool: the model asks, the application runs it, and the result goes back to the model.

## 4. Data that does not exist

The planted problem: a year with no observation. The tool returns an empty list, and the instructions say what to do with it. The cell prints what the tool returned, then the model's answer.
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
## 5. Structured output: an answer in a fixed shape

So far the model's answers are sentences, which suits a person. Suppose instead that a dashboard has a box labelled "GDP per capita, Rwanda" and must fill it from the model's answer. The cell does what a simple program might do: it takes the first number in the sentence from section 3.
"""),
    code('''
import re

first = re.findall(r"\\d[\\d,]*\\.?\\d*", final.content)[0]
print("The answer:           ", final.content)
print("The dashboard's box:  ", first)
'''),
    md("""
The box gets 2023, which is the year and not a figure. The sentence holds three figures for different years, and the words around them change from one answer to the next, so a program reading a sentence has to guess which one is wanted.

The fix is to stop asking for a sentence. We hand the model a **blank form** with a named box for each thing we need: a country, a year and a value for internet use, a year and a value for GDP per capita, a comparison, the sources. The model's job is to fill in the boxes, and the program reads each box by name. The blank form is called a **schema**, and an answer that follows one is a **structured output**. In code the form is a class written with a library called Pydantic, and `with_structured_output` sends it to the model with the request. The next cell gives the model the tool's data, asks it to fill in the form, and draws the blank form beside the filled one.
"""),
    code('''
data = [w.indicator_lookup(i, c, start=2024) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]
structured = model.with_structured_output(w.ComparisonAnswer)

w.use("nb04_structured")
parsed = structured.invoke([
    SystemMessage(content="Fill the schema from the data you are given. Use the values exactly as given."),
    HumanMessage(content="DATA:\\n" + json.dumps(data) + "\\n\\nQUESTION: " + w.MAIN_QUESTION),
])
w.show_form(w.ComparisonAnswer, parsed)
'''),
    md("""
Now the dashboard does not guess. It reads the box it needs by its name, and gets the figure and its year.
"""),
    code('''
box = next(cs for cs in parsed.countries if cs.country == "RWA").gdp_per_capita_latest
print("GDP per capita, Rwanda:", f"{box.value:,.0f} US dollars, in {box.year}")
'''),
    md("""
A form also has rules, and the program checks the answer against them before using it. A box that should hold a year cannot hold words. The cell damages the answer by putting words in a year box, and shows the message the program would receive, which names the exact box. Without the check, a wrong value would pass through with no error.
"""),
    code('''
from pydantic import ValidationError

damaged = parsed.model_dump_json().replace('"year":2024', '"year":"twenty twenty-four"', 1)
try:
    w.ComparisonAnswer.model_validate_json(damaged)
except ValidationError as e:
    box = " > ".join(str(part) for part in e.errors()[0]["loc"])
    print("box with the problem:", box)
    print("what is wrong:       ", e.errors()[0]["msg"])
'''),
    md("""
The application can now return an answer that other software reads box by box, and reject one that breaks the form.

## 6. The same tool over MCP

MCP is the standard plug: a tool written once as a server can be used by any program that speaks the standard, whether a chatbot, a desktop assistant or an agent. `indicator_server.py` wraps `get_indicator` in a FastMCP server in a dozen lines. The FastMCP client below starts it, lists its tools and calls one.
"""),
    code('''
from fastmcp import Client

async with Client(w.mcp_server_transport()) as client:
    for t in await client.list_tools():
        print(f"{t.name:16s} {t.description[:90]}")
    result = await client.call_tool("get_indicator", {"indicator": "IT.NET.USER.ZS", "country": "Rwanda", "start": 2022})
    data = result.data
    print()
    w.show_table(["year", "value"], [(o["year"], f"{o['value']:,.1f}") for o in data["observations"]], source=f"{data['name']} · {data['source']}")
'''),
    md("""
A LangChain tool can wrap the MCP call, so an agent uses the server like any other tool. The cell defines one and runs an agent over it. An agent is a model that chooses its own tool calls; notebook 05 is about agents.
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
The application can now offer its tool to any program that speaks MCP.

The program's [Data360 MCP server](https://github.com/worldbank/data360-mcp) offers search, metadata and data tools over the World Bank's data platform in the same way, at a public address. The cell lists its tools and runs one search. It reaches a live server, so the list can differ from run to run, and if the call fails the cell says so and the notebook carries on.
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

- Numbers come from a tool over authoritative data, and the model's memory is for language. In this notebook the search found two of the four figures the main question needs, and the tool returned all four.
- A tool is a function with a description the model can read and a form the model fills in. The model does not run it: the application does, and hands the result back.
- Absence of data is a tool result that the application can test for, instead of a gap the model fills with a guess.
- A structured output is an answer in a fixed shape: the model fills in named fields instead of writing free text, which a program would have to guess at. The schema is the contract: checked on the way in, readable by other software on the way out.
- MCP is the standard plug: the same tool, written once as a server, is available to any program that speaks it.

## Exercise

1. In section 1, change `k=4` to `k=8` and run the cell. How many of the four figures does the search find now, and what does the extra search cost in passages? Does the tool's column change?
2. The data holds population too, series `SP.POP.TOTL`. Run `w.indicator_lookup("SP.POP.TOTL", "Rwanda", start=2024, end=2024)` and `w.indicator_lookup("IT.NET.USER.ZS", "Rwanda", start=2024, end=2024)`, and work out how many people in Rwanda used the internet in 2024. An answer like that needs two tool results and a calculation: should the model do the arithmetic, or should the application?
3. Run `w.indicator_lookup("IT.NET.USER.ZS", "Kenya", start=2024, end=2025)`. The tool returns the year that exists and says nothing about the one that does not. Why does the instruction "if the tool returns no observation for the year asked, say so" matter for a result like this?
4. In section 5, damage a different field of the answer, for example replace a value with the word "high", and read the error. Which field does it name?
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

@@DIAGRAM:agent@@

The model chooses between the two tools, in a fixed order in a chain or by its own choice, with a recursion limit and a trace.
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

@@DIAGRAM:verification@@

The verifier and the evaluation suite are what this notebook adds.
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
