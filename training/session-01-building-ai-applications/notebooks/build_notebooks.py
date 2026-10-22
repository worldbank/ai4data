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
| Number verification | [`pcn-core`](https://github.com/worldbank/pcn), the Python core of the Proof-Carrying Numbers protocol |
| Output schemas | [Pydantic](https://docs.pydantic.dev/) |

`workshop.py` holds what these do not: the worked-example data, a chat model that replays recorded outputs when no API key is configured, and a few printing and evaluation helpers."""

LINKS = """## Where this connects to the program

- [Practical Guide to Serving Official Statistics to AI Agents](https://worldbank.github.io/ai4data/cookbook/serving-statistics-to-agents/): the design of tools over a statistical catalog, in full.
- [Practical Guide to Evaluation Suites for Statistical AI](https://worldbank.github.io/ai4data/cookbook/evaluation-suites/): how to build and run the tests.
- [Proof-Carrying Numbers](https://worldbank.github.io/ai4data/docs/pcn/): the claim-tag protocol the verifier in this session follows. See also the [paper](https://arxiv.org/abs/2509.06902) and the [`pcn` repository](https://github.com/worldbank/pcn).
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
        "workflows": explainers.workflows_svg,
        "claimtag": explainers.claimtag_svg,
        "pcn": explainers.pcn_svg,
    }[name]()
    return f"![{explainers.ALT[name]}](data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()})"


def md(text: str, id: str | None = None):
    """A Markdown cell. `id` gives the cell a stable id, which a `#scrollTo=<id>` link targets in Colab."""
    text = re.sub(r"@@DIAGRAM:(\w+)@@", lambda m: diagram_md(m.group(1)), text)
    text = re.sub(r"@@EXPLAINER:(\w+)@@", lambda m: explainer_md(m.group(1)), text)
    cell = new_markdown_cell(text.strip())
    if id:
        cell["id"] = id
        cell["metadata"]["id"] = id
    return cell


def code(text: str):
    return new_code_cell(text.strip())


CLAIMS_CELL = '''
def value(ind, country):
    o = w.latest(w.indicator_lookup(ind, country, start=2023))
    return o["value"], o["year"]


def growth(country, year):
    return next(o["value"] for o in w.indicator_lookup("NY.GDP.MKTP.KD.ZG", country, start=year)["observations"] if o["year"] == year)


rows = [  # (claim, what the model said, what the data says, verdict)
    ("Kenya · internet use", "“roughly 40%” of the population", "{:.1f}% of people online in {}".format(*value("IT.NET.USER.ZS", "Kenya")), "above the data"),
    ("Rwanda · internet use", "“roughly 30–40% (estimates vary widely)”", "{:.1f}% of people online in {}".format(*value("IT.NET.USER.ZS", "Rwanda")), "close"),
    ("Kenya · GDP per capita", "“~$2,000–2,200”", "{:,.0f} US dollars in {}".format(*value("NY.GDP.PCAP.CD", "Kenya")), "close"),
    ("Rwanda · GDP per capita", "“~$950–1,050”", "{:,.0f} US dollars in {}".format(*value("NY.GDP.PCAP.CD", "Rwanda")), "close"),
    ("Kenya · GDP growth", "“~5% (2023), ~4–5% (2024)”", f"{growth('Kenya', 2023):.1f} percent in 2023, {growth('Kenya', 2024):.1f} percent in 2024", "close"),
    ("Rwanda · GDP growth", "“~8% (2023), ~9–10% (2024)”", f"{growth('Rwanda', 2023):.1f} percent in 2023, {growth('Rwanda', 2024):.1f} percent in 2024", "2024 above the data"),
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

A language model writes from two sources: the text in its prompt and what it learned in training. This notebook shows the difference. The same research question is asked with nothing but the question, then with evidence placed in the prompt. The two answers establish the first two components of an application: the instructions, which set the model's behaviour, and the context, which supplies the evidence. The application writes both, and the model reads them.

## The application after this notebook

@@DIAGRAM:prompt@@

The prompt is the only component this notebook builds. Later notebooks add the document store, retrieval, data tools, the agent loop, and the verifier around it.

The whole session uses one research question, so that each method is judged on the same case:

> How have internet access and economic development evolved in Rwanda and Kenya? Compare the recent available data, summarize the main patterns, and cite the sources.

## Contents

1. [Section 1](#scrollTo=section-1) defines the three parts of a prompt and who writes each.
2. [Section 2](#scrollTo=section-2) asks the question with no evidence. Four of the six figures checked are close to the official values, two are off, and none can be traced to a source.
3. [Section 3](#scrollTo=section-3) places four country briefs in the prompt. Every figure now carries a citation.
4. [Section 4](#scrollTo=section-4) asks for a year the brief does not cover. The model says so and gives the closest year.
5. [Section 5](#scrollTo=section-5) asks an ambiguous question. With one added instruction, the model asks which meaning is intended.
6. [Section 6](#scrollTo=section-6) prints the messages as the model received them.
7. [Section 7](#scrollTo=section-7) counts the tokens the context cost and explains why a whole collection cannot be supplied this way.

## How to run it

Run the cells in order. Without an API key, the model's outputs are recorded ones, labelled `[recorded output: ...]`. Each cell names its recorded scenario with `w.use(...)` before it calls the model. With a key in `WORKSHOP_API_KEY`, the same cells call the model live, the answer streams into its bubble as the model writes, and the wording differs from the recorded answers.
"""),
    md("""
The next cell installs the session's tools and loads its helper module. The first run takes about a minute. The cell is setup only.
"""),
    code(SETUP),
    md(STACK),
    md("""
The setup cell imported the helper module as `w` and created `model`, a LangChain chat model: a live one when a key is set, otherwise one that replays recorded answers. The helper functions used in this notebook, in the order they appear:

| Call | What it does |
|---|---|
| `w.use(name)` | Names the recorded scenario for the next model call, so that the replay returns the right answer. Has no effect on a live model. |
| `w.show(message, role=...)` | Draws a message as a chat bubble, with the text rendered as Markdown. |
| `w.answer(chain, inputs)` | Calls the chain and draws the answer as a bubble. With a live model the tokens stream into the bubble; without a key the recorded answer is drawn at once. Returns the message. |
| `w.indicator_lookup(indicator, country, start=...)` | Returns the observations of a World Development Indicators series for one country, from the session's data file. |
| `w.latest(result)` | The last observation of such a result. |
| `w.show_table(...)`, `w.show_claim_check(...)`, `w.show_prompt(...)` | Draw a table, the claims comparison, and the prompt as HTML. |
| `w.load_documents()` | Loads the eight session documents as LangChain `Document` objects, with the file's header as metadata. |
"""),
    md(id="section-1", text="""
## 1. The three parts of a prompt

A call to a chat model sends a list of messages. The first is the system message, which the model treats as standing instructions for the whole exchange. The others are user messages. An application composes three parts into these messages:

- **Instructions**: how the model should behave, written by the application's builders.
- **Context**: the evidence for this question, selected by the application. Here it is documents; later it is passages and data records.
- **Question**: the user's text.

The parts are kept in separate slots because a different person changes each one, at a different time.

The next cell writes three sets of instructions. The basic set only names the role. The grounded set adds the three rules this session depends on: answer from the context, cite the document after each figure, and say when something is missing. The clarifying set adds one more rule, for section 5.
"""),
    code('''
INSTRUCTIONS_BASIC = "You are a research assistant. Answer the question."

INSTRUCTIONS_GROUNDED = """You are a research assistant for development data.
Answer only from the supplied context. Cite the document id in square brackets after each figure.
If the context does not contain what is needed, say so and give the closest available information."""

INSTRUCTIONS_CLARIFY = INSTRUCTIONS_GROUNDED + """
If the question is ambiguous, say which readings are possible and ask which one is meant before answering."""
'''),
    md("""
A prompt template holds the three slots. `ChatPromptTemplate.from_messages` takes a list of `(role, text)` pairs, and each `{name}` in a text is a slot that is filled at call time. The instructions become the system message. The context and the question go together into one user message, in a fixed layout, so that the model can tell them apart. `prompt | model` joins the two steps into one object: the output of the template becomes the input of the model. That object, `chain`, is the whole application at this stage.
"""),
    code('''
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "{instructions}"),                                   # the system message: the instructions slot
    ("human", "CONTEXT:\\n{context}\\n\\nQUESTION: {question}"),      # one user message: the context and the question slots
])
chain = prompt | model   # fill the template, then send the messages to the model

w.show(w.MAIN_QUESTION, role="user")
'''),
    md("""
@@EXPLAINER:prompt@@

*The animation's text is this notebook's: the instructions from the cell above, the context and the answer from section 3, and the question from the worked example.*
"""),
    md(id="section-2", text="""
## 2. The model without context

Without context, the model answers from training data: text it read before its training ended, compressed into its parameters. It holds no records, only patterns of text, so its figures are reconstructions.

The next cell makes the first call. `w.answer` passes one dictionary with a value for each slot to the chain, and draws the reply. The context slot gets the text `(none)`, so the model sees that no evidence was supplied. `w.use` names the recorded scenario first, so that the replay returns the answer recorded for this exact call.
"""),
    code('''
w.use("nb01_zero_context")
answer = w.answer(chain, {"instructions": INSTRUCTIONS_BASIC, "context": "(none)", "question": w.MAIN_QUESTION})
'''),
    md("""
The answer cannot be judged without reference values. The next cell prints the official values from the World Development Indicators for the two measures the answer leans on most: the share of people using the internet and GDP per capita. `w.indicator_lookup` returns the observations of one series for one country, and the loop keeps the 2023 and 2024 values.
"""),
    code('''
rows = []
for ind in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"]:            # internet users (% of population), GDP per capita (current US$)
    fmt = "{:,.0f}" if ind == "NY.GDP.PCAP.CD" else "{:.1f}"
    for country in ["Kenya", "Rwanda"]:
        r = w.indicator_lookup(ind, country, start=2023)   # observations from 2023 on
        by_year = {o["year"]: fmt.format(o["value"]) for o in r["observations"]}
        rows.append((r["name"], country, by_year[2023], by_year[2024]))
w.show_table(["Indicator", "Country", "2023", "2024"], rows, source="Source: World Bank, World Development Indicators.")
'''),
    md("""
The next cell takes six claims from the answer and sets each beside the official value. Each row holds the claim, the model's words, the value from the data, and a verdict. `value` returns the latest observation of a series, and `growth` returns the growth rate of a given year.
"""),
    code(CLAIMS_CELL),
    md("""
Four of the six claims are close to the official values. Two are off: Kenya's internet share, given as roughly 40 percent against 35.0 in 2024, and Rwanda's 2024 growth, given as 9 to 10 percent against 7.2. The answer is hedged, structured, and ends with a reading list, and none of that helps a reader find the two wrong figures. Every figure is a rounded range, most carry no year, and the sources are institutions to consult, so no figure can be traced to a record. A different model or different instructions would change the wording. The figures would still be reconstructions, because the prompt contains no data.
"""),
    md(id="section-3", text="""
## 3. The same question with context

Context is text that the application places in the prompt for the model to read. Under the grounded instructions, the model answers only from it and cites it.

The next cell builds the context. `w.load_documents` returns the eight session documents, each with an `id` in its metadata, and the cell keeps the four country briefs. Each brief is prefixed with its id in square brackets, which is the form the instructions ask the model to cite. The size is printed as characters and as an estimate of tokens, at four characters per token.
"""),
    code('''
docs = {d.metadata["id"]: d for d in w.load_documents()}
context = "\\n\\n".join(f"[{i}]\\n{docs[i].page_content}" for i in ["kenya-digital", "rwanda-digital", "kenya-economy", "rwanda-economy"])
print(f"{len(context):,} characters of context, roughly {len(context)//4:,} tokens")
'''),
    md("""
The call is the same as in section 2, with the grounded instructions in the first slot and the four briefs in the second.
"""),
    code('''
w.use("nb01_with_context")
answer = w.answer(chain, {"instructions": INSTRUCTIONS_GROUNDED, "context": context, "question": w.MAIN_QUESTION})
'''),
    md("""
@@EXPLAINER:before_after@@

*Shortened excerpts of the recorded answers of sections 2 and 3.*
"""),
    md("""
Every figure now carries a citation to a brief, the comparison is specific, and the model notes two statements in the briefs that it could not verify from the years listed. The application chose the four briefs and put them in the prompt. The model wrote from that text. For the rest of the session, the application decides what evidence the model sees, and the quality of the answer follows from that decision.
"""),
    md(id="section-4", text="""
## 4. A question the evidence cannot answer

An instruction sets the model's behaviour when the evidence falls short. The grounded instructions say: if the context does not contain what is needed, say so and give the closest available information. The next cell supplies only the Kenya brief, prefixed with its id so that the model can cite it, and asks for 2025. The brief ends at 2024.
"""),
    code('''
w.use("nb01_missing_info")
brief = f"[kenya-digital]\\n{docs['kenya-digital'].page_content}"   # one brief, with its id so that the model can cite it
answer = w.answer(chain, {"instructions": INSTRUCTIONS_GROUNDED, "context": brief,
                          "question": "What share of Kenya's population used the internet in 2025?"})
'''),
    md("""
The model says that 2025 is absent, names 2024 as the latest year, and gives 35.0 percent with a citation. The behaviour came from one sentence in the instructions.
"""),
    md(id="section-5", text="""
## 5. An ambiguous question under two sets of instructions

"How connected is Kenya?" has at least three readings: internet use, mobile subscriptions, or broadband coverage. A model picks a reading unless it is told to ask. The next cell asks the question twice with the same context: first with the basic instructions, then with the clarifying set, whose added rule tells the model to list the readings and ask which one is meant.
"""),
    code('''
print("Without an instruction to clarify:\\n")
w.use("nb01_ambiguous_no_instruction")
first = w.answer(chain, {"instructions": INSTRUCTIONS_BASIC, "context": brief, "question": "How connected is Kenya?"})

print("\\n" + "=" * 100 + "\\nWith the instruction:\\n")
w.use("nb01_ambiguous_with_instruction")
second = w.answer(chain, {"instructions": INSTRUCTIONS_CLARIFY, "context": brief, "question": "How connected is Kenya?"})
'''),
    md("""
With the basic instructions, the model answers at length, covers two readings, and adds a growth analysis the question did not ask for. With the clarifying instruction, it names three readings and asks. An answer to the model's own reading looks as confident as an answer to the user's reading, so the user cannot tell them apart. For a research application, asking costs one exchange and removes that risk.
"""),
    md(id="section-6", text="""
## 6. The prompt as sent to the model

`prompt.format_messages` fills the template with values and returns the message list without calling the model. The next cell does this for the last call of section 5 and draws the messages in the colours of section 1. The instructions arrive as the system message. The context and the question arrive together in one user message. This is everything the model sees.
"""),
    code('''
messages = prompt.format_messages(instructions=INSTRUCTIONS_CLARIFY, context=brief, question="How connected is Kenya?")
w.show_prompt(messages)
'''),
    md(id="section-7", text="""
## 7. The cost of context

A model reads a limited amount of text per call, called its context window and measured in tokens. A token is a word or a part of a word, about four characters of English on average. The four briefs took about a thousand tokens. A collection of thousands of documents exceeds any window, the price of a call grows with its tokens, and the passages that answer one question are a small part of the whole. An application therefore selects, for each question, the part of the collection that the model should read. Notebooks 02 and 03 build that selection.

@@EXPLAINER:desk@@

*The figure of about 1,024 tokens in the diagram is the size that section 3 printed for the four briefs.*
"""),
    md("""
## Key points

- A prompt has three parts: instructions, context, and the question. A different person writes each, so they are kept in separate slots.
- Without context, the figures are reconstructions from training data. Four of the six checked were close, two were off, and none could be traced to a record.
- With the briefs as context, every figure cited a brief. The application chose the evidence, and the model wrote from it.
- Instructions set behaviour: cite, say when information is missing, ask when a question is ambiguous. They cannot supply data.
- The context window limits what one call can read, so the application has to select the evidence. The next two notebooks build the selection.

## Exercise

1. Read the answers of sections 2 and 3 side by side. For each number, note whether it can be checked and where. Then decide which answer could be sent to a colleague, and give the reason.
2. In the first cell of section 1, add a sentence to `INSTRUCTIONS_GROUNDED`, for example "Answer in no more than three sentences." Run that cell again, then run the cell of section 6. The Instructions block now shows the added sentence exactly as the model would receive it. The recorded answers do not react to new instructions. With a live key in `WORKSHOP_API_KEY`, rerun section 3 to see how the answer changes.
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

## Contents

1. [Section 1](#scrollTo=section-1) loads the eight documents. Each has its text and a header of information about it.
2. [Section 2](#scrollTo=section-2) shows what metadata records that the text cannot say about itself, such as the date of the oldest document.
3. [Section 3](#scrollTo=section-3) cuts one brief two ways, by length and by heading, and compares the chunks.
4. [Section 4](#scrollTo=section-4) turns chunks and a question into numbers and ranks the chunks by similarity.
5. [Section 5](#scrollTo=section-5) searches the same collection prepared badly and prepared well, and counts which of the needed figures each search finds.
6. [Section 6](#scrollTo=section-6) opens one record of the vector store: its text, its metadata, and its numbers.

## How to run it

Run the cells in order. This notebook only prepares material for the model that writes the answers, so it does not use that model, and no API key is needed. The model comes back in notebook 03. Everything below is computed when you run the cell. It does use a small embedding model, which turns text into numbers; it downloads about 90 MB the first time and runs inside the notebook.
"""),
    md("""
The next cell installs the session's tools and loads its helper module. The first run takes about a minute. The cell is setup only.
"""),
    code(SETUP_NO_MODEL),
    md("""
The tools in this notebook are LangChain's: `Document`, `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, and `InMemoryVectorStore`.
"""),
    md(id="section-1", text="""
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
    md(id="section-2", text="""
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
w.show(d.page_content)
'''),
    md(id="section-3", text="""
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
w.show(repr(fixed[1].page_content))
print("\\nA section chunk:\\n")
w.show(sections[1].page_content)
print("\\nmetadata:", sections[1].metadata["section"], "|", sections[1].metadata["date"], "|", sections[1].metadata["country"])
'''),
    md("""
Window 2 cuts by length, not by meaning. It starts in the middle of the Internet use section, carries no heading of its own, and ends with the heading of the next section, separated from the paragraph it introduces. A passage about internet use is then stored next to a mobile-subscriptions heading, and nothing in the chunk says which section it belongs to. The Mobile subscriptions section carries its title and heading, which both the retriever and the model can use.

"""),
    md(id="section-4", text="""
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

"""),
    md(id="section-5", text="""
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

"""),
    md(id="section-6", text="""
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

A language model answers from the text in its prompt. Notebook 01 placed four briefs in the prompt by hand. A real collection holds thousands of documents and a prompt holds a few pages, so an application has to select the pages for each question. Notebook 02 prepared the collection for that selection: it split the eight documents into 21 section chunks, kept each document's date as metadata, and stored a vector for each chunk. This notebook performs the selection and the answer. The pattern is retrieval-augmented generation (RAG): retrieve the chunks closest to the question, add them to the prompt, and generate the answer from them. A retrieved chunk is called a passage.

The notebook then shows three ways retrieval fails: a question with two parts, an outdated passage that outranks current ones, and a question the collection cannot answer. Each failure has a fix on the retrieval side, and the last section tests retrieval without the model.

## The application after this notebook

@@DIAGRAM:retrieval@@

Retrieval selects the passages. The model's step is now called generation, because the model writes from what retrieval supplied.

## Contents

1. [Section 1](#scrollTo=section-1) answers the main question from four retrieved passages. The answer covers internet use only.
2. [Section 2](#scrollTo=section-2) shows the ranking behind that answer. The economy chunks fall below the cut-off, and one search per half of the question brings them back.
3. [Section 3](#scrollTo=section-3) asks for the latest figures. The 2016 summary outranks the 2026 briefs, and the answer reports a 2015 value.
4. [Section 4](#scrollTo=section-4) removes the summary with a filter on the date. Rewording the question does not remove it.
5. [Section 5](#scrollTo=section-5) asks a question the collection cannot answer. The search returns passages anyway.
6. [Section 6](#scrollTo=section-6) tests retrieval alone on the eight evaluation questions.

## How to run it

Run the cells in order. Without an API key, the model's answers are recorded ones, labelled `[recorded output: ...]`, as in notebook 01. With a key in `WORKSHOP_API_KEY`, the same cells call the model live, the answer streams into its bubble as the model writes, and the wording differs from the recorded answers. The searches, the rankings, and the tables are computed when a cell runs, in both modes.
"""),
    md("""
The next cell installs the session's tools and loads its helper module. The first run takes about a minute. The cell is setup only.
"""),
    code(SETUP),
    md("""
The helper functions used in this notebook, in the order they appear. `w.use`, `w.show`, and `w.answer` are as in notebook 01.

| Call | What it does |
|---|---|
| `w.load_documents()`, `w.split_sections(docs)`, `w.build_store(chunks)` | The three steps of notebook 02: load the documents, split them into section chunks, embed the chunks into an in-memory vector store. |
| `store.similarity_search(q, k=...)` | LangChain's search: the `k` chunks whose vectors are closest to the question's. `similarity_search_with_score` also returns each score. |
| `w.format_context(chunks)` | Joins chunks into the context text, each prefixed with its citation. |
| `w.citation(chunk)` | The citation of a chunk: document id, section, and year, from its metadata. |
| `w.print_hits(results)`, `w.show_ranking(...)`, `w.show_search_comparison(...)` | Print or draw search results. |
| `w.load_questions()` | The eight evaluation questions of the session, each with its expected documents and numbers. |
| `w.numbers_present(text, expected)`, `w.retrieval_hit(chunks, expected_docs)` | The two retrieval checks: how many expected numbers a text contains, and whether an expected document is among the chunks. |
"""),
    md("""
Each Colab notebook starts with an empty session, so the next cell rebuilds the store of notebook 02 with the same three calls.
"""),
    code('''
docs = w.load_documents()          # the eight session documents
chunks = w.split_sections(docs)    # one chunk per Markdown section, metadata carried over
store = w.build_store(chunks)      # a vector for every chunk, in memory
print(len(chunks), "chunks in the store, the same collection as in notebook 02")
'''),
    md("""
The application is the one of notebook 01 with retrieval in front of it. The instructions now speak of passages and of a citation form, which `w.format_context` writes before each passage. `make_rag` builds the chain: a search step that takes the question and returns the `k` closest chunks, formatted as context; a pass-through that forwards the question unchanged; the prompt; the model. `hits` runs the search alone and keeps the scores. The optional `where` is a filter on metadata, used in section 4.
"""),
    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

INSTRUCTIONS = """You are a research assistant for development data.
Answer only from the supplied passages. After each figure, cite the passage in the square-bracket form given.
If the passages do not contain what is needed, say so."""

prompt = ChatPromptTemplate.from_messages([("system", INSTRUCTIONS), ("human", "PASSAGES:\\n{context}\\n\\nQUESTION: {question}")])


def make_rag(k=4, where=None):
    """RAG (retrieval-augmented generation): search the store, put the passages in the prompt, ask the model."""
    retrieve = RunnableLambda(lambda q: store.similarity_search(q, k=k, filter=where))   # question -> k closest chunks
    return {"context": retrieve | w.format_context, "question": RunnablePassthrough()} | prompt | model


def hits(question, k=4, where=None):
    """The search alone: (chunk, score) pairs, best first."""
    return store.similarity_search_with_score(question, k=k, filter=where)
'''),
    md(id="section-1", text="""
## 1. The main question answered from the store

Retrieval scores every chunk by the distance between its vector and the question's vector, and returns the `k` closest. A vector is a list of numbers that summarises what a text is about, computed by the embedding model of notebook 02. The score printed with each passage is the cosine similarity: 1.0 means identical meaning and 0 means unrelated. The next cell shows the question, prints the four passages that `hits` returns for it with their scores, and then calls the chain through `w.answer`, which repeats the same search inside and asks the model.
"""),
    code('''
w.show(w.MAIN_QUESTION, role="user")
print()
w.print_hits(hits(w.MAIN_QUESTION, k=4))     # what the model is about to read
print()
w.use("nb03_rag_main")
reply = w.answer(make_rag(k=4), w.MAIN_QUESTION)
'''),
    md("""
The answer opens by saying that the passages contain no economic indicators, and it covers internet use with a citation after each figure. The instructions require both behaviours. The GDP figures are in the store, in the two economy briefs, and retrieval did not return them. Section 2 shows why.
"""),
    md(id="section-2", text="""
## 2. The ranking behind the answer

A similarity search compares meanings. The question's vector lies closer to the internet chunks than to the economy chunks, because most of the question's words concern connectivity. The search ranks by this closeness alone. It cannot reserve places for the second half of a question, because it never identifies the halves. The next cell asks for the ten closest chunks, labels each with its topic from a small dictionary keyed by document id, and draws the ranking with the cut-off after the fourth.
"""),
    code('''
topic = {"kenya-digital": "internet", "rwanda-digital": "internet", "regional-2016": "internet", "internet-indicator": "internet",
         "kenya-economy": "economy", "rwanda-economy": "economy", "coffee": "other", "mobile-money": "other"}
legend = {"internet": ("internet use", "#2f5f9e"), "economy": ("economy", "#d9822b"), "other": ("other topics", "#8a94a3")}

ranked = hits(w.MAIN_QUESTION, k=10)
w.show_ranking([(w.citation(d), score, topic[d.metadata["id"]]) for d, score in ranked], k=4, legend=legend, question=w.MAIN_QUESTION)
'''),
    md("""
The four chunks above the line are three internet chunks, including the 2016 summary, and the mobile-money note. The first economy chunk is fifth at 0.570 against 0.591 for fourth place, and Kenya's GDP per capita is ninth. Raising `k` is a poor remedy: the ninth place costs nine passages, the coffee note enters at seventh, and every passage adds tokens.

The fix is one search per half of the question, with the results merged. An application can split the question with a model; here the two halves are written by hand. The next cell runs a search for each half, merges the results without duplicates, and compares the merged list with the single search of section 1. The comparison marks which of the four figures a correct answer needs appear in each list; `needed` comes from the evaluation file of the session, and `w.numbers_present` looks for each figure in the text of the passages.
"""),
    code('''
sub_queries = ["internet use in Kenya and Rwanda, recent figures", "GDP per capita and economic growth in Kenya and Rwanda, recent figures"]
merged = []
for q in sub_queries:
    for d in store.similarity_search(q, k=4):   # four passages per half
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
'''),
    md("""
The merged search returns eight passages where the single search returned four, and all four needed figures are among them. The 2016 summary and the mobile-money note are among them too. The next cell gives the merged passages to the model. It calls `prompt | model` directly, because the chain of `make_rag` would run its own search.
"""),
    code('''
w.use("nb03_rag_two_queries")
reply = w.answer(prompt | model, {"context": w.format_context(merged), "question": w.MAIN_QUESTION})
'''),
    md("""
The answer covers both halves with a citation after each figure. The two extra passages did no harm: the model used the 2016 summary for a projection and said the projection could not be checked, and it reported that the mobile-money note holds no figures. Section 3 shows what happens when the 2016 summary is the passage an answer depends on.
"""),
    md(id="section-3", text="""
## 3. Outdated evidence in the ranking

A vector encodes what a text is about and nothing about when it was written. A 2016 summary that uses the phrase "latest available figures" scores high for a question that uses the same phrase, whatever its date. The next cell asks for the latest figures on internet use in Kenya and Rwanda with `k = 2`, prints the two passages with their scores, and shows the answer.
"""),
    code('''
q = "What are the latest available figures for internet use in Kenya and Rwanda?"
w.print_hits(hits(q, k=2))
print()
w.use("nb03_rag_outdated")
reply = w.answer(make_rag(k=2), q)
'''),
    md("""
The 2016 summary ranks first at 0.816 and Rwanda's 2026 brief second at 0.765. Kenya's brief is absent, so the model's Kenya figure is the summary's: 16.6 percent in 2015. The model states that the summary dates from 2016 and that 16.6 is the most recent value in its passages. The answer is correct for its passages and wrong for the user, who asked for the latest figures while Kenya's 2024 value sits in the store. Generation cannot repair a selection made before the model saw anything. The fix belongs to retrieval.
"""),
    md(id="section-4", text="""
## 4. Three fixes for outdated evidence

1. **A filter on the date.** The store checks metadata before it scores similarity, so a document dated before 2020 is never ranked. Notebook 02 kept the date for this.
2. **A reworded question.** Rewriting the question to resemble the expected passage changes the question's vector. The ranking can move either way.
3. **Sorting by date after the search**, for a store without a filter. It does the same job as the filter and is not shown.

The next cell defines the filter as a function of a chunk, `since_2020`, which the store calls on each chunk's metadata before scoring. It then runs the same search with and without the filter and marks which of the two needed figures each returns.
"""),
    code('''
since_2020 = lambda d: str(d.metadata.get("date", "")) >= "2020"   # the filter: keep chunks dated 2020 or later
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
With the filter, Kenya's brief replaces the summary and both 2024 figures are present. The next cell passes the filter to `make_rag` and shows the answer from the filtered passages.
"""),
    code('''
w.use("nb03_rag_filtered")
reply = w.answer(make_rag(k=2, where=since_2020), q)
'''),
    md("""
The second fix, rewording, is tried next without the filter. The question is rewritten in the vocabulary of the 2026 briefs.
"""),
    code('''
w.print_hits(hits("Kenya and Rwanda internet use 2024, percent of population", k=2))
'''),
    md("""
The reworded question puts the summary first at 0.871, above Rwanda's brief at 0.833. The summary shares more vocabulary with the new wording than the brief does, so the rewording strengthened the match. The date is the only property that separates the summary from the briefs, and only metadata carries it.
"""),
    md(id="section-5", text="""
## 5. A question outside the collection

A similarity search ranks every chunk and returns the top `k`. It has no notion of a chunk being unrelated: the top three for an unanswerable question are the three least unrelated chunks. The next cell asks for the population of Rwanda, which the collection holds, and for inflation in Uganda in 2023, which it does not, and prints three passages for each.
"""),
    code('''
for q2 in ["What is the population of Rwanda?", "What was inflation in Uganda in 2023?"]:
    print(q2)
    w.print_hits(hits(q2, k=3))
    print()
'''),
    md("""
The population question returns the right chunk first at 0.739. The inflation question returns three GDP chunks about Kenya and Rwanda, the best at 0.534. The scores separate the two cases, and a cut-off between them would stop the second search from reaching the model. The cut-off depends on the embedding model and has to be set from examples. The second safeguard is the instruction from notebook 01 that tells the model to say when the passages do not answer, which notebook 05 shows in use. An application uses both.
"""),
    md(id="section-6", text="""
## 6. A test of the search without the model

Retrieval can be tested alone, because its output is a list of passages that can be checked without reading an answer. The next cell runs the eight evaluation questions through the search and makes two checks per question. `w.retrieval_hit` reports whether one of the expected documents is among the passages. `w.numbers_present` counts how many of the numbers a correct answer needs appear in the passages' text. `K` sets the number of passages, and the exercise changes it.
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
At `K = 4` every question has an expected document among its passages, and q1, the main question, has two of its four numbers: the problem of section 2. A missing number here cannot be fixed by any instruction to the model. When both checks pass and the answer is still wrong, the fault lies in generation: the instructions or the model. Testing the two halves separately shows which one to change. Notebook 06 adds a check of a different kind, of each number in the answer against its source record.
"""),
    md("""
## Key points

- Retrieval scores every chunk by the distance between its vector and the question's, and returns the `k` closest as passages for the prompt.
- A vector encodes meaning. A question with two parts matches the chunks of one part and misses the other; one search per part fixes it.
- A vector encodes nothing about dates. A filter on metadata removes outdated passages; rewording the question does not.
- A search returns the top `k` chunks even when none is related. A cut-off score stops unrelated passages from reaching the model, and an instruction tells the model to say when the passages do not answer.
- Retrieval can be tested on its own by checking that the expected documents and numbers are among the passages. When a number is absent from the passages, the model never saw it, so no instruction to the model can put it in the answer; the fix is in retrieval.

## Exercise

1. Look at the ranking in section 2. Which chunk takes fourth place, and is it a good use of one of the four slots? Say what in the chunk explains its score.
2. In the cell of section 6, set `K` to `2`, run it, then set it to `8` and run it again. Which questions lose numbers at 2, and which gain at 8? At 8, does every question find all of its needed numbers?
3. Set `K` to `2` and look at q8, the question from section 3. Explain why it passes at 4 and fails at 2, using the two searches in section 4.
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

## Contents

1. [Section 1](#scrollTo=section-1) compares the search of notebook 03 with the tool on the four figures the main question needs.
2. [Section 2](#scrollTo=section-2) shows the tool as the model sees it: a name, a description, and a form to fill in.
3. [Section 3](#scrollTo=section-3) runs one exchange: the model asks, the application runs the tool, the result goes back.
4. [Section 4](#scrollTo=section-4) asks for a year that has no data. The tool returns nothing and the model says so.
5. [Section 5](#scrollTo=section-5) returns the answer as a structured output, with named fields that a program reads, and shows what happens when a field is damaged.
6. [Section 6](#scrollTo=section-6) serves the same tool over MCP and calls it from a client and an agent.

## How to run it

Run the cells in order. Without an API key, the model's answers are recorded ones, labelled `[recorded output: ...]`, as in notebook 01. With a key in `WORKSHOP_API_KEY`, the same cells call the model live, the answer streams into its bubble as the model writes, and the wording differs from the recorded answers. The tool results, tables and checks are computed when the cell runs. The last cell reaches a public server and needs the internet.
"""),
    code(SETUP),
    md(id="section-1", text="""
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
reply = w.answer(model, "What was Rwanda's GDP per capita in 2023?")
print()
r = w.indicator_lookup("NY.GDP.PCAP.CD", "Rwanda", start=2023, end=2023)
w.show_table(["year", "value"], [(o["year"], f"{o['value']:,.1f}") for o in r["observations"]], source=f"{r['name']} · {r['source']}")
'''),
    md("""
The answer from memory gives a range and a figure of about 966 dollars, hedged, and nothing tells a reader how far that is from the record. The tool's answer is exact, 1,061, and carries its source. The application can now get an exact number, with its source, instead of hoping that a passage holds it.

"""),
    md(id="section-2", text="""
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
    md(id="section-3", text="""
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
reply = w.answer(model_with_tools, messages)
for call in reply.tool_calls:
    result = w.get_indicator.invoke(call["args"])
    print()
    print("2. The application runs it, and the tool returns:")
    w.show_table(["year", "value"], [(o["year"], f"{o['value']:,.1f}") for o in result["observations"]], source=f"{result['name']} · {result['source']}")
    messages += [reply, ToolMessage(content=json.dumps(result), tool_call_id=call["id"])]
print()
print("3. The model writes its answer from the result:")
final = w.answer(model_with_tools, messages)
'''),
    md("""
The application can now use a tool: the model asks, the application runs it, and the result goes back to the model.

"""),
    md(id="section-4", text="""
## 4. Data that does not exist

The planted problem: a year with no observation. The tool returns an empty list, and the instructions say what to do with it. The model may ask again, for example for a wider range of years, so the cell allows up to three rounds, prints each tool result, and then shows the answer.
"""),
    code('''
messages = [SystemMessage(content=INSTRUCTIONS), HumanMessage(content="What share of Kenya's population used the internet in 2025?")]
w.use("nb04_missing_year")
for round in range(3):   # the model may ask again, for example for a wider range of years
    reply = model_with_tools.invoke(messages)
    if not reply.tool_calls:
        break
    messages.append(reply)
    for call in reply.tool_calls:
        result = w.get_indicator.invoke(call["args"])
        print(f"tool result for {call['args']}: {len(result['observations'])} observations")
        messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))
print()
w.show(reply)
'''),
    md(id="section-5", text="""
## 5. Structured output: an answer in a fixed shape

So far the model's answers are sentences, which suits a person. Suppose instead that a dashboard has a box labelled "GDP per capita, Rwanda" and must fill it from the model's answer. The cell does what a simple program might do: it takes the first number in the sentence from section 3.
"""),
    code('''
import re

first = re.findall(r"\\d[\\d,]*\\.?\\d*", final.content)[0]
print("The answer:")
w.show(final.content, role="assistant")
print("\\nThe dashboard's box: ", first)
'''),
    md("""
The box gets 2023, which is the year and not a figure. The sentence holds a year and a figure, and the words around them change from one answer to the next, so a program reading a sentence has to guess which number is wanted.

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

"""),
    md(id="section-6", text="""
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
w.show(w.final_text(result), role="assistant")
'''),
    md("""
The application can now offer its tool to any program that speaks MCP.

The program's [Data360 MCP server](https://github.com/worldbank/data360-mcp) offers search, metadata and data tools over the World Bank's data platform in the same way, at a public address. The cell lists its tools and runs one search. It reaches a live server, so the list can differ from run to run, and if the call fails the cell says so and the notebook carries on.
"""),
    code('''
try:
    async with Client(w.DATA360_MCP_URL) as client:
        tools = await client.list_tools()
        print(len(tools), "tools, for example:")
        for t in tools[:5]:
            print("  ", t.name)
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

**Session 01 · Building AI applications for research and data · notebook 5 of 6**

## Objective

Notebook 04 gave the model one tool and one question at a time. The main question of this session needs more than that: figures for two countries, which come from the tool, and context from the documents, which comes from the search of notebook 03. Someone has to decide which calls to make, and in what order. This notebook shows two ways of deciding, runs both on the main question, and then looks at what an answer built from several calls still lacks.

The application so far can search a store of documents (notebook 03) and call a tool for exact numbers (notebook 04). Here the search becomes a second tool, next to the first, and the application gains the means to coordinate them.

Four terms, in plain words:

- A **workflow** is a fixed order of steps written in code: search, then look up, then write. Every run makes the same calls.
- An **agent** is a model that chooses its own next step. It asks for a tool, sees the result, and decides what to do next, until it can answer. The agent here is built with LangGraph, a library for loops of this kind.
- A **trace** is the list of every call the agent made and what came back.
- A **step limit** (`recursion_limit` in LangGraph) is the number of steps after which a run is stopped, so that a loop that never ends cannot run for ever. A step is one turn of the loop: the model's turn, or the tools' turn.

## The application after this notebook

@@DIAGRAM:agent@@

## Contents

1. [Section 1](#scrollTo=section-1) answers the main question with a workflow: the steps are fixed in code.
2. [Section 2](#scrollTo=section-2) answers it with an agent that chooses its own calls, and shows the trace.
3. [Section 3](#scrollTo=section-3) asks a question the tools cannot answer, and shows what the agent does.
4. [Section 4](#scrollTo=section-4) makes a call fail, and shows the agent recover from the error.
5. [Section 5](#scrollTo=section-5) runs a loop that never ends, and stops it with the step limit.
6. [Section 6](#scrollTo=section-6) compares the two designs and when each fits.
7. [Section 7](#scrollTo=section-7) asks whether the numbers in the answer can be trusted, the question for notebook 06.

## How to run it

Run the cells in order. Without an API key, the model's answers are recorded ones, labelled `[recorded output: ...]`, as in notebook 01. With a key in `WORKSHOP_API_KEY`, the same cells call the model live, the answer streams into its bubble as the model writes, and the wording differs from the recorded answers. The searches, the tool results, the tables and the checks are computed when the cell runs. The exercises at the end run without a key.
"""),
    code(SETUP),
    code('''
import json
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

docs = w.load_documents()
store = w.build_store(w.split_sections(docs))   # the store of notebooks 02 and 03
search_documents = w.make_search_tool(store, k=3, since="2020")   # the search of notebook 03, offered as a tool
tools = [search_documents, w.get_indicator]   # the tool of notebook 04 is the second one

INSTRUCTIONS = """You are a research assistant for development data.
Use get_indicator for every statistic and search_documents for definitions, context, and explanations.
Name the indicator and the source with each figure; cite passages in the square-bracket form given.
If the tools do not provide what is needed, say so. Do not quote figures from memory."""
'''),
    md(id="section-1", text="""
## 1. A workflow: the order is written in code

The picture contrasts the two designs. In a workflow, the steps are written in code in advance: search the documents, look up the four series, then ask the model to write. The model is called once, at the end, and every run makes the same calls. The cell runs the workflow on the main question and prints each step as it happens.

@@EXPLAINER:workflows@@
"""),
    code('''
compose = ChatPromptTemplate.from_messages([("system", INSTRUCTIONS), ("human", "PASSAGES:\\n{passages}\\n\\nDATA:\\n{data}\\n\\nQUESTION: {question}")]) | model


def fixed_workflow(question):
    passages = search_documents.invoke({"query": question})
    print(f"step 1: searched the documents, found {len(passages)} passages")
    series = [w.indicator_lookup(i, c, start=2010) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]
    print(f"step 2: looked up {len(series)} series: internet use and GDP per capita, for Kenya and Rwanda")
    answer = w.answer(compose, {
        "passages": "\\n\\n".join(f"{p['citation']}\\n{p['text']}" for p in passages),
        "data": json.dumps(series),
        "question": question,
    })
    print("step 3: the model wrote the answer\\n")
    return answer, series


w.use("nb05_compose")
workflow_answer, workflow_series = fixed_workflow(w.MAIN_QUESTION)
'''),
    md("""
The application can now answer the main question with a fixed set of steps. It is predictable: the same question makes the same calls, so it is easy to test.

"""),
    md(id="section-2", text="""
## 2. An agent: the model chooses the steps

In an agent, the model is offered the tools and their descriptions, and chooses. `create_agent` builds the loop with LangGraph: the model replies with the calls it wants, the loop runs them, the results go back to the model, and the loop stops when the model answers in text. The cell asks the same question, then shows the trace as a table, one row per call, grouped by round.
"""),
    code('''
agent = create_agent(model, tools, system_prompt=INSTRUCTIONS)

w.use("nb05_agent_main")
result = agent.invoke({"messages": [HumanMessage(content=w.MAIN_QUESTION)]}, config={"recursion_limit": 10})
w.show_trace(result)
print()
w.show(w.final_text(result), role="assistant")
'''),
    md("""
The model chose the same five calls that the workflow made. On a different question it would choose differently. That is the point of an agent, and also its risk: the calls are not known in advance.

The first column of the table is the round: one message from the model, which can ask for several tools at once. Here it asked for all five in its first message, so they share round 1; in section 4, a call that fails takes a second round to correct. The application can now let the model choose between its tools.

"""),
    md(id="section-3", text="""
## 3. A question the tools cannot answer

A good agent searches, finds nothing usable, and says so. The question is about coffee exports. The collection holds one note on coffee that carries no figures, and the data tool has no series on exports.
"""),
    code('''
w.use("nb05_agent_offscope")
result = agent.invoke({"messages": [HumanMessage(content="How have Kenya's coffee export earnings changed in recent years?")]}, config={"recursion_limit": 10})
w.show_trace(result)
print()
w.show(w.final_text(result), role="assistant")
'''),
    md("""
The agent made one search and tried two indicator ids that the tool rejected, then declined to invent figures and said what it had looked for. The application can now say that it cannot answer, and what would be needed.

"""),
    md(id="section-4", text="""
## 4. A call that fails

The model asks for an indicator id that does not exist. The tool does not crash: it returns an error as data, with the list of valid ids. The model reads it and tries again. Returning errors as data is what lets the loop recover.
"""),
    code('''
w.use("nb05_agent_error")
result = agent.invoke({"messages": [HumanMessage(content="What share of Rwanda's population used the internet in 2023?")]}, config={"recursion_limit": 10})
w.show_trace(result)
print()
w.show(w.final_text(result), role="assistant")
'''),
    md("""
The first row is the failed call, in round 1; the second is the retry that worked, in round 2. The application can now recover from a wrong call, as long as its tools return errors that the model can read.

"""),
    md(id="section-5", text="""
## 5. A loop that never ends

A model can also keep calling tools without ever answering. The step limit stops it: after the given number of steps, the run ends with an error (`GraphRecursionError`), and the application decides what to do next. It can answer with what it has, ask the user, or give up. The recorded scenario below is a model that never stops: the cell sets the limit to 6, and the model keeps asking for tools until the loop is cut off.
"""),
    code('''
from langgraph.errors import GraphRecursionError

w.use("nb05_runaway")
try:
    agent.invoke({"messages": [HumanMessage(content=w.MAIN_QUESTION)]}, config={"recursion_limit": 6})
except GraphRecursionError as e:
    print("stopped:", str(e)[:80])
    print("The application now chooses: here, it tells the user that it could not finish.")
'''),
    md("""
The application can now stop a loop that does not end, instead of waiting for it.

"""),
    md(id="section-6", text="""
## 6. Choosing between the two

| | Workflow | Agent |
|---|---|---|
| Calls per run | Known in advance | Varies with the question |
| Behaviour on a new kind of question | Same calls, possibly irrelevant | Adapts, possibly wrongly |
| Cost | Predictable | Bounded by the step limit |
| Testing | Test the final step and the data | Test the trace and the answer |
| Use when | The questions have one shape | The questions vary and the tools are few and safe |

The worked example has one shape, so a workflow is the right design for it. The agent earns its place when users ask questions the designer did not anticipate, and only with a step limit, a trace, and tools that fail as data.

"""),
    md(id="section-7", text="""
## 7. Can we trust the numbers?

Both designs produce an answer that holds several numbers. Some come straight from a tool, some come from a passage, and some are worked out by the model itself: "rose by about 23.7 percentage points" is a subtraction, which no tool returned. The answer is plain text, and a reader, or a program, has no way to tell which number is which.

The tools have already laid some groundwork. Every figure the data tool returns carries a label, called a `claim_id`, made of the indicator, the country and the year. The cell shows the latest figure of each series that the workflow fetched.
"""),
    code('''
latest = [(s["indicator"], s["country"], o["claim_id"], f"{o['value']:,.1f}" if o["value"] < 1000 else f"{o['value']:,.0f}") for s in workflow_series for o in s["observations"][-1:]]
w.show_table(["indicator", "country", "claim_id", "value"], latest, source="Every figure the tool returns has a claim_id: indicator, country, year.")
'''),
    md("""
The labels sit on the tool's side, and nothing in the answer points back to them. What if every number in an answer carried its label, and a program checked each number against the record it names? The program has a protocol for exactly that, **Proof-Carrying Numbers** (PCN), its approach to trustworthy data, and notebook 06 builds it into the application.

## Key points

- A workflow is a fixed order of steps written in code: predictable and easy to test. Prefer it when the questions have one shape.
- An agent is a model that chooses its next step. It adapts to the question, so it needs a step limit, a trace, and tools that return their errors as data.
- Both designs need the same from their tools: clear descriptions and results the model can read.
- An answer built from several calls mixes numbers that come from tools, from passages and from the model's own arithmetic, and plain text does not say which is which. The labels the tool already attaches are the groundwork for checking that, in notebook 06.

## Exercise

1. In section 2, change `recursion_limit` from 10 to 3 and run the cell, then try 4. What is the smallest limit that lets this question finish? What does the application see when the limit is too low?
2. After section 2, run `sum(len(m.tool_calls) for m in result["messages"] if getattr(m, "tool_calls", None))`. How many calls did the agent make, and how many did the workflow make in section 1? Why are the two numbers equal here, and when would they differ?
3. Run `w.indicator_lookup("IT.NET.USER.ZS", "Uganda", start=2010)["observations"]`. The tool returns an empty list. What would the workflow of section 1 do with this result, and what would the agent do?
"""),
    md(LINKS),
]

# =============================================================================
# 06
# =============================================================================
NB06 = [
    md("""
# 06 · Verification

**Session 01 · Building AI applications for research and data · notebook 6 of 6**

## Objective

Notebooks 01 to 05 put the right evidence in front of the model: context, retrieved passages, exact records from a data tool, and a bounded agent loop. The answer still contains numbers in plain text, and nothing in the application checks them against the records. This notebook adds the check.

Verification follows the Proof-Carrying Numbers protocol (PCN), the program's approach to trustworthy numbers ([paper](https://arxiv.org/abs/2509.06902), [code](https://github.com/worldbank/pcn)). The model writes every figure from the data tool inside a claim tag that names the record the figure came from and the rule for comparing the two. A verifier reads each tag, fetches the record, compares, and marks the number verified or flagged. A number without a tag is unmarked.

Two terms:

- A **claim tag** is a label the model writes around a number. It names the record the number comes from and how to compare the two. It is the `claim_id` of notebook 05 with a comparison rule added.
- A **verifier** reads the tags and checks each number against its record. Here it is `pcn-core`, the Python core of the protocol. Each number gets one mark: **verified** (it matches), **flagged** (it does not), or **unmarked** (no tag, so nothing to check against).

The session has used three kinds of check:

| Check | Question it answers | Where |
|---|---|---|
| Retrieval | Did the application find the right evidence? | Notebook 03, section 6 |
| Grounding | Is the answer written from that evidence? | Citations in the answer |
| Verification | Does each number match its source record? | Claim tags and `pcn-core`, this notebook |

## The application after this notebook

@@DIAGRAM:verification@@

## Contents

1. [Section 1](#scrollTo=section-1) shows a claim tag and the main question answered with every tagged number checked: 14 verified, none flagged, and 21 untagged numbers left unmarked.
2. [Section 2](#scrollTo=section-2) replays an answer with one altered figure and one untagged figure, and shows the flag and the absence of a mark.
3. [Section 3](#scrollTo=section-3) states what the check covers and what it leaves to the reader.

## How to run it

Run the cells in order. Without an API key, the model's answers are recorded ones, labelled `[recorded output: ...]`, as in notebook 01. With a key in `WORKSHOP_API_KEY`, the same cells call the model live, and the answers will differ in wording from the recorded ones. The checks and the tables are computed when a cell runs, in both modes. The exercises at the end run without a key.
"""),
    md("""
The next cell installs the session's tools and loads its helper module. The first run takes about a minute. The cell is setup only.
"""),
    code(SETUP),
    md("""
The helper functions used in this notebook, in the order they appear. `w.use` and `w.show` are as in notebook 01.

| Call | What it does |
|---|---|
| `w.indicator_lookup(indicator, country, start=...)` | The function behind the `get_indicator` tool of notebook 04; here its results are placed in the prompt directly. |
| `w.observations_from_results(results)` | A dictionary from claim id to the value the tool returned. |
| `w.show_chat(question, answer, observations, results, raw)` | The answer as a chatbot draws it, with a mark after each checked number. |
| `w.verify(answer, observations)` | The verifier: one check per number in the answer, with its status. |
| `w.show_checks(checks)` | The checks as a table. |
"""),
    md("""
The next cell writes the instructions. The new rule is the claim tag. Its `id` is `indicator:country:year`, the key under which the tool returned the value. `policy="rounded"` with `decimals="1"` means that the record, rounded to one decimal, must equal the number shown; `policy="exact"` requires equality.
"""),
    code('''
import json
from langchain_core.messages import HumanMessage, SystemMessage

INSTRUCTIONS_CLAIMS = """You are a research assistant for development data.
Use get_indicator for every statistic and search_documents for definitions and context. Do not quote figures from memory.
Write every number that comes from get_indicator inside a claim tag that names its claim_id, like
<claim id="IT.NET.USER.ZS:KEN:2024" policy="rounded" decimals="1">35.0</claim>.
Name the indicator and the source. If the tools do not provide what is needed, say so.
If the question is ambiguous, ask which reading is meant."""
'''),
    md(id="section-1", text="""
## 1. Claim tags

A claim tag is a label around a number. The verifier takes the record named by the tag's id, applies the tag's rule, and compares the result with the number shown. The animation shows the three outcomes: a tagged figure that matches its record is verified, a tagged figure that differs is flagged, and a figure without a tag stays unmarked.

@@EXPLAINER:pcn@@

The next cell gives the model the four series the main question needs, placed in the prompt as notebook 04 did, and asks it to answer with tags. `w.observations_from_results` builds the dictionary from claim id to the value the tool returned, which is what the verifier compares against. `w.show_chat` draws the answer as a chatbot would: a mark after each checked number, a popover on hover or focus with the record and the rule, and, with `raw=True`, a fold that holds the tagged text as the model wrote it.
"""),
    code('''
results = [w.indicator_lookup(i, c, start=2010) for i in ["IT.NET.USER.ZS", "NY.GDP.PCAP.CD"] for c in ["Kenya", "Rwanda"]]
observations = w.observations_from_results(results)   # claim_id -> the value the tool returned

w.use("nb06_claims_main")
reply = model.invoke([SystemMessage(content=INSTRUCTIONS_CLAIMS), HumanMessage(content="DATA:\\n" + json.dumps(results) + "\\n\\nQUESTION: " + w.MAIN_QUESTION)])
w.show_chat(w.MAIN_QUESTION, reply.content, observations, results, raw=True)
'''),
    md("""
The model put its figures in two tables, one tag per cell, and wrote four patterns under them. Every tagged number carries a ✓. The numbers in the patterns carry no mark: they are the same figures repeated without tags, or values the model computed, such as a gain of 27.8 points or a ratio of 1.95. The next cell lists every check as a table.
"""),
    code('''
w.show_checks(w.verify(reply.content, observations))
'''),
    md("""
Fourteen numbers are verified, none is flagged, and 21 are unmarked. The record for Kenya's 2024 internet use holds 34.9762; rounded to one decimal it is 35.0, so the comparison holds. The 21 unmarked numbers are the repeats and the computed values in the patterns. A computed value has no single record to check against, so the protocol leaves it unmarked, and the reader sees that it was not checked. A check by value alone, without tags, could not tell a source from a coincidence: 28, a difference the model computed, equals Rwanda's 2022 internet share of 28.0. Years are not figures and are skipped.

The chat view imitates `ClaimMark`, the component of the program's `@pcn-js/ui` package that draws the same marks in a web chatbot ([`worldbank/pcn`](https://github.com/worldbank/pcn)). The marks here come from the Python verifier.
"""),
    md(id="section-2", text="""
## 2. A flagged number and an unmarked number

A flag means that the number does not match the record named in its tag. An unmarked number carries no tag, so there is nothing to compare, and no mark is drawn whether or not the number is right. The next cell replays a hand-made answer with one of each: Rwanda's internet use altered to 35.7 against the record's 31.7, and a figure of about 70 percent for mobile money that did not come from the tool.
"""),
    code('''
w.use("nb06_claims_error")
flawed = model.invoke([SystemMessage(content=INSTRUCTIONS_CLAIMS), HumanMessage(content="DATA:\\n" + json.dumps(results) + "\\n\\nQUESTION: " + w.MAIN_QUESTION)])
w.show_chat(w.MAIN_QUESTION, flawed.content, observations, results)
w.show_checks(w.verify(flawed.content, observations))
'''),
    md("""
The verifier flags 35.7 with a ⚠, and the popover shows the record, the value returned, and the failed comparison. The 70 carries no mark. The absence is deliberate. The protocol states the principle as: *trust is earned only by proof, and the absence of a verification mark communicates uncertainty.*
"""),
    md(id="section-3", text="""
## 3. What the check covers

The verifier answers one question about each number: does it match the record it names? It says nothing about whether that record is the right one for the question, whether the method behind the record is sound, or whether the sentence around the number interprets it correctly. Those judgements remain with the reader, and the marks tell the reader which numbers need none of that scrutiny on the first question. A test of the application across many questions, with a fixed set of expected answers, is the subject of the program's [evaluation cookbook](https://worldbank.github.io/ai4data/cookbook/evaluation-suites/); notebook 03, section 6, showed the retrieval half of such a test.

## Key points

- A claim tag names the record a number comes from and the rule for comparing them. The verifier compares the two and marks the number verified or flagged. A number without a tag is unmarked.
- An unmarked number is neither right nor wrong; it is unchecked, and the reader is told so.
- Verification checks a number against its record. Whether the record answers the question, and whether the sentence reads it correctly, stay with the reader.

## Exercise

All of these run without a key, on the answer of section 1 (`reply`).

1. Change one number inside a tag: `tampered = reply.content.replace(">35.0<", ">35.7<")`, then `w.show_checks(w.verify(tampered, observations))`. Which mark changes, and what does the record hold?
2. Remove all the tags: `plain = w.strip_claims(reply.content)`, then `w.show_checks(w.verify(plain, observations))`. What happens to every number, and why is the result not a "verified" one?
3. Change the policy of one tag from rounded to exact: `strict = reply.content.replace('policy="rounded" decimals="1"', 'policy="exact"', 1)`, then verify it. Why does 35.0 stop matching a record of 34.9762, and which policy suits a published percentage?
"""),
    md(LINKS),
]

NB07 = [
    md("""
# 07 · Lab: chat with your documents

**Session 01 · Building AI applications for research and data · the lab, after notebook 6**

## Objective

Build a small application that answers questions from documents of your choice, and use it. You start with the eight documents of the session, so that everything runs at once, and then you can replace them with your own: reports, notes, papers or a data dictionary. The documents stay inside the notebook. Nothing is sent to an outside service, and no key is needed.

Every piece comes from a notebook you have already seen:

| Piece | From |
|---|---|
| Split the documents into chunks and store them | Notebook 02 |
| Retrieve the passages closest to a question, and say so when none is close | Notebook 03 |
| Ask a model to answer from those passages | Notebooks 01 and 03 |
| Look up every number of the answer in the passages, and mark it | The idea of notebook 06, on documents |

The application is deliberately simple: one question at a time, with no memory of the conversation. The enhancements that a full chatbot adds are listed at the end of the notebook.

The model is new: a small open one, **Qwen2.5-1.5B-Instruct**, that runs in the notebook itself. It is far weaker than the models behind the chatbots you know. That is useful here, because its mistakes show why the checks of the earlier notebooks are needed.

## The application

@@DIAGRAM:lab@@

## Contents

1. [Section 1](#scrollTo=section-1) chooses the documents: the session's eight, or your own.
2. [Section 2](#scrollTo=section-2) splits the documents into chunks and stores them.
3. [Section 3](#scrollTo=section-3) loads the model that runs inside the notebook.
4. [Section 4](#scrollTo=section-4) asks one question from code: retrieve, answer, and check every number.
5. [Section 5](#scrollTo=section-5) asks three questions that show what the checks can and cannot tell you.
6. [Section 6](#scrollTo=section-6) opens the chat box, where you ask your own questions.

## How to run it

This notebook runs a real language model inside Colab, so it needs more than the others. No account key and no payment are needed. Do these steps in this order:

1. **Choose a GPU first.** In the Colab menu, click **Runtime > Change runtime type**, pick **T4 GPU**, and click **Save**. Do this before you run anything, because changing it later restarts the notebook. A GPU makes each answer take a few seconds instead of up to a minute. If Colab says no GPU is available, the notebook still works on the CPU, only more slowly.
2. **Run everything.** Click **Runtime > Run all**. If Colab warns that the notebook comes from GitHub, click **Run anyway**.
3. **Wait for the first run.** It downloads the model, about 3 GB, and takes a few minutes. The cell of step 3 prints "loaded on the GPU" when it is ready. Cells that show a spinner are still working.
4. **Chat.** Scroll to step 6 at the end, type a question in the box, and press Enter or click **Ask**. The first answer is the slowest.

If something goes wrong, click **Runtime > Restart session**, then run all again. The chat box in step 6 appears only in a running Colab or Jupyter notebook, so it is empty when you read the saved copy. To use your own documents, set `USE_MY_DOCUMENTS = True` in step 1 before you run all, and upload your files when asked.
"""),
    code(SETUP_NO_MODEL),
    md(id="section-1", text="""
## 1. Choose the documents

The session documents are used unless you say otherwise. The cell lists them. To use your own, set `USE_MY_DOCUMENTS = True` and run the cell again: Colab asks you to pick files. Plain text, Markdown and PDF files work, and a zip of several files is unpacked. A scanned PDF has no text to read, so it will not work. If you would like to see what a document looks like first, `lab.download_session_documents()` saves the eight session documents as a zip file, which you can open, change and upload again.
"""),
    code('''
import lab

USE_MY_DOCUMENTS = False   # True: upload your own files (Colab asks you to pick them)

paths = lab.upload_documents() if USE_MY_DOCUMENTS else lab.session_documents()
w.show_table(["document", "size"], [(p.name, f"{p.stat().st_size / 1000:.1f} KB") for p in paths], source=f"{len(paths)} documents")
'''),
    md(id="section-2", text="""
## 2. Split and store

As in notebook 02, each document is split into chunks, by section when it has headings and in windows of about 700 characters when it does not, and every chunk is stored with its meaning, so that a question can find it. The table shows what was made.
"""),
    code('''
chunks = lab.prepare(paths)
store = w.build_store(chunks)
lab.chunk_table(chunks)
'''),
    md(id="section-3", text="""
## 3. The model

The cell downloads the model on the first run and loads it. If Colab has given you a GPU, the model goes onto it and the cell says which one; if not, it runs on the CPU and the cell tells you how to switch. Because the model runs here, the documents are never sent anywhere. This cell has no saved output, because the message depends on the machine that runs it.
"""),
    code('''
llm = lab.load_model()
chat = lab.DocumentChat(llm, store, k=4, min_score=0.25)
'''),
    md(id="section-4", text="""
## 4. One question, asked from code

`chat.ask(question)` does the whole round: it retrieves the closest passages, drops any that are not close enough (`min_score`), asks the model to answer from what is left, and looks up every number of the answer in those passages. The answer is shown as a chat. A ✓ after a number means that the number appears in one of the passages; hover over it to see where. A ⚠ means that it does not. Years and citations are not checked. Under the answer, the passages that the answer was written from can be opened.
"""),
    code('''
from IPython.display import HTML, display

display(HTML(lab.reply_html(chat.ask("What share of Rwanda's population used the internet in 2024?"))))
'''),
    md(id="section-5", text="""
## 5. What the checks can and cannot tell you

Three questions on the session documents show the three situations that you will meet. They are asked from code, so that the answers are saved in this copy of the notebook.
"""),
    code('''
for question in [
    "What is the capital of France?",
    "How have Kenya's coffee export earnings changed in recent years?",
    "What is Kenya's GDP per capita and how does it compare with Rwanda's?",
]:
    display(HTML(lab.reply_html(chat.ask(question))))
'''),
    md("""
- **No close passage.** The documents say nothing about France, so no passage reaches `min_score` and the model is not asked. The application says so instead of guessing. This is the "no relevant result" case of notebook 03, handled by the application and not by the model.
- **A close passage with no numbers.** The coffee note is close to the question, so the model answers. It has no figures to give, and a model that is weak may say more than the note does. No mark can appear, because there are no numbers, and nothing here checks the sentences. This is what a check on numbers does not cover.
- **Numbers.** The marks say whether each number is in a passage. They do not say that it is the right number for the question. In the saved copy of the third answer, the model wrote that Rwanda's GDP per capita "had risen to $803" by 2024. The figure 803 is in the Rwanda document, so it carries a ✓, but it is the figure for 2020: the value for 2024 is 1,060. The check found the number and could not see that the sentence puts it in the wrong year. Your own run may word the answers differently, and a small model makes this kind of slip often. In notebook 06 each number named its own record, with its country and year, which is the stronger check; here the check is what can be done on arbitrary documents.

"""),
    md(id="section-6", text="""
## 6. Chat

Now use the application. Type a question in the box under the cell and press **Ask** (or Enter). Some to try on the session documents: *What is mobile money? How did Kenya's GDP per capita change between 2010 and 2020? Which sources mention coffee?* With your own documents, ask what you would ask a colleague who had read them. The chat box appears when you run the cell, so it is empty in a saved copy of this notebook.

Each question is answered on its own. The application does not remember the earlier ones, so a follow-up such as "and for Rwanda?" has nothing to refer to; write the full question each time.
"""),
    code('''
lab.launch(chat)
'''),
    md("""
## A complete version: the Data360 chat

What you built here is the small version. The World Bank's **[Data360 chat](https://data360.worldbank.org/en/int/chat)** is a complete version that puts the ideas of this series to work in one application: a language model that answers questions about development data, tools that fetch the records, and numbers that can be checked against them (the protocol of notebook 06). Open it, ask the main question of the session, and compare what you see with your own chat box: where does each answer come from, and how does it show that a number can be trusted?

## What this application leaves out

This is the simplest version that works, and it leaves out what a full chatbot has. Each of these is an enhancement that you can add later, and none is needed for a useful first version:

- **Memory of the conversation**, so that follow-up questions work.
- **Managing the context**: the application sends the four closest passages and the question, and nothing more. It does not count how much text the model can take, and it does not shorten or summarise long passages.
- **Speed-ups**: reusing the model's work on text it has already read (a **KV cache**), and showing the answer as it is written.
- **Better search**: re-ranking the passages, combining a search by words with the search by meaning, or rewriting the question before it is searched.
- **Keeping the index**, so that the documents are not split and stored again each time the notebook starts.
- **A test suite of your own questions**, as in section 6 of notebook 03 for retrieval, to see whether a change made the application better.

## Key points

- An application that chats with documents is the pieces of the earlier notebooks put together: split and store, retrieve, answer from the passages, check.
- A small model runs on your own machine, and the documents stay there. It is also weaker, so the checks and the retrieval threshold carry more of the work.
- Saying "I could not find that" when no passage is close is the application's job, and costs nothing.
- A check on numbers finds numbers that are not in the sources. It cannot find a claim that has no number.
- The Data360 chat is the complete version of this application, with the enhancements below.

## Exercise

1. Set `USE_MY_DOCUMENTS = True`, upload two or three files of your own, and rerun from step 1. Ask one question that the files answer and one that they do not. What does the application do with the second?
2. After step 3, run `chat.min_score = 0.5`, then ask the coffee question again in the chat box of step 6. What changes, and what is the risk of a high threshold? Try `0.1` as well.
3. Run `chat.k = 1`, then paste the main question of the session (`w.MAIN_QUESTION`) into the chat box. How much of the answer is left, and why?
4. On a GPU runtime, rerun from step 3 with `lab.load_model("Qwen/Qwen2.5-3B-Instruct")`. Do the answers get better? Which of the marks change?
"""),
    md(LINKS),
]

COLAB = "https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/"
ORDER = [
    ("01-from-model-to-application.ipynb", "01 · From a language model to an application", NB01),
    ("02-preparing-knowledge.ipynb", "02 · Preparing knowledge for an application", NB02),
    ("03-retrieval-augmented-generation.ipynb", "03 · Retrieval-augmented generation", NB03),
    ("04-tools-and-structured-outputs.ipynb", "04 · Tools and structured outputs", NB04),
    ("05-workflows-and-agents.ipynb", "05 · Workflows and agents", NB05),
    ("06-verification-and-evaluation.ipynb", "06 · Verification", NB06),
    ("07-lab-chat-with-your-documents.ipynb", "07 · Lab: chat with your documents", NB07),
]
for (_, _, nb), (nxt_file, nxt_title, _) in zip(ORDER, ORDER[1:]):
    nb.append(md(f"**Next:** [{nxt_title}]({COLAB}{nxt_file})"))


NOTEBOOKS = {
    "01-from-model-to-application.ipynb": NB01,
    "02-preparing-knowledge.ipynb": NB02,
    "03-retrieval-augmented-generation.ipynb": NB03,
    "04-tools-and-structured-outputs.ipynb": NB04,
    "05-workflows-and-agents.ipynb": NB05,
    "06-verification-and-evaluation.ipynb": NB06,
    "07-lab-chat-with-your-documents.ipynb": NB07,
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
