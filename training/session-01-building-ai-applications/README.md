# Session 01 · Building AI applications for research and data

90 minutes of slides and guided notebook demonstrations, followed by 15 to 30 minutes of questions.

## Who it is for

Two audiences at once:

- Colleagues who will build AI applications and want to see each capability added to a working application, with code they can rerun.
- Colleagues who commission, oversee, or evaluate such applications and need to know what each capability is for, what it costs, and how to tell whether the result is reliable.

Participants need no prior experience with language-model APIs. The notebooks run without an API key.

## What participants can do afterwards

- Describe the components of an AI application and say which problem each one solves: instructions and context, prepared knowledge, retrieval, structured outputs and tools, workflows and agents, and verification.
- Given a research or data question, choose an architecture and say what evidence would show that it works.
- Rerun the six notebooks, change the question or the documents, and read the verifier's marks and the retrieval checks.

## The worked example

One question runs through the whole session:

> How have internet access and economic development evolved in Rwanda and Kenya? Compare the recent available data, summarize the main patterns, and cite the sources.

The application starts as a model answering from its training data. Notebook by notebook it gains supplied context, a prepared corpus of documents, retrieval, structured outputs, a tool that queries authoritative indicator data, a bounded workflow that chooses between tools, and a verifier that checks every number in its answer against the data. The data are World Development Indicators values for Kenya and Rwanda retrieved from the World Bank API on 13 October 2026; the documents are short briefs written for the workshop from those values, plus one outdated summary and two distractors. See [assets/README.md](assets/README.md).

Four problems are planted so that each capability has a reason to exist: an ambiguous question, an irrelevant retrieval result, a request for data that does not exist, and a numerical claim that no source supports.

## Timing

| Segment | Slides | Demonstration | Minutes | Notebook |
|---|---|---|---|---|
| Introduction and the baseline application | 6 | 4 | 10 | 01 |
| Context, data, and metadata preparation | 7 | 8 | 15 | 01, 02 |
| Retrieval and retrieval-augmented generation | 5 | 10 | 15 | 03 |
| Structured outputs and tools | 5 | 10 | 15 | 04 |
| Workflows and agents | 5 | 10 | 15 | 05 |
| Grounding and verification | 4 | 6 | 10 | 06 |
| Data360 Chat and questions | 5 | 5 | 10 | demo |
| Total | 37 | 53 | 90 | |

Questions are taken at the notebook transitions as well as at the end. Notebooks 02 and 03 are demonstrated from one file to avoid a switch; they are distributed separately.

## Open the notebooks in Colab

| Notebook | Open in Colab |
|---|---|
| 01 · From a language model to an application | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/01-from-model-to-application.ipynb) |
| 02 · Preparing knowledge for an application | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/02-preparing-knowledge.ipynb) |
| 03 · Retrieval-augmented generation | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/03-retrieval-augmented-generation.ipynb) |
| 04 · Tools and structured outputs | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/04-tools-and-structured-outputs.ipynb) |
| 05 · Workflows and agents | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/05-workflows-and-agents.ipynb) |
| 06 · Verification | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/06-verification-and-evaluation.ipynb) |
| 07 · Lab: chat with your documents | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/07-lab-chat-with-your-documents.ipynb) |

Each notebook ends with a link to the next. Notebook 07 runs a small open model, so choose **Runtime > Change runtime type > T4 GPU** before you run it.

The eight documents of the worked example, with their metadata, their role in the session, and the chunks that notebook 02 makes of them, are shown at [worldbank.github.io/ai4data/training/session-01/documents.html](https://worldbank.github.io/ai4data/training/session-01/documents.html). The page is generated from `assets/documents/` by `scripts/docs/build_training_pages.py` when the site is built.

## Materials

| Deliverable | Where | State |
|---|---|---|
| Worked-example assets: question, documents, indicator data and metadata, evaluation questions, recorded model outputs | `assets/` | Drafted |
| Six notebooks with explanatory text, runnable code, cached outputs, and one exercise each | `notebooks/` | Drafted; outputs cached in fallback mode |
| A lab notebook (07) in which participants chat with their own documents, with a small open model that runs in Colab | `notebooks/07-lab-chat-with-your-documents.ipynb` | Drafted; needs a Colab run on a CPU and a GPU runtime |
| Shared module the notebooks import: the worked-example data and tool, a recorded-output chat model, printing and evaluation helpers; the indicator MCP server | `notebooks/workshop.py`, `notebooks/indicator_server.py` | Drafted |
| Slide deck, about 22 slides, with the outline as reviewable text | `slides/` | Drafted |
| Data360 Chat demonstration: pretested queries and a fallback | `demo/` | Drafted; queries to pretest on the live application |
| Participant resources: setup, glossary, test questions, next steps | `resources/` | Drafted |

## Technical choices

Popular libraries wherever one exists, so that what participants learn transfers to their own work:

| Need | Library |
|---|---|
| Documents, splitting, embeddings, vector store, prompts, tools, structured output | LangChain (`langchain-core`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-openai`) |
| The agent loop, with a recursion limit and a trace | LangGraph, through `langchain.agents.create_agent` |
| The tool server and client over the Model Context Protocol | FastMCP |
| Number verification | `pcn-core`, the Python core of the program's Proof-Carrying Numbers |
| Output schemas | Pydantic |
| Chat model | `ChatOpenAI` over any OpenAI-compatible API, configured by `WORKSHOP_API_KEY`, `WORKSHOP_BASE_URL`, `WORKSHOP_MODEL` |

What `workshop.py` adds: the worked-example data and the indicator tool over it (the committed CSV by default, the World Bank API on request), a chat model that replays recorded outputs when no key is configured, and printing and evaluation helpers. Embeddings come from a small sentence-transformer model downloaded on first use. Notebook 04 also lists the tools of the public Data360 MCP server and runs one search, guarded so that the session does not depend on the network.

## Roles

Recorded by role until release: session lead and instructor; notebooks 01, 04, 05; notebooks 02, 03; notebook 06, test harness, packaging, demonstration reliability. One person owns integration across the six notebooks.

## Definition of ready

- [ ] The same question runs through all six notebooks and the deck
- [ ] Every notebook runs in a fresh Colab environment without a key
- [ ] Every live demonstration has a recorded or fallback output
- [ ] At least three failure cases are demonstrated
- [ ] Deck and notebooks use the same terms and the same architecture diagram
- [ ] No participant credentials and no confidential material are required
- [ ] A full 90-minute rehearsal has been completed
- [ ] Team review of the outline and the notebooks' text is complete
