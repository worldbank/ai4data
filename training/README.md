# Training series

Materials for the training series of the AI for Data – Data for AI program: session plans, slide decks, notebooks, worked-example assets, demonstration scripts, and participant resources. Each session has its own folder, and that folder holds all of the session's materials.

| Session | Title | Format |
|---|---|---|
| 01 | [Building AI applications for research and data](session-01-building-ai-applications/README.md) | 90 minutes, slides and guided notebooks, plus 15 to 30 minutes of questions |
| 02 | Building specialized small AI using synthetic data | Planned |
| 03 | Evaluating AI systems | Planned |
| 04 | Applying AI to metadata curation and data quality | Planned |
| 05 | Building trustworthy AI applications | Planned |
| 06 | All about context: Why AI is really a search problem | Planned |

### Session 01 notebooks

They open in Google Colab from the `main` branch of `worldbank/ai4data`, and each ends with a link to the next one. For the lab (07), choose a T4 GPU runtime before you run it; the notebook explains how.

| Notebook | Open in Colab |
|---|---|
| [01 · From a language model to an application](session-01-building-ai-applications/notebooks/01-from-model-to-application.ipynb) | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/01-from-model-to-application.ipynb) |
| [02 · Preparing knowledge for an application](session-01-building-ai-applications/notebooks/02-preparing-knowledge.ipynb) | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/02-preparing-knowledge.ipynb) |
| [03 · Retrieval-augmented generation](session-01-building-ai-applications/notebooks/03-retrieval-augmented-generation.ipynb) | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/03-retrieval-augmented-generation.ipynb) |
| [04 · Tools and structured outputs](session-01-building-ai-applications/notebooks/04-tools-and-structured-outputs.ipynb) | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/04-tools-and-structured-outputs.ipynb) |
| [05 · Workflows and agents](session-01-building-ai-applications/notebooks/05-workflows-and-agents.ipynb) | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/05-workflows-and-agents.ipynb) |
| [06 · Verification](session-01-building-ai-applications/notebooks/06-verification-and-evaluation.ipynb) | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/06-verification-and-evaluation.ipynb) |
| [07 · Lab: chat with your documents](session-01-building-ai-applications/notebooks/07-lab-chat-with-your-documents.ipynb) | [Open](https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-01-building-ai-applications/notebooks/07-lab-chat-with-your-documents.ipynb) |

## Layout of a session folder

```
session-NN-short-name/
  README.md        session plan: audience, objectives, timing, what to prepare, definition of ready
  slides/          the deck (.pptx) and its outline (outline.md), which is the reviewable text of every slide
  notebooks/       the notebooks, a shared module they import, and the script that regenerates them
  assets/          the worked example: data, metadata, documents, questions, recorded model outputs
  demo/            scripts for live demonstrations, with pretested inputs and fallbacks
  resources/       what participants take home: setup guide, glossary, exercises, next steps
```

## Conventions

- **One worked example per session.** Every notebook and every slide of a session uses the same question, the same data, and the same documents, so that what changes between steps is the capability being added.
- **Runnable without credentials.** Every notebook runs in a fresh Colab environment with no API key. Model calls fall back to recorded outputs that are labelled as such; everything else, from the vector store to the MCP server, runs for real. A live key, when available, is read from the environment and never written into a notebook.
- **Popular libraries over custom code.** Sessions use the libraries participants will meet in their own work (LangChain, LangGraph, FastMCP, Pydantic, and the program's own packages such as `pcn-core`), and write custom code only for what those do not provide.
- **Real data, labelled.** Statistical values come from a named source with the retrieval date recorded. Documents written for a workshop say so in their header, state where their figures come from, and are never presented as publications.
- **Cached outputs.** Notebooks are committed with their outputs so that reviewers and participants can read them without running them. The regeneration script reruns them in fallback mode.
- **No names in the materials.** Authors and reviewers are recorded in the session README under roles until the material is released.
- **No confidential documents.** The corpus of a session is public or written for the session.
- **House style.** The writing rules of the [cookbook authoring guide](../cookbook/authoring/index.md) apply to slides, notebooks, and resources: descriptive titles, plain statements, no slogans, no antithesis as a habit.

## Review

A session is ready for delivery when its definition-of-ready checklist in the session README is complete. Reviewers open an issue or a pull request against the session folder. The outline file of the deck and the notebooks' Markdown cells are the text to review; the `.pptx` and the `.ipynb` outputs are regenerated from them.

## Adding a session

Copy the layout above, write the session README first, agree the worked example, then build the notebooks and the deck against it.

The Training section of the docs site is generated from these folders by `scripts/docs/build_training_site.py` at every site build. A folder named `session-NN-short-name` appears there as session NN, with its README as the overview page, its `preface.md` as the preface, its numbered notebooks as pages with their committed outputs, and the participant resources and the assets README as further pages.
