# Training series

Materials for the training series of the AI for Data – Data for AI program: session plans, slide decks, notebooks, worked-example assets, demonstration scripts, and participant resources. Each session lives in its own folder and is self-contained.

| Session | Title | Format | Status |
|---|---|---|---|
| 01 | [Building AI applications for research and data](session-01-building-ai-applications/README.md) | 90 minutes, slides and guided notebooks, plus 15 to 30 minutes of questions | Draft for team review |

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
- **Runnable without credentials.** Every notebook runs in a fresh Colab environment with no API key. Model calls fall back to recorded outputs that are labelled as such; retrieval falls back to a local method. A live key, when available, is read from the environment and never written into a notebook.
- **Real data, labelled.** Statistical values come from a named source with the retrieval date recorded. Documents written for a workshop say so in their header, state where their figures come from, and are never presented as publications.
- **Cached outputs.** Notebooks are committed with their outputs so that reviewers and participants can read them without running them. The regeneration script reruns them in fallback mode.
- **No names in the materials.** Authors and reviewers are recorded in the session README under roles until the material is released.
- **No confidential documents.** The corpus of a session is public or written for the session.
- **House style.** The writing rules in the repository's `CLAUDE.md` apply to slides, notebooks, and resources: descriptive titles, plain statements, no slogans, no antithesis as a habit.

## Review

A session is ready for delivery when its definition-of-ready checklist in the session README is complete. Reviewers open an issue or a pull request against the session folder. The outline file of the deck and the notebooks' Markdown cells are the text to review; the `.pptx` and the `.ipynb` outputs are regenerated from them.

## Adding a session

Copy the layout above, write the session README first, agree the worked example, then build the notebooks and the deck against it.
