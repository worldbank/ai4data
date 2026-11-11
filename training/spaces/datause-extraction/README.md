---
title: Data-use Extraction
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12.12
app_file: app.py
pinned: false
short_description: Small AI that finds datasets in text and how they are used
---

# Data-use extraction · Small AI

A [Small AI](https://worldbank.github.io/ai4data/) demo from the AI for Data – Data for AI program. It finds the datasets that a passage mentions and reads how each named dataset is used. Both models are loaded when the Space starts, so a visitor does not wait for a download. The GPU is attached for the few seconds that a request runs ([ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)).

## Pipeline

```
passage
  → rafmacalaba/gliner_agents_v3            mentions: NAMED_DATA, DESCRIPTIVE_DATA, VAGUE_DATA
  → rafmacalaba/lfm2.5-350M-focus-metadata  for each named mention, in its passage chunk:
                                             attributes (tags, geography, year, version, producer, acronym, data type)
                                             usage (role, usage action, one-sentence summary)
```

| Step | Model | Base model |
|---|---|---|
| Mentions | [`rafmacalaba/gliner_agents_v3`](https://huggingface.co/rafmacalaba/gliner_agents_v3) | `urchade/gliner_large-v2.1` |
| Attributes and usage | [`rafmacalaba/lfm2.5-350M-focus-metadata`](https://huggingface.co/rafmacalaba/lfm2.5-350M-focus-metadata) | `LiquidAI/LFM2.5-350M` |

The passage is split into windows of 200 words with an overlap of 50, because GLiNER was fine-tuned on short passages. Overlapping spans keep the one with the higher score. The metadata model reads only named mentions, and it labels a mention that is already known; it does not detect mentions. An attribute value (producer, year, geography, acronym) is shown only when it appears in the passage.

## Privacy

This is a demo. The text is processed on the Space's server and is not stored by the app or sent to another service. The page is not a browser-only tool.

## Limits

- A passage is limited to 4,000 words.
- ZeroGPU gives each visitor a daily GPU quota, which is larger for signed-in users.
- The metrics on the model cards are measured against agent-generated annotations, not human gold, so they show agreement with those annotations.

## Files

`app.py` is the app, `header.html` and `style.css` are the page, `examples.json` holds the example passages, and `requirements.txt` lists the packages. The Space has no binary files, so nothing depends on Git LFS.
