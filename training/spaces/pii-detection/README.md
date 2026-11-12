---
title: PII Detection
emoji: 🕵️
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
header: mini
pinned: false
short_description: Small AI that finds personal data in 16 languages, CPU only
---

# PII Detection · Small AI

A [Small AI](https://worldbank.github.io/ai4data/) demo from the AI for Data – Data for AI program. It finds 40 kinds of personal data in text in 16 languages and redacts them. The model has 354M parameters and runs on a basic CPU, with no GPU.

The model is [`LiquidAI/LFM2.5-Encoder-350M-PII-Detector`](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-PII-Detector): a bidirectional LFM2.5 encoder with a token-classification head.

## Privacy

This is a demo. The model runs on the Space's server, not in your browser. The page sends your text to `/api/detect`, the server finds the PII, and the spans come back. The app does not write the text to disk or to logs and does not call any other service.

## How it works

- `pii.py` loads the model through its custom modeling code (`trust_remote_code`) and runs the model's own hybrid regex decode (`pii_hybrid_decode.py` and `context_cued.py`, fetched from the model repo). The token classifier locates PII, a validator-gated regex layer owns format-bound types such as email, IBAN, credit card and JWT, and cue-gated rules recover IDs that have no structure. Each span gets a domain, a label and a redaction tag.
- `app.py` is a Flask server. It loads the model once, serves the page, and answers `POST /api/detect {text}` with `{spans, redacted, count, ms}`.
- `index.html` and `style.css` are the page: a highlighted or redacted view, a legend by domain, and detection statistics.

The model runs in f32 on the CPU, with threads matched to the container's CPU quota. Half precision has no fast path on x86, and int8 changes the token-classification boundaries.

## Origin and license

This Space started as a copy of [LiquidAI/pii-detection](https://huggingface.co/spaces/LiquidAI/pii-detection). The model and the decode scripts are Liquid AI's and fall under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-PII-Detector/blob/main/LICENSE). The page, branding and documentation are the program's. The source is in the [ai4data repository](https://github.com/worldbank/ai4data/tree/main/training/spaces/pii-detection).
