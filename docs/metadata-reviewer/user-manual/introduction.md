# Introduction

## What is Metadata Reviewer?

**Metadata Reviewer** is a Python sub-package
(ai4data.metadata.reviewer) that uses a coordinated team of
large-language-model (LLM) agents to scan dataset series metadata for
quality issues. It reads a metadata record, runs it through a five-stage
review pipeline, and returns a ranked list of detected issues — each
with a category, a severity score, the problematic field, and a
suggested correction.

The package is deliberately **provider-agnostic**. The same pipeline
runs unchanged whether the underlying model is hosted by OpenAI,
deployed on Azure OpenAI, served by Anthropic, or running locally
through Ollama. Switching providers is a one-line change at construction
time.

## The problem it solves

Reviewing metadata quality at scale is slow and inconsistent. Manually
checking hundreds of records for typos, conflicting values, invalid
units, and missing required fields is labor-intensive, and human
judgment drifts over time. The process also tends to leave no structured
record of what was checked or why.

A single LLM call lowers the labor cost but does not, on its own, give
you governance, repeatability, or an auditable trail. Metadata Reviewer
addresses these gaps by decomposing the task into specialized,
sequential agents, so that detection, filtering, classification, and
scoring happen as separate, inspectable steps that apply the same rules
to every record.

## Key features

- **Five-agent review pipeline** — two independent detectors, a critic
  that removes false positives, a categorizer, and a severity scorer.

- **Provider-agnostic** — built-in factory methods for OpenAI, Azure
  OpenAI, Anthropic, and Ollama; bring any AutoGen-compatible client of
  your own.

- **Asynchronous, job-based API** — every submission returns a Job
  handle immediately; the pipeline runs in the background while you
  poll, wait, or cancel.

- **Structured, ranked output** — each issue carries a category, a
  1--5 severity score, and a before/after field pair you can act on
  programmatically.

- **Customizable behavior** — supply your own agent manifest (YAML)
  and choose how agents are routed via team presets.

- **A companion review tool** — the browser-based **AI Suggestion
  Review Board** lets a human reviewer triage the detected issues with
  side-by-side diffs.

## How this manual is organized

- [Core Concepts](core-concepts.md) explains the pipeline, categories, and output schema.
- [Installation](installation.md) covers requirements and provider extras.
- [Quick Start](quick-start.md) is a hands-on walkthrough with a complete worked example.
- [The Client API](client-api.md) and [Jobs](jobs.md) are the API reference for submission and job lifecycle.
- [Advanced Usage](advanced-usage.md) covers team presets, custom manifests, and exclusion rules.
- [AI Suggestion Review Board](review-board.md) documents the companion triage tool.
- [End-to-End Workflow](end-to-end-workflow.md) ties the pieces into a catalogue review cycle.
- [Troubleshooting & FAQ](troubleshooting.md) and the appendices follow.
