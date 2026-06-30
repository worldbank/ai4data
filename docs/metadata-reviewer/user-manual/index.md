# Metadata Reviewer User Manual

**Package:** `ai4data.metadata.reviewer`

*A multi-agent LLM pipeline for dataset metadata quality assurance.*

Detect typos, inconsistencies, invalid content, and missing information across metadata records — automatically and consistently.

This manual is the operational reference for installing, configuring, and running the Metadata Reviewer. For a shorter overview of the pipeline and design rationale, see the [Metadata Reviewer overview](../overview.md).

## Contents

- [Introduction](introduction.md) — what the package does and how this manual is organized
- [Core Concepts](core-concepts.md) — pipeline, categories, severity, output schema, architecture
- [Installation](installation.md) — requirements and provider extras
- [Quick Start](quick-start.md) — build a client, submit metadata, retrieve results
- [The Client API](client-api.md) — constructor, factories, submission, job management
- [Jobs](jobs.md) — lifecycle, waiting, cancellation, cleanup
- [Advanced Usage](advanced-usage.md) — team presets, custom manifests, exclusion rules
- [AI Suggestion Review Board](review-board.md) — triage pipeline output in the browser
- [End-to-End Workflow](end-to-end-workflow.md) — catalogue review cycle
- [Troubleshooting & FAQ](troubleshooting.md) — common problems and fixes
- [Quick API Reference](appendix-api-reference.md)
- [Glossary](appendix-glossary.md)
- [Default Agents Manifest](appendix-agents-manifest.md)
