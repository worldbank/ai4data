"""Two-stage batch extraction over many PDFs (server use).

Stage 1 (``parse``): CPU only. Fetch/read PDFs and write page-delimited
markdown, one ``.md`` per document, with a metadata row per document.

Stage 2 (``extract``): GPU consumer. Chunk the markdown, run GLiNER
(``gliner_agents_v3``) for mentions and the LFM metadata model for usage and
attributes, and write traceable JSONL outputs.

Run with ``python -m ai4data.data_use.batch {parse,extract,run}``. See
``docs/two-stage-batch-extraction.md``.
"""
