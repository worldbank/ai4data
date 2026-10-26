# Two-stage batch extraction (CPU parse, GPU extract)

Processes a directory of PDFs, thousands of them, in two stages that only talk
through files on disk:

```
PDF directory (or manifest.jsonl)
   │  stage 1  parse    CPU, one PDF per process, no model imported
   ▼
parsed/<config>/<corpus_id>.md      page-delimited markdown, one per document
parsed/<config>.jsonl               manifest: sha256, pages, parser versions, metadata
parsed/parse_checkpoint.jsonl       done / failed
   │  stage 2  extract  GPU consumer: polls the manifest, takes new .md files
   ▼
output/extractions.jsonl            one row per chunk, GLiNER mentions (playground-compatible)
output/enrichment.jsonl             LFM usage + attributes per named mention
output/datasets.jsonl               one deduplicated record per document
output/checkpoint.jsonl             done / failed per document (written last)
output/run.json                     models + revisions, parameters, library versions, run history
output/events.jsonl                 one event per document/batch (run_id, timings, errors)
output/logs/<run_id>.log            the console log
```

Code: `src/ai4data/data_use/batch/`. It reuses `deduplicate_extraction`, the
schema vocabularies and the Call 2 field builder of the single-text path
(`DatasetExtractor`), and ports the chunking and footnote linking of
`ai4data-playground/data_use`.

## What each stage does

**Stage 1, `parse`.** Spawned worker processes (one PDF each; pymupdf4llm is
CPU-bound, so processes, not threads) run `pymupdf4llm` per page and write
`<corpus_id>.md` atomically. A per-document timeout (`--timeout`, default 600 s)
keeps a bad PDF from stalling a worker. A PDF with no extractable text fails as
`EMPTY` (OCR is not run). `pdf_url` rows in a manifest are downloaded first.
Workers import neither torch nor any model.

**Stage 2, `extract`.** One process owns the GPU. Each loop it re-reads the
manifest, takes up to `--batch-docs` documents whose `.md` exists and that are
not `done`, then:

1. chunks every page in GLiNER's own 384-word window (50 overlap) and links
   footnotes to their markers;
2. runs **GLiNER** (`gliner_agents_v3`) over all chunks of the batch in
   `--extract-batch-size` forward passes, and drops duplicate spans from
   overlapping chunks;
3. runs the **LFM** (`lfm2.5-350M-focus-metadata`) over every named mention, batched
   (`--lfm-batch-size`), once for `usage` and once for `attributes`. The passage is the
   whole chunk;
4. writes the document's rows to the three JSONL files, then its checkpoint.

When nothing is ready it sleeps `--poll-interval` seconds and checks again. It
exits when stage 1 has finished and nothing is left (`parsed/.parse_complete`),
or after `--idle-exit` seconds of waiting. Both stages can run at the same time
(`run`), or separately, even on different machines that share the directory.

## Running it on the H100 server

### Setup (once)

```bash
git clone <repo> ai4data && cd ai4data
uv sync --extra datause          # torch, transformers>=5, gliner, pymupdf4llm<1, ...
# make sure torch is a CUDA build:
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
export HF_HOME=/data/hf_cache    # model cache; both models download on first use (~2 GB)
# export HF_TOKEN=...            # only if the two model repos are private
```

### Run

```bash
export CUDA_VISIBLE_DEVICES=0    # the GPU to use
nohup uv run python -m ai4data.data_use.batch run \
    --source /data/pdfs \
    --work-dir /data/run1 \
    --device cuda --require-gpu \
    --workers 32 \
    --extract-batch-size 64 --lfm-batch-size 32 \
    > /data/run1.out 2>&1 &
```

- `--require-gpu` aborts at start if either model would land on the CPU, so a
  misconfigured server fails in seconds, not hours later.
- Start with a **smoke test**: add `--limit 20` and a throwaway `--work-dir`.
  Check the `[gpu]` line and the first `[batch]` lines, then run for real.
- To run the stages separately:
  ```bash
  python -m ai4data.data_use.batch parse   --source /data/pdfs --work-dir /data/run1 --workers 32
  python -m ai4data.data_use.batch extract --work-dir /data/run1 --device cuda --require-gpu --idle-exit 0
  ```
  `--idle-exit 0` makes `extract` wait until stage 1 writes its completion marker,
  so it can be started first.

### Is the GPU used?

Only the extraction stage uses the GPU. At start the log says so:

```
[gpu] NVIDIA H100 80GB HBM3 80 GB | models on {'entity': 'cuda:0', 'metadata': 'cuda:0'} | torch 2.x cuda 12.x
[batch] docs=8 chunks=410 prep=0s gliner=3s (130 chunks/s) lfm=5s (212 mentions) gpu_peak=9.8GB
```

Watch `nvidia-smi dmon` or `watch -n1 nvidia-smi` next to the log. Tuning:

| symptom | what to change |
|---|---|
| GPU idle, `[idle]`/waiting lines, `status` shows `waiting: 0` | parsing is the bottleneck: raise `--workers`, or run `parse` on a larger CPU machine |
| `status` `waiting` keeps growing | extraction is the bottleneck: raise `--extract-batch-size` and `--lfm-batch-size` while `gpu_peak` leaves headroom |
| CUDA out of memory | the batch is retried one document at a time; lower the two batch sizes |
| `prep` time large | chunking/footnotes are CPU work done on the GPU process; raise `--batch-docs` to amortize |

The H100 numbers are not measured yet: the pipeline was tested end to end on an
Apple GPU (MPS) only (about 10 GLiNER chunks/s there). Measure the smoke test
before estimating the full run.

### Monitor

```bash
tail -f /data/run1/output/logs/*.log                       # one line per document, docs/min, ETA
python -m ai4data.data_use.batch status --work-dir /data/run1   # counts, row totals, failures
```

### Stop, resume, retry

- Re-run the **same command**. Documents with a `done` checkpoint are skipped.
  On start, rows of documents without a `done` checkpoint (a crash mid-write) are
  removed, so nothing is duplicated.
- `--retry-failed` re-parses documents that failed in stage 1. Documents that failed
  in stage 2 are retried on the next run automatically. The reason is in
  `status` and in `events.jsonl`.
- `--limit N` means "at most N *pending* documents in this run".
- `--no-resume` deletes the outputs of the work dir and starts over.
- Resuming with different settings (models, chunk size, threshold, ...) is refused
  because `run.json` stamps them; use a fresh `--output-dir`.
- Run **one** `extract` per output directory. For several GPUs, split the PDFs into
  subdirectories and run one process per GPU, each with its own `--work-dir`.

## Inputs and discovery

A source directory is walked at every folder level; `.pdf` is matched in any case, and `.zip`, `.tar*`,
`.tgz` and `.pdf.gz` archives are unpacked (nested down to `--max-archive-depth`, default 2) into
`parsed/unpacked/`. Identical PDFs are one document; hidden files, non-PDF files, unsupported archives (7z, rar),
corrupt archives and unsafe archive members are skipped **and reported** in `parsed/inventory.json`. Preview a tree
with `scan`:

```bash
python -m ai4data.data_use.batch scan --source /data/pdfs --work-dir /data/run1
python scripts/data_use/make_messy_corpus.py /tmp/messy   # a corpus with every case, for testing
```

The full table of cases and the report format are in `src/ai4data/data_use/README.md`. A `.jsonl` manifest
(`pdf_url` or `path`, optional `title`, extra columns kept as `metadata`) is also accepted, and `pdf_url` rows are
downloaded in stage 1. `corpus_id` is `<config>:<sha256[:16]>` of the PDF bytes (of the URL for downloads), so adding
files never changes the ids of existing ones; `--config` names the corpus (default `local`). Archive members keep
their provenance (`pdf_url` with `!/`, plus `origin`, `container`, `archive_depth` in the manifest metadata).

## Output contract and traceability

| file | key | notes |
|---|---|---|
| `extractions.jsonl` | `corpus_id, page, chunk` | `corpus_id, config, pdf_url, title, page, chunk, input_text, has_data_score, entities[{text,label,score,start,end}]`. Same shape as the playground's `extractions.jsonl`; offsets are relative to `input_text`; `page` is the 0-based PDF page |
| `enrichment.jsonl` | `corpus_id, page, chunk, start, end` | joins to the matching entity; usage (`usage_context, usage_action, usage_summary, is_used, purpose_action`), `theme_tags`, `source_attributes`, plus the raw model JSON |
| `datasets.jsonl` | `corpus_id` | per document: `datasets` = `deduplicate_extraction` output (canonical names, counts, pages, occurrences, merged Call 2 fields) |
| `checkpoint.jsonl` | `corpus_id` | `done` / `failed` + error |

From any row: `corpus_id` → `parsed/<config>.jsonl` (PDF path/URL, sha256, parser
versions, `.md` file) → `output/run.json` (model revisions and parameters of every run)
→ `events.jsonl` (`run_id`, timings, errors).

`--threshold` (GLiNER score, default 0.5), `--chunk-unit words|tokens`
(`tokens` reproduces the playground's subword chunks, which can exceed GLiNER's
384-word window and lose their tail), `--no-enrich` (skip the LFM),
`--no-footnote-link`, `--no-dedupe-overlap` change what is extracted.

## Limitations

- No OCR: scanned PDFs fail as `EMPTY`.
- A native crash of a parse worker breaks the pool and ends stage 1 early (a re-run continues).
- No lock against two `extract` processes on one output directory.
- No SIGTERM handler: a kill loses the current batch (resume cleans up after it).
- Startup scans the output files and hashes PDFs once (cached in `parsed/hash_cache.json`); the first run over thousands
  of PDFs takes minutes.
- The LFM output is agreement with agent labels, not human gold (see its model card),
  and chart-source lines such as "Survey data" are still extracted as mentions.
