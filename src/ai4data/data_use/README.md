# Data Use

Extracts **dataset mentions** from text and PDFs, and says how each mention is
used. Two ways to use it:

| | for | entry point |
|---|---|---|
| **Single text / one PDF** | notebooks, services, a few documents | `extract_from_text`, `DatasetExtractor.extract_from_document` |
| **Batch of PDFs (server)** | thousands of PDFs, CPU parse + GPU extract, resumable | `python -m ai4data.data_use.batch` |

Both use the same two models:

| step | model | what it does |
|---|---|---|
| mentions | [`rafmacalaba/gliner_agents_v3`](https://huggingface.co/rafmacalaba/gliner_agents_v3) (GLiNER) | spans labelled `NAMED_DATA` (a named source), `DESCRIPTIVE_DATA` (described, not named), `VAGUE_DATA` |
| usage + attributes | [`rafmacalaba/lfm2.5-350M-focus-metadata`](https://huggingface.co/rafmacalaba/lfm2.5-350M-focus-metadata) (LFM2.5) | for each **named** mention: `role` (primary / supporting / background), `usage_action` (source / analyze / validate / curate / inform), a one-sentence `summary`, `theme_tags`, and `source_attributes` (geography, year, version, producer, acronym, data_type) |

The LFM model labels a mention it is given; it does not find mentions. It reads
the whole chunk the mention is in, as it was trained on passage chunks. Its output
is agreement with agent labels, not human gold (see the model card).

## Install

```bash
uv pip install "ai4data[datause]"      # torch, transformers>=5, gliner, pymupdf4llm<1, ...
```

`transformers>=5` is required (the LFM tokenizer does not load on 4.x).

## Single text or one PDF

```python
from ai4data.data_use import extract_from_text, deduplicate_extraction

out = extract_from_text("We use the 2015 Kenya Integrated Household Budget Survey (KIHBS).",
                        use_classifier=False)
for d in out["datasets"]:
    print(d["mention_name"]["text"], d["specificity_tag"], d["usage_context"]["text"],
          d["usage_action"]["text"], d["theme_tags"], d["source_attributes"])
```

```python
from ai4data.data_use import DatasetExtractor
ex = DatasetExtractor(relation_adapter_id="")           # see "Legacy parts" below
pages = ex.extract_from_document("report.pdf", use_classifier=False, parallel=False)
merged = deduplicate_extraction(pages)                   # one record per dataset
```

Model ids can be changed with `entity_model_id=` and `metadata_model_id=`.

### Output fields of a mention

`mention_name`, `specificity_tag` (`named` / `descriptive` / `vague`) and, for **named** mentions only:
`usage_context` (role), `is_used`, `usage_action`, `usage_summary`, `purpose_action`,
`theme_tags`, `source_attributes`. Descriptive and vague mentions carry empty values for
these. The vocabularies and the mapping to the legacy `purpose_action` labels are in
`schemas/dataset_schema.py`. `deduplicate_extraction` merges the new fields (most frequent
`usage_action`, union of `theme_tags`, most frequent non-null `source_attributes` value per key).

## Batch of PDFs on a GPU server (two stages)

```
PDF directory ──► stage 1: parse (CPU, one PDF per process) ──► parsed/*.md + manifest
                                                                      │  polled
                  stage 2: extract (GPU, one process) ◄───────────────┘
                  chunk → GLiNER → LFM → extractions / enrichment / datasets
```

The stages only share files on disk. Stage 2 waits for new `.md` files, so both can run at
the same time (`run`) or separately, even on different machines sharing the directory.
More detail and the output contract: `docs/data_use/two-stage-batch-extraction.md`.

### Run it on the H100 server

Setup, once:

```bash
git clone <repo> ai4data && cd ai4data
uv sync --extra datause
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must be True
export HF_HOME=/data/hf_cache        # model cache; both models (~2 GB) download on first use
export CUDA_VISIBLE_DEVICES=0        # the GPU to use
# export HF_TOKEN=...                # only if the model repos are private
```

1. **Smoke test** on a small slice with a throwaway work dir. Check the `[gpu]` line and the first
   `[batch]` lines, and note docs/min for the estimate of the full run.
   ```bash
   uv run python -m ai4data.data_use.batch run \
       --source /data/pdfs --work-dir /data/smoke --device cuda --require-gpu --limit 20
   ```
2. **Full run**, detached:
   ```bash
   nohup uv run python -m ai4data.data_use.batch run \
       --source /data/pdfs --work-dir /data/run1 \
       --device cuda --require-gpu \
       --workers 32 --extract-batch-size 64 --lfm-batch-size 32 \
       > /data/run1.out 2>&1 &
   ```
3. **Monitor**:
   ```bash
   tail -f /data/run1/output/logs/*.log
   python -m ai4data.data_use.batch status --work-dir /data/run1
   watch -n1 nvidia-smi
   ```
4. **Stop / resume**: run the same command again. Documents with a `done` checkpoint are skipped;
   partial rows of a document that was mid-write are removed on start.
5. **Retry** documents whose parse failed: add `--retry-failed`. Documents that failed in stage 2
   are retried automatically on the next run.

Separate stages (for example parse on a big CPU box, extract on the GPU box):

```bash
python -m ai4data.data_use.batch parse   --source /data/pdfs --work-dir /data/run1 --workers 32
python -m ai4data.data_use.batch extract --work-dir /data/run1 --device cuda --require-gpu --idle-exit 0
```

`--idle-exit 0` makes `extract` wait until stage 1 has finished, so it can be started first.
Without it, `extract` exits after 600 s with nothing to do and no completion marker.

### What uses the GPU

Only stage 2. GLiNER runs over all chunks of a batch (`--extract-batch-size` per forward pass)
and the LFM over all named mentions (`--lfm-batch-size` prompts per `generate`). Both models are on
the device; `--require-gpu` aborts at start if one is on the CPU. The log shows it:

```
[gpu] NVIDIA H100 80GB HBM3 80 GB | models on {'entity': 'cuda:0', 'metadata': 'cuda:0'} | torch ... cuda ...
[batch] docs=8 chunks=410 prep=0s gliner=3s (130 chunks/s) lfm=5s (212 mentions) gpu_peak=9.8GB
```

(Illustrative numbers. The pipeline was tested end to end on an Apple GPU, about 10 GLiNER chunks/s;
H100 throughput is not measured yet, so use the smoke test.)

| symptom | change |
|---|---|
| GPU idle, `[idle]` waits, `status` shows `waiting: 0` | parsing is the bottleneck: more `--workers`, or run `parse` on a bigger CPU machine |
| `status` `waiting` keeps growing | extraction is the bottleneck: raise `--extract-batch-size` and `--lfm-batch-size` while `gpu_peak` leaves headroom |
| CUDA out of memory | the batch is retried one document at a time; lower the batch sizes |
| large `prep` time | chunking and footnote linking run on the GPU process; raise `--batch-docs` |

### Inputs and discovery

The source is a directory tree or a `.jsonl` manifest.

**A directory** is walked at **every folder level** and flattened to one list of documents (sorted by path). What it finds:

| in the tree | what happens |
|---|---|
| `.pdf` in any case (`REPORT.PDF`), any depth, accents/spaces in names | processed |
| `.zip`, `.tar`, `.tar.gz`/`.tgz`/`.tar.bz2`/`.tar.xz`, single-file `.pdf.gz` | unpacked to `<work-dir>/parsed/unpacked/<sha256[:16]>/` (cached; a restart reuses it), their PDFs processed. Archives inside archives are opened down to `--max-archive-depth` (default 2); deeper ones are reported, not opened. `--no-archives` leaves archives alone |
| identical PDFs (same bytes, any folder or archive) | one document. The copy outside an archive wins; the dropped copies are logged and listed in `inventory.json` |
| `txt`, `csv`, `docx`, other files | ignored, counted by extension |
| hidden files and folders (`.x`), `__MACOSX`, `~$` temp files | skipped, counted |
| symlinked folders | not followed |
| `.7z`, `.rar`, `.zst`, bare `.bz2`/`.xz` | reported as unsupported, not opened |
| corrupt archive, encrypted member | reported as a problem; the rest of the tree continues |
| archive members with absolute paths, `..`, links or devices | rejected (never written outside the unpack dir) and reported |
| huge archives | capped (2 GiB per member, 50 GiB total, 200 000 members) and reported |

Nothing is dropped silently: every skipped file or archive is in the **inventory**. Look at it before a big run:

```bash
python -m ai4data.data_use.batch scan --source /data/pdfs --work-dir /data/run1      # prints the report, writes parsed/inventory.json
```
```
folders=17 files=30 hidden/system skipped=4
PDFs found: 16 (direct 8, in archives 8) -> unique 14 after dropping 2 identical copies
archives: {'archive_tar': 1, 'archive_zip': 7, 'archive_gz': 2, 'archive_unsupported': 1}  (deepest nesting seen: 2)
ignored by extension: {'.txt': 2, '.csv': 2, '.docx': 1}
problems: {'corrupt_archive': 1, 'unsafe_member': 2, 'archive_not_opened': 1, 'unsupported_archive': 1}
  - corrupt_archive: broken.zip BadZipFile: File is not a zip file
  - unsafe_member: evil.zip!/../../escape.pdf path escapes the archive; skipped
  ...
```

`scan` already unpacks the archives (into the same cache `parse` uses), and `parse`/`run` write the same
`parsed/inventory.json` and log the report at the start. Each document keeps its provenance in the manifest:
`pdf_url` is the original file's URI with `!/` into archives (`file:///data/pdfs/bundle.zip!/nested.zip!/n1.pdf`), and
`metadata` has `origin` (path under the source), `folder` (first folder level), and for archive members `container`
and `archive_depth`. Each PDF is hashed once and the hashes are cached in `parsed/hash_cache.json` (keyed by path, size and
mtime), so restarts do not re-read thousands of files.

**Try it on a deliberately messy corpus** (every case in the table):

```bash
python scripts/data_use/make_messy_corpus.py /tmp/messy
python -m ai4data.data_use.batch scan --source /tmp/messy --work-dir /tmp/messy_run
python -m ai4data.data_use.batch run  --source /tmp/messy --work-dir /tmp/messy_run
```

**A `.jsonl` manifest**: one row per document with `pdf_url` (URL or file path) or `path`, optional `title`; other
columns are kept as `metadata`. URLs are downloaded in stage 1. A directory may also hold a `manifest.jsonl` with
`{"path": "sub/a.pdf", "title": "..."}` rows to attach a title and metadata to its direct (non-archive) PDFs.

`corpus_id` = `<config>:<sha256[:16]>` of the PDF bytes (of the URL for downloads), so adding files never
changes existing ids. `--config` names the corpus (default `local`).

### Work directory

```
<work-dir>/parsed/<config>/<corpus_id>.md       page-delimited markdown, one per document
<work-dir>/parsed/<config>.jsonl                manifest: path/url, sha256, pages, parser versions, metadata
<work-dir>/parsed/parse_checkpoint.jsonl        done / failed
<work-dir>/parsed/events.jsonl, logs/           stage 1 trace
<work-dir>/output/extractions.jsonl             one row per chunk: GLiNER mentions
<work-dir>/output/enrichment.jsonl              LFM usage + attributes per named mention
<work-dir>/output/datasets.jsonl                one deduplicated record per document
<work-dir>/output/checkpoint.jsonl              done / failed per document (written last)
<work-dir>/output/run.json                      models + revisions, parameters, library versions, run history
<work-dir>/output/events.jsonl, logs/           stage 2 trace
```

`extractions.jsonl` has the same strict shape as the playground's, so its downstream scripts work:
`corpus_id, config, pdf_url, title, page, chunk, input_text, has_data_score, entities[{text,label,score,start,end}]`
(offsets relative to `input_text`; `page` is the 0-based PDF page). `enrichment.jsonl` joins to the entity on
`(corpus_id, page, chunk, start, end)`; `datasets.jsonl` is keyed by `corpus_id`.

**Traceability**: row → `corpus_id` → manifest (PDF path/URL, sha256, `.md`, parser versions) →
`run.json` (model revisions and parameters of every run) → `events.jsonl` (`run_id`, per-document timings, errors).

### Options that change what is extracted

`--threshold` (GLiNER score, default 0.5), `--chunk-tokens` / `--overlap` (384 / 50), `--chunk-unit words|tokens`
(default `words`: GLiNER's real 384-word window; `tokens` reproduces the playground's subword chunks, which can exceed
that window and silently lose their tail), `--no-enrich` (skip the LFM), `--no-footnote-link`,
`--no-dedupe-overlap`. Changing them on an existing output dir is refused (stamped in `run.json`); use a new
`--output-dir`, or `--allow-stamp-mismatch`.
`--limit N` means at most N **pending** documents in this run. `--no-resume` deletes the outputs and starts over.

### Limitations

- No OCR: a PDF without a text layer fails as `EMPTY`.
- Parse workers are a process pool: if one dies natively (segfault, OOM kill) on a PDF, the pool breaks and stage 1 stops early; the
  documents not reached are not checkpointed, so a re-run continues. Not yet handled by rebuilding the pool.
- Encrypted PDFs and password-protected archives fail (reported).
- Run **one** `extract` per output directory (no lock). For several GPUs, split the PDFs into subdirectories and run
  one process per GPU, each with its own `--work-dir`.
- No SIGTERM handler: a kill loses the batch in flight (the next start cleans up).
- Start-up scans the output files (and hashes PDFs once; cached afterwards): minutes for thousands of PDFs on the first run.
- Chart-source lines such as "Survey data" are still extracted as mentions.

## Legacy parts

The page classifier (`use_classifier=True`) and the relation step (`relation_adapter_id`) still run on GLiNER2
adapters (`ai4data/datause-classifier`, `ai4data/datause-relation-v0`) and are **not** part of the batch pipeline.
The single-text path works without them: `use_classifier=False, relation_adapter_id=""`. `example_extraction.py`
still shows the old GLiNER2 adapter.

## Tests

```bash
uv run --extra datause --with pytest python -m pytest tests/test_batch_pipeline.py tests/test_call2_fields.py \
    tests/test_extractors.py tests/test_chunking.py tests/test_schemas.py tests/test_deduplication.py
```

The batch tests use generated PDFs and a fake GLiNER/LFM, so they need no GPU and no download. They cover stage 1
(corrupt and scanned PDFs, resume, retry), stage 2 (strict output shapes and joins, crash recovery without
duplicates, settings stamp, one bad document does not stop the batch), and a stage 2 that waits for a slow stage 1.
`tests/test_e2e_swarm_pipeline.py` loads real models and is marked `slow`.
