# Data Snapshot

Extract figure and table images from PDFs, then extract validated Schema v1.4
metadata from those images. DocLayout-YOLO is the default extractor. Value
transcription and submission to external services are not implemented.

## Installation

Requires Python 3.11 or newer. Use your preferred Python environment and package
manager. Data Snapshot is part of `ai4data`, with optional extractor and metadata
dependencies.

**Release availability:** `ai4data==0.1.0` does not include these extras. Once a
release containing this module is published, the standard installation is:

```bash
python -m pip install "ai4data[data-snapshot-doclayoutyolo,data-snapshot-metadata]"
```

Until then, install the checkout from the repository root:

```bash
python -m pip install ".[data-snapshot-doclayoutyolo,data-snapshot-metadata]"
```

This builds and installs the package normally. Reinstall after changing the
source. Contributors who want edits to take effect immediately can use the
editable installation described under [Development](#development).

For snapshot extraction only, omit `data-snapshot-metadata`.

Install only the extractor you need. Weights download on the first extractor
run and are cached under `.cache/models/`, relative to the working directory.

| Extractor | Extra | Weights | Inference size |
| --- | --- | --- | ---: |
| `doclayoutyolo` (default) | `data-snapshot-doclayoutyolo` | `juliozhao/DocLayout-YOLO-DocStructBench` | 1024 |
| `yolo11` | `data-snapshot-yolo11` | `Armaggheddon/yolo11-document-layout`, `yolo11m_doc_layout.pt` | 1280 |
| `yolo26` | `data-snapshot-yolo26` | `Armaggheddon/yolo26-document-layout`, `yolo26m_doc_layout.pt` | 1280 |
| `tfid` | `data-snapshot-tfid` | `yifeihu/TF-ID-large` | Model-defined |

The `data-snapshot` extra supplies rendering and schema dependencies without a
model runtime or metadata API SDK. `ai4data[all]` includes DocLayout-YOLO and
snapshot metadata extraction; it excludes TF-ID.

### Metadata API setup

Required only for end-to-end and metadata-only runs. Set the appropriate key
in the environment or a `.env` file in the working directory. When running
commands from the repository root, put `.env` in that root, alongside
`pyproject.toml`. It is Git-ignored:

```dotenv
OPENROUTER_API_KEY=your-key
OPENAI_API_KEY=your-key
```

Only the selected provider's key is required. Existing environment variables
take precedence over `.env`. OpenRouter is the default; select OpenAI with
`--metadata-provider=openai`. Both packaged presets use GPT-5.6 Luna through the
Responses API. Requests consume provider credits; automatic SDK retries are
disabled. Snapshot extraction alone makes no metadata API requests.

### TF-ID installation

**Treat TF-ID as incompatible with the rest of the repository's model stack.**
It pins `transformers==4.46.0` and `huggingface-hub<1`; installing it alongside
other capabilities can downgrade shared libraries and break those workflows.
Install it in its own environment, without `all` or other capability extras:

```bash
python -m venv .venv-tfid
.venv-tfid/bin/python -m pip install ".[data-snapshot-tfid]"
.venv-tfid/bin/python -m ai4data.data_snapshot.extract_data_snapshot \
  --model=tfid --input=data/example.pdf
```

For end-to-end TF-ID runs, also install `data-snapshot-metadata` in that same
environment. TF-ID executes model-provided code at the pinned revision
`a91ffef47039768b2deff82681b75d77252703f2`.

## Commands

After installation, use Python from that environment. Paths are relative to
the working directory; inputs can also be absolute paths. Add `--help` to any
command for options.

### End-to-end

Extract snapshots, then metadata:

```bash
python -m ai4data.data_snapshot --input=data/example.pdf
```

### Snapshot-only

```bash
python -m ai4data.data_snapshot.extract_data_snapshot --input=data/example.pdf
```

Both PDF commands accept a PDF or a directory searched recursively for PDFs.
Use `--model=yolo11`, `--model=yolo26`, or `--model=tfid` to select another
installed extractor. Exactly one extractor runs per invocation.

### Metadata-only

Fill in missing metadata without rerunning the PDF extractor:

```bash
python -m ai4data.data_snapshot.extract_metadata \
  --input=output/doclayoutyolo/snapshots
```

Accepts a PNG, JPEG, GIF, or WebP image, or a directory searched recursively for
images. A `snapshots` input directory, or an image directly inside one, defaults
to its sibling `metadata` directory. For other locations, supply `--output-dir`:

```bash
python -m ai4data.data_snapshot.extract_metadata \
  --input=path/to/snapshot.png --output-dir=path/to/metadata \
  --metadata-provider=openai
```

### Outputs and reruns

PDF commands default to `output/<model>/`, with `snapshots/` and `metadata/`
subdirectories. `--output-dir` changes the root. For `example.pdf`:

```text
output/doclayoutyolo/
├── snapshots/example_figure_000.png
├── snapshots/example_table_000.png
├── metadata/example_figure_000.json
└── metadata/example_table_000.json
```

Counters start at zero and are independent for figures and tables. PDF stems
must be unique within a run, including across subdirectories. Metadata is
validated against the canonical Pydantic Schema v1.4 model. Requests use only
image-visible evidence, excluding filenames and source-document metadata.

| Command | Default | With `--force` / `-f` |
| --- | --- | --- |
| End-to-end | Skip a PDF if any matching snapshot or metadata exists, including partial output. | Delete matching snapshots and metadata; rerun both stages. |
| Snapshot-only | Same PDF-level skip behavior. | Delete matching snapshots and metadata; extract snapshots only. |
| Metadata-only | Skip images with an existing JSON output. | Delete and regenerate JSON; preserve images. |

**Forced replacement deletes existing files before extraction starts. Failure
does not restore them.** Use metadata-only to fill missing JSON after a failed
metadata stage. Fully skipped PDF runs do not load the extractor.

## Python API

```python
from ai4data.data_snapshot import run_extraction, run_pipeline

# DocLayout-YOLO is the default; snapshot-only requires no metadata API key.
snapshots = run_extraction("data/example.pdf", "output")

# Use fresh input/output or the metadata-only command after snapshot-only runs:
paths = run_pipeline("data/another.pdf", "output", metadata_provider="openai")
```

## Development

Install `data-snapshot-metadata` and `pytest` to run the offline tests; model
runtimes and weights are unnecessary. From the repository root, use an editable
installation so source changes take effect without reinstalling:

```bash
python -m pip install -e ".[data-snapshot-metadata]" pytest
python -m pytest tests/data_snapshot -q
```

Ported from `ajdajd/data-snapshot-pipeline` at
`b12c27780a5082d4a96860de868da9af6b03c094`. This namespace replaces
`data_snapshot_pipeline`; no old import or command aliases are provided.
The [experimental package](../../../experimental/data-snapshot/README.md)
remains the separate research and evaluation implementation.
