# Artifact-Driven Metadata Schema Development

This repository contains the source code and evidence artifacts for
*Artifact-Driven Metadata Schema Development: A Human-in-the-Loop Methodology
with Multimodal LLMs*.

It reproduces the paper's schema-development case study: metadata discovery,
field-profile aggregation, canonical concept synthesis, expert refinement,
concept coverage assessment, operational schema design, and two held-out
validation exercises. Annotation, layout-model benchmarking, and downstream
metadata extraction are intentionally excluded.

## Install

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```shell
uv sync --extra notebooks --extra dev
cp .env.example .env
```

Add an OpenAI API key to `.env` only when rerunning model calls. The checked-in
artifacts support analysis without API access.

## Data

The images and document metadata are stored in
[`ai4data/data-snapshot`](https://huggingface.co/datasets/ai4data/data-snapshot).
The exact paper samples are frozen in `data/manifests/`.

```shell
uv run python scripts/fetch_data.py
```

The fetcher pins Hugging Face dataset revision
`626f3c54ba6d8f6e0ba0795f9e544e24b2650faf`; the CSV manifests select the
paper's exact development and held-out files from that revision.

## Reproduction modes

For exact offline reproduction, use the checked-in files under `artifacts/`
and run the analysis notebooks listed below. This route requires no API key and
recomputes the reported summaries and figures from the retained model outputs.

A fresh model-supported run is a replication attempt, not a byte-identical
reproduction. It requires the source data, an OpenAI API key, and the execution
notebooks or scripts listed below. Model nondeterminism and API changes can
produce different responses; fresh outputs remain under the gitignored
`runs/` directory and do not replace the paper artifacts.

## Repository layout

```text
.
├── artifacts/                        # Checked-in model outputs and analysis evidence
│   ├── appendix_h/
│   ├── discovery/
│   ├── validation1/
│   └── validation2/
├── data/
│   └── manifests/                    # Exact development and held-out selections
├── docs/                             # Human schema-design decisions
├── notebooks/
│   ├── schema_discovery/             # Discovery, profiling, synthesis, and figures
│   ├── schema_validation1/
│   └── schema_validation2/
├── schemas/                          # Frozen v1.1 and v1.1.1 schema documents
├── scripts/
│   └── fetch_data.py                 # Source-data materialization
├── src/
│   └── schema_development/           # Pipeline code, prompts, and configuration
├── supplement/
│   └── appendix_h/                   # Post-study v1.2 comparison
└── tests/                            # Offline unit and artifact-invariant tests
```

## Reproduce the reported analyses

Run notebooks in this order:

1. `notebooks/schema_discovery/2.0-analysis.ipynb`
2. `notebooks/schema_discovery/3.0-initial_profiling.ipynb`
3. `notebooks/schema_discovery/4.1-results_viewer.ipynb`
4. `notebooks/schema_discovery/5.0-canonical_field_frequency_analysis.ipynb`
5. `notebooks/schema_discovery/7.0-eda_for_paper.ipynb`
6. `notebooks/schema_validation1/3.0-schema_validation1_analysis.ipynb`
7. `notebooks/schema_validation1/3.1-schema_validation1_viewer.ipynb`
8. `notebooks/schema_validation2/2.0-schema_validation2_analysis.ipynb`

These notebooks read the checked-in files under `artifacts/`. The raw
Responses API outputs are retained because rerunning an LLM is not expected to
produce byte-identical results.

## Expected evidence

| Workflow boundary | Checked-in evidence | Expected count |
| --- | --- | ---: |
| Development sample | `data/manifests/development_snapshots.csv` | 210 snapshots |
| Held-out sample | `data/manifests/heldout_snapshots.csv` | 202 snapshots |
| Discovery | `artifacts/discovery/2.0-discovery_results.csv` | 3,041 observations |
| Field profiling | `artifacts/discovery/3.0-field_profiles.csv` | 833 profiles |
| Concept synthesis | `artifacts/discovery/3.1-ontology_v0.md` | 70 concepts |
| Human refinement | `artifacts/discovery/3.1-ontology_v1.csv` | 41 retained concepts |
| Operational design | `schemas/Data Snapshot Metadata Schema v1.1.1.md` | 36 fields |
| Validation 1 | `artifacts/validation1/results.jsonl` | 202 results |
| Validation 2 | `artifacts/validation2/results.jsonl` | 202 results |
| Appendix H | `artifacts/appendix_h/results.jsonl` | 202 results |

## Rerun model-supported stages

The execution notebooks are:

- `notebooks/schema_discovery/1.0-schema_discovery.ipynb`
- `src/schema_development/concept_synthesis.py`
- `notebooks/schema_discovery/4.0-field_labeling_using_ontology.ipynb`
- `notebooks/schema_validation1/2.0-schema_validation1.ipynb`
- `notebooks/schema_validation2/1.0-schema_validation2.ipynb`

Fresh outputs are written under `runs/`, which is gitignored. Model names,
reasoning effort, prompts, frozen schemas, and Structured Output contracts are
preserved, but API availability and nondeterminism may affect a fresh run.

## Human decisions

LLM outputs are candidates, not schema decisions. The refined concept inventory
is `artifacts/discovery/3.1-ontology_v1.csv`; the adjudicated profile mapping is
`artifacts/discovery/4.2-final_labeled_field_profiles.csv`; and the validation
reports record why proposed gaps were accepted, rejected, or treated as
representation limitations. See `docs/schema_design_decisions.md` for the
41-concept to 36-field operational design step.

## Appendix H

`supplement/appendix_h/` reproduces the paper's post-study comparison with the
later machine-readable v1.2 schema. It is supplemental and does not alter the
main case-study evidence. Its runner uses
`artifacts/appendix_h/evaluated_schema_v1.2.schema.json`, the exact pre-amendment
contract used by that run.

The package and notebook names retain `validation3` as a historical internal
label for this Appendix H post-study comparison. Canonical prompts and
configuration live under `src/schema_development/validation3/`.

## Tests

```shell
uv run pytest -q
```

## License

Repository code is released under the included MIT License with the World Bank
IGO rider. The source images and parent-document metadata are distributed
separately through `ai4data/data-snapshot` and are not relicensed here. The
study draws from UNHCR, World Bank Policy Research Working Paper, and World Bank
Refugee corpora; consult the dataset card and each source document's metadata
for applicable attribution and reuse terms.
