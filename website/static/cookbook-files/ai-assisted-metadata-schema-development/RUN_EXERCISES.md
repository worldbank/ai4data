# Offline exercises for Chapters 2–3

Use Python 3.10 or newer. The scripts use only the standard library.
Download the scripts and the files named below into a new working directory.
The prepared evidence files retain their original contents. No images or
dataset download, provider credentials, API request, or model execution is needed.

## Inputs and commands

The commands use `python` for a Python 3.10 or newer interpreter. If the local
command is `python3`, substitute it in every command below. No package
installation is required.

Download each row's files into the same new working directory:

| Chapter and exercise | Files to download together | Output |
|---|---|---|
| Chapter 2: response inspection | inspect_discovery.py; discovery_excerpt.jsonl; source_images.csv | The script prints structural checks and writes nothing. |
| Chapter 3: exact-name profiling | profile_observations.py; geographic_observations.json | The script creates outputs/geographic_profiles.json. |

Run these commands from that directory:

    python inspect_discovery.py discovery_excerpt.jsonl source_images.csv
    python profile_observations.py geographic_observations.json --output outputs/geographic_profiles.json

The inspection command reads three selected response excerpts. They contain
four title-related fields from original responses containing 43 fields in total.
The fields, identities, model names, statuses, and original response counts are
recorded research evidence. The excerpt wrapper is newly prepared. Other
fields, raw responses, token accounting, and full input messages are omitted.
This is a structural inspection, not a semantic accuracy assessment.

The profiling command aggregates three unchanged geographic_scope observations
selected from the exact-name-profile example in evidence.json. It writes a
new JSON profile with count 3, n_snapshots 3, total_snapshots 3, and support 1.0.
All three input records are retained as evidence. The full-study profile's
192 observations and the later canonical concept's 194-snapshot support have
different scope and provenance.

## Supplemental sample check

Chapter 1 explains how to separate development and holdout samples without
requiring code. The optional check_manifest.py tool can check the prepared
source_images.csv excerpt. Download both into the same directory and run:

    python check_manifest.py source_images.csv

It reports six development snapshots and two holdout snapshots, with no shared
snapshot or parent report in this eight-item excerpt. It does not recompute the
full-study counts of 210 development snapshots, 202 holdout snapshots, or 65 shared
parent reports. This supplement is not a required chapter exercise.

## Output safety and verification

The response inspector and supplemental sample checker write nothing. The
profiling script creates its output with exclusive creation and refuses to replace an existing file. Use a new
output filename for a repeat run. Invalid input is checked before an output
directory is created. Exit status 2 indicates invalid input or a path error;
the response inspector returns 1 when selected records contain logged failures.

Inspect the generated profile: all three snapshot identities, descriptions,
source levels, values, and reasoning should remain unchanged. Exact field names
are never normalized or semantically merged. The denominator is the number of
distinct snapshots represented in the input observation file. A collection
containing artifacts with no observations would require a separate manifest
denominator and an explicit different policy.

The executable examples are new teaching implementations. They do not reproduce
the research runners. The request documents are unexecuted teaching adaptations;
fresh model-supported runs could produce different observations or concepts.
Research attribution: Dy & Solatorio (2026). Artifact-Driven Metadata Schema
Development: A Human-in-the-Loop Methodology with Multimodal LLMs. Working paper.
