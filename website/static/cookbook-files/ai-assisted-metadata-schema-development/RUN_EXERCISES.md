# Offline exercises for Chapters 1–3

Use Python 3.10 or newer. The three scripts use only the standard library.
Download the scripts and the files named below into a new working directory.
The prepared evidence files retain their original contents. No images or
dataset download, provider credentials, API request, or model execution is needed.

## Inputs and commands

The three exercises use six files in total. Download the files in each row
into the same working directory before running that exercise.

| Exercise | Files to download together | Output |
|---|---|---|
| Manifest boundary | check_manifest.py; source_images.csv | The script prints counts and writes nothing. |
| Response inspection | inspect_discovery.py; discovery_excerpt.jsonl; source_images.csv | The script prints structural checks and writes nothing. |
| Exact-name profiling | profile_observations.py; geographic_observations.json | The script creates outputs/geographic_profiles.json. |

Run the following commands from that working directory:

    python check_manifest.py source_images.csv
    python inspect_discovery.py discovery_excerpt.jsonl source_images.csv
    python profile_observations.py geographic_observations.json --output outputs/geographic_profiles.json

The first command checks the eight selected identities. It reports six
development snapshots and two held-out snapshots, with no snapshot or parent
overlap within this excerpt. It cannot recompute the full-study counts of
210 development snapshots, 202 held-out snapshots, and 65 shared parents.

The second command inspects three selected response excerpts. They contain
four title-related fields from original responses containing 43 fields in total.
The fields, identities, model names, statuses, and original response counts are
recorded research evidence. The excerpt wrapper is newly prepared. Other
fields, raw responses, token accounting, and full input messages are omitted.
This is a structural inspection, not a semantic accuracy assessment.

The third command aggregates three unchanged geographic_scope observations
selected from the exact-name-profile example in evidence.json. It writes a
new JSON profile with count 3, n_snapshots 3, total_snapshots 3, and support 1.0.
All three input records are retained as evidence. The full-study profile's
192 observations and the later canonical concept's 194-snapshot support have
different scope and provenance.

## Output safety and verification

The first two scripts write nothing. The profiling script creates its output
with exclusive creation and refuses to replace an existing file. Use a new
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
