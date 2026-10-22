---
title: End-to-end reference implementation
description: One reproducible run that documents records in the World Bank schemas, validates them, publishes a catalog, serves it through an MCP server, composes an answer from the tool results, and verifies every number with Proof-Carrying Numbers.
---

# End-to-end reference implementation

The reference implementation is a script that runs six steps on the cookbook's example organization: documentation, validation, publication, agent access over MCP, an answer composed from the tool results, and verification of the numbers in that answer. It needs no model and no network access, and a CI workflow runs it from a clean checkout. The code is in the [`reference/end_to_end`](https://github.com/worldbank/ai4data/tree/main/reference/end_to_end) folder of the repository. Its status on the [tools catalogue](/tools#reference-implementation) is deployable.

## The six steps

| Step | What happens | What it produces |
|---|---|---|
| 1. Document | Five series records in the World Bank indicator schema and one survey record in the microdata schema are loaded with the published values, whose columns follow the SDMX cross-domain concepts | The inputs of the run, in the form the Metadata Editor exports and NADA stores |
| 2. Validate | Each record is checked in three layers: structure against the pinned World Bank schema, completeness against a profile at the chosen level, and validity rules for dates, periods, URLs, vocabularies, and placeholders | A decision per record, published or withheld, with the findings |
| 3. Publish | The records that pass are written to a catalog directory with a values file and a schema.org description per product and an index; with a URL and an API key, the same records go to a NADA instance through its catalog administration API | A catalog that software and search engines can read |
| 4. Serve | An MCP server over the catalog is started as a subprocess and called over the protocol: search, record, values; every observation carries a claim identifier | Tool results in the shape the Proof-Carrying Numbers Data360 preset reads |
| 5. Answer | A question is answered from the tool results, each number inside a claim tag naming its observation; a template writes it so that the run needs no model, and an answer from any model can be supplied in its place | The text a chat application would show |
| 6. Verify | Every claim tag is compared with its claim under its policy by the Python core of Proof-Carrying Numbers; numbers outside a tag are listed as unmarked | A table of verified, flagged, and unmarked numbers, and a PASS or FAIL |

One record in the fixtures, the unemployment rate, has no definition. It passes the schema, fails the completeness profile at the foundational level, is withheld from the catalog, is absent from the server's search, and the answer says so. That is the gate the cookbooks describe, shown end to end.

## Running it

```bash
git clone https://github.com/worldbank/ai4data.git
cd ai4data/reference/end_to_end
pip install -r requirements.txt
python run.py --out build                  # every number verifies: PASS, exit 0
python run.py --out build --inject-error   # one wrong and one untagged number: FAIL, exit 1
```

The requirements are `jsonschema`, `mcp`, and `pcn-core`, the last installed from the PCN repository at a pinned commit until the package is published. Python 3.10 or later.

Options:

- `--level ai-ready` requires the AI-ready fields of the profile for publication.
- `--answer FILE` verifies an answer you supply, for example one a model wrote, against the same claims.
- `--nada-url URL` with the `NADA_API_KEY` environment variable publishes to a NADA instance; `--nada-dry-run` reports the requests without sending them.
- `--schema-dir DIR` points at another copy of the World Bank schemas.

## A run with planted errors

The console of the committed run with `--inject-error`:

```text file=../../reference/end_to_end/sample_run/planted/console.txt
```

The answer it verified, with the wrong figure inside its tag and the untagged figure:

```text file=../../reference/end_to_end/sample_run/planted/answer.md
```

The verification table of the report reads:

| Number | Claim | Outcome | Detail |
|---|---|---|---|
| 7.2 | FS_UNDERNOURISH_PCT:NAT:2024 | flagged | record holds 7.6 for 2024 |
| 92.0 | ED_PRIMARY_NER_PCT:NAT:2024 | verified | matches 92.0 for 2024 under rounded |
| 41 | none | unmarked | no claim tag, so no mark is drawn |

The flagged number is what a reader sees when a model changes a figure. The unmarked number is the fail-closed rule: a figure the model did not tie to a record receives no mark, whether or not it happens to be right. The full reports of both runs are under [`sample_run`](https://github.com/worldbank/ai4data/tree/main/reference/end_to_end/sample_run), and a test fails when a code change makes them stale.

## What a verified number means

A verified number matches the record it cites under the stated policy. The check says nothing about whether the record is the right one for the question, whether the statistical method behind it is sound, or whether the sentence around the number interprets it correctly. The report repeats this sentence under every verification table. The [Proof-Carrying Numbers page](../pcn/index.md) gives the policies and the packages.

## Replacing a step with a real system

| Step | In this implementation | In an organization |
|---|---|---|
| Document | JSON exports in the fixtures folder | Exports from the [Metadata Editor](/tools#metadata-editor), or records from its API |
| Validate | The validation module with the pinned schemas and the profiles | The same code on the organization's exports, with its own profile |
| Publish | A directory, and the NADA API when configured | The [NADA](/tools#nada) instance, or any catalog that takes the schema |
| Serve | The example server over the directory | A server over the catalog's API with access control; the [Data360 MCP server](/tools#data360-mcp) is the production form |
| Answer | A template | A chat application such as the [Data AI Chatbot](/tools#data-ai-chatbot), with a system prompt that asks for claim tags |
| Verify | The Python core in the verification module | The PCN packages in the application's renderer |

The data shapes are the join points. A record in the World Bank schema, a values file with SDMX column names, and a tool result with a claim identifier per observation are what let one step be replaced without touching the others.

## Tests

The tests in `tests/test_reference_end_to_end.py` run the pipeline in both modes, verify an external answer with a rounded claim, an exact claim, and an unknown identifier, build the NADA requests in dry-run mode, and compare the committed sample runs with fresh ones. The [workflow](https://github.com/worldbank/ai4data/blob/main/.github/workflows/reference-implementation.yml) runs them on every change to the folder.

## Limitations

- No model is called. The template stands in for one so that the run is reproducible; the `--answer` option is where a model's output enters.
- The pipeline deploys none of the applications it stands in for. It shows the pattern and the shapes they share.
- The NADA call is implemented against the catalog administration API and exercised in dry-run mode; CI has no NADA instance to send to.
- The server is a minimal catalog server. Access control, quotas, and terms are the subject of the [agents cookbook](/cookbook/serving-statistics-to-agents/access).
- The example organization has five series and one survey; the run shows the steps, not the scale.

## Related

- [Implementation hub](/hub): the areas the steps belong to.
- [Dissemination cookbook](/cookbook/ai-ready-dissemination/): findability, retrieval, meaning, and trust, with the completeness checker the validation step follows.
- Microdata documentation cookbook: the survey record and its catalog.
- [Agents cookbook](/cookbook/serving-statistics-to-agents/): the full design of a server over official statistics.
