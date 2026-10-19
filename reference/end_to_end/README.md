# End-to-end reference implementation

One command that runs the pattern the AI for Data – Data for AI program
describes, on the cookbook's example organization:

1. **Document.** Five series records in the World Bank indicator schema, one
   survey record in the microdata schema, and the published values with
   SDMX column names (`fixtures/`).
2. **Validate.** Each record against the pinned World Bank schema for its
   type, a completeness profile at the chosen level, and validity rules.
   A record that fails is withheld with its reasons.
3. **Publish.** The records that pass, as a catalog directory with one JSON
   record, one values CSV, and one schema.org description per product, and
   an index. With a URL and an API key, the same records go to a NADA
   instance through its catalog administration API.
4. **Serve.** An MCP server over the catalog (`server.py`), started as a
   subprocess and called over the protocol: search, record, values. Every
   observation carries a claim identifier.
5. **Answer.** A question answered from the tool results, with each number
   inside a claim tag. A template composes it so that no model is needed;
   a model's answer can be verified with `--answer`.
6. **Verify.** Every claim tag checked with `pcn-core`, the Python engine of
   Proof-Carrying Numbers: verified, flagged, or unmarked.

The report of a run is in `report.md` and `report.json` under the output
directory. Two runs are committed under `sample_run/`: `clean`, where every
number verifies, and `planted`, where one wrong number is flagged and one
untagged number receives no mark.

## Running it

```bash
pip install -r requirements.txt
python run.py --out build                 # result: PASS, exit 0
python run.py --out build --inject-error  # result: FAIL, exit 1
```

Options: `--level ai-ready` requires the AI-ready fields for publication;
`--answer FILE` verifies an answer you supply; `--nada-url URL` with
`NADA_API_KEY` set publishes to a NADA instance, and `--nada-dry-run`
reports the requests without sending them; `--schema-dir` points at another
copy of the World Bank schemas.

The tests in `tests/test_reference_end_to_end.py` run both modes, verify an
external answer, build the NADA requests in dry-run mode, and check that
the committed sample runs match a fresh run. They run in CI on every change
to this folder.

## Replacing a step with a real system

| Step | In this implementation | In an organization |
|---|---|---|
| Document | JSON exports in `fixtures/` | Exports from the Metadata Editor, or records from its API |
| Validate | `e2e/validate.py` with the pinned schemas and the profiles | The same code on the organization's exports, with its own profile |
| Publish | A directory, and the NADA API when configured | The NADA instance, or any catalog that takes the schema |
| Serve | `server.py` over the directory | A server over the catalog's API, with access control; the Data360 MCP server shows the production form |
| Answer | A template | A chat application such as the Data AI Chatbot, with a system prompt that asks for claim tags |
| Verify | `pcn-core` in `e2e/verify.py` | The PCN packages in the application's renderer |

## What this does not do

- It does not call a language model. The template stands in for one so
  that the run is reproducible and the verification step always has
  something to check.
- It does not deploy any of the program's applications. It shows the
  pattern and the data shapes they share.
- The NADA call is implemented against the catalog administration API and
  reported in dry-run mode; it is not run in CI, which has no NADA instance.
- A verified number matches the record it cites under the stated policy.
  The check says nothing about whether the record, the method behind it,
  or the sentence around the number is right.

## Layout

```
run.py              the command
server.py           the MCP server over the published catalog
e2e/records.py      step 1
e2e/validate.py     step 2
e2e/publish.py      step 3
e2e/answer.py       step 5
e2e/verify.py       step 6
e2e/report.py       the report
fixtures/           series.json, survey.json, values.csv, the two profiles
sample_run/         committed output of the clean and the planted run
requirements.txt    jsonschema, mcp, pcn-core (from the PCN repository)
```
