# World Bank metadata schemas (test fixture)

A pinned snapshot of the schema files that the cookbook's `check_metadata.py`
needs for the four example data types (indicator, microdata, geospatial,
document) and the files they reference. The tests point the script at this
folder so that they run without network access.

- Source: https://github.com/worldbank/metadata-schemas, folder `schemas/`
- Commit: `ff91990fc766` (2026-09-30), copied on 2026-10-02
- Licence: MIT (see the source repository)

Files: `timeseries-schema.json`, `microdata-schema.json`, `ddi-schema.json`,
`datafile-schema.json`, `variable-schema.json`, `variable-group-schema.json`,
`geospatial-schema.json`, `document-schema.json`, `datacite-schema.json`,
`provenance-schema.json`, `table-schema.json`.

To refresh, copy the same files from the source repository and update the
commit line above. The script downloads the current files on first use when
no `--schema-dir` is given, so readers are never pinned to this snapshot.
