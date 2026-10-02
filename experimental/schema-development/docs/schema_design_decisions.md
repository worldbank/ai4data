# Human schema-design decisions

The refined inventory contained 41 retained metadata concepts. Human schema
design transformed that inventory into the 36-field Data Snapshot Metadata
Schema v1.1. This was an operational design step, not another model inference.

| Decision | Representative transformation | Rationale |
|---|---|---|
| Merge | `series_labels` into `indicator_name` | Separate fields added little annotation or retrieval value. |
| Merge | `calculation_method` into `analysis_method` | The broader field retained the useful distinction without parallel vocabulary. |
| Reintroduce | `panel_title` | Multi-panel figures required an independently annotatable title. |
| Remove | `indicator_definition` | Its operational value did not justify an independent field. |
| Rename | `financial_metric` to `financial_measure` | The latter was clearer and more consistent with the schema vocabulary. |
| Refine | `comparison_group`, `reference_date` | Definitions were broadened to apply across the three corpora. |

The schemas in `schemas/` are the frozen outputs of these decisions. Validation
reports under `notebooks/schema_validation1/` and `notebooks/schema_validation2/`
preserve subsequent human adjudication decisions.

