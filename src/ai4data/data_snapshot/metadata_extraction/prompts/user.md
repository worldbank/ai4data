Inspect the attached data snapshot and extract all Schema v1.4 metadata explicitly supported by the image. Return the structured metadata only.

## Field-boundary guidance

- Preserve the complete visible title or caption in `title`. Also populate
  `document_label` when a visible Figure, Table, Annex, or Exhibit label is present.
- Use `source_text` only for exact visible wording. Never invent source wording for
  a value inferred from visual form or structure.
- Variables are measured concepts, indicators, or metrics. Dimensions classify,
  group, organize, or compare those measures. Do not duplicate a concept as both
  unless the image clearly gives it both roles.
- Coefficient, standard error, count, percentage, and confidence interval are
  statistical forms or qualifiers on the applicable variable; they are not
  standalone measured variables when the measured concept is identifiable.
- Use `category_groups` for explicit group headings and their contained categories.
  Use `panel_titles` only for explicit panels in a multi-panel snapshot, not ordinary
  section headings within one table.
- Express a visible comparative relationship as one comparison, such as
  `Baseline vs Closing Period`, rather than separate unrelated comparison items.
- Put derivation datasets and surveys in `provenance.sources`; put credited agents
  in `provenance.attributions`. Do not repeat a source statement in
  `interpretive_notes` when its meaning is fully represented elsewhere.
- Temporal coverage describes when the represented data apply, not publication or
  footer dates. Normalize visible dates only when the mapping is deterministic.
- Geographic scope is the overall represented area. Use locations for additional
  named places and level for the level at which the data are reported.
- Always populate `languages` when the language of the visible snapshot text is
  unambiguous. Use the corresponding BCP 47 `tag`; use `source_text` only when the
  snapshot explicitly names the language.
- Preserve complete substantive notes that are not fully represented by another
  field. Do not extract numerical observations or reproduce table cells.

{{MODEL_FACING_SCHEMA_SECTION}}
