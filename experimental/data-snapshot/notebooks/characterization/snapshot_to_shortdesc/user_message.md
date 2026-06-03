You are given a single data snapshot extracted from a report.

DOCUMENT METADATA
- Source type: {{SOURCE_LABEL}}
- Document title: {{DOCUMENT_TITLE}}
- Publication year: {{YEAR}}
- Language: {{LANGUAGE}}

DOCUMENT-LEVEL CONTEXT (may not apply to this snapshot)
- Abstract: {{ABSTRACT}}

SNAPSHOT CONTEXT
- Snapshot type: {{SNAPSHOT_TYPE}}

TASK
Describe the data contained in this snapshot.

If any title, subtitle, header, or label text is visible *within the snapshot image itself*,
extract it verbatim (or approximately if partially visible).
If none is visible, return "none".

Return your answer in the following JSON format:

{
  "visible_text_label": "...",
  "data_modality": "...",
  "topic_domain": "...",
  "population": "...",
  "geographic_scope": "...",
  "temporal_aspect": "...",
  "measurement_units": "...",
  "short_description": "..."
}

FIELD DEFINITIONS
- visible_text_label: title, subtitle, header, or label text visible in the snapshot image itself (not from metadata); "none" if absent
- data_modality: type of data (e.g. survey results, administrative counts, projections, monitoring indicators)
- topic_domain: subject matter (e.g. protection, displacement, livelihoods, education, health)
- population: group described (e.g. IDPs, refugees, households, women and children, unclear)
- geographic_scope: spatial coverage if visible (e.g. country-level, regional, global, unclear)
- temporal_aspect: time coverage (e.g. single year, multi-year trend, quarterly, unclear)
- measurement_units: how values are expressed (e.g. counts, percentages, rates, index, categories)
- short_description: 1–2 sentence neutral summary of what data is reported