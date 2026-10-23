# Standards-Alignment Findings and Decision Matrix

Status: Final standards-alignment decision record

## Reporting boundary

The standards review begins from the 36 flat fields in the validated v1.1.1
schema. Each field was compared with authoritative metadata, statistical,
provenance, research-data, publishing, project-finance, and value-normalization
standards.

The crosswalk results and the schema-design decisions are reported separately:

1. the crosswalk records how closely each existing field corresponds to the
   reviewed standards; and
2. the decision matrix records whether a candidate refinement should be
   accepted, rejected, or deferred for the flat v1.1.2 schema.

This separation prevents a useful standards pattern from being treated as an
automatic schema change. Artifact evidence, the intended use of the schema,
and human judgment remain part of every final decision.

## Relationship between schema fields and reviewed standards

| Reviewed relationship | Count | Representative example | Interpretation |
|---|---:|---|---|
| Exact | 10 | `title` and `source_document_title` align with DCMI `title` at their respective resource levels; `language` aligns with DCMI `language` and Schema.org `inLanguage` | The external semantics can be reused directly while preserving a clear application-facing field. |
| Close | 11 | `variable_name` aligns closely with Schema.org `variableMeasured`, DDI Variable, and SDMX measure concepts | The local scope should be preserved and the bounded difference documented. |
| Standard broader | 6 | General identifier properties are broader than `internal_identifier`; DCMI `subject` is broader than `subject_domain` | The local definition should remain narrower than the external term. |
| Standard narrower | 3 | Schema.org Project and targeted finance classifications cover only parts of the project and finance fields | The broader application concept should remain available. |
| Related or structural | 1 | PROV-O supplies separate derivation and attribution relations for meanings combined in `data_source` | The standard suggests a semantic distinction rather than a direct field equivalent. |
| No direct match | 5 | Geographic role, comparison group, intervention type, and analysis method | A local source-grounded field remains appropriate; optional mappings may be added when exact. |
| **Total** | **36** |  |  |

### Fields in each relationship category

| Relationship | v1.1.1 fields |
|---|---|
| Exact | `title`, `time_period`, `geographic_scope`, `unit_of_measure`, `currency`, `source_document_title`, `language`, `project_identifier`, `financing_source`, `data_collection_method` |
| Close | `subject_summary`, `panel_title`, `variable_name`, `category_dimension`, `category_labels`, `population_group`, `temporal_granularity`, `geographic_entities`, `geographic_granularity`, `measure_type`, `financing_instrument` |
| Standard broader | `internal_identifier`, `subject_domain`, `row_dimension`, `column_dimension`, `visualization_type`, `interpretive_note` |
| Standard narrower | `project_name`, `project_component`, `financial_measure` |
| Related or structural | `data_source` |
| No direct match | `geographic_role`, `location_type`, `comparison_group`, `intervention_type`, `analysis_method` |

## Decision criteria

A candidate change should be accepted for v1.1.2 only when it:

1. improves semantic accuracy or reduces ambiguity;
2. remains consistent with evidence observed in data snapshots;
3. preserves the accepted information coverage of the validated schema;
4. remains understandable in a flat, directly inspectable specification; and
5. does not imply relationships or normalization precision that the flat
   representation cannot preserve.

The decision labels below record the approved dispositions:

- **Accept:** incorporate the candidate change into v1.1.2.
- **Partially accept:** incorporate the terminology or definition but not the
  proposed structure or mandatory normalization.
- **Reject:** do not incorporate the candidate because it conflicts with the
  evidence or intended use.
- **Defer:** the candidate is useful but belongs to the later nested,
  machine-readable implementation.

## Human-adjudication decisions

| v1.1.1 field | Crosswalk finding | Candidate refinement | Decision | Rationale and v1.1.2 outcome |
|---|---|---|---|---|
| `title` | Exact title semantics | Add DCMI and Schema.org mappings; distinguish the snapshot resource from the parent document | Accept | Keep `title`; refine the definition and mapping. |
| `internal_identifier` | General identifier properties are broader; JATS `label` is closer | Rename to `document_label` | Accept | The new name accurately describes visible labels such as “Figure 3” without implying global uniqueness. |
| `subject_domain` | Broader subject and theme properties are useful but not exact | Rename to `subject_domains`; require an external vocabulary | Partially accept | Use the plural name because multiple coequal domains occur. Reject mandatory external coding because no single domain vocabulary fits all artifacts. |
| `subject_summary` | Schema.org `abstract` is close; generic description is broader | Clarify that the summary describes the snapshot's analytical subject | Accept | Keep `subject_summary`; refine its definition. |
| `panel_title` | JATS supplies a useful figure-group pattern | Rename to `panel_titles`; introduce panel objects | Partially accept | Use the plural flat field for explicit headings. Defer panel objects and panel-specific relationships. |
| `variable_name` | Close variable and measure concepts exist | Rename to `variable_names`; attach qualifiers to variable objects | Partially accept | Use a plural flat field. Defer variable objects and qualifier relationships. |
| `category_dimension` | SDMX and DDI provide close dimension concepts | Rename to `category_dimensions`; attach categories and presentation roles | Partially accept | Use a plural flat field. Defer explicit dimension-category and presentation-role relationships. |
| `category_labels` | SDMX, DDI, and SKOS support categories, codes, and hierarchies | Preserve labels; require codes and nested groups | Partially accept | Keep `category_labels` and preserve source order. Reject mandatory codes; defer explicit grouping structures. |
| `population_group` | DDI Universe and Schema.org population concepts are close | Require a controlled vocabulary | Partially accept | Keep `population_group` and refine its boundary. Reject a mandatory vocabulary because populations are domain-specific. |
| `time_period` | Exact temporal-coverage semantics and established date formats exist | Preserve the source expression; add normalized bounds | Partially accept | Keep `time_period` with improved guidance. Defer separate normalized bounds and precision properties. |
| `temporal_granularity` | SDMX frequency is close but does not cover all temporal resolutions | Require `CL_FREQ` values | Partially accept | Keep `temporal_granularity`; use SDMX terminology when exact but reject forced frequency mappings for irregular or event-based periods. |
| `geographic_scope` | Exact spatial-coverage semantics | Clarify overall coverage and introduce a structured place object | Partially accept | Keep `geographic_scope` with a clearer definition. Defer identifiers and place objects. |
| `geographic_entities` | Close place and geographic-location concepts | Require standard place identifiers | Partially accept | Keep `geographic_entities`; permit recognizable names but reject mandatory codes when mappings are ambiguous. |
| `geographic_granularity` | DDI GeographicLevel is closer than general granularity terminology | Rename to `geographic_level`; require an administrative-level vocabulary | Partially accept | Adopt `geographic_level`. Reject forced administrative coding without jurisdictional evidence. |
| `geographic_role` | No direct equivalent; standards provide only related patterns | Rename to `geographic_roles`; impose a closed vocabulary | Partially accept | Use the plural field but reject a closed vocabulary; preserve explicit source-grounded roles. |
| `location_type` | No direct equivalent; IATI and SKOS offer optional patterns | Rename to `location_types`; require IATI codes | Partially accept | Use the plural field but reject mandatory IATI coding outside applicable project contexts. |
| `unit_of_measure` | Exact unit semantics and a standard code list exist | Rename to `units_of_measure`; require UN/CEFACT codes | Partially accept | Use the plural field. Apply codes only when exact; defer source/code value objects and variable attachment. |
| `currency` | Exact currency semantics and ISO 4217 exist | Rename to `currencies`; require ISO codes | Partially accept | Use the plural field. Prefer an unambiguous ISO code but preserve source wording where a symbol or label carries necessary evidence. |
| `measure_type` | The field represents statistical form rather than the measured concept | Rename to `statistical_forms`; adopt controlled values | Partially accept | Adopt the clearer name and standards-grounded terminology. Defer machine-enforceable controlled-term objects. |
| `comparison_group` | No direct equivalent covers complete comparisons and named benchmarks | Rename to `comparisons`; force categories or relation triples | Partially accept | Adopt `comparisons`. Reject forced category or relation structures that would invent an implicit comparison side. |
| `row_dimension` | General dimension standards are broader | Rename to `row_dimensions`; fold into dimension presentation roles | Partially accept | Use the plural flat field. Defer role-bearing dimension objects. |
| `column_dimension` | General dimension standards are broader | Rename to `column_dimensions`; fold into dimension presentation roles | Partially accept | Use the plural flat field. Defer role-bearing dimension objects. |
| `visualization_type` | General type properties are broader; Vega-Lite offers partial terminology | Rename to `visualization_types`; require a closed vocabulary | Partially accept | Use the plural field for composite artifacts. Use preferred terms but retain unfamiliar explicit types. |
| `data_source` | PROV-O distinguishes derivation from attribution | Split into flat `data_sources` and `attributions`; later introduce role-bearing objects | Accept split; defer objects | The distinction is semantically useful without requiring nesting. Preserve explicit attribution roles as flat source-grounded text. |
| `source_document_title` | Exact title semantics at the parent-document resource level | Clarify the resource being named | Accept | No field-name change; refine its mapping and definition. |
| `language` | Exact language semantics and BCP 47 guidance exist | Rename to `languages`; require language tags | Partially accept | Use the plural field for multilingual artifacts. Do not require tags when the visible language can be recorded reliably only as text. |
| `interpretive_note` | Generic descriptions are broader; SDMX annotations are structurally relevant | Rename to `interpretive_notes`; parse internal facts into fields | Partially accept | Use the plural field and preserve complete statements. Reject automatic decomposition into unsupported fields. |
| `project_name` | Schema.org Project is narrower than the application concept | Adopt the narrower Project scope | Reject | Keep `project_name` broad enough for projects, programs, operations, and initiatives. Add optional mappings only when applicable. |
| `project_identifier` | Exact identifier semantics | Rename to `project_identifiers`; require scheme-qualified objects | Partially accept | Use the plural field. Defer structured scheme and issuer properties. |
| `project_component` | `hasPart` and IATI patterns are narrower | Rename to `project_components`; adopt an external project hierarchy | Partially accept | Use the plural field. Reject a mandatory external hierarchy that would narrow the observed component types. |
| `intervention_type` | No direct equivalent; domain classifications are contextual | Rename to `intervention_types`; require IATI classification | Partially accept | Use the plural field. Reject mandatory IATI coding for non-IATI and cross-domain artifacts. |
| `financial_measure` | External finance classifications cover narrower subsets | Rename to `financing_measures`; force transaction categories | Partially accept | Use the clearer name. Reject a mandatory classification that would collapse project cost, allocation, disbursement, and funding gap. |
| `financing_source` | Exact funder semantics | Rename to `funders`; introduce entity identifiers | Partially accept | Adopt `funders`. Defer structured identifiers and entity reconciliation. |
| `financing_instrument` | IATI FinanceType and Schema.org finance types are close | Rename to `financing_instruments`; require IATI codes | Partially accept | Use the plural field. Apply an IATI code only when the domain and mapping are exact. |
| `analysis_method` | No direct universal equivalent; method vocabularies are discipline-specific | Rename to `analysis_methods`; require an external method vocabulary | Partially accept | Use the plural field. Reject a mandatory vocabulary; preserve explicitly stated methods. |
| `data_collection_method` | DDI Mode of Collection is an exact or close fit for many values | Rename to `data_collection_methods`; require DDI values | Partially accept | Use the plural field. Apply a DDI term only when exact and preserve other explicit methods. |

## Deferred structural refinements

The standards review supports richer relationships among variables and their
qualifiers, dimensions and their categories, geographic entities and their
roles, provenance entities and attributions, and project or financing
concepts. Human review accepted terminology and semantic distinctions that
improved the flat schema but deferred these nested structures as a whole. They
were judged valuable for a subsequent machine-readable implementation rather
than necessary for the directly inspectable schema representation reported in
this study.

## Decision outcome

The approved decisions transform the 36-field v1.1.1 baseline into a 37-field
flat v1.1.2 specification. The additional field results from separating
derivation sources from attributions. Accepted field names, definitions,
mappings, and usage guidance are recorded in the v1.1.2 specification; nested
and machine-enforceable representations are outside this decision set.

## Authoritative sources consulted

- Schema.org: [data model](https://schema.org/docs/datamodel.html),
  [Dataset](https://schema.org/Dataset), and the term pages referenced in the
  decision matrix.
- Dublin Core Metadata Initiative: [DCMI Metadata
  Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/).
- World Wide Web Consortium: [Data Catalog Vocabulary
  (DCAT) 3](https://www.w3.org/TR/vocab-dcat-3/),
  [PROV-O](https://www.w3.org/TR/prov-o/),
  [SKOS](https://www.w3.org/TR/skos-reference/), and the
  [RDF Data Cube Vocabulary](https://www.w3.org/TR/vocab-data-cube/).
- SDMX: [technical standards](https://sdmx.org/standards-2/) and
  [cross-domain concepts and code lists](https://sdmx.org/sdmx_cdcl/).
- DDI Alliance: [DDI Lifecycle 3.3
  documentation](https://docs.ddialliance.org/DDI-Lifecycle/3.3/).
- International Aid Transparency Initiative: [IATI Standard
  2.03](https://reference.iatistandard.org/en/iati-standard/203/).
- National Information Standards Organization and National Library of
  Medicine: [JATS 1.3](https://jats.nlm.nih.gov/publishing/1.3/).
- International Organization for Standardization: [ISO
  8601](https://www.iso.org/iso-8601-date-and-time-format.html), [ISO
  3166](https://www.iso.org/iso-3166-country-codes.html), and [ISO
  4217](https://www.iso.org/iso-4217-currency-codes.html).
- United Nations: [UN M49](https://unstats.un.org/unsd/methodology/m49/) and
  [UN/CEFACT Recommendation
  20](https://unece.org/code-list-recommendations).
- Internet Engineering Task Force: [BCP
  47](https://www.rfc-editor.org/info/bcp47).
