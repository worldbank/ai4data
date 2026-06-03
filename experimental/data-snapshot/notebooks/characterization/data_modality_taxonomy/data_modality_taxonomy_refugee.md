# Data Modality Taxonomy for Refugee / Humanitarian PDF Snapshots

Source dataset analyzed: `shortdesc_refugee_batch1.csv` (499 snapshot descriptions). fileciteturn0file0L1-L20

## Methodology Summary

### 1. Dataset Scan
The snapshot descriptions exhibit recurring semantic patterns centered around:

- Project finance and budget allocation
- Operational implementation and procurement
- Socio-economic and humanitarian indicators
- Population displacement and beneficiary statistics
- Geographic and regional exposure/risk patterns
- Institutional coordination and governance
- Risk, conflict, protection, and vulnerability analysis
- Monitoring, evaluation, and performance tracking

A major observation is that many snapshots mix multiple semantic intents inside a single visual artifact (e.g., a table containing both budget allocations and beneficiary counts). Therefore, the taxonomy prioritizes:

- Dominant analytical intent
- Stable semantic distinctions
- Minimal overlap
- Human annotation consistency
- Downstream evaluation utility

### 2. Design Principles Used

The taxonomy intentionally:

- Separates semantic content from rendering format
  - e.g., maps, charts, tables, and infographics are NOT modalities
- Uses low-cardinality, evaluation-stable categories
- Avoids excessively domain-specific labels
- Keeps categories broad enough to generalize across document corpora
- Supports optional multi-label annotation for hybrid cases

---

# Recommended Taxonomy

| rollup_group | data_modality | definition | typical_content | examples |
|---|---|---|---|---|
| Humanitarian & Socioeconomic Conditions | Socioeconomic Indicators | Quantitative indicators describing demographic, economic, health, education, welfare, livelihood, or social conditions of populations. Focus is descriptive measurement of population status or outcomes. | Poverty rates, mortality, nutrition, enrollment, labor statistics, GDP indicators, inflation, disease prevalence, access-to-service metrics, household indicators | Refugee population statistics by region; child malnutrition rates; unemployment trends; health service utilization indicators |
| Humanitarian & Socioeconomic Conditions | Population & Beneficiary Statistics | Counts, distributions, characteristics, or movements of people targeted, affected, displaced, served, or enrolled in programs. Emphasis is on populations rather than broader socioeconomic systems. | IDP counts, refugee flows, beneficiary registries, household counts, gender/age breakdowns, migration flows, caseload statistics | Internally displaced persons by district; beneficiary counts by intervention type; gender distribution of affected households |
| Humanitarian & Socioeconomic Conditions | Risk, Protection & Vulnerability Analysis | Information describing threats, incidents, exposure, vulnerability, insecurity, protection risks, or humanitarian harm affecting populations. | Protection incidents, conflict exposure, gender-based violence, disaster risk, food insecurity, explosive hazards, vulnerability scoring | Protection incident trends; GBV case statistics; explosive incident mapping; vulnerability severity classifications |
| Finance & Resources | Financial & Budgetary Information | Information about funding, costs, expenditures, financing structures, disbursements, or resource allocation. The primary intent is monetary accounting or financial planning. | Budgets, cost tables, financing plans, expenditure breakdowns, procurement values, disbursement schedules, donor allocations | Project financing by component; operational expenditure summary; procurement budget allocation |
| Finance & Resources | Economic & Market Analysis | Information about markets, production systems, trade, prices, fiscal conditions, or macroeconomic performance. Focus is economic system behavior rather than social welfare indicators alone. | Commodity prices, fiscal balances, trade flows, agricultural production, market access, inflation trends, sector performance | Agricultural price monitoring; fiscal deficit projections; market supply-chain analysis |
| Operations & Delivery | Program Operations & Implementation | Information describing operational execution, implementation progress, logistics, procurement processes, timelines, work plans, or delivery mechanisms. | Procurement plans, implementation schedules, staffing plans, construction progress, operational workflows, service delivery tracking | Procurement method tables; implementation timeline charts; operational deployment plans |
| Operations & Delivery | Infrastructure & Service Systems | Information describing physical infrastructure, utilities, facilities, transport systems, or service networks and their operational status or coverage. | Roads, schools, hospitals, utilities, telecommunications, water systems, facility inventories, service coverage maps | School closure status; road rehabilitation plans; health facility coverage; water infrastructure inventory |
| Governance & Coordination | Governance, Policy & Institutional Systems | Information about institutional arrangements, legal frameworks, governance structures, policy actions, coordination mechanisms, or organizational responsibilities. | Administrative structures, policy frameworks, coordination matrices, institutional mandates, compliance structures, stakeholder roles | Cluster coordination structure; ministry responsibility matrix; policy reform framework |
| Governance & Coordination | Monitoring, Evaluation & Performance | Information evaluating progress, effectiveness, outcomes, targets, compliance, benchmarks, or program results over time. | KPIs, baseline/endline comparisons, scorecards, evaluation metrics, results frameworks, progress dashboards | Results indicator matrix; program outcome tracking; target vs actual performance table |
| Spatial & Environmental Context | Geographic & Spatial Analysis | Information whose primary meaning depends on geographic distribution, spatial relationships, territorial coverage, or location-based exposure. | Administrative maps, hotspot mapping, spatial accessibility, regional comparisons, geocoded incident density, territorial coverage | Conflict hotspot map; refugee settlement distribution; operational coverage by district |

---

# Why These Modalities Were Selected

## 1. Stable Across Corpora
The modalities generalize beyond refugee and humanitarian documents and should remain valid across:

- World Bank PADs
- UNHCR assessments
- Humanitarian situation reports
- Development finance documents
- Government monitoring reports

## 2. Semantic Rather Than Visual
The taxonomy intentionally avoids labels such as:

- table
- chart
- map
- infographic
- timeline

Those belong to layout or rendering taxonomies rather than semantic modality taxonomies.

## 3. Distinct Analytical Intent
Each modality captures a distinct question being answered:

| Modality | Primary Question |
|---|---|
| Socioeconomic Indicators | What is the condition/status of a population or system? |
| Population & Beneficiary Statistics | Who is affected or served, and how many? |
| Risk, Protection & Vulnerability Analysis | What threats or harms exist? |
| Financial & Budgetary Information | How are resources allocated or spent? |
| Economic & Market Analysis | How is the economy or market behaving? |
| Program Operations & Implementation | How is the intervention being executed? |
| Infrastructure & Service Systems | What systems/facilities/services exist or function? |
| Governance, Policy & Institutional Systems | Who governs or coordinates what? |
| Monitoring, Evaluation & Performance | Is the intervention achieving intended results? |
| Geographic & Spatial Analysis | Where are conditions, risks, or services located? |

---

# Key Boundary Rules (Important for Annotation Stability)

| Potential Ambiguity | Resolution Rule |
|---|---|
| Socioeconomic Indicators vs Population & Beneficiary Statistics | Use Population & Beneficiary Statistics when the central object is people counts/composition/displacement. Use Socioeconomic Indicators when measuring broader conditions/outcomes. |
| Financial & Budgetary Information vs Program Operations & Implementation | Use Financial when money/allocation is primary. Use Operations when execution/process/logistics is primary. |
| Governance vs Monitoring & Evaluation | Governance describes structures/responsibilities/policies. Monitoring describes measurement of results/performance. |
| Geographic & Spatial Analysis vs Any Other Modality | Geographic is only dominant when spatial distribution itself is analytically central, not merely displayed on a map. |
| Risk & Vulnerability vs Socioeconomic Indicators | Risk focuses on threats/exposure/incidents. Socioeconomic Indicators focus on observed conditions or outcomes. |
| Infrastructure & Service Systems vs Operations | Infrastructure describes systems/assets/services themselves. Operations describes implementation activities around them. |
| Economic & Market Analysis vs Financial & Budgetary Information | Economic focuses on markets/macroeconomic behavior. Financial focuses on organizational/project funding and expenditures. |

---

# Recommended Annotation Strategy

## Preferred Approach: Single Dominant Label

For evaluation stability, assign:

- One dominant `data_modality`
- Based on primary analytical intent

This reduces:

- Inter-annotator disagreement
- Label sparsity
- Evaluation ambiguity

## Optional Secondary Labels

Allow optional secondary labels only when:

- Two semantic intents are equally central
- The snapshot is genuinely composite

Example:

- A map showing conflict incidents by district:
  - Primary: Geographic & Spatial Analysis
  - Secondary: Risk, Protection & Vulnerability Analysis

---

# Common Edge Cases

| Edge Case | Recommended Handling |
|---|---|
| Budget table with beneficiary counts | Label Financial & Budgetary Information if resource allocation dominates; otherwise Population & Beneficiary Statistics |
| Map of health facilities | Geographic & Spatial Analysis if spatial distribution is central; Infrastructure & Service Systems if facility inventory/service capacity is central |
| Results framework with target budgets | Monitoring, Evaluation & Performance if performance tracking dominates |
| Procurement tables with costs | Program Operations & Implementation unless monetary allocation is the dominant analytical purpose |
| Protection incident charts by region | Risk, Protection & Vulnerability Analysis unless spatial comparison is the central narrative |
| Infographics mixing indicators and funding | Assign dominant analytical intent; avoid multi-label unless truly necessary |

---

# Recommended Refinements (Optional)

## Possible Future Merge

If annotation consistency becomes difficult, consider merging:

- Socioeconomic Indicators
- Population & Beneficiary Statistics

into:

- Population & Social Conditions

This would reduce taxonomy size from 10 to 9 modalities.

## Possible Future Split

If future corpora heavily emphasize environmental data, split:

- Geographic & Spatial Analysis

into:

- Spatial Analysis
- Environmental & Climate Conditions

However, this split is NOT currently recommended because it increases overlap and annotation complexity.

---

# Final Recommendation

The strongest evaluation-ready configuration for this dataset is:

1. Socioeconomic Indicators
2. Population & Beneficiary Statistics
3. Risk, Protection & Vulnerability Analysis
4. Financial & Budgetary Information
5. Economic & Market Analysis
6. Program Operations & Implementation
7. Infrastructure & Service Systems
8. Governance, Policy & Institutional Systems
9. Monitoring, Evaluation & Performance
10. Geographic & Spatial Analysis

This configuration achieves:

- Low overlap
- Strong semantic separation
- Stable annotation behavior
- Good extensibility
- Broad applicability across humanitarian and development corpora
- Clear utility for downstream evaluation and error analysis

Relevant source material analyzed included humanitarian protection reports discussing protection incidents, displacement statistics, explosive hazards, operational coverage, and coordination structures. fileciteturn0file2L1-L120

