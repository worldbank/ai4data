# Data Modality Taxonomy for PRWP Document Snapshots

Source corpus derived from PRWP snapshot descriptions. The dataset is dominated by economics, governance, development, poverty, trade, finance, labor, demographic, and institutional analysis. The proposed taxonomy prioritizes:

- Stable semantic meaning over visual form
- Low-cardinality categories
- Clear annotation boundaries
- Extensibility across heterogeneous PDF corpora
- Utility for evaluation and downstream error analysis

The taxonomy below is designed for snapshot-level classification of tables, charts, maps, diagrams, and composite analytical figures.

| rollup_group | data_modality | definition | typical_content | examples |
|---|---|---|---|---|
| Economic Systems & Markets | Macroeconomic & Financial Indicators | Aggregated economic, monetary, fiscal, trade, or financial system indicators describing economy-level conditions, trends, or performance. Focuses on national, sectoral, or market-wide quantitative indicators rather than household-level outcomes. | GDP, inflation, exchange rates, public finance, fiscal balances, debt, banking indicators, lending rates, trade balances, oil prices, bond maturity, remittances, imports/exports, sectoral fuel intensity | CPI inflation trends; oil price history; budget revenues/expenditures as % of GDP; lending interest rate trends; bank loan portfolio composition |
| Economic Systems & Markets | Production, Trade & Industry | Economic activity at the sector, firm, production-chain, or industry level, including productivity, industrial structure, trade composition, market participation, supply chains, or export specialization. Emphasizes how goods/services are produced, exchanged, or organized economically. | Sectoral production, export networks, tariffs, import demand, manufacturing activity, agricultural production, firm competitiveness, value chains, buyer requirements | Syrian export product network; tariff impacts on trade flows; wood log buyer requirements; fuel consumption by sector |
| Human Development & Welfare | Poverty, Inequality & Welfare | Distributional and welfare-oriented measures describing living standards, deprivation, inequality, vulnerability, or socioeconomic well-being of individuals or households. | Poverty rates, poverty gaps, Gini coefficients, consumption/income measures, welfare indices, household assets, living conditions | Regional poverty estimates in Vietnam; inequality indicators; household welfare comparisons |
| Human Development & Welfare | Population, Labor & Social Outcomes | Demographic, labor-market, education, migration, health, or other human-centered social indicators describing population characteristics or social conditions. | Employment, hours worked, migration, education access, age/gender distributions, health outcomes, school closures, workforce participation | Hours worked by older workers in China; migrant/remittance indicators; school access impacts; labor participation trends |
| Governance & Institutions | Governance, Policy & Institutions | Institutional coordination, public administration, governance systems, policy implementation, operational management, program execution, budgeting processes, service delivery oversight, regulatory structures, or organizational responsibilities. Captures how institutions, projects, programs, and operational systems are managed and governed rather than economy-wide outcomes themselves. | Governance indicators, institutional responsibilities, operational updates, implementation tracking, procurement monitoring, budget execution, donor coordination, service delivery systems, public administration, decentralization, policy instruments, implementation bottlenecks, organizational workflows | Project implementation dashboards; operational status updates; budget utilization tracking; institutional coordination matrices; procurement milestone trackers; governance regressions; service delivery monitoring tables |
| Governance & Institutions | Conflict, Protection & Humanitarian Conditions | Security incidents, displacement, protection risks, humanitarian access, crisis exposure, conflict dynamics, or emergency-response conditions affecting civilian populations. | Protection incidents, forced displacement, GBV, explosive incidents, humanitarian coverage, conflict events, emergency needs | Protection incident counts; humanitarian access constraints; displacement movements; explosive hazard trends |
| Environment & Resources | Environment, Energy & Natural Resources | Environmental conditions, energy systems, pollution, climate exposure, ecosystem management, natural resource use, or environmental sustainability indicators. | Pollution regressions, energy consumption, ecosystem indicators, land use, environmental compliance, natural resource management | Pollution governance regressions; electricity consumption relationships; environmental compliance requirements |
| Spatial & Relational Analytics | Spatial, Network & Relational Structures | Spatially or relationally structured information where the primary analytical value comes from geographic distribution, connectivity, flows, or network topology rather than scalar indicators alone. | Maps, geospatial distributions, transport corridors, export/product networks, actor connectivity, regional clustering | Operational presence maps; export-product network graphs; geographic incident distribution maps |
| Analytical Evidence | Statistical & Econometric Results | Outputs of formal statistical, econometric, causal, predictive, or optimization models where the central information is analytical inference rather than raw descriptive indicators. | Regression tables, coefficients, confidence intervals, Tobit models, IV estimates, fixed effects, fitted models, correlation matrices | Tobit model outputs; IV estimation tables; regression coefficient summaries; correlation matrices |

---

# Design Rationale

## 1. Separation by Analytical Intent

The taxonomy intentionally separates:

- Descriptive indicators from analytical inference
- Human outcomes from macroeconomic systems
- Institutional structures from socioeconomic outcomes
- Environmental/resource systems from general economic activity
- Relational/spatial structures from scalar tabular indicators

This improves:

- Annotation consistency
- Multi-label reasoning
- Error analysis for models
- Stability across corpora

---

# Important Boundary Rules

## Governance, Policy & Institutions vs Macroeconomic & Financial Indicators

This distinction is especially important for operational and budget-related snapshots.

Use:

- **Governance, Policy & Institutions** for:
  - project implementation updates
  - operational monitoring
  - budget execution tracking
  - procurement pipelines
  - institutional coordination
  - donor/resource management
  - service delivery oversight
  - program administration

Use:

- **Macroeconomic & Financial Indicators** for:
  - fiscal deficits
  - debt composition
  - inflation
  - national expenditure indicators
  - banking and financial system metrics
  - economy-wide fiscal conditions

Key heuristic:

> If the snapshot describes how a project, institution, or program is being managed or implemented, classify it as Governance, Policy & Institutions.

> If the snapshot describes economy-level financial or fiscal conditions, classify it as Macroeconomic & Financial Indicators.

Examples:

| Snapshot | Recommended Modality |
|---|---|
| Project implementation progress dashboard | Governance, Policy & Institutions |
| Budget utilization by ministry/project | Governance, Policy & Institutions |
| Operational presence tracking | Governance, Policy & Institutions |
| Procurement milestone tracker | Governance, Policy & Institutions |
| Government expenditure as % GDP | Macroeconomic & Financial Indicators |
| Fiscal deficit trends | Macroeconomic & Financial Indicators |
| Public debt composition | Macroeconomic & Financial Indicators |

---

## Macroeconomic & Financial Indicators vs Poverty, Inequality & Welfare

Use:

- **Macroeconomic & Financial Indicators** for economy-wide aggregates and financial systems
- **Poverty, Inequality & Welfare** for distributional or household welfare outcomes

Example:

- Inflation rate chart → Macroeconomic & Financial Indicators
- Poverty gap estimates → Poverty, Inequality & Welfare

---

## Production, Trade & Industry vs Macroeconomic & Financial Indicators

Use:

- **Production, Trade & Industry** when the unit of analysis is sectoral, industrial, firm-level, or trade-structure oriented
- **Macroeconomic & Financial Indicators** when describing aggregate economy-wide indicators

Example:

- Export product network → Production, Trade & Industry
- Exchange rate trend → Macroeconomic & Financial Indicators

---

## Governance, Policy & Institutions vs Statistical & Econometric Results

This is the most important modeling distinction.

- **Governance, Policy & Institutions** describes the subject matter
- **Statistical & Econometric Results** describes analytical inference outputs

Recommendation:

- Treat econometric/model outputs as the dominant modality whenever the snapshot primarily communicates estimated relationships or statistical inference.
- Allow secondary labels for substantive domain context if multi-label classification is enabled.

Example:

- Governance regression table → Primary: Statistical & Econometric Results; Secondary: Governance, Policy & Institutions

---

## Spatial, Network & Relational Structures vs Other Modalities

Use this modality only when:

- Geography, topology, or connectivity is central to interpretation
- The structure itself conveys the analytical meaning

Do not use for ordinary charts merely segmented by region.

Example:

- Choropleth map → Spatial, Network & Relational Structures
- Regional poverty bar chart → Poverty, Inequality & Welfare

---

## Environment, Energy & Natural Resources vs Production, Trade & Industry

Environmental modality should be used when:

- Ecological systems
- Energy systems
- Pollution
- Sustainability
- Resource management

are central analytical concepts.

Otherwise, sectoral production data belongs in Production, Trade & Industry.

---

# Recommended Annotation Policy

## Preferred Strategy: Single Dominant Label

For evaluation stability:

- Assign one dominant modality per snapshot whenever possible.
- Use the modality representing the primary analytical intent.

This minimizes:

- Annotation disagreement
- Boundary instability
- Label sparsity
- Evaluation ambiguity

---

## Optional Multi-Label Extension

If richer semantic analysis is needed later, allow:

- One primary modality
- Optional secondary modality labels

Recommended secondary-label use cases:

| Example | Primary | Secondary |
|---|---|---|
| Governance regression table | Statistical & Econometric Results | Governance, Policy & Institutions |
| Poverty map | Spatial, Network & Relational Structures | Poverty, Inequality & Welfare |
| Energy trade network | Production, Trade & Industry | Environment, Energy & Natural Resources |

---

# Edge Cases & Ambiguities

## 1. Regression Tables

These appear extremely frequently in PRWP-style corpora.

Potential ambiguity:

- Is the modality economics/governance/labor/etc.?
- Or is it statistical inference?

Recommendation:

- Use Statistical & Econometric Results as the dominant label.
- Preserve domain semantics as optional secondary labels.

This greatly improves taxonomy stability.

---

## 2. Composite Figures

Some figures combine:

- Maps
- Charts
- Tables
- Narrative annotations

Recommendation:

- Label according to the dominant analytical message.
- If no dominant message exists, prefer:
  1. Spatial/Network modality if geography/connectivity drives interpretation
  2. Statistical/Econometric modality if inference dominates
  3. Domain modality otherwise

---

## 3. Descriptive Statistics Tables

Descriptive-statistics tables may overlap with:

- Macroeconomic indicators
- Labor outcomes
- Poverty indicators

Recommendation:

- Use the substantive domain modality unless the table primarily communicates modeling/inference structure.

Example:

- Summary statistics for labor outcomes → Population, Labor & Social Outcomes
- Regression diagnostics → Statistical & Econometric Results

---

## 4. Infrastructure & Service Access

Infrastructure/service-access snapshots may span:

- Governance
- Human development
- Economic systems

Recommendation:

- Use Human Development when access affects human welfare directly
- Use Governance when institutional delivery or administration is central
- Use Production/Industry when infrastructure is treated as an economic production system

---

# Stability Assessment

This taxonomy is likely to generalize well across:

- Policy research working papers
- World Bank reports
- UN/NGO humanitarian reports
- Development economics literature
- Governance assessments
- Socioeconomic monitoring documents

because it separates:

- Subject matter
- Analytical intent
- Structural representation

without overfitting to any single document genre.

---

# Recommended Final Label Set

Recommended production label set:

1. Macroeconomic & Financial Indicators
2. Production, Trade & Industry
3. Poverty, Inequality & Welfare
4. Population, Labor & Social Outcomes
5. Governance, Policy & Institutions
6. Conflict, Protection & Humanitarian Conditions
7. Environment, Energy & Natural Resources
8. Spatial, Network & Relational Structures
9. Statistical & Econometric Results

This satisfies the desired constraints:

- Compact but expressive
- Low overlap
- Stable across corpora
- Suitable for evaluation
- Human-interpretable
- Extensible without taxonomy explosion
- Compatible with single-label or multi-label annotation

