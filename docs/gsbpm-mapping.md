# Mapping to the GSBPM

The Generic Statistical Business Process Model (GSBPM) is the UNECE reference model that national statistical offices and international organizations use to describe the steps of producing official statistics. It has eight phases, from specifying needs to evaluation, with overarching processes such as quality management and metadata management that apply to every phase.

This page maps each AI for Data workstream to the GSBPM phases and sub-processes where it applies. Organizations that plan and report in GSBPM terms can use the mapping to see where each tool fits into their existing process.

:::note[Version]
The phase and sub-process names follow GSBPM version 5.1. Confirm the numbering against the version your organization uses.
:::

---

## Mapping by workstream

| Workstream | GSBPM phase and sub-process | What the workstream contributes |
|---|---|---|
| [Generative AI for Metadata Quality](metadata-quality/generative-ai-for-metadata-quality.md) and the [Metadata Reviewer](metadata-reviewer/overview.md) | Overarching: metadata management and quality management. Applied at 6.5 (finalise outputs) and 7.1 (update output systems) | Scores metadata records on completeness, semantic alignment, specificity, and consistency, and proposes corrections before records are published |
| [Metadata Augmentation](metadata-augmentation/index.md) | 2.2 (design variable descriptions) and 5.2 (classify and code) | Groups survey variables into thematic categories using semantic clustering |
| [Anomaly Detection and Explanation](anomaly-detection/anomaly-detection.md) | 5.3 (review and validate), 6.2 (validate outputs), 6.3 (interpret and explain outputs) | Flags unusual values and produces structured, evidence-backed explanations for review |
| [Data Discoverability](data-discoverability/data-discoverability.md) | 7.2 (produce dissemination products) and 7.5 (manage user support) | Semantic search over the indicator and dataset catalog |
| [Model Context Protocol](mcp/mcp.md) | 7.2 (produce dissemination products) and 7.3 (manage release of dissemination products) | Exposes the catalog as tools that AI clients can call |
| Proof-Carrying Numbers | 6.2 (validate outputs) and 7.2 (produce dissemination products) | Checks each number in an AI-generated answer against the official record before it is shown |
| [Monitoring of Data Use](data_use/data_use.md) | 8.1 (gather evaluation inputs), 8.2 (conduct evaluation), and 7.5 (manage user support) | Identifies where datasets are cited in reports and papers |
| [Inclusive AI Applications](inclusive-ai/inclusive-ai.md) | Cross-cutting | Multilingual embedding models and batch inference used by the other workstreams |
| Statistical classification and coding | 5.2 (classify and code) | AI-assisted coding of survey responses and records against standard classifications, with human review |
| [Synthetic data](https://github.com/avsolatorio/RealTabFormer) | 6.4 (apply disclosure control) | Synthetic tabular and relational data generated with REaLTabFormer, for sharing and methodological research |
| Global Question Bank | 2.2 (design variable descriptions) | Multilingual semantic mappings of survey questions and variables across surveys and countries |
| AI-ready metadata and standards, and AI-assisted metadata platforms | Overarching: metadata management. Applied at 7.1 (update output systems) | Implementation guidance for metadata standards, and AI-assisted metadata creation and search in the Metadata Editor and NADA |
| Responsible AI guidance | Overarching: quality management | Guidance and evaluation methods for using AI in official statistics |
| [Data Snapshots](https://arxiv.org/abs/2606.06242) | 4.3 (run collection) and 5.1 (integrate data) | Locates figures and tables in PDF documents so that the data they contain can be extracted as structured data |
| Small and Agentic AI | Cross-cutting | Small language models and agentic workflows for statistical processes and specialized tasks, and AI assistants built on them |
| Ontologies and knowledge graphs | Overarching: metadata management | Shared concept definitions and links between indicators, variables, and datasets |
| [AI-readiness assessment framework](pathname:///ai-readiness-assessment/) | Overarching: quality management | Assesses the readiness of an organization and of its data products and services for AI |

---

## Coverage by phase

| Phase | Workstreams |
|---|---|
| 1. Specify needs | Open area |
| 2. Design | Metadata Augmentation (2.2), Global Question Bank (2.2) |
| 3. Build | Open area |
| 4. Collect | Data Snapshots (4.3) |
| 5. Process | Metadata Augmentation (5.2), Anomaly Detection (5.3), Statistical classification and coding (5.2), Data Snapshots (5.1) |
| 6. Analyse | Anomaly Detection (6.2, 6.3), Proof-Carrying Numbers (6.2), Metadata Quality (6.5), Synthetic data (6.4) |
| 7. Disseminate | Metadata Quality (7.1), Data Discoverability, MCP, Proof-Carrying Numbers, Monitoring of Data Use (7.5), AI-assisted metadata platforms (7.1) |
| 8. Evaluate | Monitoring of Data Use |
| Overarching | Metadata management (including ontologies and knowledge graphs), quality management (including responsible AI guidance and the AI-readiness assessment framework) |

Most of the work sits in phases 5 to 8, where the data and metadata already exist and the task is to validate, explain, publish, and track them. Newer workstreams extend into phase 5 (classification and coding), phase 6 (synthetic data), phase 2 (the Global Question Bank), and phase 4 (extracting data from documents). Phases 1 and 3 are open areas. Candidate applications there include assisting with questionnaire design and concept harmonization (phases 1 to 3), and supporting data collection with automated checks (phase 4).

---

## How this relates to the AI-ready data framework

The [AI-ready data framework](ai-ready-framework.md) organizes the program by what AI needs from data: findable, accessible, interoperable, reusable, trustworthy, and inclusive. The GSBPM mapping is a complementary view. It shows where in the statistical production process each workstream applies, so that statistical organizations can place the tools in their own process.
