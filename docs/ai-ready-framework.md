# AI-ready data framework

AI systems are becoming a common way for people to put questions about poverty, employment, climate, and food security. Whether the answers draw on official statistics depends on whether AI can find, interpret, and verify the data. When it cannot, AI relies on secondary sources that may be outdated or wrong. The AI for Data – Data for AI program works to keep official statistics the trusted foundation of those answers.

This page defines what the program means by AI-ready data, and shows how each workstream contributes to it. AI supports statistical systems and does not replace them. The program follows the UN Fundamental Principles of Official Statistics, keeps human review in AI-assisted workflows, and develops its methods and software as open resources.

---

## FAIR for AI

The program builds on the FAIR principles (findable, accessible, interoperable, reusable) and extends them for AI systems as consumers of data. Two dimensions are added: trustworthy and inclusive.

| Dimension | Question for AI | Attributes covered | Workstreams |
|---|---|---|---|
| Findable | Can AI find the right data? | Data discoverability, comprehensive metadata | [Data Discoverability](data-discoverability/data-discoverability.md), [Metadata Augmentation](metadata-augmentation/index.md), open benchmarks and evaluation |
| Accessible | Can AI retrieve it, with context? | Openly accessible, machine-readable, real-time accessibility | [Model Context Protocol](mcp/mcp.md), AI-assisted metadata platforms, [Data Snapshots](https://arxiv.org/abs/2606.06242) |
| Interoperable | Can AI combine and interpret it? | Integrative, machine-understandable, contextual relevance | AI-ready metadata and standards, Global Question Bank, ontologies and knowledge graphs, statistical classification and coding |
| Reusable | Is it documented well enough to reuse? | Comprehensive metadata, high data quality, licensing and privacy | [Generative AI for Metadata Quality](metadata-quality/generative-ai-for-metadata-quality.md), [Monitoring of Data Use](data_use/data_use.md), [synthetic data](https://github.com/avsolatorio/RealTabFormer) |
| Trustworthy | Can answers be traced and verified? | High data quality, ethical and governance standards | [Anomaly Detection and Explanation](anomaly-detection/anomaly-detection.md), Proof-Carrying Numbers, responsible AI guidance |
| Inclusive | Does it work across languages and countries? | Diversity and representativeness | [Inclusive AI Applications](inclusive-ai/inclusive-ai.md), Small and Agentic AI |

---

## AI-ready data attributes

The program organizes the characteristics of AI-ready data in three groups.

**Foundational attributes** apply to every use case.

| Attribute | Meaning |
|---|---|
| Machine-readable | Open formats that machines can process automatically |
| Comprehensive metadata | Clear metadata on provenance, processing, and limitations |
| Openly accessible | Openly available when legally and ethically permitted |
| High data quality | Accurate, complete, timely, and appropriately granular data |
| Data discoverability | Findable through search, catalogs, and APIs |
| Licensing and privacy | Clear licensing, with privacy protected through governance |
| Ethical and governance standards | Aligned with ethical, legal, and governance standards |

**Attributes for generative AI context** apply when AI answers questions using data at the time of the query.

| Attribute | Meaning |
|---|---|
| Real-time accessibility | Low-latency access to current, authoritative data |
| Contextual relevance | Relevant context for the specific task or query |
| Machine-understandable | Semantic context that AI can interpret accurately |

**Attributes for training AI models** apply when data is used to train or fine-tune models.

| Attribute | Meaning |
|---|---|
| Integrative | Linkable across datasets, time, geography, and categories |
| Quantity | Enough high-quality data for the specific task (use-case dependent) |
| Diversity and representativeness | Represents target populations fairly across diverse dimensions |

---

## AI-readiness assessment framework

The program also develops an AI-readiness assessment framework for national statistical organizations. It assesses both the institution and the data products and services the institution provides. It has two pillars and twelve dimensions, with 69 questions plus a gateway question, and places each dimension on one of four maturity levels (A to D). Each question maps to the recommendations of the Committee for the Coordination of Statistical Activities (CCSA) for national statistical organizations.

The [AI-readiness assessment page](pathname:///ai-readiness-assessment/) describes the pillars, lists the dimensions, and shows how the results form a readiness profile.

---

## Program structure

The workstreams fall under two pillars.

### AI for Data

AI is applied to produce, curate, and disseminate development data with less manual effort and higher quality.

| Group | Workstream |
|---|---|
| Data production | Statistical classification and coding |
| Data production | [Data Snapshots](https://arxiv.org/abs/2606.06242) |
| Data production | Small and Agentic AI |
| Data production | [Synthetic data](https://github.com/avsolatorio/RealTabFormer) |
| Data quality and metadata | [Generative AI for Metadata Quality](metadata-quality/generative-ai-for-metadata-quality.md) |
| Data quality and metadata | [Metadata Augmentation](metadata-augmentation/index.md) |
| Data quality and metadata | [Anomaly Detection and Explanation](anomaly-detection/anomaly-detection.md) |
| Data quality and metadata | Responsible AI guidance |
| Discovery and trustworthy dissemination | [Data Discoverability](data-discoverability/data-discoverability.md) |
| Discovery and trustworthy dissemination | Proof-Carrying Numbers |
| Discovery and trustworthy dissemination | [Monitoring of Data Use](data_use/data_use.md) |
| Methods and open tools | [Inclusive AI Applications](inclusive-ai/inclusive-ai.md) |
| Methods and open tools | Open benchmarks and evaluation |

### Data for AI

Development data is made discoverable, interpretable, and usable by AI systems through open standards and infrastructure.

| Group | Workstream |
|---|---|
| Standards and metadata | [AI-ready data framework](ai-ready-framework.md) |
| Standards and metadata | AI-ready metadata and standards |
| Infrastructure | [Model Context Protocol](mcp/mcp.md) |
| Infrastructure | AI-assisted metadata platforms |
| Semantic knowledge | Global Question Bank |
| Semantic knowledge | Ontologies and knowledge graphs |

For where each workstream applies in the statistical production process, see the [mapping to the GSBPM](gsbpm-mapping.md).
