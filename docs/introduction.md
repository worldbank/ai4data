# Introduction

**AI for Data – Data for AI** is a work program of the World Bank's Development Data Group and Office of the Chief Statistician. It applies AI to the production, curation, and dissemination of development data, and it prepares development data for use by AI systems. Its methods, software, and guidance are published as open resources.

## Why the program exists

People increasingly put questions about poverty, employment, climate, and food security to AI systems. Whether the answers draw on official statistics depends on whether an AI system can find the data, interpret what a number means, retrieve the authoritative value, and show where it came from. When it cannot, the system falls back on secondary sources that may be outdated or wrong, and official statistics lose influence while remaining the most trusted evidence.

Four gaps recur across data producers:

- **Metadata is incomplete or inconsistent.** A dataset with a vague title, missing keywords, or a definition that does not match its name does not appear in a search for what it measures.
- **Context is thin.** Units, classifications, breaks in series, and revisions are often absent from the machine-readable record, so a number is easy to misread.
- **Data are hard for software to reach.** Many statistics are published in PDF tables and spreadsheets, and where an API exists, each organization has its own.
- **Answers cannot be traced.** A figure in a generated answer rarely links to the record it came from, so a wrong or outdated value goes unnoticed.

The program's position follows the UN Fundamental Principles of Official Statistics: AI supports statistical systems and does not replace them, and a person remains responsible for what an organization publishes.

## Three pillars

**AI for Data.** AI applied inside the statistical process, under human review: assessment and enrichment of metadata, thematic organization of data dictionaries, detection and explanation of anomalies in time series, extraction of data from documents, search by meaning over catalogs, monitoring of how datasets are used, and verification of the numbers in generated text.

**Data for AI.** Development data made usable by AI systems through open standards and infrastructure: the World Bank metadata schemas, the Metadata Editor and the NADA catalog, Model Context Protocol servers over official statistics, and the question-level and concept-level metadata that let a system resolve what a term refers to.

**Country capacity.** An assessment framework that scores a national statistical organization on twelve dimensions of AI readiness, an operational companion that maps each dimension to tools and recipes, and a cookbook series that gives the steps.

The [AI-ready data framework](ai-ready-framework.md) defines the attributes of AI-ready data under FAIR and shows which workstream serves each. The [mapping to the GSBPM](gsbpm-mapping.md) places each workstream in the statistical production process.

## Where to go

| Need | Start at |
|---|---|
| An overview of the tools, guides, and cookbooks by what an organization does with them | The [implementation hub](/hub) |
| The status, stack, licence, installation guide, and known limits of each tool | The [tools catalogue](/tools) |
| A score for an organization on twelve dimensions of AI readiness | The [assessment framework](/ai-readiness-assessment) |
| What to do about a dimension with a low score | The [operational companion](/ai-readiness-in-practice) |
| Step-by-step guides with tested scripts | The [cookbooks](/cookbook/) |
| One run that documents, validates, publishes, serves, and verifies | The [end-to-end reference implementation](reference-implementation/index.md) |
| The method behind one workstream | The workstream pages in the sidebar |

## How this documentation is organized

The pages in the **Workstreams** section describe one method each: the problem it addresses, how it works, how to run it, and its limits. Each opens with a summary that needs no technical background. The **Metadata Reviewer** section is the user manual of one tool. The **Implementation** section holds the reference implementation and the pages on Proof-Carrying Numbers and MCP that the hub links to.

For data scientists and engineers, the pipeline pages, the notebooks, and the reference implementation give the code. For curators and data producers, the metadata pages and the cookbooks on documentation and curation give the procedures. For managers, the introduction, the assessment framework, and the governance cookbook give the decisions.

## Status of the work

Every tool on the [tools catalogue](/tools) carries one of five statuses, from in operation to in development, with the evidence behind it and the date it was checked. The status describes the software. It makes no claim about adoption outside the World Bank, which the [adoption area of the hub](/hub#adopt) addresses with an installation test any team can run and record.

## Contributing and contact

Issues and contributions go to the [ai4data repository](https://github.com/worldbank/ai4data) and to the repository of each tool. Organizations planning an adoption or a joint implementation can write to the Development Data Group at [aifordata@worldbank.org](mailto:aifordata@worldbank.org).
