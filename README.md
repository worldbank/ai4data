# AI for Data - Data for AI

## Project Name

**AI for Data – Data for AI**

## Description

**AI for Data – Data for AI** is a work program under the World Bank's Development Data Group, advancing the strategic use of artificial intelligence (AI) to improve the quality, usability, and impact of development data. The program focuses on two complementary missions: (1) **AI for Data** - applying AI to improve data and metadata quality, data discoverability and dissemination, monitoring of data use, and user experience in producing and accessing development datasets; and (2) **Data for AI** - ensuring development data is structured, documented, and made available in ways that enable effective and trustworthy use by AI systems. This repository provides Python tools and a JavaScript library, along with methodologies for data scientists, researchers, and development practitioners working with development data and AI applications.

## Overview

The program is spearheading the application of AI, along with the development of methodologies, tools, and models to drive improvements and efficiencies across the development data lifecycle. The work ranges from applying generative AI to enhance metadata quality to building low-resource AI models that support semantic search, data dissemination, and knowledge discovery—particularly for underserved contexts and users.

By innovating across the data lifecycle—from curation to dissemination to monitoring its use—this program ensures that development data becomes more accessible, useful, and actionable, while fostering responsible AI use across global development applications.

---

## Flagship Workstreams

### Generative AI for Metadata Quality

We leverage generative models to improve and scale metadata generation and refinement, including:

- Automated rewriting of indicator names and definitions for clarity and consistency.
- LLM-based validation of metadata completeness, coherence, and alignment with global standards.
- Multi-agent systems for iterative metadata improvement and enrichment.

### Anomaly Detection in Data

We combine statistical methods, machine learning, and frontier AI models to detect, interpret, and contextualize irregularities in development data. This includes:

- Automated detection of anomalies using a hybrid approach that integrates rule-based checks, classical ML, and multimodal LLMs.
- Use of large language models to generate human-readable explanations for anomalies—especially when linked to identifiable causes such as conflict, economic shocks, or climate events.
- Exploration of AI agents that flag anomalies in near real-time and help data producers triage and investigate irregular data patterns.

### Semantic Discovery & Retrieval

We design and deploy AI-powered systems to enhance the discoverability of development datasets and indicators. This includes:

- Semantic search and recommendation systems for survey microdata, indicators, and documents.
- Retrieval-augmented generation (RAG) for better user interaction with statistical content.
- AI agents that assist users in refining their queries and identifying relevant data assets.

### Monitoring of Data Use

We develop AI systems that track how development data is used in research, policy, and public discourse. This includes:

- Fine-tuned language models that identify and classify mentions of datasets in research publications and project documents.
- A data citation taxonomy that distinguishes between named, unnamed, and superficial data references to evaluate the depth of use.
- Systems to quantify the visibility, reuse, and policy relevance of datasets across countries, sectors, and languages—laying the groundwork for open, citation-based metrics for data impact.

### Efficient & Inclusive AI Applications

We focus on bridging the computational and socio-linguistic gaps in AI to ensure that tools are effective, fair, and usable across all development contexts, helping ensure that low-income countries are not left behind. This includes:

- Developing methods to evaluate and surface biases in AI systems, particularly those affecting low-resource languages and underrepresented populations.
- Training compact, low-resource embedding models to enable efficient semantic search, classification, and retrieval in bandwidth- or compute-constrained settings.
- Advising on and implementing practical AI strategies for countries with limited data coverage, including guidance on leveraging AI with non-traditional or incomplete datasets.

---

## Partnerships

While also supporting the various programs in the Development Data Group, we also collaborate across World Bank units and with external partners to advance the responsible use of AI in development data:

- With World Bank's Operational teams to advise and develop AI methods to support projects in countries.
- With National Statistical Offices (NSOs) and data producers to support AI-assisted metadata curation and catalog integration.
- With global AI communities, contributing to open-source tools and standards that enable fair, energy-efficient, and multilingual AI.

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Installation

Install the base package:

```bash
pip install ai4data
```

Or using uv:

```bash
uv pip install ai4data
```

### Optional Dependencies

Install specific capabilities:

```bash
# For dataset mention extraction from text/documents
pip install ai4data[datause]

# For anomaly detection in data
pip install ai4data[anomaly]

# For dataset name harmonization and deduplication
pip install ai4data[harmonization]

# For all capabilities
pip install ai4data[all]
```

### Usage Example

```python
from ai4data import data_use, anomaly_detection

# Extract dataset mentions from text
# See documentation for detailed examples
```

For detailed usage examples and API documentation, please refer to the [full documentation](https://worldbank.github.io/ai4data).

### JavaScript / TypeScript

JavaScript libraries are published under the **@ai4data** npm organization. Each package lives in `packages/ai4data/<library>/` and is installed as `@ai4data/<library>`:

```bash
npm install @ai4data/core
npm install @ai4data/search   # semantic search client (HNSW + BM25, Web Worker)
```

To add or work on packages, use the root workspace: run `npm install` at the repo root, then build or test per package (e.g. `npm run build --workspace=@ai4data/core`) or for all workspaces (`npm run build`). See [Repository structure](docs/repo-structure.md) for the full layout and how to add new libraries.

## Documentation

Comprehensive documentation is available at: **https://worldbank.github.io/ai4data**

The documentation includes:
- Detailed API reference
- Usage examples and tutorials
- Methodology descriptions
- Best practices and guidelines

## Contact Information

For questions, issues, or contributions, please contact:

- **Development Data Group**: ai4data@worldbank.org
- **GitHub Issues**: [https://github.com/worldbank/ai4data/issues](https://github.com/worldbank/ai4data/issues)

## Vision

The **AI for Data – Data for AI** program reflects the World Bank's commitment to harnessing frontier technologies for development impact. By aligning technical innovation with the principles of openness, quality, and inclusion, we are building a future where development data is not only easier to find—but more meaningful, more relevant, and more empowering for everyone.

---

## License

This project is licensed under the MIT License together with the World Bank IGO Rider. The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.
