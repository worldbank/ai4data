// The tools catalogue: one entry per piece of software or method the program
// offers, with its maturity stated from what the repository shows. The tools
// page renders every entry; the implementation hub groups them by area.
//
// Statuses are a fixed set. Change an entry's status only with evidence
// (a release, a running endpoint, a test suite), and record the date the
// evidence was checked in `checked`.

export const STATUSES = {
  operation: {
    label: 'In operation',
    short: 'Operation',
    description:
      'A service the World Bank runs with a public endpoint. Other organizations use it as a client or deploy the same software themselves.',
  },
  reference: {
    label: 'Deployable',
    short: 'Deployable',
    description:
      'Released software with installation instructions, versioned releases or tags, and tests. The adopting organization deploys and operates it.',
  },
  prerelease: {
    label: 'Pre-release',
    short: 'Pre-release',
    description:
      'Working software that runs from its repository, with documentation. A versioned release has not been made, and interfaces may change until it is.',
  },
  method: {
    label: 'Method',
    short: 'Method',
    description:
      'A documented method with a paper, a notebook, a specification, or a reference guide. There is no maintained software to deploy.',
  },
  development: {
    label: 'In development',
    short: 'In development',
    description: 'Announced work with nothing to install or read yet.',
  },
};

export const STATUS_ORDER = ['operation', 'reference', 'prerelease', 'method', 'development'];

// Areas of the implementation hub. The order is the order on the hub page.
export const AREAS = [
  {
    id: 'assess',
    label: 'Assess',
    title: 'Readiness framework and diagnostics',
    summary:
      'The assessment framework, its operational companion, and the cookbook that turns a readiness profile into an improvement pathway.',
  },
  {
    id: 'document',
    label: 'Document',
    title: 'Metadata standards, the Editor, and AI-assisted curation',
    summary:
      'Documentation of indicators, microdata, geospatial data, and documents in the World Bank schemas, and AI-assisted review and enrichment under curator review.',
  },
  {
    id: 'connect',
    label: 'Connect',
    title: 'MCP and interoperable data access',
    summary:
      'Access to official statistics by software and AI agents: files at stable addresses, APIs, and Model Context Protocol servers, with provenance in each response.',
  },
  {
    id: 'verify',
    label: 'Verify',
    title: 'Proof-Carrying Numbers and checked outputs',
    summary:
      'Verification of the numbers in generated text against the records they cite, evaluation of retrieval and answers, and quality checks on data and metadata.',
  },
  {
    id: 'implement',
    label: 'Implement',
    title: 'Cookbooks and the reference implementation',
    summary:
      'The cookbooks with their tested scripts, and the reference implementation that runs documentation, validation, publication, agent access, and verification in one pipeline.',
  },
  {
    id: 'adopt',
    label: 'Adopt',
    title: 'Installation tests, support boundaries, and contribution',
    summary:
      'The installation test, the support boundaries, the release practice of each repository, and where issues and contributions go.',
  },
];

const GITHUB = 'https://github.com/worldbank/';

// `area` is the hub area the tool belongs to. `verifies` lists the evidence
// behind the status. `operator` says who runs it as a service, if anyone.
export const TOOLS = [
  // ---------------------------------------------------------------- assess
  {
    id: 'assessment-framework',
    name: 'AI-readiness assessment framework',
    area: 'assess',
    kind: 'Framework',
    status: 'method',
    summary:
      'Scores a national statistical organization on two pillars and twelve dimensions, with 69 questions and a gateway question, and places it at one of four maturity levels per dimension.',
    stack: 'Reference guide (draft), interactive summary page, scoring annex',
    licence: 'Published on this site',
    docs: '/ai-readiness-assessment',
    install: null,
    repo: null,
    releases: null,
    operator: 'Applied by the program with the organization; the summary page is public.',
    limits: [
      'The reference guide is a draft; question wording and scoring may change between versions.',
      'The score is a self-assessment with evidence; it is not an audit.',
      'The framework places an organization; the improvement pathway cookbook and the operational companion say what to do next.',
    ],
    verifies: ['Reference guide v0.0.8 and its scoring annex', 'Interactive summary page on this site'],
    checked: '2026-10-13',
  },
  {
    id: 'operational-companion',
    name: 'Operational companion to the assessment',
    area: 'assess',
    kind: 'Guidance',
    status: 'method',
    summary:
      'For each dimension of the assessment, the workstreams, tools, and cookbook recipes that support progress to a higher level, and the evidence each produces for the assessment form.',
    stack: 'Page on this site, generated from the cookbook investment view',
    licence: 'MIT',
    docs: '/ai-readiness-in-practice',
    install: null,
    repo: GITHUB + 'ai4data',
    releases: null,
    operator: null,
    limits: [
      'Coverage flags count the public cookbooks; a hidden draft cookbook is not credited until it is published.',
      'Effort estimates per recipe are ranges taken from the cookbook; a project estimate needs the organization’s own rates.',
    ],
    verifies: ['Generated from the cookbook content in CI'],
    checked: '2026-10-13',
  },

  // -------------------------------------------------------------- document
  {
    id: 'metadata-editor',
    name: 'Metadata Editor',
    area: 'document',
    kind: 'Application',
    status: 'reference',
    summary:
      'Web application for documenting microdata (DDI Codebook 2.5), indicators, documents, geospatial data, images, videos, and scripts in the World Bank metadata schemas, with validation, templates, a REST API, and publication to NADA catalogs.',
    stack: 'PHP 8.2 or later with the listed extensions; MySQL 8 or MariaDB 10 or later; Apache 2.4, NGINX, or IIS 10; Python 3.11 or later for the FastAPI service',
    licence: 'MIT with the World Bank IGO rider',
    docs: 'https://worldbank.github.io/metadata-editor-docs',
    install: 'https://worldbank.github.io/metadata-editor-docs/tech_installation.html',
    repo: GITHUB + 'metadata-editor',
    releases: GITHUB + 'metadata-editor/releases',
    operator: 'Deployed by the adopting organization.',
    limits: [
      'The Metadata Reviewer is an optional add-on to the Editor\'s FastAPI service, installed from its own requirements file and enabled in site administration; it adds AI-generated issues with suggested corrections under Assess metadata on a project\'s Issues page. The augmentation package runs beside the Editor on exported records.',
      'Publication targets NADA catalogs through catalog connections; other catalogs take the exported JSON or DDI.',
      'Requires a PHP web stack, a relational database, and the Python service for importing and processing data, so it is deployed on a server.',
    ],
    verifies: ['Tagged releases with notes', 'Installation guide, with the Metadata Reviewer add-on guide', 'OpenAPI description of the REST API in the repository'],
    checked: '2026-10-19',
  },
  {
    id: 'nada',
    name: 'NADA catalog',
    area: 'document',
    kind: 'Application',
    status: 'reference',
    summary:
      'Open-source data cataloguing system for surveys, indicators, documents, tables, and geospatial data, with search, managed access requests, and a catalog administration REST API with API-key authentication.',
    stack: 'PHP, MySQL or MariaDB',
    licence: 'MIT',
    docs: 'https://nada.ihsn.org/',
    install: 'https://github.com/ihsn/nada#readme',
    repo: 'https://github.com/ihsn/nada',
    releases: 'https://github.com/ihsn/nada/releases',
    operator: 'Deployed by the adopting organization. The World Bank Microdata Library runs on NADA.',
    limits: [
      'The released software has keyword search and a REST API. Semantic search and the Model Context Protocol interface come from NADA AI, listed below, which is being integrated into NADA.',
      'A catalog is as findable as its records are complete; the completeness checks in the cookbooks apply before publication.',
    ],
    verifies: ['Repository and releases', 'Catalog administration API description (OpenAPI) in the repository'],
    checked: '2026-10-13',
  },
  {
    id: 'metadata-schemas',
    name: 'World Bank metadata schemas',
    area: 'document',
    kind: 'Standard',
    status: 'reference',
    summary:
      'JSON Schema definitions for indicators, indicator databases, microdata (DDI), geospatial (ISO 19115), documents, tables, images, videos, scripts, and resources, with a Python library of pydantic models and Excel templates.',
    stack: 'JSON Schema; Python package metadataschemas on PyPI',
    licence: 'MIT',
    docs: 'https://worldbank.github.io/metadata-schemas/',
    install: 'https://pypi.org/project/metadataschemas/',
    repo: GITHUB + 'metadata-schemas',
    releases: GITHUB + 'metadata-schemas/releases',
    operator: null,
    limits: [
      'The schemas require very little (an identifier and a title), so a record that validates can still be incomplete. Completeness is decided by a profile, as in the cookbooks.',
      'The Python package version lags the schema files in the repository; pin the commit when validation must match a catalog.',
    ],
    verifies: ['Schema files in the repository', 'Package on PyPI', 'Pinned snapshot used by the cookbook tests'],
    checked: '2026-10-13',
  },
  {
    id: 'metadata-reviewer',
    name: 'Metadata Reviewer',
    area: 'document',
    kind: 'Python package',
    status: 'prerelease',
    summary:
      'Five-agent pipeline that scans metadata records for typos, inconsistencies, missing fields, and incorrect content, and returns ranked issues with suggested corrections for a curator to accept or reject. It runs as a Python package, through the review app, and inside the Metadata Editor as an optional add-on.',
    stack: 'Python 3.11 or later; any AutoGen-compatible model client (OpenAI, Azure OpenAI, Anthropic, Ollama)',
    licence: 'MIT',
    docs: '/docs/metadata-reviewer/overview',
    install: '/docs/metadata-reviewer/user-manual/installation',
    repo: GITHUB + 'ai4data',
    releases: 'https://pypi.org/project/ai4data/',
    operator: null,
    limits: [
      'Every suggestion needs a curator decision; the pipeline detects and proposes. As a package it writes no catalog record; inside the Metadata Editor the curator accepts or rejects each issue on the project\'s Issues page.',
      'Quality depends on the model behind the agents; results differ between providers and model versions.',
    ],
    verifies: ['User manual on this site', 'Package extras on PyPI', 'Tests in the repository', 'Metadata Reviewer add-on guide in the Metadata Editor documentation'],
    checked: '2026-10-19',
  },
  {
    id: 'metadata-augmentation',
    name: 'Metadata Augmentation',
    area: 'document',
    kind: 'Python package',
    status: 'prerelease',
    summary:
      'Generates DDI-style variable groups for a data dictionary: embeds variable labels, clusters them, asks a model to curate each group, and checks the result for self-consistency before a curator reviews it.',
    stack: 'Python 3.11 or later; sentence-transformers, scikit-learn, litellm',
    licence: 'MIT',
    docs: '/docs/metadata-augmentation/',
    install: '/docs/metadata-augmentation/#installation',
    repo: GITHUB + 'ai4data',
    releases: 'https://pypi.org/project/ai4data/',
    operator: null,
    limits: [
      'Groups that fail the self-consistency check are kept and flagged, so the output always needs review.',
      'Cluster counts and group quality vary with the embedding model and the dictionary language.',
    ],
    verifies: ['Methodology pages on this site', 'Package extras on PyPI'],
    checked: '2026-10-13',
  },
  {
    id: 'metadata-quality',
    name: 'Generative AI for metadata quality',
    area: 'document',
    kind: 'Notebook',
    status: 'method',
    summary:
      'Scores each metadata record on completeness, semantic alignment, specificity, and consistency with structured model output, and aggregates the scores to find systematic gaps in a catalog.',
    stack: 'Jupyter notebook; a model provider',
    licence: 'MIT',
    docs: '/docs/metadata-quality/generative-ai-for-metadata-quality',
    install: null,
    repo: GITHUB + 'ai4data',
    releases: null,
    operator: null,
    limits: [
      'The notebook demonstrates the workflow; there is no packaged command to run it on a catalog.',
      'Scores are model judgments and are calibrated against curators before they are used to prioritize work.',
    ],
    verifies: ['Notebook in the repository, rendered on this site'],
    checked: '2026-10-13',
  },
  {
    id: 'data-snapshots',
    name: 'Data Snapshots',
    area: 'document',
    kind: 'Method and benchmark',
    status: 'method',
    summary:
      'Layout detection models locate figures and tables in PDF documents so that the data they hold can be extracted and documented; the paper benchmarks open-source models on the task.',
    stack: 'Paper and benchmark dataset',
    licence: 'See the paper',
    docs: 'https://arxiv.org/abs/2606.06242',
    install: null,
    repo: null,
    releases: null,
    operator: null,
    limits: [
      'Extraction of values from a located figure is a separate step; the cookbook on data from documents covers verification of extracted values.',
    ],
    verifies: ['Paper'],
    checked: '2026-10-13',
  },

  // --------------------------------------------------------------- connect
  {
    id: 'data360-mcp',
    name: 'Data360 MCP server',
    area: 'connect',
    kind: 'MCP server',
    status: 'operation',
    summary:
      'Model Context Protocol server over the World Bank Data360 platform: search of indicators with country coverage checks, metadata, disaggregation, codelist lookup, and time-series data for any MCP client, with chart specifications and a system-prompt resource.',
    stack: 'Python 3.11 or later; published on PyPI as data360-mcp; HTTP and SSE transports',
    licence: 'MIT with the World Bank IGO rider',
    docs: 'https://worldbank.github.io/data360-mcp',
    install: 'https://github.com/worldbank/data360-mcp#installation--version-pinning',
    repo: GITHUB + 'data360-mcp',
    releases: GITHUB + 'data360-mcp/releases',
    endpoint: 'https://maimcpext.worldbank.org/ext/data360/mcp',
    operator: 'Development Data Group, World Bank. The public endpoint serves the Data360 catalog.',
    limits: [
      'Serves the indicators in Data360; microdata and documents are out of scope.',
      'Text search in the upstream API is unreliable, so the server fetches broadly and ranks in application code; results depend on that ranking.',
      'A dimension listed in metadata can have data for only some of its values; the disaggregation tool reports what exists.',
      'Availability follows the Data360 API; the server degrades to cached metadata for a short cooldown when the API is slow.',
      'Adapting the server to another organization’s data API means replacing the data layer; the tool contracts and the client setup guides carry over.',
    ],
    verifies: [
      'Public endpoint named in the documentation site',
      'Semantic-versioned releases with a documented rollback',
      'Test suite in the repository',
      'Client setup guides for three MCP clients',
    ],
    checked: '2026-10-13',
  },
  {
    id: 'nada-ai',
    name: 'NADA AI',
    area: 'connect',
    kind: 'Python service',
    status: 'prerelease',
    summary:
      'Semantic search and a Model Context Protocol server over a NADA catalog: the service ingests the catalog records through the NADA REST API, indexes them with local or hosted embeddings, answers keyword, vector, and hybrid queries, and exposes the catalog to MCP clients through tools, interactive apps, resources, and prompts. It is being integrated into NADA, and its release is tied to the NADA release.',
    stack: 'Python 3.11 or later; OpenSearch or Qdrant as the index; a local sentence-transformer or a hosted embedding model; Docker files for the index',
    licence: 'MIT with the World Bank IGO rider',
    docs: 'https://github.com/avsolatorio/nada-ai#readme',
    install: 'https://github.com/avsolatorio/nada-ai#readme',
    repo: 'https://github.com/avsolatorio/nada-ai',
    releases: null,
    operator: 'Deployed beside a NADA catalog by the adopting organization until the integrated NADA release.',
    limits: [
      'Runs as a separate service beside the catalog and reads it through the REST API until the integration into NADA is released; interfaces may change before then.',
      'Repository in transfer to a World Bank organization; the link above is the current location.',
      'Needs a vector index (OpenSearch or Qdrant) and an embedding model, local or hosted.',
    ],
    verifies: ['Repository with installation instructions and Docker files', 'The MCP server tool list in the repository'],
    checked: '2026-10-16',
  },
  {
    id: 'data-ai-chatbot',
    name: 'Data AI Chatbot',
    area: 'connect',
    kind: 'Application',
    status: 'reference',
    summary:
      'Self-hostable chat application that connects to any MCP server over HTTP or SSE, routes model calls through LiteLLM, renders charts and documents in a side panel, and marks numbers in answers with Proof-Carrying Numbers. The World Bank customization, Data360 Chat, ships pre-configured for the Data360 MCP server.',
    stack: 'Next.js frontend and FastAPI backend; PostgreSQL, with Redis optional for resumable streams and caching; Docker Compose',
    licence: 'Apache-2.0 with the World Bank IGO rider',
    docs: 'https://worldbank.github.io/data-ai-chatbot',
    install: 'https://github.com/worldbank/data-ai-chatbot#getting-started',
    repo: GITHUB + 'data-ai-chatbot',
    releases: GITHUB + 'data-ai-chatbot/releases',
    operator: 'The World Bank runs Data360 Chat. Other organizations deploy the application with their own MCP server and model provider.',
    limits: [
      'Needs a model provider through LiteLLM and a PostgreSQL database, with Redis optional; it is a full-stack deployment.',
      'The verification marks depend on the MCP server returning claim identifiers with values; a server without them yields unverified numbers.',
      'Verification covers the numbers the model tagged; the PCN page says what a mark does and does not mean.',
    ],
    verifies: [
      'Deployment, operations, and security documentation in the repository',
      'Docker Compose files for development and production',
    ],
    checked: '2026-10-13',
  },
  {
    id: 'mcp-guidance',
    name: 'MCP for official statistics',
    area: 'connect',
    kind: 'Guidance',
    status: 'method',
    summary:
      'Architecture and tool design for an MCP server over official statistics, with a minimal example server in the cookbook and a full treatment of tools, context, access control, safety, and evaluation in the agents cookbook.',
    stack: 'Documentation and example scripts',
    licence: 'MIT',
    docs: '/docs/mcp/',
    install: null,
    repo: GITHUB + 'ai4data',
    releases: null,
    operator: null,
    limits: [
      'The example server reads two CSV files; a production server calls the organization’s API and adds access control.',
    ],
    verifies: ['Pages on this site', 'Example server tested in CI with the cookbook scripts'],
    checked: '2026-10-13',
  },
  {
    id: 'semantic-search',
    name: 'Semantic search and the PI-FT toolkit',
    area: 'connect',
    kind: 'Python package and toolkit',
    status: 'prerelease',
    summary:
      'Indexing and search by meaning over catalog records, and a toolkit that fine-tunes embedding models on structured metadata with permutation-invariant training, reporting Recall@k, MRR, and nDCG.',
    stack: 'Python 3.11 or later; sentence-transformers, FAISS; the @ai4data/search package on npm for in-browser search',
    licence: 'MIT',
    docs: '/docs/data-discoverability/',
    install: '/pift-toolkit/deployment',
    repo: GITHUB + 'ai4data',
    releases: null,
    operator: null,
    limits: [
      'Fine-tuning needs question sets in the organization’s languages; the cookbooks give the method for building them.',
    ],
    verifies: ['Toolkit documentation on this site', 'Package extras in the repository'],
    checked: '2026-10-13',
  },

  // ---------------------------------------------------------------- verify
  {
    id: 'pcn',
    name: 'Proof-Carrying Numbers',
    area: 'verify',
    kind: 'Protocol and packages',
    status: 'reference',
    summary:
      'Presentation-layer protocol for numeric fidelity: a model emits each number as a claim tag naming its source record, and the renderer verifies the number against the record under a declared policy and draws the mark. TypeScript packages for the core, the UI, and a Data360 preset; a Python core for comparison; a specification with shared fixtures.',
    stack: 'TypeScript packages on npm under the @pcn-js scope (React 18 or later for the UI); Python 3.10 or later for the core',
    licence: 'MIT with the World Bank IGO rider',
    docs: '/docs/pcn/',
    install: 'https://github.com/worldbank/pcn/blob/main/docs/integration.md',
    repo: GITHUB + 'pcn',
    releases: 'https://www.npmjs.com/package/@pcn-js/core',
    operator: 'Used inside Data360 Chat. Other applications integrate the packages.',
    limits: [
      'A verified mark means the number matches the record it cites under the stated policy. It says nothing about whether the record, the method behind it, or the interpretation is right.',
      'The application must ingest tool results and render claim tags; PCN does not extract claims from free text on its own.',
      'The TypeScript packages are published on npm; the Python core is installed from the repository until it is published to PyPI.',
      'The Python core compares a token with a claim under a policy; parsing and rendering are in the TypeScript packages.',
    ],
    verifies: ['Packages @pcn-js/core, @pcn-js/ui, and @pcn-js/data360 on npm', 'Specification and fixtures shared across languages', 'Package tests in the repository', 'Paper with the formal results'],
    checked: '2026-10-13',
  },
  {
    id: 'anomaly',
    name: 'Anomaly detection and explanation',
    area: 'verify',
    kind: 'Python package and review app',
    status: 'prerelease',
    summary:
      'Takes anomaly scores from any detector on indicator time series, builds the context for each flagged point, elicits a structured explanation from a model with a classification, a confidence, and cited evidence, and exports it to a web application where a reviewer approves, rejects, or corrects it.',
    stack: 'Python 3.11 or later; batch APIs of OpenAI or Gemini; FastAPI review app',
    licence: 'MIT',
    docs: '/docs/anomaly-detection/',
    install: '/docs/anomaly-detection/#quick-start',
    repo: GITHUB + 'ai4data',
    releases: 'https://pypi.org/project/ai4data/',
    operator: null,
    limits: [
      'Explanations are hypotheses with cited evidence; the reviewer decides, and the feedback export is the record.',
      'Evidence quality depends on what the model can retrieve for a country and period; sparse contexts produce low-confidence outputs.',
      'Batch runs are asynchronous and take hours for large sets.',
    ],
    verifies: ['Methodology handbook and notebook on this site', 'Review application in the repository'],
    checked: '2026-10-13',
  },
  {
    id: 'data-use',
    name: 'Monitoring of data use',
    area: 'verify',
    kind: 'Python package',
    status: 'prerelease',
    summary:
      'Extracts dataset mentions from documents with zero-shot named-entity recognition, harmonizes name variants to canonical datasets, and deduplicates them, so that use of a dataset in research and policy documents can be counted.',
    stack: 'Python 3.11 or later; GLiNER, PyMuPDF, RapidFuzz; sentence-transformers optional',
    licence: 'MIT',
    docs: '/docs/data_use/',
    install: '/docs/data_use/#installation',
    repo: GITHUB + 'ai4data',
    releases: 'https://pypi.org/project/ai4data/',
    operator: null,
    limits: [
      'Recall on informal mentions is lower than on formal citations; the cookbook on monitoring data use combines the two.',
      'Harmonization needs a dictionary of the organization’s dataset names and their variants.',
    ],
    verifies: ['Package extras on PyPI', 'Tests in the repository'],
    checked: '2026-10-13',
  },

  // ------------------------------------------------------------- implement
  {
    id: 'reference-implementation',
    name: 'End-to-end reference implementation',
    area: 'implement',
    kind: 'Reproducible pipeline',
    status: 'reference',
    summary:
      'A script that documents a set of series and a survey in the World Bank schemas, validates them against the pinned schemas and a completeness profile, publishes a catalog with schema.org records, serves it through an MCP server, composes an answer from the tool results, and verifies the numbers in the answer with the PCN engine.',
    stack: 'Python 3.10 or later; jsonschema, mcp, pcn-core',
    licence: 'MIT',
    docs: '/docs/reference-implementation/',
    install: '/docs/reference-implementation/#running-it',
    repo: GITHUB + 'ai4data/tree/main/reference/end_to_end',
    releases: null,
    operator: null,
    limits: [
      'The catalog is a directory of files, and the publication step to a NADA instance runs only when a catalog URL and an API key are given.',
      'The answer is composed from tool results by a template, so that the run needs no model; a model’s answer can be verified with the same step.',
      'The pipeline shows the pattern on the cookbook’s example organization; it is not a deployment of any of the applications above.',
    ],
    verifies: ['Test in CI that runs the pipeline and checks every step’s outcome'],
    checked: '2026-10-13',
  },
  {
    id: 'cookbook-scripts',
    name: 'Cookbook scripts',
    area: 'implement',
    kind: 'Scripts',
    status: 'reference',
    summary:
      'The scripts shipped with the cookbooks: checkers, scorers, builders, and converters that run on the shared example organization, each with a docstring stating its purpose, usage, exit status, and what it does not check.',
    stack: 'Python 3.11 or later; pandas, jsonschema, PyYAML, scikit-learn, mcp',
    licence: 'MIT',
    docs: '/cookbook/',
    install: '/cookbook/authoring/#checking-a-cookbook',
    repo: GITHUB + 'ai4data/tree/main/website/static/cookbook-files',
    releases: null,
    operator: null,
    limits: [
      'Each script does one check on one input layout; adapting it to an organization’s files is the first recipe of each chapter.',
    ],
    verifies: ['Lint, structure checks, and tests in CI; quoted outputs regenerated from the scripts'],
    checked: '2026-10-13',
  },
];

// Work announced by the program that has nothing to install or read yet.
export const IN_DEVELOPMENT = [
  {name: 'AI-assisted enrichment inside the Metadata Editor', area: 'document', note: 'Enrichment in the Editor itself; review is in the Editor as the Metadata Reviewer add-on, and the augmentation package runs beside it.'},
  {name: 'Global Question Bank', area: 'document', note: 'Question- and variable-level metadata with multilingual mappings across surveys.'},
  {name: 'Ontologies and knowledge graphs', area: 'document', note: 'Concept links between indicators, variables, and datasets.'},
  {name: 'Responsible AI guidance', area: 'verify', note: 'Guidance for AI use in official statistics under the UN Fundamental Principles; the governance instruments are in the cookbooks.'},
  {name: 'Statistical classification and coding', area: 'document', note: 'AI-assisted coding against standard classifications with human review; the production cookbook gives the method.'},
  {name: 'Small and agentic models for statistical tasks', area: 'verify', note: 'Small models and agentic workflows for production tasks; the small and open models cookbook gives the sizing and evaluation method.'},
  {name: 'Open benchmarks and evaluation', area: 'verify', note: 'Reusable benchmarks for search, metadata quality, and dissemination; the evaluation cookbook gives the suite method.'},
];

export const toolsByArea = (areaId) => TOOLS.filter((t) => t.area === areaId);
export const toolById = (id) => TOOLS.find((t) => t.id === id);
