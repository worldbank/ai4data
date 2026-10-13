// Content of the implementation hub: for each area, what an adopting team
// does first, the tools (by id from tools.js), the guides on this site, the
// cookbooks (by id from cookbooks.js; hidden drafts are filtered out at
// render time), and the assessment dimensions the area serves.

export const HUB = {
  assess: {
    start:
      'Score the organization on the twelve dimensions with the assessment framework, or take the five-minute cookbook self-assessment for a first orientation. Read the operational companion for the dimensions with the largest gaps.',
    tools: ['assessment-framework', 'operational-companion'],
    guides: [
      {title: 'AI-readiness assessment framework', to: '/ai-readiness-assessment', note: 'Two pillars, twelve dimensions, four maturity levels; the pathway for early-stage organizations.'},
      {title: 'Operational companion', to: '/ai-readiness-in-practice', note: 'Resources and evidence per dimension; effort and roles per recipe.'},
      {title: 'A modern statistical office', to: '/modern-nso', note: 'A day in an office that has implemented the recipes, by role and maturity level.'},
      {title: 'AI-ready data framework', to: '/docs/ai-ready-framework', note: 'The attributes of AI-ready data under FAIR, and how the workstreams map to them.'},
    ],
    cookbooks: ['assessment-to-pathway', 'governance-and-risk'],
    dimensions: ['1.1', '1.5'],
  },
  document: {
    start:
      'Document each product in the World Bank schema for its type with the Metadata Editor or the CSV template, run the completeness check at the foundational level, and publish to a catalog. Add AI-assisted review and enrichment once the records exist.',
    tools: ['metadata-editor', 'nada', 'metadata-schemas', 'metadata-reviewer', 'metadata-augmentation', 'metadata-quality', 'data-snapshots'],
    guides: [
      {title: 'Metadata Reviewer', to: '/docs/metadata-reviewer/overview', note: 'Five-agent review pipeline with a user manual and a review board.'},
      {title: 'Metadata Augmentation', to: '/docs/metadata-augmentation/', note: 'DDI-style variable groups generated for a data dictionary.'},
      {title: 'Generative AI for metadata quality', to: '/docs/metadata-quality/generative-ai-for-metadata-quality', note: 'Scoring records on completeness, alignment, specificity, and consistency.'},
      {title: 'Mapping to the GSBPM', to: '/docs/gsbpm-mapping', note: 'Where each workstream applies in the statistical production process.'},
    ],
    cookbooks: ['microdata-documentation', 'metadata-curation-with-llms', 'data-from-documents', 'ai-ready-dissemination'],
    dimensions: ['2.1', '2.2', '2.3'],
  },
  connect: {
    start:
      'Publish tidy files at stable addresses with a licence, then an API with provenance in every response, then an MCP server over the API. Connect a client to the Data360 MCP server to see the pattern before building one.',
    tools: ['data360-mcp', 'data-ai-chatbot', 'mcp-guidance', 'semantic-search'],
    guides: [
      {title: 'Model Context Protocol for statistics', to: '/docs/mcp/', note: 'Architecture, tool design, and the servers in operation.'},
      {title: 'MCP for official statistics', to: '/docs/mcp/mcp-for-official-statistics', note: 'Use cases for statistical offices and what a server over a catalog returns.'},
      {title: 'Data discoverability', to: '/docs/data-discoverability/', note: 'Search by meaning over catalog records, and fine-tuning embeddings on structured metadata.'},
      {title: 'PI-FT toolkit', to: '/pift-toolkit/pipeline', note: 'Pipeline, configuration, and deployment of fine-tuned embedding models.'},
    ],
    cookbooks: ['ai-ready-dissemination', 'serving-statistics-to-agents', 'ml-ready-datasets', 'synthetic-data-for-sharing'],
    dimensions: ['2.3', '2.4', '2.5'],
  },
  verify: {
    start:
      'Put a source and a citation on every published figure, return provenance from the API, and verify the numbers in generated answers against the records they cite. Build an evaluation suite before a model-assisted step goes into production.',
    tools: ['pcn', 'anomaly', 'data-use'],
    guides: [
      {title: 'Proof-Carrying Numbers', to: '/docs/pcn/', note: 'The protocol, the packages, the policies, and what a verified mark does and does not mean.'},
      {title: 'Anomaly detection and explanation', to: '/docs/anomaly-detection/', note: 'Evidence-backed explanations of flagged values with a reviewer workflow.'},
      {title: 'Monitoring of data use', to: '/docs/data_use/', note: 'Dataset mentions extracted from documents and harmonized.'},
    ],
    cookbooks: ['evaluation-suites', 'monitoring-data-use', 'language-models-in-production', 'small-and-open-models'],
    dimensions: ['1.6', '2.5', '2.6'],
  },
  implement: {
    start:
      'Run the end-to-end reference implementation on the example organization to see every step and its output. Then follow the cookbook for the data type or the service the organization is working on, with its scripts on the organization’s own files.',
    tools: ['reference-implementation', 'cookbook-scripts'],
    guides: [
      {title: 'End-to-end reference implementation', to: '/docs/reference-implementation/', note: 'Documentation, validation, publication, MCP access, and verification in one run.'},
      {title: 'Cookbook landing page', to: '/cookbook/', note: 'All cookbooks, by group and by reading path.'},
      {title: 'How the cookbooks are written', to: '/cookbook/authoring/', note: 'The standard every cookbook follows and the checks that enforce it.'},
    ],
    cookbooks: ['ai-ready-dissemination', 'microdata-documentation', 'serving-statistics-to-agents', 'evaluation-suites', 'language-models-in-production', 'ai-assistants-at-work'],
    dimensions: [],
  },
  adopt: {
    start:
      'Read the status, the installation guide, and the known limits of a tool on the tools page. Install it from the documentation alone, record what blocked you, and file the record as an issue in the tool’s repository.',
    tools: [],
    guides: [
      {title: 'Tools catalogue', to: '/tools', note: 'Every tool with its status, stack, licence, operator, installation guide, and known limits.'},
      {title: 'Contributing', to: 'https://github.com/worldbank/ai4data/blob/main/CONTRIBUTING.md', note: 'How changes to the methods, scripts, and documentation are proposed.'},
      {title: 'Code of conduct', to: 'https://github.com/worldbank/ai4data/blob/main/CODE_OF_CONDUCT.md', note: 'Expected behaviour in the repositories and their discussions.'},
      {title: 'Issues', to: 'https://github.com/worldbank/ai4data/issues', note: 'Bug reports, installation test records, and feature requests for this repository.'},
    ],
    cookbooks: ['governance-and-risk', 'ai-assistants-at-work'],
    dimensions: ['1.1', '1.2', '1.3'],
  },
};

// The installation test an adopting team records. The steps are the same for
// every tool; the issue carries the answers.
export const INSTALL_TEST = [
  {step: 'Choose the tool and read its row on the tools page: status, stack, licence, installation guide, and known limits.'},
  {step: 'Install from the installation guide alone, on a machine the original developers have not touched, and note every point where the guide was not enough.'},
  {step: 'Run the tool’s own smoke test: the test suite, the quick start, or the first recipe of the matching cookbook.'},
  {step: 'Record the elapsed time, the operating system and versions, the blockers, and the workarounds.'},
  {step: 'File the record as an issue in the tool’s repository with the title “Installation test: <tool> on <platform>”. A test that passed is as useful as one that failed.'},
];

// What the program offers to an adopting organization, and what it does not.
export const SUPPORT = {
  offers: [
    'Public repositories with an issue tracker for each tool, read by its maintainers.',
    'Documentation on this site and in each repository: user guides, installation guides, API descriptions where the tool has an API, and the cookbooks.',
    'Versioned releases where the tool has them, with a documented way to pin and roll back.',
    'Contact with the Development Data Group for organizations planning an adoption or a joint implementation.',
  ],
  excludes: [
    'Hosting or operation of a tool on behalf of another organization. The services in operation serve World Bank data.',
    'Service levels, on-call support, or guaranteed response times on issues.',
    'Adaptation of a tool to an organization’s systems, languages, and classifications, which the cookbooks describe and the organization carries out, alone or in a joint engagement.',
  ],
};

// Release and versioning practice per repository, for adopters who pin.
export const RELEASE_PRACTICE = [
  {repo: 'data360-mcp', practice: 'Semantic versioning with git tags and GitHub releases; the Python package on PyPI; rollback by installing a prior release.', to: 'https://github.com/worldbank/data360-mcp/releases'},
  {repo: 'metadata-editor', practice: 'Tagged releases with notes; upgrade by replacing the application files and running the installer.', to: 'https://github.com/worldbank/metadata-editor/releases'},
  {repo: 'metadata-schemas', practice: 'Tagged releases of the Python package; the schema files on the main branch are the reference, so pin a commit for validation that must match a catalog.', to: 'https://github.com/worldbank/metadata-schemas/releases'},
  {repo: 'pcn', practice: 'Package versions in the monorepo with changesets; packages are built from the repository until they are published to npm and PyPI.', to: 'https://github.com/worldbank/pcn'},
  {repo: 'data-ai-chatbot', practice: 'Releases from the repository; Docker Compose files for development and production; the deployment guide covers upgrades.', to: 'https://github.com/worldbank/data-ai-chatbot/releases'},
  {repo: 'ai4data', practice: 'The Python package on PyPI with extras per workstream; the cookbook scripts and the reference implementation are tested in CI on every change.', to: 'https://pypi.org/project/ai4data/'},
];
