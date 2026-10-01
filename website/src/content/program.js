// Single source of truth for the program structure shown on the home page.

export const workstreams = {
  // AI for Data: data quality and metadata
  metadataQuality: {
    title: 'Generative AI for Metadata Quality',
    to: '/docs/metadata-quality/generative-ai-for-metadata-quality',
    description:
      'Scores metadata on four dimensions: completeness, semantic alignment, specificity, and consistency. The LLM output is structured and auditable.',
    gsbpm: 'Metadata management',
  },
  metadataAugmentation: {
    title: 'Metadata Augmentation',
    to: '/docs/metadata-augmentation/',
    description:
      'Organizes hundreds of survey variables into DDI-style thematic groups using semantic clustering.',
    gsbpm: '5.2',
  },
  anomaly: {
    title: 'Anomaly Detection and Explanation',
    to: '/docs/anomaly-detection/',
    description:
      'Classifies flagged anomalies as an external driver, data error, measurement update, or insufficient data, and cites the evidence.',
    gsbpm: '5.3, 6.2, 6.3',
  },
  responsibleAi: {
    title: 'Responsible AI guidance',
    description:
      'Guidance and evaluation methods for using AI in official statistics, aligned with the UN Fundamental Principles of Official Statistics.',
    gsbpm: 'Quality management',
  },

  // AI for Data: data production
  classification: {
    title: 'Statistical classification and coding',
    description:
      'AI-assisted coding of survey responses and records against standard classifications, with human review of the results.',
    gsbpm: '5.2',
  },
  dataSnapshots: {
    title: 'Data Snapshots',
    to: 'https://arxiv.org/abs/2606.06242',
    description:
      'Layout detection models that locate figures and tables in PDF documents, so that the data they contain can be extracted. Includes a benchmark of open-source models.',
  },
  smallAgentic: {
    title: 'Small and Agentic AI',
    description:
      'Small language models and agentic workflows for statistical processes and specialized tasks, and AI assistants built on them.',
  },
  synthetic: {
    title: 'Synthetic data',
    to: 'https://github.com/avsolatorio/RealTabFormer',
    description:
      'Generates synthetic tabular and relational data with REaLTabFormer, an open-source transformer model, for data sharing and methodological research.',
    gsbpm: '6.4',
  },

  // AI for Data: discovery and trustworthy dissemination
  discoverability: {
    title: 'Data Discoverability',
    to: '/docs/data-discoverability/',
    description:
      'Conceptual search returns food insecurity indicators for the query “how many people go hungry”, although none of their titles contains those words.',
    gsbpm: '7.2, 7.5',
  },
  pcn: {
    title: 'Proof-Carrying Numbers',
    to: 'https://arxiv.org/abs/2509.06902',
    description:
      'Checks each number in a chatbot answer against the official record and marks it as verified or flagged.',
    gsbpm: '6.2, 7.2',
  },
  dataUse: {
    title: 'Monitoring of Data Use',
    to: '/docs/data_use/',
    description:
      'Extracts dataset mentions from reports and papers with zero-shot NER, including name variants such as “WDI” and “World Development Indicators.”',
    gsbpm: '8.1, 8.2',
  },

  // AI for Data: methods and open tools
  inclusive: {
    title: 'Inclusive AI Applications',
    to: '/docs/inclusive-ai/',
    description:
      'Multilingual embedding models covering 50+ languages, and batch inference at roughly half the cost of synchronous calls.',
  },
  evaluation: {
    title: 'Open benchmarks and evaluation',
    description:
      'Reusable benchmarks and evaluation methods for semantic search, metadata quality, and trustworthy dissemination.',
  },

  // Data for AI
  framework: {
    title: 'AI-ready data framework',
    to: '/docs/ai-ready-framework',
    description:
      'Defines the attributes of AI-ready data under FAIR+ and maps each dimension to the workstreams of the program.',
  },
  assessment: {
    title: 'AI-readiness assessment framework',
    to: '/ai-readiness-assessment',
    description:
      'Assesses the AI readiness of national statistical organizations across two pillars and twelve dimensions, covering both the institution and the data products and services it provides.',
  },
  standards: {
    title: 'AI-ready metadata and standards',
    description:
      'Implementation guidance for DDI, SDMX, DCAT, Croissant, and schema.org, so that metadata is machine-readable and multilingual.',
    gsbpm: 'Metadata management',
  },
  mcp: {
    title: 'Model Context Protocol',
    to: '/docs/mcp/',
    description:
      'An MCP server that any compatible AI client, such as Claude, ChatGPT, or a custom agent, can call.',
    gsbpm: '7.2, 7.3',
  },
  platforms: {
    title: 'AI-assisted metadata platforms',
    description:
      'AI-assisted metadata creation, semantic search, and AI-native APIs in the open-source Metadata Editor and NADA platforms.',
    gsbpm: '7.1',
  },
  knowledgeGraphs: {
    title: 'Ontologies and knowledge graphs',
    description:
      'Shared concept definitions and links between indicators, variables, and datasets, so that AI systems can resolve what a term refers to.',
  },
  questionBank: {
    title: 'Global Question Bank',
    description:
      'A shared repository of question- and variable-level metadata, with multilingual semantic mappings across surveys and countries.',
    gsbpm: '2.2',
  },
};

export const pillars = [
  {
    id: 'ai-for-data',
    label: 'AI for Data',
    lede: 'Apply AI to produce, curate, and disseminate development data with less manual effort and higher quality.',
    groups: [
      {
        label: 'Data production',
        items: ['classification', 'dataSnapshots', 'smallAgentic', 'synthetic'],
      },
      {
        label: 'Data quality and metadata',
        items: ['metadataQuality', 'metadataAugmentation', 'anomaly', 'responsibleAi'],
      },
      {
        label: 'Discovery and trustworthy dissemination',
        items: ['discoverability', 'pcn', 'dataUse'],
      },
      {
        label: 'Methods and open tools',
        items: ['inclusive', 'evaluation'],
      },
    ],
  },
  {
    id: 'data-for-ai',
    label: 'Data for AI',
    lede: 'Make development data discoverable, interpretable, and usable by AI systems through open standards and infrastructure.',
    groups: [
      {
        label: 'Assessment',
        items: ['assessment'],
      },
      {
        label: 'Standards and metadata',
        items: ['framework', 'standards'],
      },
      {
        label: 'Infrastructure',
        items: ['mcp', 'platforms'],
      },
      {
        label: 'Semantic knowledge',
        items: ['questionBank', 'knowledgeGraphs'],
      },
    ],
  },
];

// FAIR plus two dimensions that AI use adds. Attribute names follow the
// program's AI-ready data attributes.
export const dimensions = [
  {
    id: 'findable',
    label: 'Findable',
    question: 'Can AI find the right data?',
    attributes: ['Data discoverability', 'Comprehensive metadata'],
    items: ['discoverability', 'metadataAugmentation', 'evaluation'],
  },
  {
    id: 'accessible',
    label: 'Accessible',
    question: 'Can AI retrieve it, with context?',
    attributes: ['Openly accessible', 'Machine-readable', 'Real-time accessibility'],
    items: ['mcp', 'platforms', 'dataSnapshots'],
  },
  {
    id: 'interoperable',
    label: 'Interoperable',
    question: 'Can AI combine and interpret it?',
    attributes: ['Integrative', 'Machine-understandable', 'Contextual relevance'],
    items: ['standards', 'questionBank', 'knowledgeGraphs', 'classification'],
  },
  {
    id: 'reusable',
    label: 'Reusable',
    question: 'Is it documented well enough to reuse?',
    attributes: ['Comprehensive metadata', 'High data quality', 'Licensing and privacy'],
    items: ['metadataQuality', 'synthetic', 'dataUse'],
  },
  {
    id: 'trustworthy',
    label: 'Trustworthy',
    question: 'Can answers be traced and verified?',
    attributes: ['High data quality', 'Ethical and governance standards'],
    items: ['anomaly', 'pcn', 'responsibleAi'],
    plus: true,
  },
  {
    id: 'inclusive',
    label: 'Inclusive',
    question: 'Does it work across languages and countries?',
    attributes: ['Diversity and representativeness'],
    items: ['inclusive', 'smallAgentic'],
    plus: true,
  },
];
