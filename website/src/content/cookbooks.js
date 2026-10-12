import {isHiddenCookbook, isHiddenPath} from './visibility';

// Cookbooks listed on the /cookbook landing page. Each cookbook is a folder
// under /cookbook with its own sidebar in sidebars-cookbook.js. The ids in
// website/cookbook-visibility.json are dropped from every list exported here.
const allCookbooks = [
  {
    id: 'ai-ready-dissemination',
    audience: 'National statistical organizations',
    title: 'Practical Guide to AI-Ready Data Dissemination',
    description:
      'Helps a statistical organization make its published statistics findable, retrievable, and understandable by people and by AI systems. Nine chapters cover catalog records, open files and APIs, documented meaning, natural-language access, provenance, evaluation, monitoring of use, governance, and sustainability, each with recipes, worked examples, and a checklist.',
    chapters: 9,
    to: '/cookbook/ai-ready-dissemination/',
  },
  {
    id: 'microdata-documentation',
    audience: 'Data curators in national statistical organizations',
    title: 'Practical Guide to AI-Ready Microdata Documentation',
    description:
      'Helps data curators document survey microdata so that people and AI systems can find a survey, interpret its variables, and obtain the data under the stated conditions. Eight chapters run from producing documentation out of existing files to comparability across rounds and access tiers, built on DDI Codebook and the World Bank microdata schema.',
    chapters: 8,
    to: '/cookbook/microdata-documentation/',
  },
  {
    id: 'data-from-documents',
    audience: 'Dissemination, library, and research data teams',
    title: 'Practical Guide to Extracting Data from Documents',
    description:
      'Helps a team recover the statistics locked in its reports and PDFs as structured, verified, and citable data. Nine chapters cover the inventory of documents, text from scans, locating and extracting tables and figures, verification against the page, series across documents, publication, citation, and running the pipeline over a whole backlog.',
    chapters: 9,
    to: '/cookbook/data-from-documents/',
  },
  {
    id: 'monitoring-data-use',
    audience: 'Dissemination teams and management in national statistical organizations',
    title: 'Practical Guide to Monitoring Data Use',
    description:
      'Helps a dissemination team see where its data are used, in research, policy, news, and AI answers. Nine chapters cover what counts as use, access and citation counts, collecting documents, finding and matching mentions of the data, reporting use with its caveats, and acting on the results.',
    chapters: 9,
    to: '/cookbook/monitoring-data-use/',
  },
  {
    id: 'serving-statistics-to-agents',
    audience: 'Developers and dissemination teams in national statistical organizations',
    title: 'Practical Guide to Serving Official Statistics to AI Agents',
    description:
      'Helps developers and dissemination teams expose official statistics to AI assistants through the Model Context Protocol. Nine chapters cover the scope of agent access, the design of the tools, the context an assistant needs, provenance in every response, access control, safety, evaluation, distribution to clients, and operation, built on the Data360 MCP server.',
    chapters: 9,
    to: '/cookbook/serving-statistics-to-agents/',
  },
  {
    id: 'metadata-curation-with-llms',
    audience: 'Metadata curators and their managers in national statistical organizations',
    title: 'Practical Guide to Metadata Curation with Language Models',
    description:
      'Helps a curation team use language models on catalog metadata with a curator deciding every change. Eight chapters cover which tasks a model may assist, migration of legacy records, drafting of missing fields, quality scoring, review of suggestions, vocabularies, translation, and improvement from the decisions curators record.',
    chapters: 8,
    to: '/cookbook/metadata-curation-with-llms/',
  },
  {
    id: 'evaluation-suites',
    audience: 'Teams that deploy AI components in national statistical organizations',
    title: 'Practical Guide to Evaluation Suites for Statistical AI',
    description:
      'Helps a team build, run, and maintain the evaluation suites that decide whether a search, an assistant, an extractor, or a drafting tool is good enough to publish and safe to change. Nine chapters cover what to evaluate, question sets and labels, the measures for each kind of component, graders, gates on changes, evaluation in production, and reporting.',
    chapters: 9,
    to: '/cookbook/evaluation-suites/',
  },
  {
    id: 'language-models-in-production',
    audience: 'Methodologists and production managers in national statistical organizations',
    title: 'Practical Guide to Language Models in Statistical Production',
    description:
      'Helps methodologists and production managers use language models inside statistical production, from questionnaire design and open-text coding to editing, anomaly explanation, and release commentary. Seven chapters keep a person at every point of decision and document model use in the quality report.',
    chapters: 7,
    to: '/cookbook/language-models-in-production/',
  },
  {
    id: 'synthetic-data-for-sharing',
    audience: 'Microdata teams and methodologists in national statistical organizations',
    title: 'Practical Guide to Synthetic Data for Sharing',
    description:
      'Helps microdata teams produce synthetic files that applicants, students, and developers can work with while the real file stays protected. Seven chapters cover what synthetic data are for, the choice of method, utility and disclosure risk, linked tables and panels, documentation and release, and the place of synthetic files in the access workflow.',
    chapters: 7,
    to: '/cookbook/synthetic-data-for-sharing/',
  },
  {
    id: 'small-and-open-models',
    audience: 'IT and data science leads in national statistical organizations',
    title: 'Practical Guide to Small and Open Models for Statistical Offices',
    description:
      'Helps IT and data science leads run small and open-weight models on the organization\'s own servers for its narrow tasks. Seven chapters cover when a small model is enough, candidates and licences, hardware and serving, adaptation, comparison with larger models on the organization\'s own suite, security, and cost.',
    chapters: 7,
    to: '/cookbook/small-and-open-models/',
  },
  {
    id: 'ml-ready-datasets',
    audience: 'Data managers and dissemination teams in national statistical organizations',
    title: 'Practical Guide to Publishing ML-Ready Datasets',
    description:
      'Helps data managers publish the examples a statistical organization produces, such as coded responses and transcribed tables, as datasets for training and evaluating models. Eight chapters cover selection, file form, Croissant records, dataset cards, representativeness, splits, licence and privacy, and versions.',
    chapters: 8,
    to: '/cookbook/ml-ready-datasets/',
  },
];

export const cookbooks = allCookbooks.filter((c) => !isHiddenCookbook(c.id));

// The cookbooks grouped by what they are about, in the order a reader meets them.
const allGroups = [
  {
    title: 'Data products that AI can use',
    ids: ['ai-ready-dissemination', 'microdata-documentation', 'data-from-documents', 'ml-ready-datasets'],
  },
  {
    title: 'Metadata and interfaces',
    ids: ['metadata-curation-with-llms', 'serving-statistics-to-agents'],
  },
  {
    title: 'Models inside the office',
    ids: ['language-models-in-production', 'small-and-open-models', 'synthetic-data-for-sharing'],
  },
  {
    title: 'Measurement',
    ids: ['evaluation-suites', 'monitoring-data-use'],
  },
];

export const groups = allGroups
  .map((g) => ({...g, ids: g.ids.filter((id) => !isHiddenCookbook(id))}))
  .filter((g) => g.ids.length > 0);

// Where to start, by role. Each path is three or four chapters in order.
const allReadingPaths = [
  {
    role: 'Dissemination lead',
    steps: [
      {to: '/cookbook/ai-ready-dissemination/find', label: 'Findable records and search by meaning'},
      {to: '/cookbook/ai-ready-dissemination/trust', label: 'Source, citation, and verified numbers'},
      {to: '/cookbook/serving-statistics-to-agents/purpose', label: 'What an agent interface should do'},
      {to: '/cookbook/monitoring-data-use/define', label: 'What counts as use of the data'},
    ],
  },
  {
    role: 'Microdata curator',
    steps: [
      {to: '/cookbook/microdata-documentation/produce', label: 'A dictionary from the files the office has'},
      {to: '/cookbook/microdata-documentation/variables', label: 'The dictionary check before release'},
      {to: '/cookbook/metadata-curation-with-llms/draft', label: 'Model drafts under a curator\'s decision'},
      {to: '/cookbook/synthetic-data-for-sharing/purpose', label: 'When a synthetic file helps'},
    ],
  },
  {
    role: 'Methodologist',
    steps: [
      {to: '/cookbook/language-models-in-production/map', label: 'Where a model may assist production'},
      {to: '/cookbook/language-models-in-production/coding', label: 'Coding with a threshold and a re-coded sample'},
      {to: '/cookbook/evaluation-suites/questions', label: 'Test questions and labels'},
      {to: '/cookbook/ml-ready-datasets/represent', label: 'Representativeness of training data'},
    ],
  },
  {
    role: 'IT and data science lead',
    steps: [
      {to: '/cookbook/small-and-open-models/when', label: 'When a small model is enough'},
      {to: '/cookbook/small-and-open-models/serving', label: 'Sizing and running a model server'},
      {to: '/cookbook/evaluation-suites/gates', label: 'Gates on every change'},
      {to: '/cookbook/serving-statistics-to-agents/safety', label: 'Keeping the interface safe'},
    ],
  },
  {
    role: 'Management',
    steps: [
      {to: '/cookbook/ai-ready-dissemination/govern', label: 'The AI-use policy and the component register'},
      {to: '/cookbook/language-models-in-production/assurance', label: 'The statement of model use'},
      {to: '/cookbook/small-and-open-models/cost', label: 'What models cost to sustain'},
    ],
  },
];

export const readingPaths = allReadingPaths
  .map((p) => ({...p, steps: p.steps.filter((st) => !isHiddenPath(st.to))}))
  .filter((p) => p.steps.length > 0);
