import {isHiddenCookbook, isHiddenPath, isIndexedCookbook} from './visibility';

// Cookbooks listed on the /cookbook landing page. Each cookbook is a folder
// under /cookbook with its own sidebar in sidebars-cookbook.js. The ids in
// website/cookbook-visibility.json are dropped from every list exported here.
export const allCookbooks = [
  {
    id: 'ai-ready-dissemination',
    version: '0.1',
    date: 'October 2026',
    audience: 'National statistical organizations',
    title: 'Practical Guide to AI-Ready Data Dissemination',
    description:
      'Helps a statistical organization make its published statistics findable, retrievable, and understandable by people and by AI systems. Nine chapters cover catalog records, open files and APIs, documented meaning, natural-language access, provenance, evaluation, monitoring of use, governance, and sustainability, each with recipes, worked examples, and a checklist.',
    chapters: 9,
    to: '/cookbook/ai-ready-dissemination/',
  },
  {
    id: 'microdata-documentation',
    version: '0.1',
    date: 'October 2026',
    audience: 'Data curators in national statistical organizations',
    title: 'Practical Guide to AI-Ready Microdata Documentation',
    description:
      'Helps data curators document survey microdata so that people and AI systems can find a survey, interpret its variables, and obtain the data under the stated conditions. Eight chapters run from producing documentation out of existing files to comparability across rounds and access tiers, built on DDI Codebook and the World Bank microdata schema.',
    chapters: 8,
    to: '/cookbook/microdata-documentation/',
  },
  {
    id: 'data-from-documents',
    version: '0.1',
    date: 'October 2026',
    audience: 'Dissemination, library, and research data teams',
    title: 'Practical Guide to Extracting Data from Documents',
    description:
      'Helps a team recover the statistics locked in its reports and PDFs as structured, verified, and citable data. Nine chapters cover the inventory of documents, text from scans, locating and extracting tables and figures, verification against the page, series across documents, publication, citation, and running the pipeline over a whole backlog.',
    chapters: 9,
    to: '/cookbook/data-from-documents/',
  },
  {
    id: 'monitoring-data-use',
    version: '0.1',
    date: 'October 2026',
    audience: 'Dissemination teams and management in national statistical organizations',
    title: 'Practical Guide to Monitoring Data Use',
    description:
      'Helps a dissemination team see where its data are used, in research, policy, news, and AI answers. Nine chapters cover what counts as use, access and citation counts, collecting documents, finding and matching mentions of the data, reporting use with its caveats, and acting on the results.',
    chapters: 9,
    to: '/cookbook/monitoring-data-use/',
  },
  {
    id: 'serving-statistics-to-agents',
    version: '0.1',
    date: 'October 2026',
    audience: 'Developers and dissemination teams in national statistical organizations',
    title: 'Practical Guide to Serving Official Statistics to AI Agents',
    description:
      'Helps developers and dissemination teams expose official statistics to AI assistants through the Model Context Protocol. Nine chapters cover the scope of agent access, the design of the tools, the context an assistant needs, provenance in every response, access control, safety, evaluation, distribution to clients, and operation, built on the Data360 MCP server.',
    chapters: 9,
    to: '/cookbook/serving-statistics-to-agents/',
  },
  {
    id: 'metadata-curation-with-llms',
    version: '0.1',
    date: 'October 2026',
    audience: 'Metadata curators and their managers in national statistical organizations',
    title: 'Practical Guide to Metadata Curation with Language Models',
    description:
      'Helps a curation team use language models on catalog metadata with a curator deciding every change. Eight chapters cover which tasks a model may assist, migration of legacy records, drafting of missing fields, quality scoring, review of suggestions, vocabularies, translation, and improvement from the decisions curators record.',
    chapters: 8,
    to: '/cookbook/metadata-curation-with-llms/',
  },
  {
    id: 'evaluation-suites',
    version: '0.1',
    date: 'October 2026',
    audience: 'Teams that deploy AI components in national statistical organizations',
    title: 'Practical Guide to Evaluation Suites for Statistical AI',
    description:
      'Helps a team build, run, and maintain the evaluation suites that decide whether a search, an assistant, an extractor, or a drafting tool is good enough to publish and safe to change. Nine chapters cover what to evaluate, question sets and labels, the measures for each kind of component, graders, gates on changes, evaluation in production, and reporting.',
    chapters: 9,
    to: '/cookbook/evaluation-suites/',
  },
  {
    id: 'language-models-in-production',
    version: '0.1',
    date: 'October 2026',
    audience: 'Methodologists and production managers in national statistical organizations',
    title: 'Practical Guide to Language Models in Statistical Production',
    description:
      'Helps methodologists and production managers use language models inside statistical production, from questionnaire design and open-text coding to editing, anomaly explanation, and release commentary. Seven chapters keep a person at every point of decision and document model use in the quality report.',
    chapters: 7,
    to: '/cookbook/language-models-in-production/',
  },
  {
    id: 'synthetic-data-for-sharing',
    version: '0.1',
    date: 'October 2026',
    audience: 'Microdata teams and methodologists in national statistical organizations',
    title: 'Practical Guide to Synthetic Data for Sharing',
    description:
      'Helps microdata teams produce synthetic files that applicants, students, and developers can work with while the real file stays protected. Seven chapters cover what synthetic data are for, the choice of method, utility and disclosure risk, linked tables and panels, documentation and release, and the place of synthetic files in the access workflow.',
    chapters: 7,
    to: '/cookbook/synthetic-data-for-sharing/',
  },
  {
    id: 'small-and-open-models',
    version: '0.1',
    date: 'October 2026',
    audience: 'IT and data science leads in national statistical organizations',
    title: 'Practical Guide to Small and Open Models for Statistical Offices',
    description:
      'Helps IT and data science leads run small and open-weight models on the organization\'s own servers for its narrow tasks. Seven chapters cover when a small model is enough, candidates and licences, hardware and serving, adaptation, comparison with larger models on the organization\'s own suite, security, and cost.',
    chapters: 7,
    to: '/cookbook/small-and-open-models/',
  },
  {
    id: 'ml-ready-datasets',
    version: '0.1',
    date: 'October 2026',
    audience: 'Data managers and dissemination teams in national statistical organizations',
    title: 'Practical Guide to Publishing ML-Ready Datasets',
    description:
      'Helps data managers publish the examples a statistical organization produces, such as coded responses and transcribed tables, as datasets for training and evaluating models. Eight chapters cover selection, file form, Croissant records, dataset cards, representativeness, splits, licence and privacy, and versions.',
    chapters: 8,
    to: '/cookbook/ml-ready-datasets/',
  },
  {
    id: 'governance-and-risk',
    version: '0.1',
    date: 'October 2026',
    audience: 'Management, policy owners, and legal advisers in national statistical organizations',
    title: 'Practical Guide to Governing AI Use in a Statistical Organization',
    description:
      'Helps management and the owner of the AI-use policy govern every AI component the organization runs: a written position, a register with owners, risk ratings with matching controls, a board that records decisions, provider dependence, licences and terms for AI use, the statement of model use, and the review before a re-assessment.',
    chapters: 8,
    to: '/cookbook/governance-and-risk/',
  },
  {
    id: 'assessment-to-pathway',
    version: '0.1',
    date: 'October 2026',
    audience: 'Management and planning teams in national statistical organizations, and the advisers who support them',
    title: 'Practical Guide from the Readiness Assessment to an Improvement Pathway',
    description:
      'Helps management and the advisers who support them turn an AI-readiness profile into an agreed improvement pathway: reading the profile, setting target levels and sizing the gaps, ordering the work by dependency, estimating effort and kinds of cost, writing the pathway document, keeping an evidence register per question, and measuring progress at the re-assessment.',
    chapters: 7,
    to: '/cookbook/assessment-to-pathway/',
  },
  {
    id: 'ai-assistants-at-work',
    version: '0.1',
    date: 'October 2026',
    audience: 'All staff of national statistical organizations, and the managers who set the rules for assistant use',
    title: 'Practical Guide to AI Assistants in Everyday Statistical Work',
    description:
      'Helps every member of staff use a general AI assistant for everyday tasks under the organization\'s rules: which tasks an assistant may help with, what may be entered into one, how to ask so that the answer can be checked, how to check an answer before it is used, how to record use, and how a unit adopts assistants together.',
    chapters: 6,
    to: '/cookbook/ai-assistants-at-work/',
  },
];

export const cookbooks = allCookbooks.filter((c) => !isHiddenCookbook(c.id));

// The cookbook a page belongs to, from its pathname (with or without the
// site's base URL), and whether the page is the cookbook's index. Returns
// null outside the cookbooks and for hidden cookbooks, whose pages are
// unlisted by the site config. `draft` is true until the cookbook is listed
// as indexed in website/cookbook-visibility.json.
export function cookbookPage(pathname) {
  const m = /\/cookbook\/([^/#?]+)\/?([^#?]*)/.exec(pathname || '');
  if (!m) {
    return null;
  }
  const entry = allCookbooks.find((c) => c.id === m[1]);
  if (!entry || isHiddenCookbook(entry.id)) {
    return null;
  }
  return {entry, isIndex: m[2] === '', draft: !isIndexedCookbook(entry.id)};
}

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
  {
    title: 'Governance and planning',
    ids: ['assessment-to-pathway', 'governance-and-risk', 'ai-assistants-at-work'],
  },
];

export const groups = allGroups
  .map((g) => ({...g, ids: g.ids.filter((id) => !isHiddenCookbook(id))}))
  .filter((g) => g.ids.length > 0);

// Where to start, by role. Each path is three or four chapters in order.
const allReadingPaths = [
  {
    role: 'All staff',
    steps: [
      {to: '/cookbook/ai-assistants-at-work/tasks', label: 'The task card for the role'},
      {to: '/cookbook/ai-assistants-at-work/inputs', label: 'The three tiers before the paste'},
      {to: '/cookbook/ai-assistants-at-work/checking', label: 'Every figure checked before use'},
    ],
  },
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
      {to: '/cookbook/assessment-to-pathway/targets', label: 'Target levels and the gap per dimension'},
      {to: '/cookbook/governance-and-risk/position', label: 'The organization\'s position on AI'},
      {to: '/cookbook/governance-and-risk/risk', label: 'Risk tiers and the controls each requires'},
      {to: '/cookbook/ai-ready-dissemination/govern', label: 'The AI-use policy and the component register'},
      {to: '/cookbook/language-models-in-production/assurance', label: 'The statement of model use'},
      {to: '/cookbook/small-and-open-models/cost', label: 'What models cost to sustain'},
    ],
  },
];

export const readingPaths = allReadingPaths
  .map((p) => ({...p, steps: p.steps.filter((st) => !isHiddenPath(st.to))}))
  .filter((p) => p.steps.length > 0);
