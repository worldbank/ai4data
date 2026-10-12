// Planning draft for an operational project that supports the full
// implementation of AI-readiness in one national statistical office (NSO).
// The page that renders this is unlisted. It carries no budget figures,
// dates, counterpart names, funders, or numeric targets: the sequence is
// given by dependencies and the money by kind of cost.
//
// Vocabulary. "The project" is the operational project that finances the
// office. "The program" is AI for Data - Data for AI, which provides tools,
// methods, cookbooks, and advisory work. Each area says what the office
// needs, what the project would finance, what the program provides today,
// and where the program could expand.

const CB = '/cookbook/';

export const LEVELS = [
  {
    id: 'available',
    label: 'Available now',
    short: 'Available',
    description:
      'The program provides this today as an open-source tool, a method, a cookbook recipe, or a reference implementation.',
  },
  {
    id: 'adapt',
    label: 'Adaptation financed by the project',
    short: 'Adapt',
    description:
      'The program has the component. The project finances its adaptation to the office: languages, classifications, existing systems, and staff.',
  },
  {
    id: 'expand',
    label: 'Expansion opportunity',
    short: 'Expand',
    description:
      'The program does not provide this yet. The project is an occasion to develop it once and reuse it in later engagements.',
  },
  {
    id: 'project',
    label: 'Project only',
    short: 'Project',
    description:
      'The project finances this outside the program: procurement, salaries, legal drafting, and facilities.',
  },
];

export const PHASES = [
  {
    id: 'p0',
    n: '0',
    title: 'Assessment and project design',
    gate: 'Entry: a request from the office and agreement on the scope of the assessment.',
    summary:
      'The office is scored on the twelve dimensions of the assessment framework. The scores, the gaps between current and target levels, and the dependencies between dimensions give the project its components and their order. The results framework of the project takes its indicators from the assessment so that a re-assessment at the end measures the change.',
    project: [
      'A project preparation mission with the office and its main data users.',
      'An inventory of the systems, datasets, classifications, and staff the office already has.',
      'A procurement and staffing plan derived from the component list.',
    ],
    program: [
      'The assessment framework and its scoring, applied with the office.',
      'The operational companion that maps each dimension to tools and recipes.',
      'The investment view of the cookbook, which gives effort, roles, and kinds of cost per recipe.',
    ],
  },
  {
    id: 'p1',
    n: '1',
    title: 'Foundations',
    gate: 'Entry: a signed project with a named project implementation unit inside the office.',
    summary:
      'The office puts in place what every later component depends on: an AI-use policy with an owner for each component, a metadata standard applied to the catalog, a quality procedure with status codes and a revisions log, a licence on every published product, and a catalog with stable URLs. Training starts with the staff who will own these artefacts.',
    project: [
      'Staff time of curators and methodologists, which is the largest cost of this phase.',
      'A catalog host and version control if the office has neither.',
      'Legal review of the statistics act and of data licences for use in AI.',
      'The first training cycle for curators, methodologists, and the dissemination unit.',
    ],
    program: [
      'The Metadata Editor, the metadata schemas, and the NADA catalog.',
      'The completeness checker, the dictionary check, and the release procedure of the cookbook.',
      'The AI-use policy template and the component register.',
      'The cookbook as the training material of the first cycle.',
    ],
  },
  {
    id: 'p2',
    n: '2',
    title: 'AI-ready operations',
    gate: 'Entry: the foundations are in production and the catalog passes the completeness profile.',
    summary:
      'The office adds the components that make its products usable by AI systems and that use AI in production under review. Semantic search and a documented API sit beside the catalog. An MCP server and an assistant that verifies every number sit on the API. Anomaly detection, classification and coding, and metadata review run with a human decision on every output. Evaluation suites gate each change.',
    project: [
      'A GPU server, or hosted-model usage for public inputs, and an API gateway with monitoring.',
      'A data science and engineering team inside the office, recruited or seconded.',
      'Reviewer time for calibration samples and a monthly sample of answers in each language.',
      'The second training cycle, which covers model operation, evaluation, and review.',
    ],
    program: [
      'Reference implementations of the MCP server, Proof-Carrying Numbers, semantic search, and anomaly detection.',
      'The evaluation suite method and the question sets of the cookbook.',
      'The small and open models guidance for sizing and serving a local model.',
      'Adaptation of models and retrieval to the office\'s languages and records.',
    ],
  },
  {
    id: 'p3',
    n: '3',
    title: 'Institutionalization',
    gate: 'Entry: the AI-ready components have run through at least one full release cycle with their evaluation suites.',
    summary:
      'The office owns and funds the components from its own budget, publishes a statement of model use with each release, monitors data use, and re-assesses itself. The project closes with a re-assessment on the same framework and with the office contributing its question sets, evaluation suites, and records to the shared resources of the program.',
    project: [
      'Transfer of hosting, maintenance contracts, and model licences to the office\'s own budget.',
      'A final re-assessment and a published results report.',
      'Training of trainers so that the office runs later training cycles itself.',
    ],
    program: [
      'The data-use monitoring method and its reports.',
      'The statement of model use and the governance recipes of the cookbook.',
      'Shared resources that the office joins: the Global Question Bank, evaluation suites, and the peer network of offices.',
    ],
  },
];

export const CATEGORIES = [
  {id: 'governance', label: 'Governance and legal'},
  {id: 'people', label: 'People and training'},
  {id: 'assets', label: 'Infrastructure and assets'},
  {id: 'metadata', label: 'Metadata and catalog'},
  {id: 'access', label: 'Access for AI systems'},
  {id: 'production', label: 'AI in production'},
  {id: 'measurement', label: 'Measurement and sustainability'},
];

// Each area: what the office needs, what the project finances, what the
// program contributes (with a level), where the program could expand, the
// phases in which the area is active, the assessment dimensions it serves,
// and the areas it depends on.
export const AREAS = [
  {
    id: 'ai-strategy',
    category: 'governance',
    title: 'AI strategy and governance of the office',
    dimensions: ['1.1'],
    need:
      'A written position of the office on AI: which tasks a model may assist, which decisions stay with people, who owns each AI component, and how model use is reported to users of the statistics. The assessment framework scores this under strategy and governance, and it gates the dimensions that follow.',
    finances: [
      'Management time for a strategy adopted at the level of the head of the office.',
      'A governance board that meets on a schedule and records its decisions.',
    ],
    contributes: [
      {level: 'available', text: 'The one-page AI-use policy template and the component register with review dates.', to: `${CB}ai-ready-dissemination/govern`},
      {level: 'available', text: 'The map of model-assisted tasks and the tasks that stay with people.', to: `${CB}language-models-in-production/map`},
      {level: 'available', text: 'Responsible AI guidance aligned with the UN Fundamental Principles of Official Statistics (in development).'},
      {level: 'adapt', text: 'A workshop series with management that turns the template into the office\'s own policy and register.'},
    ],
    expansions: [
      'A model AI strategy for a statistical office, written at the level of a national statistical plan, with the policy and register as its annexes.',
      'A governance review service: a short structured review of the policy, register, and board records, run before each re-assessment.',
    ],
    phases: ['p0', 'p1', 'p3'],
    dependsOn: [],
    evidence: ['The adopted AI-use policy', 'The component register', 'Minutes of the governance board'],
  },
  {
    id: 'legal',
    category: 'governance',
    title: 'Legal environment and data licensing',
    dimensions: ['1.2', '2.6'],
    need:
      'A statistics act and data protection rules that allow the office to use AI in production and to publish products that AI systems may use. Every published product carries a licence, and the licence states whether the data may be used to train models and under what terms.',
    finances: [
      'Legal review of the statistics act, the data protection law, and the office\'s own regulations.',
      'Drafting of licences and of terms of use for machine clients.',
    ],
    contributes: [
      {level: 'available', text: 'Licence and terms for model training, free-text screening, and disclosure review.', to: `${CB}ml-ready-datasets/license`},
      {level: 'available', text: 'Terms of use for agents, access tiers, and client registration.', to: `${CB}serving-statistics-to-agents/access`},
      {level: 'available', text: 'Reading and classifying model licences.', to: `${CB}small-and-open-models/candidates`},
    ],
    expansions: [
      'A legal checklist for statistics acts and data protection rules on the use of AI in production and on data for AI, with model clauses that a legal adviser can adapt.',
      'A licence selector for statistical products that records the choice and the reasons in the catalog.',
    ],
    phases: ['p1'],
    dependsOn: ['ai-strategy'],
    evidence: ['Licences in every catalog record', 'Terms of use published with the API', 'The legal review report'],
  },
  {
    id: 'staffing',
    category: 'people',
    title: 'Roles and staffing',
    dimensions: ['1.3', '1.5'],
    need:
      'The office has people in the roles that the components need: a metadata lead and curators, a data engineer, a data scientist, an evaluation reviewer, and an owner of the API and catalog. Many offices have the curators and methodologists and lack the engineering roles. The investment view of the cookbook lists the roles that each recipe needs.',
    finances: [
      'Recruitment or secondment of a data engineer and a data scientist into the office.',
      'A project implementation unit with a coordinator and a procurement officer.',
      'Retention measures so that trained staff stay with the office after the project.',
    ],
    contributes: [
      {level: 'available', text: 'Roles and effort per recipe, drawn from the cookbook.', to: '/ai-readiness-in-practice'},
      {level: 'available', text: 'Reading paths by role across the cookbook series.', to: '/cookbook/#reading-paths'},
      {level: 'adapt', text: 'Terms of reference for the engineering roles, derived from the component list of the project.'},
    ],
    expansions: [
      'A fellowship or secondment scheme that places a data scientist from the program or a partner in the office for the AI-ready phase and pairs them with a counterpart.',
      'A help desk for implementing offices, with a shared issue tracker and office hours.',
    ],
    phases: ['p0', 'p1', 'p2'],
    dependsOn: ['ai-strategy'],
    evidence: ['An organizational chart with the roles filled', 'Terms of reference', 'A staff register with training completed'],
  },
  {
    id: 'training',
    category: 'people',
    title: 'Training and skills',
    dimensions: ['1.3'],
    need:
      'Staff who own an artefact can produce and check it, and staff who use model outputs can read an evaluation report and record a review decision. Training follows the components as they arrive, so that each cycle teaches what the next phase needs.',
    finances: [
      'Three training cycles: foundations, model operation and evaluation, and training of trainers.',
      'Travel and venue costs for in-person cycles and the time of staff who attend.',
    ],
    contributes: [
      {level: 'available', text: 'The cookbook series, with worked examples on one running example organization and tests for each recipe.', to: '/cookbook/'},
      {level: 'available', text: 'The self-assessment that places a team and chooses a reading order.', to: `${CB}ai-ready-dissemination/start-here`},
      {level: 'available', text: 'The notebooks and reference implementations in the repository.', to: '/docs/introduction'},
      {level: 'adapt', text: 'A curriculum assembled from the cookbooks for the office, with its own data in the exercises.'},
    ],
    expansions: [
      'A structured curriculum with modules, exercises, and a completion record for each role, built from the cookbooks and maintained with them.',
      'A training-of-trainers module so that an office runs later cycles itself.',
      'A sandbox environment with the running example organization and the tools installed, so that a cycle starts without a local setup.',
    ],
    phases: ['p1', 'p2', 'p3'],
    dependsOn: ['staffing'],
    evidence: ['Completion records per role', 'Exercises done on the office\'s own data', 'A trainer roster'],
  },
  {
    id: 'compute',
    category: 'assets',
    title: 'Compute, hosting, and model serving',
    dimensions: ['1.4'],
    need:
      'A place to run models on confidential inputs inside the office\'s own perimeter, hosting for the catalog, the search index, and the API, and a development environment with version control and continuous integration. The size depends on the office: a small office needs one GPU server and a medium office a shared server with evaluation runners.',
    finances: [
      'A GPU server, or hosted-model usage restricted to public inputs, with an isolated segment for confidential inputs.',
      'Hosting for the catalog, the search index, and the API, with monitoring.',
      'Version control, continuous integration, and a development environment.',
    ],
    contributes: [
      {level: 'available', text: 'Sizing and running a model server for an office of each size.', to: `${CB}small-and-open-models/serving`},
      {level: 'available', text: 'Hosting notes per dimension in the operational companion.', to: '/ai-readiness-in-practice'},
      {level: 'adapt', text: 'Technical specifications for procurement, derived from the component list and the office\'s size.'},
    ],
    expansions: [
      'A reference architecture for a statistical office, with deployment packages (containers and configuration) for the catalog, the search index, the API, the MCP server, and the model server.',
      'A hosted option through a partner for offices that cannot run a server, with confidential inputs kept out of it.',
    ],
    phases: ['p1', 'p2'],
    dependsOn: ['staffing'],
    evidence: ['A systems inventory with owners', 'Monitoring dashboards', 'The deployment configuration under version control'],
  },
  {
    id: 'metadata',
    category: 'metadata',
    title: 'Metadata standards and classifications',
    dimensions: ['2.1'],
    need:
      'Every product has a complete, machine-readable record in a standard schema (DDI for microdata, SDMX for indicators, DCAT and schema.org for discovery), classifications are published as code lists with crosswalks, and the records are checked against a completeness profile before release.',
    finances: [
      'Curator time to document the backlog of datasets and to maintain records at each release.',
      'Migration of existing records into the editor and the catalog.',
    ],
    contributes: [
      {level: 'available', text: 'The Metadata Editor and the World Bank metadata schemas.', to: 'https://github.com/worldbank/metadata-editor'},
      {level: 'available', text: 'The NADA catalog.', to: 'https://nada.ihsn.org/'},
      {level: 'available', text: 'The Metadata Reviewer and generative AI for metadata quality.', to: '/docs/metadata-reviewer/overview'},
      {level: 'available', text: 'The completeness checker and the dictionary check.', to: `${CB}ai-ready-dissemination/find`},
      {level: 'adapt', text: 'Classification and coding adapted to the office\'s classifications and languages.', to: `${CB}language-models-in-production/coding`},
    ],
    expansions: [
      'A migration tool from the common legacy formats of statistical offices into the editor.',
      'Integration of the Global Question Bank into the editor, so that a questionnaire is documented against shared concepts as it is entered.',
      'An ontology and knowledge graph service over the office\'s records, as a shared component.',
    ],
    phases: ['p1', 'p2'],
    dependsOn: ['legal'],
    evidence: ['Records that pass the completeness profile', 'Published code lists and crosswalks', 'Review decisions in the reviewer log'],
  },
  {
    id: 'quality',
    category: 'metadata',
    title: 'Quality control with model assistance',
    dimensions: ['2.2'],
    need:
      'Status codes and a revisions log on every series, anomaly detection on the main series with an explanation for each flag, and a review board that records its decisions. Where a model assists, an evaluation suite gates each change and a statement of model use accompanies each release.',
    finances: [
      'Methodologist and curator time for calibration samples and reviews.',
      'A share of the GPU server for scoring and explanations on confidential series.',
    ],
    contributes: [
      {level: 'available', text: 'Anomaly detection and explanation.', to: '/docs/anomaly-detection/'},
      {level: 'available', text: 'Evaluation suites per model-assisted task.', to: `${CB}evaluation-suites/`},
      {level: 'available', text: 'The statement of model use and the decision points of production.', to: `${CB}language-models-in-production/assurance`},
    ],
    expansions: [
      'A review board toolkit: the board\'s terms of reference, a decision record format, and a dashboard of flags and decisions over releases.',
      'Shared evaluation suites across offices for the common tasks, so that an office starts from a suite and adds its own cases.',
    ],
    phases: ['p2', 'p3'],
    dependsOn: ['metadata', 'compute'],
    evidence: ['The revisions log', 'Evaluation reports attached to releases', 'The statement of model use'],
  },
  {
    id: 'catalog',
    category: 'metadata',
    title: 'Searchable catalog and discovery',
    dimensions: ['2.3'],
    need:
      'A catalog with a stable URL and schema.org markup on every record, DOIs for datasets, search by meaning beside keyword search, and search at the level of variables for documented microdata. Retrieval is measured on a question set that the office maintains in each of its languages.',
    finances: [
      'DataCite membership or a consortium fee for DOIs.',
      'A search host at the AI-ready level.',
      'Staff time to write and maintain question sets in each language.',
    ],
    contributes: [
      {level: 'available', text: 'Data discoverability methods and the retrieval scorer.', to: '/docs/data-discoverability/'},
      {level: 'available', text: 'The PI-FT embedding toolkit for retrieval fine-tuned on the office\'s own records.', to: '/pift-toolkit/pipeline'},
      {level: 'available', text: 'Findable records and search by meaning.', to: `${CB}ai-ready-dissemination/find`},
      {level: 'adapt', text: 'Embedding models and question sets in the office\'s languages.'},
    ],
    expansions: [
      'A semantic search module packaged for NADA, so that an office installs search by meaning without building it.',
      'Multilingual retrieval models for the languages of the regions where the program works, trained once and shared.',
    ],
    phases: ['p1', 'p2'],
    dependsOn: ['metadata'],
    evidence: ['Retrieval scores on the question set', 'DOIs resolved for each dataset', 'Schema.org markup validated on sampled pages'],
  },
  {
    id: 'api',
    category: 'access',
    title: 'API access with provenance',
    dimensions: ['2.4'],
    need:
      'Tidy files at stable URLs and a documented API (OpenAPI or an SDMX web service) with provenance in every response, versioned datasets with retrievable vintages, monitoring of availability and latency, and a deprecation policy.',
    finances: [
      'Developer time for the API and its documentation.',
      'An API gateway with rate limits and monitoring.',
      'Hosting that grows with machine traffic.',
    ],
    contributes: [
      {level: 'available', text: 'The WDI SDMX API and the Data360 API as reference designs.', to: 'https://data360.worldbank.org/en/api'},
      {level: 'available', text: 'What an agent interface should do and how an API carries provenance.', to: `${CB}serving-statistics-to-agents/purpose`},
      {level: 'available', text: 'Source, citation, and verified numbers.', to: `${CB}ai-ready-dissemination/trust`},
    ],
    expansions: [
      'An API package for NADA and for the common indicator databases of statistical offices, with provenance fields, versioning, and an OpenAPI description out of the box.',
      'A conformance test that an office runs against its API and attaches to the assessment form.',
    ],
    phases: ['p1', 'p2'],
    dependsOn: ['metadata', 'quality'],
    evidence: ['The OpenAPI description or SDMX structure', 'Monitoring reports', 'The deprecation policy'],
  },
  {
    id: 'agents',
    category: 'access',
    title: 'Agentic and generative access',
    dimensions: ['2.5'],
    need:
      'An MCP server over the API with a manifest, a guidance resource, access tiers, and a client registry; an assistant that answers from retrieved records with every number verified against the source; and evaluation suites that gate every change. Confidential inputs go to a local model.',
    finances: [
      'Data science and developer time to build and gate the components.',
      'A GPU server shared with quality control, or hosted-model usage for public inputs only.',
      'Reviewer time for a monthly sample of answers in each language.',
    ],
    contributes: [
      {level: 'available', text: 'The Model Context Protocol for statistics and the Data360 MCP server as a reference implementation.', to: '/docs/mcp/'},
      {level: 'available', text: 'Proof-Carrying Numbers for verified figures in generated text.', to: 'https://arxiv.org/abs/2509.06902'},
      {level: 'available', text: 'Serving official statistics to AI agents, chapter by chapter.', to: `${CB}serving-statistics-to-agents/`},
      {level: 'adapt', text: 'The assistant adapted to the office\'s languages, with question sets and reviewers in each.'},
    ],
    expansions: [
      'A deployable MCP server for NADA and for SDMX services, so that any office with a standard catalog or API exposes it to agents with configuration alone.',
      'A verified assistant as a hosted service for offices that publish public data only, run by a partner under the office\'s terms of use.',
      'A registry of statistical MCP servers that AI vendors can use to prefer official sources.',
    ],
    phases: ['p2', 'p3'],
    dependsOn: ['api', 'quality', 'compute'],
    evidence: ['The MCP manifest and client registry', 'Evaluation reports with verified-number rates', 'The monthly review sample'],
  },
  {
    id: 'production',
    category: 'production',
    title: 'AI-assisted data production',
    dimensions: ['1.4', '2.2'],
    need:
      'Models assist the tasks that the map allows: coding of occupations, industries, and products against standard classifications with a threshold and a re-coded sample; extraction of tables and series from documents into tidy files; and drafting of metadata under a curator\'s decision. Every output has a reviewer and a recorded decision.',
    finances: [
      'Methodologist time for the map, the thresholds, and the re-coded samples.',
      'A share of the GPU server for coding and extraction on confidential records.',
    ],
    contributes: [
      {level: 'available', text: 'Coding with a threshold and a re-coded sample.', to: `${CB}language-models-in-production/coding`},
      {level: 'available', text: 'Extracting data from documents.', to: `${CB}data-from-documents/`},
      {level: 'available', text: 'Metadata curation with language models.', to: `${CB}metadata-curation-with-llms/`},
      {level: 'available', text: 'Metadata augmentation for discovery.', to: '/docs/metadata-augmentation/'},
      {level: 'adapt', text: 'Coding models adapted to the national classifications and the languages of the questionnaires.'},
    ],
    expansions: [
      'National classification coders as shared models: one per classification family, adapted per country from a common base and evaluated on a shared suite.',
      'A document-extraction service for the archive of printed statistical publications, which many offices hold and have never digitized as data.',
    ],
    phases: ['p2', 'p3'],
    dependsOn: ['ai-strategy', 'compute', 'quality'],
    evidence: ['Agreement rates on re-coded samples', 'Extraction logs with reviewer decisions', 'The statement of model use'],
  },
  {
    id: 'sharing',
    category: 'production',
    title: 'Synthetic data and ML-ready datasets',
    dimensions: ['2.5', '2.6'],
    need:
      'Synthetic files that let users develop against microdata before an access request, and datasets published in machine-learning formats with their records, licences, and a representativeness note. Both go through disclosure review.',
    finances: [
      'Methodologist and curator time for disclosure review and representativeness notes.',
      'A share of the GPU server for generating synthetic files.',
    ],
    contributes: [
      {level: 'available', text: 'Synthetic data for sharing, with the REaLTabFormer method.', to: `${CB}synthetic-data-for-sharing/`},
      {level: 'available', text: 'Publishing ML-ready datasets.', to: `${CB}ml-ready-datasets/`},
      {level: 'available', text: 'Data Snapshots for model-ready extracts of indicator databases.', to: 'https://arxiv.org/abs/2606.06242'},
    ],
    expansions: [
      'A synthetic data service inside the Microdata Library pattern, so that a catalog offers a synthetic file for each public-use dataset.',
      'Croissant records generated from the catalog, so that ML-ready datasets are listed where model builders look.',
    ],
    phases: ['p3'],
    dependsOn: ['metadata', 'legal', 'compute'],
    evidence: ['Disclosure review reports', 'Croissant records', 'Representativeness notes'],
  },
  {
    id: 'small-models',
    category: 'production',
    title: 'Small and open models',
    dimensions: ['1.4', '1.5'],
    need:
      'The office runs models it can afford and audit: small open models on its own server for the tasks where they suffice, with an exit plan per provider for any hosted model it uses, and a cost record per component.',
    finances: [
      'The GPU server, shared with quality control and agentic access.',
      'Data scientist time for candidate evaluation and adaptation.',
    ],
    contributes: [
      {level: 'available', text: 'When a small model is enough.', to: `${CB}small-and-open-models/when`},
      {level: 'available', text: 'Efficient and inclusive AI methods for low-resource languages.', to: '/docs/inclusive-ai/'},
      {level: 'available', text: 'Small and agentic AI research (in development).'},
    ],
    expansions: [
      'A model catalog for statistical offices: open models evaluated on statistical tasks, with licences classified and sizing recorded, updated as models change.',
      'Adapted models for the languages of implementing offices, released under open licences.',
    ],
    phases: ['p2', 'p3'],
    dependsOn: ['compute'],
    evidence: ['The model register with licences', 'Evaluation reports per candidate', 'The cost record per component'],
  },
  {
    id: 'monitoring',
    category: 'measurement',
    title: 'Monitoring of data use',
    dimensions: ['1.1', '2.5'],
    need:
      'The office knows how its data are used: downloads and API calls by client type, citations in publications and policy documents, and references by AI systems. The results feed the governance board and the re-assessment.',
    finances: [
      'Developer time for logging and the monitoring reports.',
      'Analyst time to read the reports at each release.',
    ],
    contributes: [
      {level: 'available', text: 'Monitoring of data use.', to: '/docs/data_use/'},
      {level: 'available', text: 'What counts as use of the data.', to: `${CB}monitoring-data-use/define`},
    ],
    expansions: [
      'A shared monitor of references to official statistics by AI systems, run by the program across offices and reported to each.',
      'A data-use dashboard packaged for NADA and the API gateway.',
    ],
    phases: ['p2', 'p3'],
    dependsOn: ['api'],
    evidence: ['Monitoring reports per release', 'The AI-reference report', 'Decisions of the governance board that cite them'],
  },
  {
    id: 'sustain',
    category: 'measurement',
    title: 'Sustainability and re-assessment',
    dimensions: ['1.5', '1.6'],
    need:
      'The components continue after the project: hosting and maintenance in the office\'s own budget, staff retained, suites and question sets maintained, and a re-assessment on the same framework that reports the change in maturity level per dimension.',
    finances: [
      'A transition year in which the office\'s budget takes over recurring costs step by step.',
      'The re-assessment and a results report.',
      'Participation in the peer network of implementing offices.',
    ],
    contributes: [
      {level: 'available', text: 'The assessment framework, applied a second time on the same questions.', to: '/ai-readiness-assessment'},
      {level: 'available', text: 'Partnerships and shared resources of the program.', to: '/docs/partnerships/'},
      {level: 'expand', text: 'A peer network of implementing offices with shared suites, question sets, and model adaptations.'},
    ],
    expansions: [
      'A results framework template for projects of this kind, with indicators derived from the assessment dimensions and the evidence each recipe produces.',
      'A maintenance subscription model under which a partner maintains the shared components and offices contribute cases.',
    ],
    phases: ['p0', 'p3'],
    dependsOn: ['ai-strategy'],
    evidence: ['The re-assessment scores', 'The office\'s budget lines for the components', 'The results report'],
  },
];

// Roles the project staffs, where they sit, and what the program gives them.
export const ROLES = [
  {
    role: 'Project coordinator',
    where: 'The project implementation unit inside the office',
    work: 'Runs procurement, staffing, and the phase gates. Reports to the governance board.',
    program: 'The phase gates and the component list of this page; the operational companion.',
  },
  {
    role: 'Metadata lead and curators',
    where: 'The dissemination or data management unit',
    work: 'Own the records, the completeness profile, the code lists, and the review decisions.',
    program: 'The Metadata Editor, the schemas, the reviewer, and the dissemination and microdata cookbooks.',
  },
  {
    role: 'Methodologist',
    where: 'The methods or quality unit',
    work: 'Owns the map of model-assisted tasks, thresholds, calibration samples, and the revisions log.',
    program: 'The production, evaluation, and anomaly-detection resources.',
  },
  {
    role: 'Data engineer',
    where: 'The IT unit, recruited or seconded',
    work: 'Runs the catalog, the search index, the API, the gateway, version control, and continuous integration.',
    program: 'The reference implementations, the hosting notes, and the serving-to-agents cookbook.',
  },
  {
    role: 'Data scientist',
    where: 'A new AI or data science unit, or the methods unit',
    work: 'Evaluates and adapts models, builds the suites, and runs the model server.',
    program: 'The small models, evaluation, and production cookbooks; the PI-FT toolkit.',
  },
  {
    role: 'Evaluation reviewer',
    where: 'Subject-matter units, part time',
    work: 'Labels question sets, reviews samples of model outputs, and records decisions.',
    program: 'The evaluation suites cookbook and the review formats of each recipe.',
  },
  {
    role: 'Legal adviser',
    where: 'The office\'s legal function or external counsel',
    work: 'Reviews the act and the rules, drafts licences and terms of use.',
    program: 'The licensing and access recipes; the checklist listed as an expansion.',
  },
];

export const RISKS = [
  {
    risk: 'The foundations are skipped for the visible components.',
    response: 'The phase gates make the catalog, the policy, and the quality procedure a condition of the AI-ready phase. The assessment dependencies are the argument.',
  },
  {
    risk: 'Trained engineering staff leave the office after the project.',
    response: 'Retention measures in the staffing component, training of trainers, and a help desk so that one departure does not stop a component.',
  },
  {
    risk: 'Confidential records reach a hosted model.',
    response: 'An isolated segment for confidential inputs, a local model server, and the AI-use policy as a gate on every component.',
  },
  {
    risk: 'Components are built once and not maintained.',
    response: 'Deployment packages under version control, evaluation suites in continuous integration, and recurring costs moved to the office\'s budget before closing.',
  },
  {
    risk: 'The program cannot support several offices at once with bespoke work.',
    response: 'The expansion opportunities on this page are the shared components that make a second engagement cheaper than the first.',
  },
];

export const EXCLUDED = [
  'Budget figures and cost estimates. The kinds of cost are given per area; figures belong to the project document.',
  'Dates and durations. The phases are ordered by dependencies and entered through gates.',
  'The counterpart office, the financing instrument, and the funders.',
  'Numeric targets. The results framework takes its indicators from the assessment dimensions once a counterpart is known.',
];
