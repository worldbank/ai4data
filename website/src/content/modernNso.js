// A composite portrait of a fictional national statistical organization at
// three maturity levels. All figures are illustrative; the people are roles,
// never named individuals; the tools are the open-source ones the cookbooks
// use. The office is the "example organization" of the cookbooks.

export const LEVELS = [
  {id: 'foundational', label: 'Foundational', short: 'Foundational'},
  {id: 'ai-ready', label: 'AI-ready', short: 'AI-ready'},
  {id: 'ai-native', label: 'AI-native', short: 'AI-native'},
];

export const LEVEL_INTRO = {
  foundational:
    'The office documents what it publishes, keeps people in every decision, and runs one or two model-assisted tasks under full review. The data layer stands on its own.',
  'ai-ready':
    'The office measures everything a model touches. Catalog records are complete at the AI-ready profile, an API and a question set exist, and each model-assisted task has an evaluation suite and a threshold.',
  'ai-native':
    'Models run as stages of production, every output is verified or reviewed before it reaches a person outside the office, and the office publishes its own statement of model use with each release.',
};

const CB = '/cookbook/';

export const units = [
  {
    id: 'production',
    name: 'Methodology and production floor',
    short: 'Production',
    position: 'west',
    tagline: 'Coding, editing, and estimation.',
    people: ['Survey methodologists', 'Coding supervisor and coders', 'Editors', 'Subject-matter analysts'],
    dims: ['2.2', '1.4'],
    levels: {
      foundational: {
        day: 'Coders assign occupation and industry codes by hand with the ISCO index. Editors work through edit failures from a rule list. Anomalies are found by eye on release tables.',
        tools: ['Classification indexes (ISCO-08, ISIC)', 'Editing system with written rules', 'Spreadsheets and R or Stata'],
        ai: [
          {name: 'Question-wording review', role: 'flags questionnaire problems against the office\'s checklist', decision: 'the methodologist accepts or dismisses each flag', off: 'the checklist is applied by hand'},
        ],
        links: [{to: `${CB}language-models-in-production/map`, label: 'Map of model-assisted tasks'}],
        next: 'An evaluation set of coded responses and a confidence threshold turn hand coding into a production rule with a measured error.',
      },
      'ai-ready': {
        day: 'A local model proposes codes with a confidence; responses above the threshold are accepted, the rest go to coders with the proposal shown. Edit failures arrive with a two-sentence explanation. A re-coded sample measures the automatic share every round.',
        tools: ['Classification indexes as reference material', 'Editing system with rule export', 'Evaluation set of coded responses', 'R or Python for the measures'],
        ai: [
          {name: 'Occupation and industry coder', role: 'proposes a code and a confidence from the index and the response', decision: 'the supervisor sets the threshold; coders decide below it', off: 'coders code every response'},
          {name: 'Edit explanation drafter', role: 'explains why a rule fired, citing the record', decision: 'the editor decides as before', off: 'editors read the rule identifier'},
        ],
        links: [
          {to: `${CB}language-models-in-production/coding`, label: 'Classification and coding'},
          {to: `${CB}language-models-in-production/editing`, label: 'Editing and imputation'},
        ],
        next: 'Proposals for missing values checked against every rule, anomalies explained with cited evidence, and the effect on the statistics measured each release.',
      },
      'ai-native': {
        day: 'The nightly batch codes the quarter\'s open responses on the office\'s own server and resumes where it stopped. Coders see only the routed share. Editors receive value proposals that already passed every rule. The anomaly detector flags a jump and the analyst reads a drafted explanation with its evidence before deciding.',
        tools: ['Model server inside the production network', 'Editing system with rule export and proposal queue', 'Anomaly detection pipeline', 'Evaluation suites per task'],
        ai: [
          {name: 'Occupation and industry coder', role: 'codes responses above the threshold; routes the rest', decision: 'coders decide the routed share; a re-coded sample measures the rest', off: 'coders code every response; the release date holds because coding starts earlier'},
          {name: 'Value proposer', role: 'proposes a value with a reason, checked against every edit rule', decision: 'the editor accepts, changes, or rejects', off: 'editors impute by the documented method alone'},
          {name: 'Anomaly explainer', role: 'drafts the cause of a flagged value with cited evidence', decision: 'the analyst records a verdict', off: 'the analyst reads the release notes and revisions log by hand'},
        ],
        links: [
          {to: `${CB}language-models-in-production/coding`, label: 'Classification and coding'},
          {to: `${CB}language-models-in-production/editing`, label: 'Editing, imputation, and anomalies'},
          {to: `${CB}language-models-in-production/assurance`, label: 'Quality assurance and the statement of model use'},
        ],
        next: null,
      },
    },
  },
  {
    id: 'documentation',
    name: 'Data documentation and curation',
    short: 'Documentation',
    position: 'north',
    tagline: 'Records for every dataset, variable, and series.',
    people: ['Data curators', 'Metadata lead', 'Classification specialist', 'Native-speaking reviewers'],
    dims: ['2.1', '2.2', '2.3'],
    levels: {
      foundational: {
        day: 'Curators document each survey in the Metadata Editor and publish to the catalog. The completeness check runs on every record at the foundational level. Dictionaries carry labels, value codes, and universes.',
        tools: ['Metadata Editor', 'NADA catalog', 'Completeness checker with the office\'s profile', 'Dictionary check'],
        ai: [],
        links: [
          {to: `${CB}microdata-documentation/`, label: 'Microdata documentation'},
          {to: `${CB}ai-ready-dissemination/find`, label: 'Findable records'},
        ],
        next: 'Records complete at the AI-ready profile, a controlled vocabulary, and model drafts for missing fields under a curator\'s decision.',
      },
      'ai-ready': {
        day: 'A model drafts missing definitions from the record and the schema; drafts that fail the formal checks never reach the curator. The reviewer board shows each flagged issue as a diff. Keywords map to the vocabulary. Translations are checked for changed numbers before a native speaker reads them.',
        tools: ['Metadata Editor', 'NADA', 'Metadata Reviewer and its board', 'Vocabulary (SKOS) and mapping script', 'Translation checks'],
        ai: [
          {name: 'Field drafter', role: 'drafts a missing field from the record and the schema', decision: 'the curator accepts, edits, or rejects', off: 'curators write every field'},
          {name: 'Metadata reviewer', role: 'detects issues with a severity and a proposed fix', decision: 'the curator decides in the board', off: 'review by reading, at the pace of the team'},
          {name: 'Translator with glossary', role: 'drafts translations in the office\'s terms', decision: 'a native speaker reviews every one', off: 'translation by hand'},
        ],
        links: [
          {to: `${CB}metadata-curation-with-llms/`, label: 'Metadata curation with language models'},
          {to: `${CB}microdata-documentation/concepts`, label: 'Concepts and classifications'},
        ],
        next: 'Variables linked to concepts across surveys, a question bank, and records regenerated from production at each release.',
      },
      'ai-native': {
        day: 'Each release regenerates the documented parts of the record. Variables carry concept links that make a search return the same concept in every survey. Acceptance rates per field decide which drafts need lighter review. The question bank supplies wordings before a new survey is designed.',
        tools: ['Metadata Editor and NADA', 'Metadata Reviewer', 'Concept scheme (SKOS, XKOS) and question bank', 'Acceptance tracking and the evaluation set'],
        ai: [
          {name: 'Field drafter and reviewer', role: 'drafts, flags, and proposes across the catalog', decision: 'curators decide; rules handle categories they always accept', off: 'curation at the pace of the team; the catalog stays valid'},
          {name: 'Variable grouper', role: 'groups dictionary variables into themes', decision: 'a curator renames, splits, or merges', off: 'groups follow the questionnaire sections'},
          {name: 'Question-bank matcher', role: 'proposes matches between new questions and the bank', decision: 'the methodologist confirms the concept', off: 'search of the bank by keyword'},
        ],
        links: [
          {to: `${CB}metadata-curation-with-llms/improve`, label: 'Measuring and improving curation'},
          {to: `${CB}microdata-documentation/navigate`, label: 'Variable search across surveys'},
        ],
        next: null,
      },
    },
  },
  {
    id: 'dissemination',
    name: 'Dissemination and the catalog',
    short: 'Dissemination',
    position: 'east',
    tagline: 'Pages, files, the API, and the assistant.',
    people: ['Dissemination officers', 'Web and API developers', 'Catalog administrator', 'Analysts who write commentary'],
    dims: ['2.3', '2.4', '2.5', '2.6'],
    levels: {
      foundational: {
        day: 'Series pages show source, release, and a citation. Tidy CSV files sit at stable addresses. The catalog carries schema.org markup. A known-item question set measures search once a month.',
        tools: ['NADA with schema.org on every page', 'Tidy CSV at stable URLs', 'Question set and retrieval scorer', 'DataCite DOIs'],
        ai: [],
        links: [
          {to: `${CB}ai-ready-dissemination/retrieve`, label: 'Retrieval by programs'},
          {to: `${CB}ai-ready-dissemination/trust`, label: 'Source, citation, and verification'},
        ],
        next: 'An API described with OpenAPI that returns provenance with every response, semantic search scored against the baseline, and a licence readable by machines.',
      },
      'ai-ready': {
        day: 'The API returns unit, period, release, and source with every value. Semantic search sits beside keyword search, scored per language. The licence is a URL in every record. Downloads, citations, and mentions appear side by side in the quarterly use report.',
        tools: ['OpenAPI-described API or SDMX web service', 'Semantic search index', 'Access log counted under the COUNTER rules', 'Citation collection from identifier services'],
        ai: [
          {name: 'Semantic search', role: 'ranks records by meaning', decision: 'none needed; scored by the question set', off: 'keyword search answers; exact codes still win'},
          {name: 'Mention extractor', role: 'finds the office\'s datasets in publications', decision: 'a person reviews unmatched mentions', off: 'use counted from downloads and citations alone'},
        ],
        links: [
          {to: `${CB}ai-ready-dissemination/evaluate`, label: 'Evaluating search'},
          {to: `${CB}monitoring-data-use/`, label: 'Monitoring data use'},
        ],
        next: 'An agent interface over the API, an assistant that answers only from retrieved records with every number verified, and release commentary drafted from the tables.',
      },
      'ai-native': {
        day: 'An MCP server serves four read-only tools over the API with provenance in every response, behind quotas and a registry of clients. The assistant answers from retrieved records and shows no number that the verifier did not match. Commentary arrives as a draft with every figure checked against the release table. Datasets built from coded responses are published for model training with a Croissant record and a card.',
        tools: ['MCP server over the API', 'Number verifier in front of the assistant', 'Commentary check', 'Croissant records and dataset cards', 'AI visibility check, monthly'],
        ai: [
          {name: 'Statistics assistant', role: 'answers questions from retrieved records and cites the series', decision: 'a subject-matter reviewer samples thirty answers a month', off: 'the catalog, the API, and search keep working; the assistant page says it is paused'},
          {name: 'Commentary drafter', role: 'drafts release text from the output tables in the office\'s style', decision: 'the analyst edits and approves; the check runs again on the final text', off: 'the analyst writes from the tables'},
          {name: 'Agent interface (MCP)', role: 'lets any assistant call the office\'s data with provenance', decision: 'the evaluation suite gates every change', off: 'developers use the API directly'},
        ],
        links: [
          {to: `${CB}serving-statistics-to-agents/`, label: 'Serving statistics to AI agents'},
          {to: `${CB}ai-ready-dissemination/ask`, label: 'Answering from retrieved records'},
          {to: `${CB}ml-ready-datasets/`, label: 'Publishing ML-ready datasets'},
        ],
        next: null,
      },
    },
  },
  {
    id: 'dataScience',
    name: 'IT, data engineering, and data science',
    short: 'IT and data science',
    position: 'south',
    tagline: 'Model servers, evaluation suites, and access control.',
    people: ['Data engineers', 'Data scientists', 'Systems and security staff', 'Evaluation owner'],
    dims: ['1.4', '2.5', '1.2'],
    levels: {
      foundational: {
        day: 'The catalog and the API run on the office\'s servers with monitoring. A version-controlled repository holds the scripts, the question set, and the completeness profile. Hosted models are used only on public text.',
        tools: ['Version control and CI', 'Catalog and API hosts with monitoring', 'Completeness checker, dictionary check, retrieval scorer'],
        ai: [],
        links: [
          {to: `${CB}evaluation-suites/inventory`, label: 'Inventory of AI components'},
          {to: `${CB}small-and-open-models/when`, label: 'When a small model is enough'},
        ],
        next: 'A model server inside the production network, an evaluation suite per task, and a cost estimate before any batch job.',
      },
      'ai-ready': {
        day: 'An open-weight model runs on a server inside the production network, its files verified against a lock file. Each task has a suite with a labelled set, measures, and a required score. A change to a prompt, a model, or an index ships only when the gate passes.',
        tools: ['vLLM or llama.cpp on a GPU server', 'Model lock file and verification', 'Evaluation suites with report cards and gates', 'Batch cost estimator'],
        ai: [
          {name: 'Local model server', role: 'serves the coding, editing, and curation models', decision: 'operations fix versions and run the switch-off test', off: 'every job falls back to its manual path by design'},
          {name: 'Judge model in the suites', role: 'grades answers with a rubric where no script can', decision: 'a reviewer checks a fifth of the grades each run', off: 'reviewers grade the sample'},
        ],
        links: [
          {to: `${CB}evaluation-suites/`, label: 'Evaluation suites'},
          {to: `${CB}small-and-open-models/serving`, label: 'Serving models locally'},
          {to: `${CB}small-and-open-models/security`, label: 'Model security and provenance'},
        ],
        next: 'Models adapted on the office\'s own examples, compared with larger ones on cost and quality, and jobs that resume, log their versions, and report their cost.',
      },
      'ai-native': {
        day: 'A small model fine-tuned on the office\'s coded responses beats the hosted model on the national language at a tenth of the cost, and the cost frontier shows why. Batch jobs run from manifests and write their versions into every output. Prompt-injection tests run with every change. Synthetic files are generated and measured as a release stage.',
        tools: ['Fine-tuning pipeline on the office\'s examples', 'Cost and quality frontier', 'Job manifests with resumable stages', 'REaLTabFormer for synthetic data', 'Injection tests in the suites'],
        ai: [
          {name: 'Adapted small models', role: 'the office\'s own coder and retriever, tuned on its examples', decision: 'the suite and the cost frontier choose; a person signs the register', off: 'the generic model, or the manual path'},
          {name: 'Synthetic data generator', role: 'produces test files that mirror the real microdata', decision: 'the utility and risk reports pass the unit\'s limits', off: 'applicants wait for the licensed file'},
        ],
        links: [
          {to: `${CB}small-and-open-models/adaptation`, label: 'Adapting a model to the office'},
          {to: `${CB}small-and-open-models/cost`, label: 'Cost of running models'},
          {to: `${CB}synthetic-data-for-sharing/`, label: 'Synthetic data for sharing'},
        ],
        next: null,
      },
    },
  },
  {
    id: 'management',
    name: 'Management and governance',
    short: 'Management',
    position: 'centre',
    tagline: 'The policy, the register, and the statement of model use.',
    people: ['Head of the office', 'Head of methodology', 'AI-use policy owner', 'Legal and data protection'],
    dims: ['1.1', '1.2', '1.3', '1.5', '1.6'],
    levels: {
      foundational: {
        day: 'A one-page AI-use policy says what models may and may not be used for and who approves a new use. A register lists every AI component with an owner and a review date. The list of tasks that stay with people is published.',
        tools: ['AI-use policy', 'AI component register', 'The readiness assessment, once'],
        ai: [],
        links: [
          {to: `${CB}ai-ready-dissemination/govern`, label: 'Governance and responsible operation'},
          {to: '/ai-readiness-assessment', label: 'The readiness assessment'},
        ],
        next: 'Decision points and suites required before any task leaves its pilot, confidentiality rules enforced in the map of tasks, and a readiness profile used as a work plan.',
      },
      'ai-ready': {
        day: 'No task uses a model without a row in the map, a decision point, and a suite. The map check flags any confidential input sent to a provider. The readiness profile from the assessment orders the year\'s work by the largest gaps.',
        tools: ['Map of model-assisted tasks with its check', 'Register with review dates', 'Readiness profile and improvement pathway', 'Training path through the cookbooks'],
        ai: [],
        links: [
          {to: `${CB}language-models-in-production/map`, label: 'The map of model-assisted tasks'},
          {to: '/ai-readiness-in-practice', label: 'From the assessment to resources'},
        ],
        next: 'A statement of model use with every release, an audit of one run a year from statement to reproduced result, and contributions back to the community.',
      },
      'ai-native': {
        day: 'The quality report of each release carries a statement of where models were used, with measured performance and the effect on the statistics, checked by script before publication. Once a year one run is audited end to end. The office contributes recipes and examples back and runs peer exchanges with other offices.',
        tools: ['Statement of model use and its check', 'Yearly audit from statement to reproduced run', 'Contributions to the open cookbooks', 'Repeat assessment on a schedule'],
        ai: [],
        links: [
          {to: `${CB}language-models-in-production/assurance`, label: 'Statement of model use'},
          {to: `${CB}ai-ready-dissemination/contributing`, label: 'Contributing a recipe'},
        ],
        next: null,
      },
    },
  },
];

// A labour force survey release day, hour by hour. `min` is the lowest level
// at which the step exists; `text` may vary by level; `off` is what happens
// with the AI layer switched off.
export const releaseDay = [
  {
    time: '06:30',
    unit: 'dataScience',
    actor: 'Data engineer',
    min: 'ai-ready',
    kind: 'ai',
    text: {
      'ai-ready': 'The nightly job finished coding the quarter\'s open responses on the local model server. The manifest shows which records remain for coders.',
      'ai-native': 'The nightly batch coded the quarter\'s 250,000 open responses on the office\'s own server, resuming after an interruption at record 90,000. Seventy percent were accepted above the threshold of 0.80; the rest are queued for coders.',
    },
    off: 'Coders code every response. Coding started two weeks earlier, so the release date holds.',
    link: {to: `${CB}language-models-in-production/running`, label: 'Running models in production'},
  },
  {
    time: '08:00',
    unit: 'production',
    actor: 'Coding supervisor',
    min: 'foundational',
    kind: 'decision',
    text: {
      foundational: 'The coding team works through the open responses with the ISCO index. The supervisor checks a sample.',
      'ai-ready': 'The supervisor draws a random sample of three hundred automatically coded responses for re-coding, without the model\'s code shown. Coders work the routed share with the proposal beside each response.',
      'ai-native': 'The supervisor draws the re-coded sample and reads last round\'s agreement by major group. Coders see only the routed share; accepting or changing a proposal takes seconds.',
    },
    off: 'Coders code everything; the sample is still drawn, to measure the coders.',
    link: {to: `${CB}language-models-in-production/coding`, label: 'Coding with a re-coded sample'},
  },
  {
    time: '09:00',
    unit: 'production',
    actor: 'Editors',
    min: 'foundational',
    kind: 'decision',
    text: {
      foundational: 'Editors work through the edit failures with the rule list and the household roster.',
      'ai-ready': 'Each edit failure arrives with a two-sentence explanation that names the fields and a plausible cause. Editors decide as before and rate a sample of explanations.',
      'ai-native': 'Proposed values that passed every rule wait in the queue with their reasons; proposals that failed went back with the rule named. Nothing has been applied; the editor decides each one.',
    },
    off: 'Editors read the rule identifiers and impute by the documented method.',
    link: {to: `${CB}language-models-in-production/editing`, label: 'Editing with explanations and proposals'},
  },
  {
    time: '10:30',
    unit: 'production',
    actor: 'Subject-matter analyst',
    min: 'ai-ready',
    kind: 'decision',
    text: {
      'ai-ready': 'The anomaly detector flags a jump in youth unemployment in the Northern Region. The analyst reads the release notes and the revisions log and records a verdict.',
      'ai-native': 'The anomaly detector flags the jump; the explainer drafts a cause with the evidence it rests on: a coverage change in the regional sample noted in the methodology log. The analyst confirms the verdict, and the confirmed explanation goes into the release.',
    },
    off: 'The analyst reads the notes and logs by hand.',
    link: {to: `${CB}language-models-in-production/editing`, label: 'Anomaly explanations with cited evidence'},
  },
  {
    time: '11:30',
    unit: 'documentation',
    actor: 'Data curator',
    min: 'foundational',
    kind: 'check',
    text: {
      foundational: 'The curator updates the survey record in the Metadata Editor and runs the completeness check at the foundational level before publishing to the catalog.',
      'ai-ready': 'The reviewer board shows three issues on the updated record, each with the current and proposed text side by side. The curator accepts two and rejects one with a reason. The French labels pass the translation check and go to the reviewer.',
      'ai-native': 'The record\'s measured sections regenerate from the release. The board shows one new issue; the categories the curators always accept were applied by rule. The question bank already holds the new question\'s concept.',
    },
    off: 'The curator reviews the record by reading; the completeness check still runs.',
    link: {to: `${CB}metadata-curation-with-llms/review`, label: 'Review in the board'},
  },
  {
    time: '13:00',
    unit: 'dissemination',
    actor: 'Analyst and dissemination officer',
    min: 'foundational',
    kind: 'check',
    text: {
      foundational: 'The analyst writes the release commentary from the output tables and checks every figure against them by hand.',
      'ai-ready': 'The analyst writes the commentary; the number check matches every figure to the release table and lists the comparative claims that need a source.',
      'ai-native': 'A draft arrives from the output tables in the office\'s style. The check reports six figures verified, one unverified, and one claim needing a source; the analyst removes the unverified sentence, sources the claim, and approves. The check runs again on the final text.',
    },
    off: 'The analyst writes from the tables; the number check still runs.',
    link: {to: `${CB}language-models-in-production/commentary`, label: 'Commentary with every number verified'},
  },
  {
    time: '14:00',
    unit: 'dataScience',
    actor: 'Evaluation owner',
    min: 'ai-ready',
    kind: 'check',
    text: {
      'ai-ready': 'The new catalog records change what the search index returns. The suite runs; retrieval per language and the answer checks are compared with the stored scores. The gate passes.',
      'ai-native': 'The suite runs on the assistant, the search, and the agent interface with the new records. The gate compares every slice with the previous run and passes; the report card with its intervals is stored with the release.',
    },
    off: 'The suite runs on the search alone.',
    link: {to: `${CB}evaluation-suites/gates`, label: 'Gates on every change'},
  },
  {
    time: '15:00',
    unit: 'dissemination',
    actor: 'Catalog administrator',
    min: 'foundational',
    kind: 'release',
    text: {
      foundational: 'The release goes out: series pages with source and citation, tidy CSV at the stable addresses, the revisions log updated, the catalog record with schema.org markup.',
      'ai-ready': 'The release goes out through the pages, the files, and the API, which returns the new values with unit, release, and source. Status codes mark the latest quarter as provisional.',
      'ai-native': 'The release goes out through the pages, the API, and the MCP server; an assistant asked for the new rate answers with the verified figure and the citation. The statement of model use is published with the quality report.',
    },
    off: 'The release goes out through the pages, the files, and the API, unchanged.',
    link: {to: `${CB}serving-statistics-to-agents/`, label: 'Serving statistics to agents'},
  },
  {
    time: '16:00',
    unit: 'management',
    actor: 'AI-use policy owner',
    min: 'ai-native',
    kind: 'decision',
    text: {
      'ai-native': 'The statement check passed before publication: every use names its production step and the performance section carries measured numbers. The register\'s review dates are updated, and the monthly AI visibility check is scheduled for next week.',
    },
    off: 'The statement says which components were switched off and why.',
    link: {to: `${CB}language-models-in-production/assurance`, label: 'Statement of model use'},
  },
  {
    time: '16:30',
    unit: 'dissemination',
    actor: 'Dissemination officer',
    min: 'ai-ready',
    kind: 'ai',
    text: {
      'ai-ready': 'The access log is counted under the COUNTER rules and the citation services are queried for the survey\'s DOI. The quarterly use report will show downloads, citations, and mentions side by side.',
      'ai-native': 'Mention extraction runs over the week\'s collected documents; unmatched mentions go to a review queue. The dataset page\'s "used in" list grows from confirmed matches.',
    },
    off: 'Downloads and citations are counted; mentions wait.',
    link: {to: `${CB}monitoring-data-use/`, label: 'Monitoring data use'},
  },
];

export const SIZES = [
  {id: 'small', label: 'Small office', staff: 'about 150 staff'},
  {id: 'medium', label: 'Medium office', staff: 'about 600 staff'},
  {id: 'large', label: 'Large office', staff: 'about 2,000 staff'},
];

// Illustrative ranges at the AI-native level. A column per office size.
export const resources = {
  people: {
    columns: ['Role', 'Skills', 'Small', 'Medium', 'Large'],
    rows: [
      ['Methodologists with evaluation skills', 'Survey methods; designing labelled sets, thresholds, and measures', '1 to 2', '3 to 5', '8 to 12'],
      ['Data curators', 'Metadata standards (DDI, SDMX); the Metadata Editor; vocabulary work', '2', '4 to 8', '12 to 25'],
      ['Data engineers', 'Python, SQL, pipelines, version control, CI', '1', '2 to 4', '6 to 10'],
      ['Data scientists', 'Model serving and adaptation, evaluation suites, synthetic data', '1 to 2', '3 to 6', '10 to 20'],
      ['Web and API developers', 'OpenAPI or SDMX, schema.org, the MCP SDK', '1 to 2', '3 to 5', '8 to 12'],
      ['Systems and security staff', 'GPU servers, access control, monitoring, model-file verification', '1 to 2', '3 to 5', '8 to 15'],
      ['AI-use policy owner', 'Governance, the register, the statement of model use; usually part of an existing role', '0.5', '1', '2 to 3'],
      ['Coders and editors', 'Unchanged in number at first; their work shifts to the routed share and the re-coded sample', 'as today', 'as today', 'as today'],
    ],
    note: 'Counts are full-time equivalents and illustrative. The first three roles can be one person each in a small office; the skills matter more than the headcount, and the cookbooks are written as the training path.',
  },
  compute: {
    columns: ['Resource', 'Purpose', 'Small', 'Medium', 'Large'],
    rows: [
      ['Model server', 'Coding, editing explanations, curation drafts, the assistant', '1 GPU with 24 GB: an 8B model at 4 bits for 8 concurrent requests needs about 9 GB', '2 GPUs with 48 GB: a 14B to 32B model, batch coding overnight', '4 to 8 GPUs with 80 GB: a 70B model at 4 bits needs about 45 GB for weights alone'],
      ['Catalog and API hosts', 'NADA, the API or SDMX service, the MCP server behind a gateway', '2 virtual machines', '4 virtual machines or a small cluster', 'A cluster with autoscaling'],
      ['Storage', 'Microdata, documents and their text layers, synthetic files, model files', '2 to 5 TB', '20 to 50 TB', '100 TB and more'],
      ['Evaluation runners', 'Suites on every change, nightly report cards', 'The CI service of the repository', 'One dedicated runner with a GPU', 'Several runners'],
      ['Secure enclave', 'Confidential microdata and the models that touch it', 'One isolated server', 'An isolated network segment', 'An isolated segment with its own GPUs'],
    ],
    note: 'GPU figures come from the sizing recipe of the small models cookbook (weights plus key-value cache plus a margin). Measure throughput on a test run before procurement.',
  },
  tools: {
    columns: ['Tool', 'Purpose', 'Licence'],
    rows: [
      ['Metadata Editor', 'Documenting every data type in the World Bank schemas; export to schema.org, Croissant, DCAT', 'Open source'],
      ['NADA', 'The catalog: schema.org markup on catalog pages, search, a catalog administration REST API, managed access; semantic search and MCP access through NADA AI, being integrated into NADA', 'Open source'],
      ['SDMX tooling', 'Aggregate data and code lists in the standard the statistical community shares', 'Open standards'],
      ['vLLM, llama.cpp, Ollama', 'Serving open-weight models inside the production network', 'Open source'],
      ['scikit-learn, pandas, scipy', 'Measures, agreement, bootstrap intervals, nearest neighbours in the cookbook scripts', 'Open source'],
      ['REaLTabFormer, synthpop, sdcMicro', 'Synthetic data and disclosure control', 'Open source'],
      ['PyMuPDF, Tesseract, Docling', 'Text layers, OCR, and table extraction from documents', 'Open source'],
      ['GLiNER and the data-use pipeline', 'Finding mentions of datasets in publications', 'Open source'],
      ['mlcroissant', 'Validating and loading ML-ready dataset records', 'Open source'],
      ['The MCP Python SDK', 'The agent interface over the API', 'Open source'],
      ['Git and a CI service', 'Version control of prompts, question sets, profiles, and the gates', 'Open source'],
      ['DataCite', 'Persistent identifiers that make citations countable', 'Membership'],
    ],
    note: 'Open source first, so that a switch of provider or model is a configuration change and the office keeps every prompt, test, and record under its own version control.',
  },
};

export const dimensionsByUnit = {
  '1.1': 'Strategy and governance',
  '1.2': 'Legal environment',
  '1.3': 'Skills and capacity',
  '1.4': 'Technical environment',
  '1.5': 'Funding and resources',
  '1.6': 'Partnerships',
  '2.1': 'Metadata standards and ontologies',
  '2.2': 'Quality control',
  '2.3': 'Searchable catalog',
  '2.4': 'API access',
  '2.5': 'Agentic AI access',
  '2.6': 'Data licensing',
};
