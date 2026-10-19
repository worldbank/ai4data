// What the recipes cannot say by themselves, per Pillar II dimension: the
// systems that have to exist at each level, the dimensions a step depends on,
// and hardware and hosting by office size. The effort and roles come from the
// recipes (investment.json, written by scripts/docs/investment_view.py).
// Money is given as the kind of cost, never as a figure: prices differ too
// much between countries and years for a number to be honest.

export const LEVEL_TO_GRADE = {Foundational: 'A to B', 'AI-ready': 'B to C', 'AI-native': 'C to D'};

export const investmentNotes = {
  '2.1': {
    systems: {
      Foundational: ['The Metadata Editor and a catalog (NADA or equivalent)', 'The World Bank metadata schemas and a completeness profile', 'Version control for records and profiles'],
      'AI-ready': ['A controlled vocabulary (SKOS) and its mapping script', 'Published code lists and crosswalks', 'The Metadata Reviewer or an equivalent review step'],
      'AI-native': ['A concept scheme and question bank shared across surveys', 'Records regenerated from production at each release'],
    },
    dependsOn: [],
    hosting: {small: 'The catalog host; no additional hardware', medium: 'The catalog host and a small server for the review pipeline', large: 'A catalog cluster and a review pipeline that runs a local model'},
    money: ['Staff time of curators (the largest cost at every level)', 'No licence costs: the editor, the catalog, and the schemas are open source'],
  },
  '2.2': {
    systems: {
      Foundational: ['Status codes and a revisions log in the values files', 'The completeness checker and the dictionary check in the release procedure'],
      'AI-ready': ['A review board with decisions recorded', 'Anomaly detection on the main series', 'A local model server where confidential records are scored or explained'],
      'AI-native': ['Evaluation suites per model-assisted task with gates in continuous integration', 'A statement of model use generated and checked at each release'],
    },
    dependsOn: ['2.1'],
    hosting: {small: 'One GPU with 24 GB for scoring and explanations', medium: 'One GPU server shared by the units', large: 'A GPU server with evaluation runners'},
    money: ['Staff time of methodologists and curators for calibration samples and reviews', 'One GPU server at the AI-ready level, shared with 2.5'],
  },
  '2.3': {
    systems: {
      Foundational: ['A catalog with stable URLs and schema.org markup on every page', 'DOIs through DataCite', 'A known-item question set and the retrieval scorer'],
      'AI-ready': ['A semantic search index beside keyword search', 'Variable-level search over documented dictionaries', 'A release calendar served as data'],
      'AI-native': ['Retrieval fine-tuned on the organization\'s own records', 'Lineage and quality fields surfaced in search results'],
    },
    dependsOn: ['2.1'],
    hosting: {small: 'The catalog host; a CPU is enough for a few thousand records', medium: 'A search index on its own host', large: 'A search cluster with a vector index and re-ranking'},
    money: ['DataCite membership or a consortium fee for DOIs', 'Staff time to write and maintain question sets in each language', 'A search host at the AI-ready level'],
  },
  '2.4': {
    systems: {
      Foundational: ['Tidy CSV at stable URLs and a bulk file', 'NADA\'s catalog and metadata API where the catalog is NADA'],
      'AI-ready': ['An OpenAPI-described API or an SDMX web service with provenance in every response', 'Monitoring of availability and latency', 'Rate limits and quotas behind a gateway'],
      'AI-native': ['Pagination, filtering, and versioned datasets with retrievable vintages', 'Published service targets and a deprecation policy'],
    },
    dependsOn: ['2.1', '2.2'],
    hosting: {small: 'One API host behind the web server', medium: 'An API host and a gateway with monitoring', large: 'An autoscaling API tier with a gateway'},
    money: ['Developer time for the API and its documentation', 'A gateway or API-management service, open source or hosted', 'Hosting that grows with machine traffic'],
  },
  '2.5': {
    systems: {
      Foundational: ['The API of 2.4 and the provenance fields of 2.2', 'A question set with expected answers', 'An AI-use policy and a component register'],
      'AI-ready': ['An MCP server over the API with a manifest, a guidance resource, tiers, and a client registry', 'An assistant that answers from retrieved records with every number verified', 'Evaluation suites that gate every change', 'A local model server for confidential inputs'],
      'AI-native': ['Models adapted on the organization\'s own examples', 'Synthetic files and ML-ready datasets published with their records', 'Production evaluation on logged traffic'],
    },
    dependsOn: ['2.1', '2.2', '2.4'],
    hosting: {small: 'The API host plus one GPU with 24 GB (an 8B model at 4 bits for 8 concurrent requests)', medium: 'A GPU server with two 48 GB cards and an evaluation runner', large: 'Four to eight 80 GB GPUs, an isolated segment for confidential inputs, several evaluation runners'},
    money: ['A GPU server, or hosted-model usage for public inputs only', 'Data science and developer time to build and gate the components', 'Reviewer time: a monthly sample of answers in each language'],
  },
  '2.6': {
    systems: {
      Foundational: ['One open licence chosen and stated on every dataset page', 'The licence field filled in every record'],
      'AI-ready': ['The licence as a URL in schema.org markup, API responses, and Croissant records', 'Terms of use for agents and for model training beside the licence'],
      'AI-native': ['Access tiers enforced on every automated path, tested from outside', 'Use of the data by models tracked through citations and model cards'],
    },
    dependsOn: ['2.1'],
    hosting: {small: 'None beyond the catalog', medium: 'None beyond the catalog and the API', large: 'None beyond the catalog and the API'},
    money: ['Legal review of the licence text for AI training and derived products', 'Curator time to fill the licence field across the catalog'],
  },
};
