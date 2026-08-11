// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    {
      type: 'html',
      value: '<div class="sidebarSectionLabel">Get Started</div>',
      defaultStyle: true,
    },
    'introduction',

    {
      type: 'html',
      value: '<div class="sidebarSectionLabel">Workstreams</div>',
      defaultStyle: true,
    },
    {
      type: 'category',
      label: 'AI for Metadata Quality',
      items: [
        'metadata-quality/generative-ai-for-metadata-quality',
        'notebooks/metadata-quality-assessment-with-llm',
      ],
    },
    {
      type: 'category',
      label: 'Metadata Augmentation',
      items: [
        'metadata-augmentation/index',
        'metadata-augmentation/methodology',
      ],
    },
    {
      type: 'category',
      label: 'Anomaly Detection in Data',
      items: [
        'anomaly-detection/anomaly-detection',
        'anomaly/explanation/index',
        'anomaly/explanation/motivation',
        'anomaly/explanation/elicitation-pipeline',
        'anomaly-detection/feedback-system',
        'notebooks/data-anomaly/timeseries-anomaly-explanation-with-llms',
      ],
    },
    {
      type: 'category',
      label: 'Data Discoverability',
      items: ['data-discoverability/data-discoverability'],
    },
    {
      type: 'category',
      label: 'Monitoring of Data Use',
      items: ['data_use/data_use'],
    },
    {
      type: 'category',
      label: 'Model Context Protocol (MCP)',
      items: ['mcp/mcp'],
    },
    {
      type: 'category',
      label: 'Efficient and Inclusive AI Applications',
      items: ['inclusive-ai/inclusive-ai'],
    },

    {
      type: 'html',
      value: '<div class="sidebarSectionLabel">Metadata Reviewer</div>',
      defaultStyle: true,
    },
    {
      type: 'category',
      label: 'Overview',
      items: [
        'metadata-reviewer/overview',
        'metadata-reviewer/agentic-approach',
        'metadata-reviewer/implementation',
      ],
    },
    {
      type: 'category',
      label: 'User Manual',
      items: [
        'metadata-reviewer/user-manual/index',
        'metadata-reviewer/user-manual/introduction',
        'metadata-reviewer/user-manual/core-concepts',
        'metadata-reviewer/user-manual/installation',
        'metadata-reviewer/user-manual/quick-start',
        'metadata-reviewer/user-manual/client-api',
        'metadata-reviewer/user-manual/jobs',
        'metadata-reviewer/user-manual/advanced-usage',
        'metadata-reviewer/user-manual/review-board',
        'metadata-reviewer/user-manual/end-to-end-workflow',
        'metadata-reviewer/user-manual/troubleshooting',
        'metadata-reviewer/user-manual/appendix-api-reference',
        'metadata-reviewer/user-manual/appendix-glossary',
        'metadata-reviewer/user-manual/appendix-agents-manifest',
      ],
    },

    {
      type: 'html',
      value: '<div class="sidebarSectionLabel">Program</div>',
      defaultStyle: true,
    },
    {
      type: 'category',
      label: 'Partnerships',
      items: ['partnerships/partnerships'],
    },
  ],
};

export default sidebars;
