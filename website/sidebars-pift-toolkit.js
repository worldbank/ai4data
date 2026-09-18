// @ts-check

// Mirrors the top-level structure of ../sidebars.js so that pages served by
// this plugin instance (research/pift-toolkit/docs) don't strand visitors in
// an isolated 4-item sidebar. Every entry here is a `link` back into the main
// docs plugin except the local pift-toolkit pages, which stay as `doc` ids.
//
// This is a manual mirror, not a derived one: Docusaurus collapses a
// `<dir>/<same-name>.md` or `<dir>/index.md` doc's route to `/<dir>/` (drops
// the trailing segment), so a naive `id -> /docs/<id>` mapping would produce
// broken links for several entries below. Verified by the site build's own
// `onBrokenLinks: 'throw'` check — if this drifts from sidebars.js, the build
// fails loudly on the wrong href rather than shipping a silent 404.

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  piftSidebar: [
    {
      type: 'html',
      value: '<div class="sidebarSectionLabel">Get Started</div>',
      defaultStyle: true,
    },
    {type: 'link', label: 'Introduction', href: '/docs/introduction'},

    {
      type: 'html',
      value: '<div class="sidebarSectionLabel">Workstreams</div>',
      defaultStyle: true,
    },
    {
      type: 'link',
      label: 'AI for Metadata Quality',
      href: '/docs/metadata-quality/generative-ai-for-metadata-quality',
    },
    {
      type: 'link',
      label: 'Metadata Augmentation',
      href: '/docs/metadata-augmentation/',
    },
    {
      type: 'link',
      label: 'Anomaly Detection in Data',
      href: '/docs/anomaly-detection/',
    },
    {
      type: 'category',
      label: 'Data Discoverability',
      items: [
        {type: 'link', label: 'Data Discoverability', href: '/docs/data-discoverability/'},
        {
          type: 'category',
          label: 'Fine-Tuning Embedding Models',
          items: [
            {
              type: 'link',
              label: 'Overview',
              href: '/docs/data-discoverability/embedding-fine-tuning',
            },
            'method',
            'pipeline',
            'configuration',
            'deployment',
          ],
        },
      ],
    },
    {
      type: 'link',
      label: 'Monitoring of Data Use',
      href: '/docs/data_use/',
    },
    {
      type: 'link',
      label: 'Model Context Protocol (MCP)',
      href: '/docs/mcp/',
    },
    {
      type: 'link',
      label: 'Efficient and Inclusive AI Applications',
      href: '/docs/inclusive-ai/',
    },

    {
      type: 'html',
      value: '<div class="sidebarSectionLabel">Metadata Reviewer</div>',
      defaultStyle: true,
    },
    {
      type: 'link',
      label: 'Overview',
      href: '/docs/metadata-reviewer/overview',
    },
    {
      type: 'link',
      label: 'User Manual',
      href: '/docs/metadata-reviewer/user-manual/',
    },

    {
      type: 'html',
      value: '<div class="sidebarSectionLabel">Program</div>',
      defaultStyle: true,
    },
    {type: 'link', label: 'Partnerships', href: '/docs/partnerships/'},
  ],
};

export default sidebars;
