// @ts-check
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {themes as prismThemes} from 'prism-react-renderer';
import codeImport from 'remark-code-import';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI for Data - Data for AI',
  tagline:
    'Applying AI to improve development data, and making development data AI-ready.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://worldbank.github.io',
  baseUrl: '/ai4data/',

  organizationName: 'worldbank',
  projectName: 'ai4data',

  onBrokenLinks: 'throw',

  markdown: {
    format: 'detect',
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],
  clientModules: [path.join(__dirname, 'src/clientModules/fonts.js')],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '../docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.js',
          remarkPlugins: [
            [codeImport, {rootDir: repoRoot, allowImportingFromOutside: true}],
          ],
          editUrl: 'https://github.com/worldbank/ai4data/edit/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'AI for Data - Data for AI',
        logo: {
          alt: 'World Bank Group',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            href: 'https://github.com/worldbank/ai4data',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Program',
            items: [
              {label: 'Introduction', to: '/docs/introduction'},
              {label: 'Partnerships', to: '/docs/partnerships/'},
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/worldbank/ai4data',
              },
              {
                label: 'Issues',
                href: 'https://github.com/worldbank/ai4data/issues',
              },
              {
                label: 'Contact',
                href: 'mailto:ai4data@worldbank.org',
              },
            ],
          },
        ],
        copyright: `
          <div>Country borders or names do not necessarily reflect the World Bank Group's official position. All maps are for illustrative purposes and do not imply the expression of any opinion on the part of the World Bank, concerning the legal status of any country or territory or concerning the delimitation of frontiers or boundaries.</div>
          <div style="margin-top: 0.5rem">All content (unless otherwise specified) is subject to the <a href="https://opensource.org/license/mit">MIT License</a>. Copyright © ${new Date().getFullYear()} World Bank Group, Development Data Group.</div>
        `,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'json', 'yaml', 'python'],
      },
    }),

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      /** @type {import('@docusaurus/plugin-content-docs').Options} */
      ({
        id: 'pift-toolkit',
        path: '../research/pift-toolkit/docs',
        routeBasePath: 'pift-toolkit',
        sidebarPath: path.join(__dirname, 'sidebars-pift-toolkit.js'),
        remarkPlugins: [
          [codeImport, {rootDir: repoRoot, allowImportingFromOutside: true}],
        ],
        editUrl:
          'https://github.com/worldbank/ai4data/edit/main/research/pift-toolkit/docs/',
      }),
    ],
    [
      '@easyops-cn/docusaurus-search-local',
      /** @type {import('@easyops-cn/docusaurus-search-local').PluginOptions} */
      ({
        hashed: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: true,
        docsRouteBasePath: ['/docs', '/pift-toolkit'],
      }),
    ],
  ],
};

export default config;
