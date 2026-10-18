// @ts-check
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {themes as prismThemes} from 'prism-react-renderer';
import codeImport from 'remark-code-import';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

// Cookbooks that are built but not public. The list is kept in one JSON file
// read here, by src/content/visibility.js, and by scripts/docs/check_cookbooks.py.
const hiddenCookbooks = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'cookbook-visibility.json'), 'utf8'),
).hidden;
const hiddenCookbookPath = hiddenCookbooks.length
  ? new RegExp(`^cookbook/(${hiddenCookbooks.map((id) => id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})/`)
  : null;

/** Marks every page of a hidden cookbook as unlisted at build time. */
async function parseFrontMatter(params) {
  const result = await params.defaultParseFrontMatter(params);
  const rel = path.relative(repoRoot, params.filePath).split(path.sep).join('/');
  if (hiddenCookbookPath && hiddenCookbookPath.test(rel)) {
    result.frontMatter.unlisted = true;
  }
  return result;
}

/** Removes the downloadable files of hidden cookbooks from the build output. */
function hideCookbookFiles() {
  return {
    name: 'hide-cookbook-files',
    async postBuild({outDir}) {
      for (const id of hiddenCookbooks) {
        await fs.promises.rm(path.join(outDir, 'cookbook-files', id), {recursive: true, force: true});
      }
    },
  };
}

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI for Data - Data for AI',
  tagline:
    'The AI for Data – Data for AI program applies AI to producing, reviewing, and sharing development data. It also prepares that data so AI systems can find, interpret, and cite it correctly.',
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
    parseFrontMatter,
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
        sitemap: {
          ignorePatterns: [
            '/ai4data/nso-roadmap',
            '/ai4data/nso-roadmap/**',
            ...hiddenCookbooks.map((id) => `/ai4data/cookbook/${id}/**`),
          ],
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/social-card.png',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: '',
        logo: {
          alt: 'AI for Data - Data for AI',
          src: 'img/program-logo.png',
          srcDark: 'img/program-logo-dark.png',
          height: 40,
          width: 119,
        },
        items: [
          {
            type: 'dropdown',
            label: 'Hub',
            position: 'left',
            to: '/hub',
            items: [
              {to: '/hub', label: 'AI readiness in practice'},
              {to: '/hub#assess', label: 'Assess'},
              {to: '/hub#document', label: 'Document'},
              {to: '/hub#connect', label: 'Connect'},
              {to: '/hub#verify', label: 'Verify'},
              {to: '/hub#implement', label: 'Implement'},
              {to: '/hub#adopt', label: 'Adopt'},
            ],
          },
          {
            to: '/tools',
            label: 'Tools',
            position: 'left',
          },
          {
            to: '/cookbook/',
            label: 'Cookbook',
            position: 'left',
            activeBaseRegex: '/cookbook/',
          },
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            type: 'dropdown',
            label: 'AI readiness',
            position: 'left',
            items: [
              {to: '/ai-readiness-assessment', label: 'Assessment framework'},
              {to: '/ai-readiness-in-practice', label: 'Operationalizing the assessment'},
              {to: '/modern-nso', label: 'A modern statistical office'},
            ],
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
              {label: 'A modern statistical office', to: '/modern-nso'},
            ],
          },
          {
            title: 'Readiness',
            items: [
              {label: 'AI-readiness assessment', to: '/ai-readiness-assessment'},
              {label: 'Operationalizing the assessment', to: '/ai-readiness-in-practice'},
              {label: 'Roadmap', to: '/nso-roadmap', className: 'footer__link-quiet'},
            ],
          },
          {
            title: 'Implementation',
            items: [
              {label: 'Implementation hub', to: '/hub'},
              {label: 'Tools', to: '/tools'},
              {label: 'Reference implementation', to: '/docs/reference-implementation/'},
              {label: 'Cookbook', to: '/cookbook/'},
            ],
          },
          {
            title: 'Repository',
            items: [
              {label: 'GitHub', href: 'https://github.com/worldbank/ai4data'},
              {label: 'Issues', href: 'https://github.com/worldbank/ai4data/issues'},
            ],
          },
        ],
        copyright: `
          <div class="fb-brand">
            <a href="https://www.worldbank.org/en/about/unit/unit-dec/dev" class="fb-logo"><img src="/ai4data/img/ddg-logo-dark.png" alt="World Bank Group, Development Data Group"></a>
            <strong>AI for Data - Data for AI</strong>
            <span>A program of the <a href="https://www.worldbank.org/en/about/unit/unit-dec/dev">Office of the WBG Chief Statistician and Development Data Group</a>, World Bank Group.</span>
            <a href="mailto:aifordata@worldbank.org">aifordata@worldbank.org</a>
          </div>
          <div class="fb-legal">
            <div>Country borders or names do not necessarily reflect the World Bank Group's official position. All maps are for illustrative purposes and do not imply the expression of any opinion on the part of the World Bank, concerning the legal status of any country or territory or concerning the delimitation of frontiers or boundaries.</div>
            <div>All content (unless otherwise specified) is subject to the <a href="https://opensource.org/license/mit">MIT License</a>. Copyright © ${new Date().getFullYear()} World Bank Group, <a href="https://www.worldbank.org/en/about/unit/unit-dec/dev">Office of the WBG Chief Statistician &amp; Development Data Group</a>.</div>
          </div>
        `,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'json', 'yaml', 'python'],
      },
    }),

  plugins: [
    hideCookbookFiles,
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
      '@docusaurus/plugin-content-docs',
      /** @type {import('@docusaurus/plugin-content-docs').Options} */
      ({
        id: 'cookbook',
        path: '../cookbook',
        routeBasePath: 'cookbook',
        exclude: ['**/_*/**', '**/_*.{md,mdx}'],
        sidebarPath: path.join(__dirname, 'sidebars-cookbook.js'),
        remarkPlugins: [
          [codeImport, {rootDir: repoRoot, allowImportingFromOutside: true}],
        ],
        editUrl: 'https://github.com/worldbank/ai4data/edit/main/cookbook/',
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
        docsRouteBasePath: ['/docs', '/pift-toolkit', '/cookbook'],
      }),
    ],
  ],
};

export default config;
