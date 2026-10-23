import Head from '@docusaurus/Head';
import Link from '@docusaurus/Link';
import useGlobalData from '@docusaurus/useGlobalData';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {allCookbooks} from '@site/src/content/cookbooks';
import {hiddenCookbooks} from '@site/src/content/visibility';
import styles from './unlisted.module.css';

// A catalogue of the pages that exist on the site without being listed on it: the hidden
// cookbooks, read from website/cookbook-visibility.json, and the standalone pages below.
// This page is itself unlisted: no link leads here, it carries a noindex tag, and the site
// config keeps it out of the sitemap and the search index.

const STANDALONE_PAGES = [
  {to: '/nso-roadmap', title: 'Roadmap for AI-readiness in a statistical office', note: 'Planning draft; linked only from the footer.'},
  {to: '/modern-nso', title: 'A modern statistical office', note: 'Composite portrait at three maturity levels.'},
];

function chapterLabel(path, cookbookPath) {
  const slug = path.replace(cookbookPath, '').replace(/\/$/, '');
  if (!slug) return 'Overview';
  return slug.charAt(0).toUpperCase() + slug.slice(1).replace(/-/g, ' ');
}

export default function Unlisted() {
  const globalData = useGlobalData();
  const docsPlugins = globalData['docusaurus-plugin-content-docs'] || {};
  const allDocs = Object.values(docsPlugins).flatMap((plugin) => (plugin.versions || []).flatMap((v) => v.docs || []));
  const hidden = hiddenCookbooks.map((id) => {
    const card = allCookbooks.find((c) => c.id === id);
    const prefix = `/cookbook/${id}/`;
    const chapters = allDocs
      .filter((d) => d.path.startsWith(prefix) && d.path !== prefix)
      .sort((a, b) => a.path.localeCompare(b.path));
    return {id, card, prefix, chapters};
  });
  const otherUnlisted = allDocs.filter((d) => d.unlisted && !hiddenCookbooks.some((id) => d.path.startsWith(`/cookbook/${id}/`)));

  return (
    <Layout title="Unlisted pages" description="The pages of this site that exist without being listed: hidden cookbooks and standalone drafts, with direct links for reviewers.">
      <Head>
        <meta name="robots" content="noindex, nofollow" />
      </Head>
      <main className={styles.page}>
        <Heading as="h1">Unlisted pages</Heading>
        <p className={styles.lede}>
          These pages are built and deployed with the site but are not listed on it: no card, no navigation entry, no search result, no sitemap entry, and a noindex tag. They open from
          these links for review. This page is unlisted in the same way; keep its address.
        </p>

        <Heading as="h2">Hidden cookbooks</Heading>
        <p className={styles.note}>
          Listed in <code>website/cookbook-visibility.json</code> under <code>hidden</code>. Removing an id from that list publishes the cookbook; nothing else changes.
        </p>
        {hidden.map(({id, card, prefix, chapters}) => (
          <section key={id} className={styles.cookbook}>
            <Heading as="h3">
              <Link to={prefix}>{card ? card.title : id}</Link>
            </Heading>
            {card && (
              <p className={styles.meta}>
                {card.audience} · version {card.version}, {card.date} · {card.chapters} chapters
              </p>
            )}
            <ul className={styles.chapters}>
              {chapters.map((d) => (
                <li key={d.path}>
                  <Link to={d.path}>{chapterLabel(d.path, prefix)}</Link>
                  <span className={styles.path}>{d.path}</span>
                </li>
              ))}
            </ul>
          </section>
        ))}

        <Heading as="h2">Standalone pages</Heading>
        <ul className={styles.chapters}>
          {STANDALONE_PAGES.map((p) => (
            <li key={p.to}>
              <Link to={p.to}>{p.title}</Link>
              <span className={styles.path}>{p.to}</span>
              <span className={styles.note}>{p.note}</span>
            </li>
          ))}
        </ul>

        {otherUnlisted.length > 0 && (
          <>
            <Heading as="h2">Other unlisted documents</Heading>
            <ul className={styles.chapters}>
              {otherUnlisted.map((d) => (
                <li key={d.path}>
                  <Link to={d.path}>{d.id}</Link>
                  <span className={styles.path}>{d.path}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </main>
    </Layout>
  );
}
