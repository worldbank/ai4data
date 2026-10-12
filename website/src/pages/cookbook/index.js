import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import ThemedImage from '@theme/ThemedImage';
import {cookbooks, groups, readingPaths} from '@site/src/content/cookbooks';
import {countWord} from '@site/src/content/visibility';
import styles from './styles.module.css';

const DDG_URL = 'https://www.worldbank.org/en/about/unit/unit-dec/dev';

export default function CookbookLanding() {
  const logoLight = useBaseUrl('/img/ddg-logo.png');
  const logoDark = useBaseUrl('/img/ddg-logo-dark.png');
  return (
    <Layout
      title="Cookbooks"
      description="Practical, problem-oriented guides from the AI for Data – Data for AI program.">
      <header className={styles.hero}>
        <div className="container">
          <Link className={styles.logo} to={DDG_URL}>
            <ThemedImage
              alt="World Bank Group, Development Data Group"
              sources={{light: logoLight, dark: logoDark}}
            />
          </Link>
          <span className="eyebrow">Cookbooks</span>
          <Heading as="h1" className={styles.title}>
            Practical guides for AI-ready data
          </Heading>
          <p className={styles.lede}>
            {countWord(cookbooks.length, true)} cookbooks for national statistical organizations, each
            organized around the questions a team asks, with recipes at three
            maturity levels, scripts that run on a shared example, open
            standards, and checklists. A team new to the series starts with
            the reading path for its role; a team with an assessment profile
            starts from the dimension with the largest gap.
          </p>
        </div>
      </header>
      <main className={styles.main}>
        <div className="container">
          {groups.map((g) => (
            <section key={g.title} className={styles.group}>
              <Heading as="h2" className={styles.groupTitle}>
                {g.title}
              </Heading>
              <div className={styles.grid}>
                {g.ids.map((id) => cookbooks.find((c) => c.id === id)).map((c) => (
                  <Link className={styles.card} to={c.to} key={c.id}>
                    <span className={styles.audience}>{c.audience}</span>
                    <Heading as="h3" className={styles.cardTitle}>
                      {c.title}
                    </Heading>
                    <p className={styles.cardBody}>{c.description}</p>
                    <span className={styles.meta}>
                      {c.chapters} chapters <span aria-hidden="true">→</span>
                    </span>
                  </Link>
                ))}
              </div>
            </section>
          ))}
          <section className={styles.paths} id="reading-paths">
            <Heading as="h2" className={styles.groupTitle}>
              Reading paths by role
            </Heading>
            <div className={styles.pathGrid}>
              {readingPaths.map((p) => (
                <div key={p.role} className={styles.path}>
                  <span className={styles.pathRole}>{p.role}</span>
                  <ol className={styles.pathSteps}>
                    {p.steps.map((st) => (
                      <li key={st.to}>
                        <Link to={st.to}>{st.label}</Link>
                      </li>
                    ))}
                  </ol>
                </div>
              ))}
            </div>
          </section>
          <p className={styles.authoring}>
            New cookbooks follow the{' '}
            <Link to="/cookbook/authoring/">authoring guide</Link>, which sets
            the structure and writing rules and names the scaffold and checker.
          </p>
        </div>
      </main>
    </Layout>
  );
}
