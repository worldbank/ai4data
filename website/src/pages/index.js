import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Challenge from "@site/src/components/Challenge";
import Program from "@site/src/components/Program";
import ScrollDemo from "@site/src/components/ScrollDemo";
import HeroVisual from "@site/src/components/HeroVisual";
import CookbookCta from "@site/src/components/CookbookCta";
import GetInvolved from "@site/src/components/GetInvolved";
import HubGrid from "@site/src/components/HubGrid";

import styles from "./index.module.css";

const DDG_URL = "https://www.worldbank.org/en/about/unit/unit-dec/dev";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  const ddgLogo = useBaseUrl("/img/ddg-logo.png");
  const programLogo = useBaseUrl("/img/program-logo.png");
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className={styles.heroGrid}>
          <div className={styles.heroInner}>
            <div className={styles.logoLockup}>
              <Link className={styles.lockupItem} to={DDG_URL}>
                <img
                  src={ddgLogo}
                  alt="World Bank Group, Development Data Group"
                  className={styles.lockupDdg}
                />
              </Link>
              <span className={styles.lockupDivider} aria-hidden="true" />
              <span className={styles.lockupItem}>
                <img
                  src={programLogo}
                  alt="AI for Data, Data for AI"
                  className={styles.lockupProgram}
                />
              </span>
            </div>
            <Link className={clsx("eyebrow", styles.eyebrowLink)} to={DDG_URL}>
              Office of the WBG Chief Statistician &amp; Development Data Group
            </Link>
            <Heading as="h1" className={styles.heroTitle}>
              <strong>AI-ready</strong> development data
            </Heading>
            <p className={styles.heroSubtitle}>
              {siteConfig.tagline}
            </p>
            <div className={styles.buttons}>
              <Link
                className={clsx("button button--lg", styles.primaryButton)}
                to="/hub"
              >
                Implementation hub
              </Link>
              <Link
                className={clsx("button button--lg", styles.secondaryButton)}
                to="/docs/introduction"
              >
                Documentation
              </Link>
            </div>
          </div>
          <div className={styles.heroVisual}>
            <HeroVisual />
          </div>
        </div>
      </div>
    </header>
  );
}

function HubSection() {
  return (
    <section className={styles.hub}>
      <div className="container">
        <div className={styles.hubHead}>
          <span className="eyebrow">Implementation hub</span>
          <Heading as="h2" className={styles.hubTitle}>
            AI readiness in practice
          </Heading>
          <p className={styles.hubLede}>
            The hub lists the program&apos;s tools, guides, and cookbooks
            under six areas. The tools catalogue records a status for each
            tool: in operation, deployable, prototype, or method.
          </p>
        </div>
        <HubGrid compact headingLevel="h3" />
        <p className={styles.hubLinks}>
          <Link to="/hub">The hub →</Link>
          <Link to="/tools">Tools catalogue →</Link>
          <Link to="/docs/reference-implementation/">Reference implementation →</Link>
        </p>
      </div>
    </section>
  );
}

function ClosingCta() {
  return (
    <section className={styles.closing}>
      <div className="container">
        <div className={styles.closingRow}>
          <div className={styles.closingInner}>
            <Heading as="h2" className={styles.closingTitle}>
              Open-source tooling and documentation
            </Heading>
            <p className={styles.closingText}>
              Methods, software, and guidance are developed as open resources,
              with documentation for each released workstream.
            </p>
          </div>
          <Link
            className={clsx("button button--md", styles.closingButtonPrimary)}
            to="https://github.com/worldbank/ai4data"
          >
            Repository
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}
    >
      <HomepageHeader />
      <main>
        <Challenge />
        <HubSection />
        <ScrollDemo />
        <Program />
        <CookbookCta />
        <GetInvolved />
        <ClosingCta />
      </main>
    </Layout>
  );
}
