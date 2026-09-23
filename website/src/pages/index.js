import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import ScaleBand from '@site/src/components/ScaleBand';
import DataLifecycle from '@site/src/components/DataLifecycle';
import PipelineShowcase from '@site/src/components/PipelineShowcase';
import AnomalyExample from '@site/src/components/AnomalyExample';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className={styles.heroInner}>
          <span className="eyebrow">World Bank — Development Data Group</span>
          <Heading as="h1" className={styles.heroTitle}>
            Making development data <strong>AI-ready.</strong>
          </Heading>
          <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx('button button--lg', styles.primaryButton)}
              to="/docs/introduction">
              Read the Documentation
            </Link>
            <Link
              className={clsx('button button--lg', styles.secondaryButton)}
              to="https://github.com/worldbank/ai4data">
              View on GitHub
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function ClosingCta() {
  return (
    <section className={styles.closing}>
      <div className="container">
        <div className={styles.closingInner}>
          <span className="eyebrow" style={{color: '#7ba7f2'}}>
            Open Source
          </span>
          <Heading as="h2" className={styles.closingTitle}>
            Built in the open by the Development Data Group
          </Heading>
          <p className={styles.closingText}>
            Every workstream on this site ships as Python tooling and
            documentation you can read end to end — methodology, pipeline
            code, and API reference alike.
          </p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--lg',
                styles.closingButtonPrimary,
              )}
              to="https://github.com/worldbank/ai4data">
              Explore the Repository
            </Link>
            <Link
              className={clsx(
                'button button--lg',
                styles.closingButtonSecondary,
              )}
              to="mailto:ai4data@worldbank.org">
              Get in Touch
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="A World Bank Development Data Group program applying AI across the development data lifecycle to make development data AI-ready.">
      <HomepageHeader />
      <main>
        <ScaleBand />
        <DataLifecycle />
        <PipelineShowcase />
        <AnomalyExample />
        <HomepageFeatures />
        <ClosingCta />
      </main>
    </Layout>
  );
}
