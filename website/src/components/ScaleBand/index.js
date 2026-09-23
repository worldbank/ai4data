import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Stats = [
  {
    value: '1,400+',
    label: 'WDI Indicators',
    detail:
      'Across 217 economies — too many for one curator to hand-check every release.',
  },
  {
    value: '1,000s',
    label: 'Anomalies Per Review Cycle',
    detail:
      'Flagged across WDI and Scorecard data — far more than a team can individually investigate.',
  },
  {
    value: '500–1,000',
    label: 'Variables Per Survey',
    detail:
      'In a single DHS-style microdata catalog entry, before cross-survey comparison even starts.',
  },
  {
    value: '50+',
    label: 'Languages In Scope',
    detail: 'Most metadata tooling is built for one language. This data isn’t.',
  },
];

export default function ScaleBand() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">The Challenge</span>
          <Heading as="h2" className={styles.title}>
            Development data has outgrown manual review
          </Heading>
          <p className={styles.lede}>
            The World Bank&apos;s indicator catalogs and microdata libraries
            have grown far past what a curator or analyst can check by hand.
            A team can&apos;t hand-verify thousands of indicator
            descriptions, investigate every anomaly a monitoring dataset
            flags, or cover 50+ languages at once. This program automates
            that first pass, reserving expert time for the judgment calls
            only a human can make.
          </p>
        </div>
        <div className={styles.grid}>
          {Stats.map((stat) => (
            <div className={styles.tile} key={stat.label}>
              <div className={styles.label}>{stat.label}</div>
              <div className={styles.value}>{stat.value}</div>
              <p className={styles.detail}>{stat.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
