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
            Statistical offices and development organizations increasingly
            manage data at a scale no team can review by hand — indicator
            databases spanning hundreds of geographies, microdata libraries
            with thousands of surveys, metadata that has to work across
            dozens of languages. A curator can&apos;t hand-check every
            description, an analyst can&apos;t investigate every flagged
            anomaly, and headcount alone never catches up. The World
            Bank&apos;s own catalogs make that scale concrete:
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
