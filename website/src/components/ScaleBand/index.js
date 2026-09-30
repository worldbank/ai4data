import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Stats = [
  {
    value: '1,400+',
    label: 'WDI Indicators',
    detail:
      'Across 217 economies. One curator cannot hand-check every release.',
  },
  {
    value: '1,000s',
    label: 'Anomalies Per Review Cycle',
    detail:
      'Flagged across WDI and Scorecard data. A team cannot investigate each one individually.',
  },
  {
    value: '500–1,000',
    label: 'Variables Per Survey',
    detail:
      'In a single DHS-style microdata catalog entry, before any cross-survey comparison.',
  },
  {
    value: '50+',
    label: 'Languages In Scope',
    detail: 'Catalog metadata is written in many languages.',
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
            Statistical offices and development organizations manage data
            at a scale that manual review cannot cover. Indicator databases
            span hundreds of geographies, microdata libraries hold thousands
            of surveys, and metadata must work across dozens of languages.
            A curator cannot check every description, and an analyst cannot
            investigate every flagged anomaly. The World Bank&apos;s catalogs
            illustrate the scale:
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
