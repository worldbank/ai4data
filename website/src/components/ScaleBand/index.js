import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Stats = [
  {
    value: '1,400+',
    label: 'WDI indicators',
    detail: 'across 217 economies, some with records back to the 1960s',
  },
  {
    value: '1,000s',
    label: 'anomalies per review cycle',
    detail: 'flagged by statistical detectors across WDI and Scorecard data',
  },
  {
    value: '500–1,000',
    label: 'variables per household survey',
    detail: 'in a single DHS-style microdata catalog entry',
  },
  {
    value: '50+',
    label: 'languages supported',
    detail: 'via multilingual embedding models for search and clustering',
  },
];

export default function ScaleBand() {
  return (
    <section className={styles.section}>
      <div className="container">
        <span className={styles.eyebrow}>The Challenge</span>
        <Heading as="h2" className={styles.title}>
          Development data at a scale manual review can&apos;t match
        </Heading>
        <p className={styles.lede}>
          A single metadata curator cannot review thousands of indicator
          descriptions for accuracy. A statistical team cannot manually
          investigate every flagged anomaly. This program automates the
          first pass — so human experts spend their time validating, not
          searching.
        </p>
        <div className={styles.grid}>
          {Stats.map((stat) => (
            <div className={styles.stat} key={stat.label}>
              <div className={styles.statValue}>{stat.value}</div>
              <div className={styles.statLabel}>{stat.label}</div>
              <p className={styles.statDetail}>{stat.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
