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
    label: 'anomalies / review cycle',
    detail: 'flagged by statistical detectors across WDI and Scorecard data',
  },
  {
    value: '500–1,000',
    label: 'variables / survey',
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
        <div className={styles.head}>
          <span className="kicker">The Challenge</span>
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
        </div>
        <div className={styles.ledger}>
          {Stats.map((stat) => (
            <div className={styles.row} key={stat.label}>
              <div className={styles.rowValue}>{stat.value}</div>
              <div className={styles.rowMeta}>
                <div className={styles.rowLabel}>{stat.label}</div>
                <p className={styles.rowDetail}>{stat.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
