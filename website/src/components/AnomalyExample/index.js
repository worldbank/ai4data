import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

// Real worked example from docs/anomaly/explanation/elicitation-pipeline.md
const YEARS = [2013, 2014, 2015, 2016, 2017];
const VALUES = [6.67, 6.31, 2.65, -1.62, 0.8];

// value -3..8 mapped to y 140..10
const yFor = (v) => 140 - ((v + 3) / 11) * 130;
const X = [20, 110, 200, 290, 380];
const POINTS = X.map((x, i) => `${x},${yFor(VALUES[i]).toFixed(1)}`).join(' ');

export default function AnomalyExample() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">A Worked Example</span>
          <Heading as="h2" className={styles.title}>
            From a flagged anomaly to a verifiable explanation
          </Heading>
          <p className={styles.lede}>
            A statistical detector flags a data point as unusual. The
            elicitation pipeline turns that flag into a structured,
            evidence-backed explanation a reviewer can check.
          </p>
        </div>

        <div className={styles.panel}>
          <div className={styles.chartCol}>
            <div className={styles.chartLabel}>
              GDP growth (annual %) — Nigeria
            </div>
            <svg
              className={styles.chart}
              viewBox="0 0 400 150"
              preserveAspectRatio="xMidYMid meet">
              <rect
                x={200}
                y={0}
                width={90}
                height={150}
                className={styles.anomalyBand}
              />
              <polyline points={POINTS} className={styles.line} fill="none" />
              {X.map((x, i) => (
                <circle
                  key={x}
                  cx={x}
                  cy={yFor(VALUES[i])}
                  r={i === 3 ? 5 : 3}
                  className={i === 3 ? styles.pointFlagged : styles.point}
                />
              ))}
              {YEARS.map((year, i) => (
                <text
                  key={year}
                  x={X[i]}
                  y={144}
                  textAnchor="middle"
                  className={styles.axisLabel}>
                  {year}
                </text>
              ))}
            </svg>
            <div className={styles.chartCaption}>
              Anomaly window: 2015–2016 · value drops to −1.62%
            </div>
          </div>

          <div className={styles.outputCol}>
            <div className={styles.outputRow}>
              <span className={styles.outputKey}>Classification</span>
              <span className={styles.badge}>external_driver</span>
            </div>
            <div className={styles.outputRow}>
              <span className={styles.outputKey}>Confidence</span>
              <span className={styles.outputValue}>0.92</span>
            </div>
            <div className={styles.outputRow}>
              <span className={styles.outputKey}>Evidence strength</span>
              <span className={styles.outputValue}>strong_direct</span>
            </div>
            <div className={styles.outputRow}>
              <span className={styles.outputKey}>Verifiability</span>
              <span className={styles.outputValue}>well_documented</span>
            </div>
            <p className={styles.explanation}>
              &ldquo;Nigeria experienced a sharp GDP contraction in 2016, its
              first recession in 25 years, driven by the collapse in global
              oil prices beginning in 2014–2015 and militant attacks on oil
              infrastructure in the Niger Delta.&rdquo;
            </p>
            <Link
              className={styles.footLink}
              to="/docs/anomaly/explanation/elicitation-pipeline#example-llm-inputoutput">
              See the full input/output schema →
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
