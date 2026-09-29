import {useEffect, useRef, useState} from 'react';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import QueryToggle from '@site/src/components/QueryToggle';
import styles from './styles.module.css';

const PipelineStages = [
  {name: 'Primary', role: 'Detect'},
  {name: 'Secondary', role: 'Re-scan'},
  {name: 'Critic', role: 'Filter'},
  {name: 'Categorizer', role: 'Classify'},
  {name: 'Severity Scorer', role: 'Rank'},
];

const IssueCategories = [
  'Typo / Language',
  'Formatting / Structure',
  'Missing / Redundant Info',
  'Inconsistency / Conflict',
  'Incorrect / Invalid Content',
  'Ambiguity / Unclear',
];

// Real worked example from docs/anomaly/explanation/elicitation-pipeline.md
const YEARS = [2013, 2014, 2015, 2016, 2017];
const VALUES = [6.67, 6.31, 2.65, -1.62, 0.8];
const yFor = (v) => 140 - ((v + 3) / 11) * 130;
const X = [20, 110, 200, 290, 380];
const POINTS = X.map((x, i) => `${x},${yFor(VALUES[i]).toFixed(1)}`).join(' ');

function PipelineEvidence() {
  return (
    <div className={styles.evidenceBox}>
      <div className={styles.pipelineRow}>
        {PipelineStages.map((stage, i) => (
          <div className={styles.pipelineStage} key={stage.name}>
            <div className={styles.pipelineStageHead}>
              <span className={styles.pipelineIndex}>{i + 1}</span>
              <span className={styles.pipelineRole}>{stage.role}</span>
            </div>
            <div className={styles.pipelineName}>{stage.name}</div>
          </div>
        ))}
      </div>
      <div className={styles.pipelineFoot}>
        <span className={styles.pipelineFootLabel}>
          Output — six issue categories
        </span>
        <div className={styles.chips}>
          {IssueCategories.map((c) => (
            <span className={styles.chip} key={c}>
              {c}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function AnomalyEvidence() {
  return (
    <div className={styles.evidenceBox}>
      <div className={styles.chartLabel}>GDP growth (annual %) — Nigeria</div>
      <svg
        className={styles.chart}
        viewBox="0 0 400 150"
        preserveAspectRatio="xMidYMid meet">
        <rect x={200} y={0} width={90} height={150} className={styles.anomalyBand} />
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
          <text key={year} x={X[i]} y={144} textAnchor="middle" className={styles.axisLabel}>
            {year}
          </text>
        ))}
      </svg>
      <div className={styles.anomalyOutput}>
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
      </div>
    </div>
  );
}

const Stops = [
  {
    id: 'pipeline',
    eyebrow: 'Metadata Reviewer',
    title: 'Five agents, one quality pipeline',
    body: (
      <p>
        Detection, filtering, classification, and scoring run as
        specialized, sequential agents instead of one prompt asked to do
        everything at once — so the output stays consistent and auditable
        across thousands of records.
      </p>
    ),
    link: {
      to: '/docs/metadata-reviewer/overview',
      label: 'See the full pipeline reference',
    },
    Evidence: PipelineEvidence,
  },
  {
    id: 'anomaly',
    eyebrow: 'Anomaly Detection and Explanation',
    title: 'Every anomaly ships with evidence',
    body: (
      <p>
        A statistical detector flags a data point as unusual. The
        elicitation pipeline turns that flag into a structured,
        evidence-backed explanation — classification, confidence, and a
        cited source — that a reviewer can check in seconds.
      </p>
    ),
    link: {
      to: '/docs/anomaly/explanation/elicitation-pipeline#example-llm-inputoutput',
      label: 'See the full input/output schema',
    },
    Evidence: AnomalyEvidence,
  },
  {
    id: 'access',
    eyebrow: 'Data Discoverability & MCP',
    title: 'The same question, answered two ways',
    body: (
      <p>
        A person can search in plain language. An AI agent can issue the
        same request as a Model Context Protocol tool call. Both resolve
        against the same indicators, so a client never needs a bespoke
        integration.
      </p>
    ),
    link: {to: '/docs/mcp/', label: 'See the MCP integration guide'},
    Evidence: QueryToggle,
  },
];

export default function ScrollDemo() {
  const [active, setActive] = useState(0);
  const refs = useRef([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = Number(entry.target.dataset.stopIndex);
            setActive(idx);
          }
        });
      },
      {rootMargin: '-35% 0px -50% 0px', threshold: 0},
    );
    refs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, []);

  const ActiveEvidence = Stops[active].Evidence;

  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">See It Work</span>
          <Heading as="h2" className={styles.title}>
            Real output, not mockups
          </Heading>
          <p className={styles.lede}>
            Scroll to see the evidence behind each workstream — the panel
            updates to match. Every number and label below is drawn from
            the program&apos;s own documentation and data.
          </p>
        </div>

        <div className={styles.layout}>
          <div className={styles.rail}>
            <div className={styles.stickyPanel}>
              <ActiveEvidence />
            </div>
          </div>

          <div className={styles.stops}>
            {Stops.map((stop, i) => {
              const StopEvidence = stop.Evidence;
              return (
                <div
                  className={styles.stop}
                  key={stop.id}
                  data-stop-index={i}
                  ref={(el) => (refs.current[i] = el)}>
                  <span className={styles.stopEyebrow}>{stop.eyebrow}</span>
                  <Heading as="h3" className={styles.stopTitle}>
                    {stop.title}
                  </Heading>
                  {stop.body}
                  <div className={styles.stopEvidenceMobile}>
                    <StopEvidence />
                  </div>
                  <Link className={styles.stopLink} to={stop.link.to}>
                    {stop.link.label} →
                  </Link>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
