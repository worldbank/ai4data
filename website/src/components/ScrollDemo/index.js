import clsx from 'clsx';
import {useEffect, useRef, useState} from 'react';
import Link from '@docusaurus/Link';
import PcnEvidence from './PcnEvidence';
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
          Output: six issue categories
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
      <div className={styles.chartLabel}>GDP growth (annual %), Nigeria</div>
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
    title: 'Five-agent metadata review pipeline',
    body: (
      <p>
        Detection, filtering, classification, and scoring run as
        specialized, sequential agents. Each step has a narrow task, which
        keeps the output consistent and auditable across thousands of
        records.
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
    title: 'Evidence-backed anomaly explanations',
    body: (
      <p>
        A statistical detector flags a data point as unusual. The
        elicitation pipeline turns that flag into a structured,
        evidence-backed explanation with a classification, a confidence
        score, and a cited source. A reviewer can check it in seconds.
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
    title: 'Search and MCP access to the catalog',
    body: (
      <p>
        The catalog can be searched in plain language. The same query can
        also be sent by an AI agent as a Model Context Protocol (MCP) tool
        call. Both return results from the same indicator index.
      </p>
    ),
    link: {to: '/docs/mcp/', label: 'See the MCP integration guide'},
    Evidence: QueryToggle,
  },
  {
    id: 'pcn',
    eyebrow: 'Proof-Carrying Numbers',
    title: 'Number verification in chatbot answers',
    body: (
      <p>
        In a chatbot answer built on official statistics, each number
        carries a tag naming the record it came from. The application
        compares the number with that record and shows a check mark when
        they match. The application draws the check mark, so the model
        cannot produce one. Numbers that differ from the record are
        flagged, and numbers without a tag receive no mark.
      </p>
    ),
    link: {
      to: 'https://arxiv.org/abs/2509.06902',
      label: 'Read the PCN paper',
    },
    Evidence: PcnEvidence,
  },
];

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export default function ScrollDemo() {
  const [progress, setProgress] = useState(0);
  const trackRef = useRef(null);
  const stageRef = useRef(null);

  useEffect(() => {
    // Progress is derived from scroll position, so it is identical in both
    // scroll directions. The stage is pinned while the track scrolls past;
    // progress runs from 0 (first stop) to Stops.length - 1 (last stop).
    let frame = null;
    const update = () => {
      frame = null;
      const track = trackRef.current;
      const stage = stageRef.current;
      if (!track || !stage) {
        return;
      }
      const stickyTop = parseFloat(window.getComputedStyle(stage).top) || 0;
      const rect = track.getBoundingClientRect();
      const total = rect.height - stage.offsetHeight;
      const t = total > 0 ? clamp((stickyTop - rect.top) / total, 0, 1) : 0;
      setProgress(t * (Stops.length - 1));
    };
    const onScroll = () => {
      if (frame === null) {
        frame = window.requestAnimationFrame(update);
      }
    };
    update();
    window.addEventListener('scroll', onScroll, {passive: true});
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      if (frame !== null) {
        window.cancelAnimationFrame(frame);
      }
    };
  }, []);

  const active = Math.round(progress);

  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Examples</span>
          <Heading as="h2" className={styles.title}>
            Example outputs
          </Heading>
          <p className={styles.lede}>
            Scroll to step through the examples. The panel and its
            description change together. Every number and label below is
            drawn from the program&apos;s own documentation and data.
          </p>
        </div>

        <div
          className={styles.track}
          ref={trackRef}
          style={{'--stops': Stops.length}}>
          <div className={styles.stage} ref={stageRef}>
            <div className={styles.layout}>
              <div className={styles.rail}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeader}>
                    <span className={styles.panelLabel}>
                      Example {active + 1} of {Stops.length}
                      <span className={styles.panelSource}>
                        {Stops[active].eyebrow}
                      </span>
                    </span>
                    <span className={styles.panelDots} aria-hidden="true">
                      {Stops.map((stop, i) => (
                        <span
                          key={stop.id}
                          className={
                            i === active ? styles.dotActive : styles.dot
                          }
                        />
                      ))}
                    </span>
                  </div>
                  <div className={styles.panelBody}>
                    {Stops.map((stop, i) => {
                      const Evidence = stop.Evidence;
                      return (
                        <div
                          className={clsx(
                            styles.layer,
                            i === active && styles.layerActive,
                          )}
                          key={stop.id}
                          aria-hidden={i !== active}>
                          <Evidence active={i === active} />
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div className={styles.stops}>
                {Stops.map((stop, i) => {
                  const StopEvidence = stop.Evidence;
                  return (
                    <div
                      className={clsx(
                        styles.stop,
                        styles.layer,
                        i === active && styles.layerActive,
                      )}
                      key={stop.id}
                      aria-hidden={i !== active}>
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
        </div>
      </div>
    </section>
  );
}
