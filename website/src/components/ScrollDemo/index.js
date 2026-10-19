import clsx from 'clsx';
import {useEffect, useRef, useState} from 'react';
import Link from '@docusaurus/Link';
import AnomalyEvidence from './AnomalyEvidence';
import MetadataEvidence from './MetadataEvidence';
import PcnEvidence from './PcnEvidence';
import SearchEvidence from './SearchEvidence';
import SnapshotEvidence from './SnapshotEvidence';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Stops = [
  {
    id: 'snapshot',
    eyebrow: 'Data Snapshots',
    title: 'From a PDF figure to structured data',
    body: (
      <p>
        A layout detection model locates a figure or table on a PDF page. The
        region is saved as a data snapshot, and the snapshot is converted to
        structured data that other tools can search and compute on.
      </p>
    ),
    link: {
      to: 'https://arxiv.org/abs/2606.06242',
      label: 'Benchmark paper',
    },
    Evidence: SnapshotEvidence,
  },
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
      label: 'Pipeline reference',
    },
    Evidence: MetadataEvidence,
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
      label: 'Input and output schema',
    },
    Evidence: AnomalyEvidence,
  },
  {
    id: 'access',
    eyebrow: 'Data Discoverability & MCP',
    title: 'Search and MCP access to the catalog',
    body: (
      <p>
        The catalog can be searched in plain language, and the results are
        ranked by meaning. A result can match a query without sharing any of
        its words. The same query can also be sent by an AI agent as a Model
        Context Protocol (MCP) tool call.
      </p>
    ),
    link: {to: '/docs/mcp/', label: 'MCP integration guide'},
    Evidence: SearchEvidence,
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
    caveat:
      'A check mark means the number matches the record it cites under the stated policy. It does not say whether the record, the method behind it, or the interpretation is right.',
    link: {
      to: '/docs/pcn/',
      label: 'Proof-Carrying Numbers',
    },
    Evidence: PcnEvidence,
  },
];

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// On small screens the panels are inline under their text. Each one starts
// when it first scrolls into view.
function MobileEvidence({Evidence}) {
  const ref = useRef(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      return undefined;
    }
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          obs.disconnect();
        }
      },
      {threshold: 0.3},
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return (
    <div ref={ref}>
      <Evidence active={seen} />
    </div>
  );
}

export default function ScrollDemo() {
  const [progress, setProgress] = useState(0);
  const trackRef = useRef(null);
  const stageRef = useRef(null);

  useEffect(() => {
    // Progress is derived from scroll position, so it is identical in both
    // scroll directions. The stage is pinned while the track scrolls past;
    // progress runs from 0 (start of the first stop) to Stops.length (end of
    // the last stop). The integer part selects the example and the fraction
    // drives the steps inside it.
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
      setProgress(t * Stops.length);
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

  const active = Math.min(Stops.length - 1, Math.floor(progress));
  const scrub = Math.min(1, progress - active);

  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Examples</span>
          <Heading as="h2" className={styles.title}>
            Example outputs
          </Heading>
          <p className={styles.lede}>
            Five examples show the outputs of the program&apos;s tools on real
            and illustrative inputs. Scrolling moves through the examples and
            through the steps of each one; the interactive elements in each
            panel can be selected. Parts that are illustrations are labeled.
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
                          <Evidence active={i === active} scrub={i === active ? scrub : 0} />
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
                      {stop.caveat && <p className={styles.stopCaveat}>{stop.caveat}</p>}
                      <div className={styles.stopEvidenceMobile}>
                        <MobileEvidence Evidence={StopEvidence} />
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
