import {useEffect, useState} from 'react';
import clsx from 'clsx';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

// Page 12 of a World Bank Policy Research Working Paper from the
// ai4data/data-snapshot dataset (MIT). The page image, the bounding box and
// the snapshot crop come from that dataset. The structured output is an
// illustration: its values are the data labels printed on the chart.
// Policy Research Working Paper 4771 (World Bank). The figure is on page 12 of
// the paper, which is page 15 of the PDF.
const PAPER_URL =
  'https://documents.worldbank.org/curated/en/937071468314370786/pdf/The-typology-of-partial-credit-guarantee-funds-around-the-world.pdf#page=15';
const BBOX = [0.266, 0.525, 0.734, 0.781]; // normalized x1, y1, x2, y2

const Stages = [
  {
    name: 'Detect',
    role: 'Layout model',
    note: 'A layout detection model finds figures and tables on each page and returns a class and a bounding box.',
  },
  {
    name: 'Crop',
    role: 'Snapshot',
    note: 'The region inside the bounding box is saved as an image. This image is the data snapshot.',
  },
  {
    name: 'Extract',
    role: 'Structured data',
    note: 'The snapshot is converted to structured data: chart type, categories, series, and values.',
  },
];

const CATEGORIES = ['MUTUAL', 'PUBLIC'];
const SERIES = [
  {name: 'HIGH', values: [15, 7]},
  {name: 'LOW_MIDDLE', values: [7, 17]},
];

const JSON_OUT = `{
  "type": "bar_chart",
  "title": "Type of guarantee systems",
  "categories": ["MUTUAL", "PUBLIC"],
  "series": [
    {"name": "HIGH", "values": [15, 7]},
    {"name": "LOW_MIDDLE", "values": [7, 17]}
  ],
  "value_labels_printed": true,
  "source": {
    "page": 12,
    "bbox": [0.266, 0.525, 0.734, 0.781]
  }
}`;

const STEP_MS = 2400;

export default function SnapshotEvidence({active = true}) {
  const [stage, setStage] = useState(0); // 0 = not started, 1..3
  const [playing, setPlaying] = useState(false);
  const [run, setRun] = useState(0);
  const [view, setView] = useState('table');
  const pageSrc = useBaseUrl('/img/snapshot/page.webp');
  const snapSrc = useBaseUrl('/img/snapshot/snapshot.webp');

  useEffect(() => {
    if (!active) {
      setStage(0);
      setPlaying(false);
      return undefined;
    }
    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      setStage(Stages.length);
      setPlaying(false);
      return undefined;
    }
    setStage(0);
    setPlaying(true);
    return undefined;
  }, [active, run]);

  useEffect(() => {
    if (!playing) {
      return undefined;
    }
    const t = setTimeout(
      () => {
        setStage((s) => {
          if (s + 1 >= Stages.length) {
            setPlaying(false);
          }
          return Math.min(s + 1, Stages.length);
        });
      },
      stage === 0 ? 700 : STEP_MS,
    );
    return () => clearTimeout(t);
  }, [playing, stage]);

  const goTo = (s) => {
    setPlaying(false);
    setStage(s);
  };

  const [x1, y1, x2, y2] = BBOX;

  return (
    <div className={clsx(styles.evidenceBox, styles.dsRoot)}>
      <div className={styles.mrSteps3} role="group" aria-label="Processing step">
        {Stages.map((s, i) => (
          <button
            type="button"
            key={s.name}
            aria-pressed={stage === i + 1}
            className={clsx(
              styles.mrStep,
              stage >= i + 1 && styles.mrStepDone,
              stage === i + 1 && styles.mrStepActive,
            )}
            onClick={() => goTo(i + 1)}>
            <span className={styles.mrStepIndex}>{i + 1}</span>
            <span className={styles.mrStepText}>
              <span className={styles.mrStepName}>{s.name}</span>
              <span className={styles.mrStepRole}>{s.role}</span>
            </span>
          </button>
        ))}
      </div>

      <div className={styles.mrNote} aria-live="polite">
        {stage === 0
          ? 'A PDF page enters the pipeline. Select a step or press Run.'
          : Stages[stage - 1].note}
      </div>

      <div className={styles.dsColumns}>
        <div>
          <span className={styles.mrLabel}>PDF page</span>
          <div className={styles.dsPage}>
            <img src={pageSrc} alt="Page 12 of a World Bank working paper with a bar chart" />
            {stage >= 1 && (
              <span
                className={styles.dsBox}
                style={{
                  left: `${x1 * 100}%`,
                  top: `${y1 * 100}%`,
                  width: `${(x2 - x1) * 100}%`,
                  height: `${(y2 - y1) * 100}%`,
                }}>
                <span className={styles.dsBoxTag}>Figure</span>
              </span>
            )}
          </div>
        </div>

        <div className={styles.dsRight}>
          <div>
            <span className={styles.mrLabel}>Snapshot</span>
            {stage >= 2 ? (
              <img
                className={styles.dsSnap}
                src={snapSrc}
                alt="Cropped bar chart: Type of guarantee systems"
              />
            ) : (
              <div className={styles.dsPlaceholder}>
                {stage === 1 ? 'Region detected' : 'No region yet'}
              </div>
            )}
          </div>

          <div>
            <span className={styles.mrLabel}>
              Structured data{' '}
              {stage >= 3 && (
                <span className={styles.dsViews}>
                  <button
                    type="button"
                    aria-pressed={view === 'table'}
                    className={clsx(
                      styles.dsView,
                      view === 'table' && styles.dsViewOn,
                    )}
                    onClick={() => setView('table')}>
                    Table
                  </button>
                  <button
                    type="button"
                    aria-pressed={view === 'json'}
                    className={clsx(
                      styles.dsView,
                      view === 'json' && styles.dsViewOn,
                    )}
                    onClick={() => setView('json')}>
                    JSON
                  </button>
                </span>
              )}
            </span>
            {stage >= 3 ? (
              view === 'table' ? (
                <table className={styles.dsTable}>
                  <thead>
                    <tr>
                      <th>Type of guarantee systems</th>
                      {CATEGORIES.map((c) => (
                        <th key={c}>{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {SERIES.map((s) => (
                      <tr key={s.name}>
                        <th>{s.name}</th>
                        {s.values.map((v, i) => (
                          <td key={i}>{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <pre className={styles.dsJson}>{JSON_OUT}</pre>
              )
            ) : (
              <div className={styles.dsPlaceholder}>
                {stage === 2 ? 'Reading the chart' : 'Not extracted yet'}
              </div>
            )}
            {stage >= 3 && (
              <p className={styles.dsCheck}>
                4 of 4 values match the labels printed on the chart.
              </p>
            )}
          </div>
        </div>
      </div>

      <div className={styles.dsFootRow}>
        <button
          type="button"
          className={styles.pcnReplay}
          onClick={() => setRun((r) => r + 1)}>
          {stage === 0 && !playing ? '▶ Run' : '↻ Replay'}
        </button>
        <span className={styles.pcnFine}>
          Page, bounding box, and snapshot are from the ai4data/data-snapshot
          dataset. The page is from{' '}
          <a href={PAPER_URL} target="_blank" rel="noopener noreferrer">
            a working paper by Beck, Klapper, and Mendoza
          </a>
          . The structured output is an illustration.
        </span>
      </div>
    </div>
  );
}
