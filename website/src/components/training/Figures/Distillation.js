import React, {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Distillation in two phases (the large model writes and labels training sentences; the
// small model then labels documents alone), followed by six report abstracts with the
// reference topics, the small model's scores, and the answers of three large models.
const PHASES = [
  ['train', '1. Training: the teacher prepares examples'],
  ['use', '2. Use: the student works alone'],
];

// Two layouts of the same diagram: a row for wide columns and a column for phones.
const LAYOUTS = {
  wide: {
    view: '0 0 680 214',
    nodes: {
      large: [0, 30, 140, 60],
      data: [180, 30, 140, 60],
      small: [360, 30, 140, 60],
      out: [540, 30, 140, 60],
      docs: [360, 150, 140, 46],
    },
    edges: {
      train: [['M140,60 L180,60', 'large'], ['M320,60 L360,60', 'large']],
      use: [['M430,150 L430,90', 'small'], ['M500,60 L540,60', 'small']],
    },
  },
  narrow: {
    view: '0 0 340 380',
    nodes: {
      large: [0, 0, 200, 60],
      data: [0, 100, 200, 60],
      small: [0, 200, 200, 60],
      out: [0, 300, 200, 60],
      docs: [236, 207, 104, 46],
    },
    edges: {
      train: [['M100,60 L100,100', 'large'], ['M100,160 L100,200', 'large']],
      use: [['M236,230 L200,230', 'small'], ['M100,260 L100,300', 'small']],
    },
  },
};
const LABELS = {
  large: ['Teacher', 'large model', 'large'],
  data: ['Practice sentences', 'with their topics', ''],
  small: ['Student', 'small model', 'small'],
  out: ['Topics', 'for each document', ''],
  docs: ['New documents', '', ''],
};
const ACTIVE = {train: ['large', 'data', 'small'], use: ['small', 'docs', 'out']};

function useNarrow() {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const q = window.matchMedia('(max-width: 560px)');
    setNarrow(q.matches);
    const on = () => setNarrow(q.matches);
    q.addEventListener('change', on);
    return () => q.removeEventListener('change', on);
  }, []);
  return narrow;
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(true);
  useEffect(() => {
    const q = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(q.matches);
    const on = () => setReduced(q.matches);
    q.addEventListener('change', on);
    return () => q.removeEventListener('change', on);
  }, []);
  return reduced;
}

function Flow({phase}) {
  const reduced = useReducedMotion();
  const layout = LAYOUTS[useNarrow() ? 'narrow' : 'wide'];
  const all = [...layout.edges.train, ...layout.edges.use];
  return (
    <svg className={clsx(styles.svg, styles.flow)} viewBox={layout.view} role="img"
      style={{maxWidth: layout === LAYOUTS.narrow ? 360 : undefined, margin: '0 auto'}}
      aria-label={phase === 'train'
        ? 'Training: the teacher, a large model, writes practice sentences with their topics, and the student, a small model, learns from them.'
        : 'Use: new documents go to the student, which assigns their topics without the teacher.'}>
      <defs>
        <marker id="flow-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--faint)" />
        </marker>
      </defs>
      {all.map(([d]) => (
        <path key={d} d={d} stroke="var(--faint)" strokeWidth={2} fill="none" markerEnd="url(#flow-head)"
          opacity={layout.edges[phase].some(([e]) => e === d) ? 1 : 0.3} />
      ))}
      {Object.entries(layout.nodes).map(([key, [x, y, w, h]]) => {
        const [label, sub, kind] = LABELS[key];
        const on = ACTIVE[phase].includes(key);
        return (
          <g key={key} className={clsx(!on && styles.flowDim)} style={{transition: 'opacity 0.3s'}}>
            <rect className={styles.flowBox} x={x + 1} y={y + 1} width={w - 2} height={h - 2} rx={10}
              style={kind ? {stroke: `var(--${kind})`, strokeWidth: 2.5} : undefined} />
            <text x={x + w / 2} y={y + (sub ? h / 2 - 2 : h / 2 + 5)} textAnchor="middle" fontWeight={700}>{label}</text>
            {sub && (
              <text x={x + w / 2} y={y + h / 2 + 16} textAnchor="middle" className={styles.tick}>{sub}</text>
            )}
          </g>
        );
      })}
      {!reduced &&
        layout.edges[phase].map(([d, kind], i) =>
          [0, 1].map((k) => (
            <circle key={`${phase}-${d}-${k}`} r={5} fill={`var(--${kind})`}>
              <animateMotion dur="1.6s" begin={`${k * 0.8}s`} repeatCount="indefinite" path={d} />
            </circle>
          )),
        )}
    </svg>
  );
}

const shortTitle = (t) => {
  const head = t.split(' : ')[0].split(': ')[0];
  return head.length > 46 ? `${head.slice(0, 44)}…` : head;
};

function LabelChips({labels, reference, measured}) {
  return (
    <ul className={styles.chips}>
      {labels.length === 0 && <li className={clsx(styles.chip, styles.chipOff)}>no topic</li>}
      {labels.map((l) => {
        const hit = reference.includes(l);
        const scored = measured.includes(l);
        return (
          <li key={l} className={clsx(styles.chip, hit && styles.chipMatch, !scored && styles.chipOff)}
            title={hit ? 'In the reference' : scored ? 'Not in the reference' : 'Not scored: no reference abstract has this topic'}>
            {hit ? '✓ ' : ''}{l}
          </li>
        );
      })}
    </ul>
  );
}

export default function Distillation({data}) {
  const [phase, setPhase] = useState('train');
  const [doc, setDoc] = useState(0);
  const [threshold, setThreshold] = useState(data.threshold);
  const [open, setOpen] = useState(false);
  const d = data.documents[doc];
  const max = Math.max(d.student[0].score, threshold) * 1.05;
  const chosen = d.student.slice(0, data.top).filter((s) => s.score >= threshold).map((s) => s.label);

  return (
    <figure className={styles.figure}>
      <p className={styles.title}>A large model teaches a small one</p>
      <p className={styles.caption}>
        The teacher is needed only while the student learns. Switch between the two stages.
      </p>
      <div className={styles.segmented} role="group" aria-label="Phase">
        {PHASES.map(([key, label]) => (
          <button key={key} type="button" aria-pressed={phase === key} onClick={() => setPhase(key)}>{label}</button>
        ))}
      </div>
      <Flow phase={phase} />

      <p className={styles.title} style={{marginTop: 22}}>The student at work on six report abstracts</p>
      <p className={styles.caption}>
        Select an abstract. The student scores every topic from 0 to 1 and assigns the topics above the dashed
        line. Move the slider to raise or lower the line. ✓ marks a topic recorded for the report in the World Bank
        documents database.
      </p>
      <div className={styles.docTabs} role="group" aria-label="Abstracts">
        {data.documents.map((x, i) => (
          <button key={x.id} type="button" aria-pressed={i === doc}
            onClick={() => { setDoc(i); setOpen(false); }}>
            {shortTitle(x.title)}
          </button>
        ))}
      </div>
      <p style={{marginBottom: 4}}>
        <a href={d.url}><strong>{d.title}</strong></a> ({d.year})
      </p>
      <div className={clsx(styles.abstract, open && styles.abstractOpen)}>{d.text}</div>
      <button type="button" className={styles.linkButton} onClick={() => setOpen(!open)}>
        {open ? 'Show less' : 'Show the whole abstract'}
      </button>

      <div className={styles.twoCol} style={{marginTop: 14}}>
        <div>
          <p><strong>Student</strong>: the eight highest scores</p>
          <div className={styles.bars}>
            {d.student.map((s) => {
              const pick = chosen.includes(s.label);
              return (
                <React.Fragment key={s.label}>
                  <span style={{fontWeight: pick ? 700 : 400}}>
                    {d.reference.includes(s.label) ? '✓ ' : ''}{s.label}
                  </span>
                  <span className={styles.barTrack} title={`${s.label}: ${s.score}`}>
                    <span className={styles.barFill}
                      style={{width: `${(s.score / max) * 100}%`, background: pick ? 'var(--small)' : 'var(--line)'}} />
                    <span style={{position: 'absolute', top: -3, bottom: -3, left: `${(threshold / max) * 100}%`,
                      borderLeft: '2px dashed var(--ink)'}} />
                  </span>
                  <span className={styles.barValue}>{s.score.toFixed(3)}</span>
                </React.Fragment>
              );
            })}
          </div>
          <label className={styles.slider}>
            Threshold {threshold.toFixed(2)}
            <input type="range" min={0.01} max={0.5} step={0.01} value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))} />
          </label>
          <p className={styles.source} style={{marginTop: 0}}>
            The student assigns at most {data.top} topics. The starting line of {data.threshold} was chosen on a
            separate set of abstracts.
          </p>
        </div>
        <div>
          <p><strong>Answers</strong></p>
          <div className={styles.answers}>
            <span>Recorded topics</span>
            <LabelChips labels={d.reference} reference={d.reference} measured={data.measured} />
            <span><span className={clsx(styles.dot, styles.small)} /> Student</span>
            <LabelChips labels={chosen} reference={d.reference} measured={data.measured} />
            {Object.entries(d.teachers).map(([name, labels]) => (
              <React.Fragment key={name}>
                <span><span className={clsx(styles.dot, styles.large)} /> {name}</span>
                <LabelChips labels={labels} reference={d.reference} measured={data.measured} />
              </React.Fragment>
            ))}
          </div>
          <p className={styles.source}>
            Below the student, three large models asked directly for the topics, with the list of topics in the
            prompt. Grey topics do not appear among the topics recorded for the 300 test abstracts, so they are
            not scored.
          </p>
        </div>
      </div>
    </figure>
  );
}
