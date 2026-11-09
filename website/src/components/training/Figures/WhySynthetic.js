import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Four reasons why a team has no labelled examples for its task, each with a small picture.
// In "Rare", a button adds written emergency cases to 400 messages with 4 real emergencies,
// so the reader sees the share of the cases that the model learns from change.
const TOTAL = 400;
const REAL = 4;
const WRITTEN = 120;
const REAL_AT = new Set([37, 141, 236, 318]);

function Rare() {
  const [added, setAdded] = useState(false);
  const dots = Array.from({length: TOTAL}, (_, i) => (REAL_AT.has(i) ? 'real' : 'routine'));
  const extra = added ? Array(WRITTEN).fill('written') : [];
  return (
    <div>
      <div className={styles.rareGrid} role="img"
        aria-label={`${TOTAL} messages, of which ${REAL} are emergencies${added ? `, and ${WRITTEN} written emergency cases` : ''}.`}>
        {dots.concat(extra).map((d, i) => <span key={i} className={clsx(styles.rareDot, styles[`rare_${d}`])} />)}
      </div>
      <div className={styles.legend} style={{marginTop: 8}}>
        <span><span className={clsx(styles.rareDot, styles.rare_routine)} /> routine message</span>
        <span><span className={clsx(styles.rareDot, styles.rare_real)} /> real emergency</span>
        {added && <span><span className={clsx(styles.rareDot, styles.rare_written)} /> written emergency case</span>}
      </div>
      <button type="button" className={styles.primaryButton} style={{marginTop: 10}} onClick={() => setAdded(!added)}>
        {added ? 'Remove the written cases' : `Add ${WRITTEN} written emergency cases`}
      </button>
      <p className={styles.stepText} style={{marginTop: 8}} aria-live="polite">
        {added
          ? `The model now learns from ${WRITTEN} written emergencies of many kinds. The ${REAL} real ones are kept for testing.`
          : `Among ${TOTAL} real messages, ${REAL} are emergencies. A model trained on them sees too few to learn what an emergency looks like.`}
      </p>
    </div>
  );
}

const REASONS = [
  {
    key: 'missing',
    label: 'Never collected',
    text: 'Messages sent after a flood do not exist before the flood, and the model has to work on the first day. A new benefit, a new form, or a new list of codes poses the same problem.',
    picture: (
      <svg viewBox="0 0 240 110" className={styles.svg} aria-hidden="true">
        <rect x="20" y="30" width="80" height="56" rx="6" className={styles.iconPaper} />
        <path d="M20 58 H44 L52 70 H68 L76 58 H100" className={styles.iconLine} fill="none" />
        <text x="60" y="104" textAnchor="middle" className={styles.tick}>0 messages so far</text>
        <path d="M112 58 H138" className={styles.iconArrow} markerEnd="url(#ws-head)" />
        <rect x="150" y="22" width="70" height="72" rx="6" className={styles.iconSmall} />
        <text x="185" y="54" textAnchor="middle" className={styles.tick}>model</text>
        <text x="185" y="70" textAnchor="middle" className={styles.tick}>ready on day 1</text>
      </svg>
    ),
  },
  {
    key: 'private',
    label: 'Private',
    text: 'Applications, complaints, and job seekers\' profiles hold personal data. Sending them to an outside service for labelling or training is often not allowed. Written practice cases contain no real person.',
    picture: (
      <svg viewBox="0 0 240 110" className={styles.svg} aria-hidden="true">
        <rect x="16" y="20" width="100" height="74" rx="6" className={styles.iconPaper} />
        <text x="66" y="38" textAnchor="middle" className={styles.tick}>organization</text>
        <rect x="46" y="56" width="40" height="28" rx="4" className={styles.iconTag} />
        <path d="M54 56 V48 A12 12 0 0 1 78 48 V56" className={styles.iconLine} fill="none" />
        <path d="M124 57 H170" className={styles.iconArrow} markerEnd="url(#ws-head)" />
        <path d="M138 47 L156 67 M156 47 L138 67" className={styles.iconStrokeLarge} />
        <ellipse cx="200" cy="57" rx="30" ry="20" className={styles.iconLarge} />
        <text x="200" y="61" textAnchor="middle" className={styles.tick}>outside</text>
      </svg>
    ),
  },
  {
    key: 'rare',
    label: 'Rare',
    text: 'The cases that matter most are often the rarest: an emergency among routine messages, a wrong entry among correct forms. A large model can write many more of them.',
    picture: <Rare />,
  },
  {
    key: 'language',
    label: 'Local language',
    text: 'Real messages mix languages, abbreviations, and spelling errors. Few labelled examples exist in most local languages, and general models have seen less text in them.',
    picture: (
      <div className={styles.practice}>
        <span className={styles.practiceKind}>Text message, English and Swahili</span>
        <p className={styles.practiceText}>pls help, maji hayajatoka tangu jumatatu. contractor broke the pipe near the school</p>
        <span className={styles.practiceKind}>In English</span>
        <p className={styles.stepText} style={{margin: 0}}>
          Please help, the water has not come since Monday. The contractor broke the pipe near the school.
        </p>
      </div>
    ),
  },
];

export default function WhySynthetic() {
  const [key, setKey] = useState('rare');
  const r = REASONS.find((x) => x.key === key);
  return (
    <figure className={styles.figure}>
      <p className={styles.title}>Why there are no training examples</p>
      <p className={styles.caption}>
        The figure gives four common reasons why a team has no labelled examples for its task. Select a reason.
      </p>
      <svg width="0" height="0" style={{position: 'absolute'}} aria-hidden="true">
        <defs>
          <marker id="ws-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--faint)" />
          </marker>
        </defs>
      </svg>
      <div className={styles.segmented} role="group" aria-label="Reason">
        {REASONS.map((x) => (
          <button key={x.key} type="button" aria-pressed={key === x.key} onClick={() => setKey(x.key)}>{x.label}</button>
        ))}
      </div>
      <div className={styles.twoCol}>
        <div>
          <p className={styles.caseTask}>{r.text}</p>
        </div>
        <div>{r.picture}</div>
      </div>
      <div className={styles.analogy}>
        <strong>An office comparison.</strong> A new staff member usually learns from past files with the right
        decision recorded. When those files do not exist, an experienced colleague writes practice cases: realistic,
        varied, each with its answer, and with extra cases of the rare kind. Synthetic data apply the same idea. A large
        model writes the practice cases, and a small model learns from them. The final test always uses real cases.
      </div>
    </figure>
  );
}
