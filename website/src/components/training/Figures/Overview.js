import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// The whole method in three steps, each linked to its section of the page. A switch
// replaces the description of each step with a comparison from office work.
const VIEWS = [
  ['method', 'The method'],
  ['office', 'An office comparison'],
];

const STEPS = [
  {
    term: 'Synthetic data',
    title: 'A large model writes practice examples',
    method: 'A large AI model writes thousands of example sentences and lists the topics of each one.',
    office: 'An experienced officer writes practice cases, each with the correct answer.',
    href: '#step-1-synthetic-data',
    kind: 'large',
    icon: (
      <g>
        <rect x="6" y="14" width="44" height="44" rx="8" className={styles.iconLarge} />
        <path d="M18 30 H38 M18 38 H38 M18 46 H30" className={styles.iconStrokeLarge} />
        <path d="M56 36 H66" className={styles.iconArrow} markerEnd="url(#ov-head)" />
        <rect x="72" y="10" width="42" height="52" rx="4" className={styles.iconPaper} />
        <path d="M79 20 H107 M79 26 H101" className={styles.iconLine} />
        <rect x="79" y="31" width="18" height="6" rx="3" className={styles.iconTag} />
        <path d="M79 44 H107 M79 50 H97" className={styles.iconLine} />
        <rect x="79" y="54" width="14" height="5" rx="2.5" className={styles.iconTag} />
      </g>
    ),
  },
  {
    term: 'Distillation',
    title: 'A small model learns from the examples',
    method: 'A small model is trained on the examples until it gives the same topics to the same sentences.',
    office: 'A new staff member studies the practice cases and learns the answers.',
    href: '#step-2-the-small-model-learns',
    kind: 'small',
    icon: (
      <g>
        <rect x="6" y="10" width="42" height="52" rx="4" className={styles.iconPaper} />
        <path d="M13 20 H41 M13 26 H35" className={styles.iconLine} />
        <rect x="13" y="31" width="18" height="6" rx="3" className={styles.iconTag} />
        <path d="M13 44 H41 M13 50 H31" className={styles.iconLine} />
        <rect x="13" y="54" width="14" height="5" rx="2.5" className={styles.iconTag} />
        <path d="M54 36 H68" className={styles.iconArrow} markerEnd="url(#ov-head)" />
        <rect x="76" y="22" width="28" height="28" rx="6" className={styles.iconSmall} />
        <circle cx="86" cy="33" r="2.5" className={styles.iconDotSmall} />
        <circle cx="95" cy="40" r="2.5" className={styles.iconDotSmall} />
        <path d="M86 33 L95 40" className={styles.iconStrokeSmall} />
      </g>
    ),
  },
  {
    term: 'Small AI',
    title: 'The small model works alone',
    method:
      'The small model tags the reports on the organization\'s own computer, without the large model and without an internet connection.',
    office: 'The new staff member handles the routine files alone, and the files stay in the office.',
    href: '#step-3-the-small-model-at-work',
    kind: 'small',
    icon: (
      <g>
        <rect x="20" y="12" width="80" height="44" rx="4" className={styles.iconPaper} />
        <path d="M10 62 H110" className={styles.iconLine} strokeWidth={4} />
        <rect x="30" y="22" width="22" height="22" rx="5" className={styles.iconSmall} />
        <circle cx="38" cy="31" r="2" className={styles.iconDotSmall} />
        <circle cx="45" cy="36" r="2" className={styles.iconDotSmall} />
        <path d="M38 31 L45 36" className={styles.iconStrokeSmall} />
        <path d="M60 26 H90 M60 32 H84" className={styles.iconLine} />
        <rect x="60" y="38" width="18" height="6" rx="3" className={styles.iconTag} />
      </g>
    ),
  },
];

export default function Overview() {
  const [view, setView] = useState('method');
  return (
    <figure className={styles.figure}>
      <p className={styles.title}>The method in three steps</p>
      <p className={styles.caption}>
        Each step links to its section below. The switch replaces the description of each step with a comparison
        from office work.
      </p>
      <div className={styles.segmented} role="group" aria-label="Description">
        {VIEWS.map(([key, label]) => (
          <button key={key} type="button" aria-pressed={view === key} onClick={() => setView(key)}>{label}</button>
        ))}
      </div>
      <svg width="0" height="0" style={{position: 'absolute'}} aria-hidden="true">
        <defs>
          <marker id="ov-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--faint)" />
          </marker>
        </defs>
      </svg>
      <ol className={styles.steps}>
        {STEPS.map((s, i) => (
          <li key={s.term} className={clsx(styles.step, styles[`step_${s.kind}`])}>
            <svg className={styles.stepIcon} viewBox="0 0 120 72" aria-hidden="true">{s.icon}</svg>
            <span className={styles.stepNumber}>Step {i + 1} · {s.term}</span>
            <p className={styles.stepTitle}>{s.title}</p>
            <p className={styles.stepText} aria-live="polite">{s[view]}</p>
            <a className={styles.stepLink} href={s.href}>Read step {i + 1}</a>
          </li>
        ))}
      </ol>
      <p className={styles.source}>
        The large model is needed in steps 1 and 2 only. In step 3 the reports do not leave the organization.
      </p>
    </figure>
  );
}
