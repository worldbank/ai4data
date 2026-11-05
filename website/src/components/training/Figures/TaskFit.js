import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Six tasks from development work. The reader picks the kind of model that suits each
// one, and the figure then shows the suggested answer and the reason for it.
const KINDS = [
  ['small', 'Small model'],
  ['large', 'Large model'],
];

export default function TaskFit({data}) {
  const [picks, setPicks] = useState({});
  const [revealAll, setRevealAll] = useState(false);
  const label = (kind) => KINDS.find(([k]) => k === kind)[1].toLowerCase();

  return (
    <figure className={styles.figure}>
      <p className={styles.title}>Which kind of model suits the task?</p>
      <p className={styles.caption}>
        Choose a small or a large model for each task to show the suggested answer and the reason for it.
      </p>
      <ul className={styles.tasks}>
        {data.tasks.map((t, i) => {
          const pick = picks[i];
          const shown = revealAll || pick;
          return (
            <li key={t.task} className={styles.task}>
              <p className={styles.taskText}>{t.task}</p>
              <div className={styles.segmented} role="group" aria-label={`Model for: ${t.task}`}>
                {KINDS.map(([kind, name]) => (
                  <button key={kind} type="button" aria-pressed={pick === kind}
                    onClick={() => setPicks({...picks, [i]: kind})}>
                    <span className={clsx(styles.dot, styles[kind])} /> {name}
                  </button>
                ))}
              </div>
              {shown && (
                <p className={styles.taskWhy} aria-live="polite">
                  <strong>
                    {pick ? (pick === t.fit ? '✓ ' : '✗ ') : ''}Suggested: {label(t.fit)}.
                  </strong>{' '}
                  {t.why}
                </p>
              )}
            </li>
          );
        })}
      </ul>
      <button type="button" className={styles.linkButton} onClick={() => setRevealAll(!revealAll)}>
        {revealAll ? 'Hide the answers' : 'Show all the answers'}
      </button>
    </figure>
  );
}
