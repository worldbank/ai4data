import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// The memory a model needs, drawn as squares of 1 GB, against three kinds of computer.
// The capacities are the rounded ranges of the session's figure data.
const ICONS = {
  device: (
    <g>
      <rect x="4" y="8" width="26" height="18" rx="2" />
      <path d="M1 29 H33" />
      <rect x="34" y="6" width="11" height="22" rx="2" />
    </g>
  ),
  server: (
    <g>
      <rect x="8" y="3" width="30" height="8" rx="1.5" />
      <rect x="8" y="14" width="30" height="8" rx="1.5" />
      <rect x="8" y="25" width="30" height="8" rx="1.5" />
      <path d="M12 7 H14 M12 18 H14 M12 29 H14" />
    </g>
  ),
  datacenter: (
    <g>
      <rect x="3" y="10" width="40" height="23" rx="1.5" />
      <path d="M3 10 L23 2 L43 10" />
      <path d="M10 16 V28 M17 16 V28 M24 16 V28 M31 16 V28 M38 16 V28" />
    </g>
  ),
};

function Squares({capacity, used, cols}) {
  const filled = Math.min(used, capacity);
  return (
    <div className={styles.squares} style={{gridTemplateColumns: `repeat(${cols}, 1fr)`}} aria-hidden="true">
      {Array.from({length: capacity}, (_, i) => {
        const part = Math.max(0, Math.min(1, filled - i));
        return (
          <span key={i} className={styles.square}>
            {part > 0 && <span className={styles.squareFill} style={{width: `${part * 100}%`}} />}
          </span>
        );
      })}
    </div>
  );
}

export default function WhereItRuns({data}) {
  const [current, setCurrent] = useState(0);
  const model = data.machines.models[current];

  return (
    <figure className={styles.figure}>
      <p className={styles.title}>Where a model can run</p>
      <p className={styles.caption}>
        A model has to fit in the memory of the computer that runs it. Each square is 1 GB of memory. Choose a model
        to fill the three kinds of computer.
      </p>
      <div className={styles.segmented} role="group" aria-label="Model">
        {data.machines.models.map((m, i) => (
          <button key={m.name} type="button" aria-pressed={i === current} onClick={() => setCurrent(i)}>
            <span className={clsx(styles.dot, styles[m.kind])} /> {m.name}
          </button>
        ))}
      </div>
      <p className={styles.modelLine} aria-live="polite">
        <strong>{model.name}</strong>: {model.params}, {model.memory}. {model.role}
      </p>
      <div className={styles.machines}>
        {data.machines.computers.map((c) => {
          const fits = c.gb === null || model.gb <= c.gb;
          return (
            <div key={c.key} className={clsx(styles.machine, fits && styles.machineFits)} style={{'--fill': `var(--${model.kind})`}}>
              <svg className={styles.machineIcon} viewBox="0 0 46 36" aria-hidden="true">{ICONS[c.key]}</svg>
              <p className={styles.machineName}>{c.label}</p>
              <p className={styles.machineSub}>{c.memory}</p>
              {c.key === 'datacenter' ? (
                <div className={styles.bigMemory}>Hundreds of GB or more</div>
              ) : (
                <Squares capacity={c.gb} used={model.gb} cols={c.key === 'device' ? 10 : 20} />
              )}
              <p className={styles.verdict}>{fits ? '✓ The model fits' : '✗ Too large for this computer'}</p>
              <dl>
                <dt>Internet</dt>
                <dd>{c.internet}</dd>
                <dt>The text</dt>
                <dd>{c.text}</dd>
              </dl>
            </div>
          );
        })}
      </div>
      <p className={styles.source}>
        Memory ranges rounded for typical computers, following {data.source} Hosted chat assistants do not publish
        the size of their models.
      </p>
    </figure>
  );
}
