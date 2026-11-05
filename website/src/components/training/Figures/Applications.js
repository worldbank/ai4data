import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// The applications of WDR 2026 table S1.2 and the session's classifier, placed by where
// the model runs and by the kind of task.
const COLUMNS = [
  ['device', 'Model on the device, offline'],
  ['server', 'Model on its own server'],
  ['cloud', 'Large model in the cloud'],
];
const ROWS = [
  ['predictive', 'Predictive: picks a label'],
  ['generative', 'Generative: writes an answer'],
];

export default function Applications({data}) {
  const apps = data.applications;
  const [current, setCurrent] = useState(apps.findIndex((a) => a.session));
  const app = apps[current];

  return (
    <figure className={styles.figure}>
      <p className={styles.title}>Small AI applications by where the model runs</p>
      <p className={styles.caption}>
        Five applications from WDR 2026, table S1.2, and the classifier built in this session (blue outline).
        Select an application to show its task and how it runs.
      </p>
      <div className={styles.grid} role="table" aria-label="Applications by where the model runs and kind of task">
        <div className={styles.gridHead} role="columnheader" />
        {COLUMNS.map(([key, label]) => (
          <div key={key} className={styles.gridHead} role="columnheader">{label}</div>
        ))}
        {ROWS.map(([row, rowLabel]) => (
          <React.Fragment key={row}>
            <div className={styles.gridHead} role="rowheader">{rowLabel}</div>
            {COLUMNS.map(([col, colLabel]) => (
              <div key={col} className={styles.cell} role="cell" aria-label={`${rowLabel}, ${colLabel}`}>
                {apps.map((a, i) =>
                  a.where === col && a.kind === row ? (
                    <button key={a.name} type="button" aria-pressed={i === current}
                      className={clsx(styles.app, a.session && styles.appSession)} onClick={() => setCurrent(i)}>
                      {a.name}
                      <span className={styles.appSector}>{a.sector}</span>
                    </button>
                  ) : null,
                )}
              </div>
            ))}
          </React.Fragment>
        ))}
      </div>
      <div className={styles.detail} aria-live="polite">
        <strong>{app.name}</strong> · {app.sector}
        <dl>
          <dt>What it does</dt>
          <dd>{app.task}</dd>
          <dt>How it runs</dt>
          <dd>{app.runs}</dd>
        </dl>
      </div>
      <p className={styles.source}>Source of the five applications: {data.source}</p>
    </figure>
  );
}
