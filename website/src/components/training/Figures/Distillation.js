import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// The student at work on one report abstract: its scores, the assigned topics, and the
// topics recorded for the report. Under "go deeper", five more abstracts, the score
// threshold, and the answers of three large models.
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

function Explorer({data}) {
  const [doc, setDoc] = useState(1);
  const [threshold, setThreshold] = useState(data.threshold);
  const [open, setOpen] = useState(false);
  const d = data.documents[doc];
  const max = Math.max(d.student[0].score, threshold) * 1.05;
  const chosen = d.student.slice(0, data.top).filter((s) => s.score >= threshold).map((s) => s.label);

  return (
    <details className={styles.deeper}>
      <summary>Go deeper: five more abstracts, the score threshold, and the answers of three large models</summary>
      <p className={styles.caption} style={{marginTop: 10}}>
        Select an abstract. The student scores every topic from 0 to 1 and assigns the topics above the dashed
        line. Move the slider to raise or lower the line. ✓ marks a topic recorded for the report in the World Bank
        documents database.
      </p>
      <div className={styles.docTabs} role="group" aria-label="Abstracts">
        {data.documents.map((x, i) => i > 0 && (
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
    </details>
  );
}

function Worked({data}) {
  const d = data.documents[0];
  const max = d.student[0].score;
  const chosen = d.student.slice(0, data.top).filter((x) => x.score >= data.threshold);
  const right = chosen.filter((x) => d.reference.includes(x.label)).length;
  return (
    <div>
      <div className={styles.detail}>
        <a href={d.url}><strong>{d.title}</strong></a> ({d.year})
        <p className={styles.abstract} style={{marginTop: 6, maxHeight: '4.8em'}}>{d.text}</p>
      </div>
      <div className={styles.twoCol} style={{marginTop: 14}}>
        <div>
          <p><strong>Student's highest scores</strong></p>
          <div className={styles.bars} style={{gridTemplateColumns: 'minmax(120px, max-content) 1fr max-content'}}>
            {d.student.slice(0, 6).map((x) => {
              const pick = chosen.some((c) => c.label === x.label);
              return (
                <React.Fragment key={x.label}>
                  <span style={{fontWeight: pick ? 700 : 400, color: pick ? undefined : 'var(--muted)'}}>{x.label}</span>
                  <span className={styles.barTrack}>
                    <span className={styles.barFill}
                      style={{width: `${(x.score / max) * 100}%`, minWidth: 3, background: pick ? 'var(--small)' : 'var(--line)'}} />
                  </span>
                  <span className={styles.barValue}>{pick ? 'assigned' : ''}</span>
                </React.Fragment>
              );
            })}
          </div>
        </div>
        <div>
          <p><strong>Result</strong></p>
          <div className={styles.answers}>
            <span>Recorded topics</span>
            <LabelChips labels={d.reference} reference={d.reference} measured={data.measured} />
            <span><span className={clsx(styles.dot, styles.small)} /> Student</span>
            <LabelChips labels={chosen.map((x) => x.label)} reference={d.reference} measured={data.measured} />
          </div>
          <p style={{marginTop: 10, fontSize: '0.9rem'}}>
            The student assigned {chosen.length} topics, and {right} of them are recorded for the report.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function Distillation({data}) {
  return (
    <figure className={styles.figure}>
      <p className={styles.title}>The student at work on one real report</p>
      <p className={styles.caption}>
        The student reads the abstract of a report and gives every topic a score. It assigns the topics with a
        clear score. A check mark shows a topic recorded for this report in the World Bank documents database.
      </p>
      <Worked data={data} />
      <Explorer data={data} />
    </figure>
  );
}
