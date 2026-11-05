import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// One request of the worked example written twice by the same large model: once with the
// topic names allowed in the sentences and once with them avoided. Four steps: the plan,
// the prompt, the sentences, and what each set gave the small model.
const SETS = [
  ['allow', 'Topic names allowed'],
  ['avoid', 'Topic names avoided'],
];
const STEPS = ['Plan the request', 'Prompt the writer', 'Read the sentences', 'Compare the two sets'];

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function Highlighted({text, terms}) {
  if (!terms.length) return text;
  const parts = text.split(new RegExp(`(${terms.map(escape).join('|')})`, 'gi'));
  return parts.map((part, i) => (i % 2 === 1 ? <mark key={i}>{part}</mark> : part));
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

function Bars({rows, max, format, current}) {
  return (
    <div className={styles.bars}>
      {rows.map((r) => (
        <React.Fragment key={r.key}>
          <span style={{fontWeight: r.key === current ? 700 : 400}}>{r.label}</span>
          <span className={styles.barTrack} title={`${r.label}: ${format(r.value)}`}>
            <span className={styles.barFill}
              style={{width: `${(r.value / max) * 100}%`, background: r.key === current ? 'var(--large)' : 'var(--faint)'}} />
          </span>
          <span className={styles.barValue}>{format(r.value)}</span>
        </React.Fragment>
      ))}
    </div>
  );
}

export default function SyntheticData({data}) {
  const [names, setNames] = useState('allow');
  const [step, setStep] = useState(0);
  const [topic, setTopic] = useState(null);
  const set = data.sets[names];
  const shown = set.sentences.filter((s) => !topic || s.labels.includes(topic));
  const named = set.sentences.filter((s) => s.named).length;
  const rule = data.prompt[names].split('\n').find((line) => line.startsWith('- ') && /topic names/.test(line));

  return (
    <figure className={styles.figure}>
      <p className={styles.title}>From a list of topics to labelled training sentences</p>
      <p className={styles.caption}>
        Request {data.request} of the worked example, written by {data.writer} twice with one rule changed.
      </p>
      <div className={styles.segmented} role="group" aria-label="Writing rule">
        {SETS.map(([key, label]) => (
          <button key={key} type="button" aria-pressed={names === key} onClick={() => setNames(key)}>
            {label}
          </button>
        ))}
      </div>
      <ol className={styles.steps}>
        {STEPS.map((label, i) => (
          <li key={label}>
            <button type="button" aria-current={step === i ? 'step' : undefined} onClick={() => setStep(i)}>
              <span className={styles.stepNumber}>Step {i + 1}</span>
              {label}
            </button>
          </li>
        ))}
      </ol>

      {step === 0 && (
        <div>
          <p>
            The worked example has {data.labels} topics. Each of its {data.requests.toLocaleString('en')} requests
            names eight of them, and the writer is asked for 20 sentences about those eight. This request names:
          </p>
          <ul className={styles.chips}>
            {data.topics.map((t) => (
              <li key={t} className={styles.chip}>{t}</li>
            ))}
          </ul>
        </div>
      )}

      {step === 1 && (
        <div>
          <p>
            The prompt lists the eight topics and the rules. The two sets differ in one rule, highlighted below.
          </p>
          <div className={styles.prompt}>
            <Highlighted text={data.prompt[names]} terms={rule ? [rule] : []} />
          </div>
        </div>
      )}

      {step === 2 && (
        <div>
          <p>
            {named} of the {set.sentences.length} sentences contain the name of one of their own topics (highlighted).
            Select a topic to show only its sentences.
          </p>
          <ul className={styles.chips} style={{marginBottom: 12}}>
            {data.topics.map((t) => (
              <li key={t}>
                <button type="button" className={styles.chip} aria-pressed={topic === t}
                  onClick={() => setTopic(topic === t ? null : t)}>
                  {t}
                </button>
              </li>
            ))}
          </ul>
          <ul className={styles.sentences}>
            {shown.map((s) => (
              <li key={s.text} className={styles.sentence}>
                <Highlighted text={s.text} terms={s.labels} />
                <ul className={styles.chips}>
                  {s.labels.map((l) => (
                    <li key={l} className={clsx(styles.chip, l === topic && styles.chipMatch)}>{l}</li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </div>
      )}

      {step === 3 && (
        <div className={styles.twoCol}>
          <div>
            <p><strong>Sentences that name one of their topics</strong>, all 1,000 requests</p>
            <Bars current={names} max={1} format={(v) => `${Math.round(v * 1000) / 10}%`}
              rows={SETS.map(([key, label]) => ({key, label, value: data.sets[key].named / data.sets[key].total}))} />
          </div>
          <div>
            <p><strong>Small model trained on each set</strong>, micro F1 on 300 report abstracts</p>
            <Bars current={names} max={0.6} format={(v) => v.toFixed(3)}
              rows={SETS.map(([key, label]) => ({key, label, value: mean(data.sets[key].report_f1)}))} />
          </div>
          <p style={{gridColumn: '1 / -1', fontSize: '0.88rem'}}>
            The model trained on the sentences with names allowed scored higher on the real abstracts. The bar
            shows the mean of three training runs for that set and one run for the set with names avoided.
          </p>
        </div>
      )}
    </figure>
  );
}
