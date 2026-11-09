import React, {useEffect, useRef, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// One request of the worked example: a list of topics goes to a large model, which writes
// sentences with their topics. Under "go deeper", the same request written with topic
// names avoided, and what each version gave the small model.
const SETS = [
  ['allow', 'Topic names allowed'],
  ['avoid', 'Topic names avoided'],
];

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

function Sentence({s, highlight}) {
  return (
    <li className={styles.sentence}>
      {highlight ? <Highlighted text={s.text} terms={s.labels} /> : s.text}
      <ul className={styles.chips}>
        {s.labels.map((l) => <li key={l} className={styles.chip}>{l}</li>)}
      </ul>
    </li>
  );
}

function Deeper({data}) {
  const [names, setNames] = useState('allow');
  const [full, setFull] = useState(false);
  const set = data.sets[names];
  const rule = data.prompt[names].split('\n').find((line) => line.startsWith('- ') && /topic names/.test(line));
  return (
    <details className={styles.deeper}>
      <summary>Go deeper: two wordings of the same request</summary>
      <p>
        The same request was sent twice with one rule changed. Switch between the two versions; topic names in the
        sentences are highlighted.
      </p>
      <div className={styles.segmented} role="group" aria-label="Writing rule">
        {SETS.map(([key, label]) => (
          <button key={key} type="button" aria-pressed={names === key} onClick={() => setNames(key)}>{label}</button>
        ))}
      </div>
      <p className={styles.rule}>{rule ? rule.slice(2) : ''}</p>
      <button type="button" className={styles.linkButton} onClick={() => setFull(!full)}>
        {full ? 'Hide the full prompt' : 'Show the full prompt'}
      </button>
      {full && (
        <div className={styles.prompt} style={{marginTop: 8}}>
          <Highlighted text={data.prompt[names]} terms={rule ? [rule] : []} />
        </div>
      )}
      <ul className={styles.sentences} style={{marginTop: 12, maxHeight: 300}}>
        {set.sentences.slice(0, 8).map((s) => <Sentence key={s.text} s={s} highlight />)}
      </ul>
      <div className={styles.twoCol} style={{marginTop: 14}}>
        <div>
          <p><strong>Sentences that name one of their topics</strong>, all 1,000 requests</p>
          <Bars current={names} max={1} format={(v) => `${Math.round(v * 1000) / 10}%`}
            rows={SETS.map(([key, label]) => ({key, label, value: data.sets[key].named / data.sets[key].total}))} />
        </div>
        <div>
          <p><strong>Score of a small model trained on each version</strong>, 300 real abstracts (F1, 0 to 1)</p>
          <Bars current={names} max={0.6} format={(v) => v.toFixed(3)}
            rows={SETS.map(([key, label]) => ({key, label, value: mean(data.sets[key].report_f1)}))} />
        </div>
      </div>
      <p className={styles.source}>
        The score for names allowed is the mean of three training runs; the score for names avoided comes from one run.
      </p>
    </details>
  );
}

export default function SyntheticData({data}) {
  const sentences = data.sets.allow.sentences;
  const [shown, setShown] = useState(0);
  const [running, setRunning] = useState(false);
  const timer = useRef(null);

  useEffect(() => {
    if (!running) return undefined;
    if (shown >= sentences.length) {
      setRunning(false);
      return undefined;
    }
    timer.current = setTimeout(() => setShown(shown + 1), shown === 0 ? 300 : 650);
    return () => clearTimeout(timer.current);
  }, [running, shown, sentences.length]);

  const start = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setShown(sentences.length);
    } else {
      setShown(0);
      setRunning(true);
    }
  };

  return (
    <figure className={styles.figure}>
      <p className={styles.title}>A large model writes practice examples</p>
      <p className={styles.caption}>
        One request from the worked example. The request names eight topics; the large model {data.writer} writes
        20 sentences and lists the topics of each one.
      </p>
      <div className={styles.story}>
        <div className={styles.storyCard}>
          <span className={styles.stepNumber}>The request</span>
          <p style={{margin: '4px 0 8px'}}>Write 20 sentences about these topics:</p>
          <ul className={styles.chips}>
            {data.topics.map((t) => <li key={t} className={styles.chip}>{t}</li>)}
          </ul>
        </div>
        <div className={styles.storyArrow} aria-hidden="true">→</div>
        <div className={clsx(styles.storyCard, styles.storyModel)}>
          <span className={styles.stepNumber}>The large model</span>
          <p style={{margin: '4px 0 10px'}}>{data.writer}</p>
          <button type="button" className={styles.primaryButton} onClick={start} disabled={running}>
            {shown ? 'Write them again' : 'Write the sentences'}
          </button>
        </div>
      </div>
      <p style={{margin: '14px 0 6px'}}>
        <strong>Practice sentences</strong> {shown ? `${shown} of ${sentences.length}` : ''}
      </p>
      {shown === 0 ? (
        <p className={styles.source}>Press the button to show the sentences one by one.</p>
      ) : (
        <ul className={styles.sentences} aria-live="polite">
          {sentences.slice(0, shown).map((s) => <Sentence key={s.text} s={s} />)}
        </ul>
      )}
      <p className={styles.source}>
        The worked example repeats this for {data.requests.toLocaleString('en')} requests, each with eight of the{' '}
        {data.labels} topics. Together they give about 20,000 labelled sentences.
      </p>
      <Deeper data={data} />
    </figure>
  );
}
