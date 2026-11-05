import React, {useMemo, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// A word-counting model (naive Bayes) trained in the browser on the first n synthetic
// sentences of four topics and tested on 60 real report abstracts, 15 per topic. The
// tokenizer and the stop words match the builder of the session's figure data.
const MAX = 40;

function tokenize(text, stop) {
  return (text.toLowerCase().match(/[a-z]+/g) || []).filter((w) => w.length > 2 && !stop.has(w));
}

function train(data, stop, n) {
  const counts = {};
  const totals = {};
  const vocab = new Set();
  for (const t of data.topics) {
    counts[t] = new Map();
    totals[t] = 0;
    for (const s of data.sentences[t].slice(0, n)) {
      for (const w of tokenize(s, stop)) {
        counts[t].set(w, (counts[t].get(w) || 0) + 1);
        totals[t] += 1;
        vocab.add(w);
      }
    }
  }
  return {counts, totals, size: vocab.size + 1};
}

// Probability of each topic for a text, from the summed log likelihoods of its words.
function predict(model, topics, words) {
  const logs = topics.map((t) =>
    words.reduce((sum, w) => sum + Math.log(((model.counts[t].get(w) || 0) + 1) / (model.totals[t] + model.size)), 0),
  );
  const top = Math.max(...logs);
  const exps = logs.map((l) => Math.exp(l - top));
  const z = exps.reduce((a, b) => a + b, 0);
  return topics.map((t, i) => ({topic: t, p: exps[i] / z}));
}

// The words that most set one topic apart from the other three.
function learnedWords(model, topics, topic, k = 6) {
  const own = model.counts[topic];
  const rate = (t, w) => ((model.counts[t].get(w) || 0) + 1) / (model.totals[t] + model.size);
  return [...own.entries()]
    .filter(([, c]) => c >= 2)
    .map(([w]) => [w, rate(topic, w) / Math.max(...topics.filter((t) => t !== topic).map((t) => rate(t, w)))])
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([w]) => w);
}

function Curve({points, n}) {
  const W = 300;
  const H = 160;
  const L = 34;
  const B = 24;
  const x = (i) => L + (i / MAX) * (W - L - 8);
  const y = (v) => 8 + (1 - v) * (H - B - 8);
  const path = points.map((v, i) => `${i ? 'L' : 'M'}${x(i + 1)},${y(v)}`).join(' ');
  return (
    <svg className={styles.svg} viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Share of the 60 abstracts with the right topic, by examples per topic, from 1 to ${MAX}.`}>
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line className={styles.axis} x1={L} x2={W - 8} y1={y(v)} y2={y(v)} strokeDasharray={v === 0.25 ? '4 3' : undefined} />
          <text x={L - 6} y={y(v) + 4} textAnchor="end" className={styles.tick}>{v * 100}%</text>
        </g>
      ))}
      <text x={W - 8} y={y(0.25) - 5} textAnchor="end" className={styles.tick}>guessing</text>
      <path d={path} fill="none" stroke="var(--small)" strokeWidth={2} />
      {n > 0 && <circle cx={x(n)} cy={y(points[n - 1])} r={5} fill="var(--small)" stroke="var(--ifm-background-surface-color)" strokeWidth={2} />}
      {[1, 10, 20, 30, 40].map((k) => (
        <text key={k} x={x(k)} y={H - B + 12} textAnchor="middle" className={styles.tick} fontSize={10}>{k}</text>
      ))}
      <text x={(L + W) / 2} y={H - 1} textAnchor="middle" className={styles.tick}>examples per topic</text>
    </svg>
  );
}

const PRESETS = [1, 10, 40];

export default function Learn({data}) {
  const stop = useMemo(() => new Set(data.stop), [data]);
  const [n, setN] = useState(1);
  const [own, setOwn] = useState('The new road cuts the travel time from farms to the market.');
  const tests = useMemo(() => data.tests.map((d) => ({...d, words: tokenize(d.text, stop)})), [data, stop]);

  // Results on the test abstracts for every n, so the curve is drawn once.
  const curve = useMemo(() => {
    const out = [];
    for (let k = 1; k <= MAX; k += 1) {
      const m = train(data, stop, k);
      out.push(tests.filter((d) => predict(m, data.topics, d.words).reduce((a, b) => (b.p > a.p ? b : a)).topic === d.topic).length / tests.length);
    }
    return out;
  }, [data, stop, tests]);

  const model = useMemo(() => train(data, stop, n), [data, stop, n]);
  const guesses = tests.map((d) => {
    const best = n ? predict(model, data.topics, d.words).reduce((a, b) => (b.p > a.p ? b : a)).topic : null;
    return {...d, guess: best, right: best === d.topic};
  });
  const right = guesses.filter((g) => g.right).length;
  const ownScores = n ? predict(model, data.topics, tokenize(own, stop)) : null;
  const latest = n ? data.topics.map((t) => [t, data.sentences[t][n - 1]]) : [];

  return (
    <figure className={styles.figure}>
      <p className={styles.title}>A model learns from examples</p>
      <p className={styles.caption}>
        A simple model learns four topics from example sentences. It counts the words in the examples of each topic
        and gives a new text the topic whose words match best. Choose how many examples it reads.
      </p>

      <div className={styles.segmented} role="group" aria-label="Examples per topic">
        {PRESETS.map((k) => (
          <button key={k} type="button" aria-pressed={n === k} onClick={() => setN(k)}>
            {k} {k === 1 ? 'example' : 'examples'} per topic
          </button>
        ))}
      </div>

      <div className={styles.twoCol} style={{marginTop: 16}}>
        <div>
          <p><strong>Words the model links to each topic</strong></p>
          <div className={styles.wordTable}>
            {data.topics.map((t) => {
              const words = n ? learnedWords(model, data.topics, t) : [];
              return (
                <React.Fragment key={t}>
                  <span className={styles.wordTopic}>{t}</span>
                  <ul className={styles.chips}>
                    {words.length === 0 ? <li className={clsx(styles.chip, styles.chipOff)}>none yet</li> :
                      words.map((w) => <li key={w} className={styles.chip}>{w}</li>)}
                  </ul>
                </React.Fragment>
              );
            })}
          </div>
        </div>
        <div>
          <p style={{marginBottom: 2}}><strong>Test on 60 real reports</strong></p>
          <p className={styles.bigNumber} aria-live="polite">
            {right} <span>of 60 given the right topic</span>
          </p>
          <div className={styles.dotRows} role="img"
            aria-label={`${right} of 60 reports given the right topic.`}>
            {data.topics.map((t) => (
              <React.Fragment key={t}>
                <span className={styles.dotLabel}>{t}</span>
                <span className={styles.dotGrid}>
                  {guesses.filter((g) => g.topic === t).map((g) => (
                    <span key={g.id} className={clsx(styles.testDot, g.right && styles.testRight)}
                      title={`${g.title}\nRecorded topic: ${g.topic}${g.guess ? `\nModel's guess: ${g.guess}` : ''}`} />
                  ))}
                </span>
              </React.Fragment>
            ))}
          </div>
          <p className={styles.source} style={{marginTop: 8}}>
            Each square is one report, filled when the model gives it the right topic. A random guess among the
            four topics gets about 15 right.
          </p>
        </div>
      </div>

      <div className={styles.detail} style={{marginTop: 16}}>
        <label htmlFor="learn-own"><strong>Try a sentence</strong></label>
        <p className={styles.source} style={{margin: '2px 0 0'}}>
          The sentence below is about Transport. Compare the model's answer with 10 and with 40 examples per topic,
          or write another sentence.
        </p>
        <input id="learn-own" className={styles.textInput} value={own} onChange={(e) => setOwn(e.target.value)} />
        {ownScores && (
          <div className={styles.bars} style={{marginTop: 10}}>
            {ownScores.map(({topic, p}) => (
              <React.Fragment key={topic}>
                <span>{topic}</span>
                <span className={styles.barTrack}>
                  <span className={styles.barFill} style={{width: `${p * 100}%`, background: 'var(--small)'}} />
                </span>
                <span className={styles.barValue}>{Math.round(p * 100)}%</span>
              </React.Fragment>
            ))}
          </div>
        )}
      </div>

      <details className={styles.deeper}>
        <summary>Go deeper: any number of examples, the learning curve, and the examples themselves</summary>
        <label className={styles.slider} style={{marginTop: 12}}>
          <span><strong>{n}</strong> {n === 1 ? 'example' : 'examples'} per topic</span>
          <input type="range" min={1} max={MAX} value={n} onChange={(e) => setN(Number(e.target.value))}
            aria-label="Examples per topic" />
        </label>
        <div className={styles.twoCol} style={{marginTop: 12}}>
          <div>
            <p><strong>Share of the 60 reports with the right topic</strong></p>
            <Curve points={curve} n={n} />
            <p className={styles.source}>
              The dashed line at 25% is a random guess. After about 20 examples per topic this model stops improving.
              The model built in the session learns more from each example and reads about 20,000 of them.
            </p>
          </div>
          <div>
            <p><strong>The last example of each topic</strong></p>
            <ul className={styles.latest}>
              {latest.map(([t, s]) => (
                <li key={t}><span className={styles.chip}>{t}</span> {s}</li>
              ))}
            </ul>
          </div>
        </div>
      </details>
    </figure>
  );
}
