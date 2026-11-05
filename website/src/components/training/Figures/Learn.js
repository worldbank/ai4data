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
      aria-label={`Share of the 60 abstracts with the right topic, by practice sentences per topic, from 1 to ${MAX}.`}>
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
      <text x={(L + W) / 2} y={H - 1} textAnchor="middle" className={styles.tick}>practice sentences per topic</text>
    </svg>
  );
}

export default function Learn({data}) {
  const stop = useMemo(() => new Set(data.stop), [data]);
  const [n, setN] = useState(0);
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
      <p className={styles.title}>Training a model in this browser</p>
      <p className={styles.caption}>
        A simple model learns four topics from practice sentences written by a large language model. It counts the
        words in the sentences of each topic and guesses the topic whose words best match a new text. Add practice
        sentences and watch the test on 60 real report abstracts.
      </p>

      <div className={styles.learnControls}>
        <button type="button" className={styles.primaryButton} disabled={n >= MAX} onClick={() => setN(Math.min(MAX, n + 1))}>
          Add one sentence per topic
        </button>
        <label className={styles.slider}>
          <span><strong>{n}</strong> per topic</span>
          <input type="range" min={0} max={MAX} value={n} onChange={(e) => setN(Number(e.target.value))}
            aria-label="Practice sentences per topic" />
        </label>
      </div>

      {n === 0 ? (
        <p className={styles.detail}>The model has seen no examples yet and cannot tell the topics apart.</p>
      ) : (
        <div className={styles.detail}>
          <strong>Latest practice sentences</strong>
          <ul className={styles.latest}>
            {latest.map(([t, s]) => (
              <li key={t}><span className={styles.chip}>{t}</span> {s}</li>
            ))}
          </ul>
        </div>
      )}

      <div className={styles.twoCol} style={{marginTop: 16}}>
        <div>
          <p><strong>Words the model links to each topic</strong></p>
          <div className={styles.wordTable}>
            {data.topics.map((t) => (
              <React.Fragment key={t}>
                <span className={styles.wordTopic}>{t}</span>
                <ul className={styles.chips}>
                  {n === 0 ? <li className={clsx(styles.chip, styles.chipOff)}>none yet</li> :
                    learnedWords(model, data.topics, t).map((w) => <li key={w} className={styles.chip}>{w}</li>)}
                </ul>
              </React.Fragment>
            ))}
          </div>
        </div>
        <div>
          <p><strong>Test on 60 real abstracts</strong>: {n ? `${right} right` : 'not trained yet'}</p>
          <div className={styles.dotRows} role="img"
            aria-label={`${right} of 60 abstracts given the right topic.`}>
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
          <Curve points={curve} n={n} />
        </div>
      </div>

      <div className={styles.detail} style={{marginTop: 16}}>
        <label htmlFor="learn-own"><strong>Write a sentence for the model</strong></label>
        <p className={styles.source} style={{margin: '2px 0 0'}}>
          The sentence below is about Transport. Compare the model's answer at 5 and at 40 sentences per topic.
        </p>
        <input id="learn-own" className={styles.textInput} value={own} onChange={(e) => setOwn(e.target.value)} />
        {ownScores ? (
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
        ) : (
          <p className={styles.source}>Add practice sentences first.</p>
        )}
      </div>
      <p className={styles.source}>
        Each square is one abstract, filled when the model gives it the right topic. Each row holds the 15 abstracts of
        one topic; hover over a square for its title.
        The dashed line at 25% is the score of a random guess among four topics. The model in this figure is far
        simpler than the student model of the session.
      </p>
    </figure>
  );
}
