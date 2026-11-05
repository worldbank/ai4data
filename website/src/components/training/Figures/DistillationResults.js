import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Accuracy and speed of the small model and of the large models asked directly, on the
// same 300 report abstracts. Two charts, one measure each.
const W = 680;
const LEFT = 170;
const RIGHT = 50;
const MIN = 0.003;
const MAX = 10;
const lx = (s) => LEFT + ((Math.log10(s) - Math.log10(MIN)) / (Math.log10(MAX) - Math.log10(MIN))) * (W - LEFT - RIGHT);
const TICKS = [0.01, 0.1, 1, 10];

function seconds(s) {
  return s < 0.1 ? `${(s * 1000).toFixed(s < 0.01 ? 1 : 0)} ms` : `${s.toFixed(2)} s`;
}

export default function DistillationResults({data}) {
  const rowH = 40;
  const speedH = data.speed.length * rowH + 50;
  return (
    <figure className={styles.figure}>
      <p className={styles.title}>The student compared with large models</p>
      <p className={styles.caption}>
        The student and three large models assigned topics to the same 300 report abstracts. The large models were
        asked directly for the topics of each abstract.
      </p>
      <div className={styles.tiles}>
        <div className={styles.tile}>
          <p className={styles.tileLabel}>Accuracy</p>
          <p className={styles.tileLead}>The student scores about the same as the large models.</p>
          <p className={styles.tileRow}><span><span className={clsx(styles.dot, styles.small)} /> Student</span><strong>0.52</strong></p>
          <p className={styles.tileRow}><span><span className={clsx(styles.dot, styles.large)} /> Large models</span><strong>0.52–0.53</strong></p>
          <p className={styles.tileNote}>Score from 0 to 1. A higher score means more correct topics and fewer missed ones.</p>
        </div>
        <div className={styles.tile}>
          <p className={styles.tileLabel}>Speed</p>
          <p className={styles.tileLead}>The student is about 3 to 20 times faster.</p>
          <p className={styles.tileRow}><span><span className={clsx(styles.dot, styles.small)} /> Student</span><strong>0.19 s</strong></p>
          <p className={styles.tileRow}><span><span className={clsx(styles.dot, styles.large)} /> Large models</span><strong>0.5–3.9 s</strong></p>
          <p className={styles.tileNote}>Seconds per abstract.</p>
        </div>
        <div className={styles.tile}>
          <p className={styles.tileLabel}>Computer</p>
          <p className={styles.tileLead}>The student runs on an ordinary processor, without an internet connection.</p>
          <p className={styles.tileRow}><span><span className={clsx(styles.dot, styles.small)} /> Student</span><strong>Processor (CPU)</strong></p>
          <p className={styles.tileRow}><span><span className={clsx(styles.dot, styles.large)} /> Large models</span><strong>GPU or hosted service</strong></p>
          <p className={styles.tileNote}>The student needs 0.57 GB of memory. A GPU is a graphics processor, now also used to run AI models.</p>
        </div>
      </div>
      <details className={styles.deeper}>
      <summary>Go deeper: the score and the time of each model</summary>
      <p className={styles.title} style={{marginTop: 12}}>How well each model assigns topics</p>
      <p className={styles.caption}>
        F1 score on 300 report abstracts, from 0 to 1. The score is high when the assigned topics are correct and
        few of the recorded topics are missed. Student scores are the mean of three training runs. The first student
        learned from the first 20,296 sentences. In further rounds, a program read the student's mistakes on a
        separate set of abstracts and asked the teachers for more sentences on the topics involved.
      </p>
      <div className={styles.bars}>
        {data.scores.map((r) => (
          <React.Fragment key={r.model}>
            <span>{r.model}</span>
            <span className={styles.barTrack} title={`${r.model}: ${r.f1.toFixed(3)}`}>
              <span className={styles.barFill} style={{width: `${(r.f1 / 0.6) * 100}%`, background: `var(--${r.kind})`}} />
            </span>
            <span className={styles.barValue}>{r.f1.toFixed(3)}</span>
          </React.Fragment>
        ))}
      </div>
      <p className={styles.source}>Scale from 0 to 0.6.</p>

      <p className={styles.title} style={{marginTop: 22}}>Time to assign the topics of one abstract</p>
      <p className={styles.caption}>Each mark on the scale is ten times the one before.</p>
      <div className={styles.scroll}>
      <svg className={styles.svg} viewBox={`0 0 ${W} ${speedH}`} role="img"
        aria-label={data.speed.map((r) => `${r.model}, ${r.where}: ${seconds(r.seconds)}`).join('; ')}>
        {TICKS.map((t) => (
          <g key={t}>
            <line className={styles.axis} x1={lx(t)} x2={lx(t)} y1={6} y2={speedH - 34} />
            <text x={lx(t)} y={speedH - 16} textAnchor="middle" className={styles.tick}>{t < 0.1 ? `${t * 1000} ms` : `${t} s`}</text>
          </g>
        ))}
        {data.speed.map((r, i) => {
          const y = 24 + i * rowH;
          return (
            <g key={`${r.model}-${r.where}`}>
              <title>{`${r.model}, ${r.where}: ${seconds(r.seconds)}`}</title>
              <text x={0} y={y - 2} fontSize={13}>{r.model}</text>
              <text x={0} y={y + 14} fontSize={12} className={styles.tick}>{r.where}</text>
              <line x1={LEFT} x2={W - RIGHT} y1={y} y2={y} stroke="var(--line)" strokeWidth={1} />
              <circle cx={lx(r.seconds)} cy={y} r={7} fill={`var(--${r.kind})`}
                stroke="var(--ifm-background-surface-color)" strokeWidth={2} />
              <text x={lx(r.seconds) + 12} y={y + 4} fontSize={12}>{seconds(r.seconds)}</text>
            </g>
          );
        })}
      </svg>
      </div>
      <div className={styles.legend}>
        <span><span className={clsx(styles.dot, styles.small)} /> student (small model)</span>
        <span><span className={clsx(styles.dot, styles.large)} /> large model asked directly</span>
      </div>
      <p className={styles.source}>
        Each abstract was sent alone; the student reads its sentences together. The times of the hosted services
        include the network and the service's queue. The CPU time was measured with four threads of a server processor
        on 100 of the abstracts, and the time of Qwen3.8-27B on 50.
      </p>
      </details>
    </figure>
  );
}
