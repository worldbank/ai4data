import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Model size against where a model can run: the three bands of WDR 2026 table S1.1 on a
// log scale of parameters, the models used in the session placed on it, and the shrink
// of the smallest model about as capable as the first ChatGPT.
const W = 680;
const LEFT = 12;
const RIGHT = 12;
const MIN = 5e7;
const MAX = 3e12;
const TOP = 16;
const BAND_BOTTOM = 170;
const AXIS = 178;

const x = (p) => LEFT + ((Math.log10(p) - Math.log10(MIN)) / (Math.log10(MAX) - Math.log10(MIN))) * (W - LEFT - RIGHT);

const TICKS = [
  [1e8, '100 million'],
  [1e9, '1 billion'],
  [1e10, '10 billion'],
  [1e11, '100 billion'],
  [1e12, '1 trillion'],
];

const LABEL_PLACEMENT = [
  {dy: 28, anchor: 'start', dx: -8},
  {dy: -16, anchor: 'end', dx: 6},
  {dy: 28, anchor: 'start', dx: -6},
];

function formatParams(p) {
  if (p >= 1e9) return `${(p / 1e9).toFixed(p >= 1e10 ? 0 : 1)} billion`;
  return `${Math.round(p / 1e6)} million`;
}

function bandOf(bands, p) {
  return bands.find((b) => p >= b.from && p < b.to);
}

export default function SizeScale({data}) {
  const [selected, setSelected] = useState({type: 'model', index: 0});
  const shades = ['var(--band-1)', 'var(--band-2)', 'var(--band-3)'];
  const keyboard = (fn) => (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fn();
    }
  };

  const model = selected.type === 'model' ? data.models[selected.index] : null;
  const band = selected.type === 'band' ? data.bands[selected.index] : bandOf(data.bands, model.params);
  const shrinkFrom = x(data.shrink.from);
  const shrinkTo = x(data.shrink.to);
  const isSelected = (type, index) => selected.type === type && selected.index === index;

  return (
    <figure className={styles.figure}>
      <p className={styles.title}>Model size and where a model can run</p>
      <p className={styles.caption}>
        Bands from WDR 2026, table S1.1. Select a band or a model to see its requirements.
      </p>
      <div className={styles.segmented} role="group" aria-label="Bands">
        {data.bands.map((b, i) => (
          <button key={b.key} type="button" aria-pressed={isSelected('band', i)}
            onClick={() => setSelected({type: 'band', index: i})}>
            {b.label}
          </button>
        ))}
      </div>{' '}
      <div className={styles.segmented} role="group" aria-label="Models">
        {data.models.map((m, i) => (
          <button key={m.name} type="button" aria-pressed={isSelected('model', i)}
            onClick={() => setSelected({type: 'model', index: i})}>
            <span className={clsx(styles.dot, styles[m.kind])} /> {m.name}
          </button>
        ))}
      </div>
      <div className={styles.scroll}>
      <svg className={styles.svg} viewBox={`0 0 ${W} 222`} role="img"
        aria-label="Log scale of model parameters from 50 million to 3 trillion, divided into the bands phone or laptop, single server, and data center, with the models of the session placed on it.">
        {data.bands.map((b, i) => {
          const x0 = x(Math.max(b.from, MIN));
          const x1 = x(Math.min(b.to, MAX));
          const active = band && band.key === b.key;
          return (
            <g key={b.key} role="button" tabIndex={0} aria-label={`${b.label}: ${b.params} parameters`}
              onClick={() => setSelected({type: 'band', index: i})}
              onKeyDown={keyboard(() => setSelected({type: 'band', index: i}))}>
              <rect x={x0} y={TOP} width={x1 - x0} height={BAND_BOTTOM - TOP} fill={shades[i]}
                stroke={active ? 'var(--ink)' : 'none'} strokeWidth={active ? 1.5 : 0} rx={6} />
              <text x={x0 + 10} y={TOP + 22} fontSize={14} fontWeight={700}>{b.label}</text>
              <text x={x0 + 10} y={TOP + 40} className={styles.tick}>{b.params}</text>
            </g>
          );
        })}
        <path className={styles.shrink}
          d={`M${shrinkFrom},${TOP + 92} C${shrinkFrom - 60},${TOP + 62} ${shrinkTo + 80},${TOP + 62} ${shrinkTo + 8},${TOP + 88}`}
          markerEnd="url(#shrink-head)" />
        <defs>
          <marker id="shrink-head" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--faint)" />
          </marker>
        </defs>
        <circle cx={shrinkFrom} cy={TOP + 93} r={4} fill="var(--faint)" />
        <text x={(shrinkFrom + shrinkTo) / 2 + 20} y={TOP + 64} textAnchor="middle" className={styles.tick}>
          {formatParams(data.shrink.from)} to {formatParams(data.shrink.to)}
        </text>
        {data.models.map((m, i) => {
          const cx = x(m.params);
          const place = LABEL_PLACEMENT[i % LABEL_PLACEMENT.length];
          const active = isSelected('model', i);
          return (
            <g key={m.name} role="button" tabIndex={0} aria-label={`${m.name}: ${formatParams(m.params)} parameters`}
              onClick={() => setSelected({type: 'model', index: i})}
              onKeyDown={keyboard(() => setSelected({type: 'model', index: i}))}>
              <circle cx={cx} cy={128} r={16} fill="transparent" />
              <circle cx={cx} cy={128} r={active ? 9 : 7} fill={`var(--${m.kind})`}
                stroke="var(--ifm-background-surface-color)" strokeWidth={2} />
              <text x={cx + place.dx} y={128 + place.dy} textAnchor={place.anchor} fontSize={13}
                fontWeight={active ? 700 : 400}>
                {m.name}
              </text>
            </g>
          );
        })}
        <line className={styles.axis} x1={LEFT} x2={W - RIGHT} y1={AXIS} y2={AXIS} />
        {TICKS.map(([p, label]) => (
          <g key={p}>
            <line className={styles.axis} x1={x(p)} x2={x(p)} y1={AXIS} y2={AXIS + 5} />
            <text x={x(p)} y={AXIS + 20} textAnchor="middle" className={styles.tick}>{label}</text>
          </g>
        ))}
        <text x={W / 2} y={AXIS + 40} textAnchor="middle" className={styles.tick}>Parameters (log scale)</text>
      </svg>
      </div>
      <div className={styles.detail} aria-live="polite">
        {model ? (
          <>
            <strong>{model.name}</strong> · {model.detail}
            <dl>
              <dt>Parameters</dt>
              <dd>{formatParams(model.params)}</dd>
              <dt>Memory</dt>
              <dd>{model.memory}</dd>
              <dt>Band</dt>
              <dd>{band.label}, connectivity needed: {band.connectivity.toLowerCase()}</dd>
              <dt>Role in the session</dt>
              <dd>{model.role}</dd>
            </dl>
          </>
        ) : (
          <>
            <strong>{band.label}</strong>
            <dl>
              <dt>Parameters</dt>
              <dd>{band.params}</dd>
              <dt>Memory needed</dt>
              <dd>{band.memory}</dd>
              <dt>Connectivity needed</dt>
              <dd>{band.connectivity}</dd>
            </dl>
          </>
        )}
      </div>
      <div className={styles.legend}>
        <span><span className={clsx(styles.dot, styles.small)} /> small model</span>
        <span><span className={clsx(styles.dot, styles.large)} /> large model</span>
        <span>Dashed arrow: {data.shrink.text}, {data.shrink.years}</span>
      </div>
      <p className={styles.source}>
        {data.hosted} The figures for the bands are rounded orders of magnitude for typical cases. Source: {data.source}
      </p>
    </figure>
  );
}
