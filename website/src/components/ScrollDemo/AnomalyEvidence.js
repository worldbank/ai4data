import {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Worked example from docs/anomaly/explanation/elicitation-pipeline.md.
// Series: WDI NY.GDP.MKTP.KD.ZG, Nigeria (values from the WDI API). The
// classification, confidence, evidence strength and explanation are the
// documented pipeline output. The news excerpt is verbatim from CNBC and is
// shown to illustrate how a reviewer can check a cited event.
const YEARS = [2013, 2014, 2015, 2016, 2017];
const VALUES = [6.67, 6.31, 2.65, -1.62, 0.81];
const X = [50, 145, 240, 335, 430];
const yFor = (v) => 20 + ((8 - v) / 11) * 78;
const POINTS = X.map((x, i) => [x, yFor(VALUES[i])]);
const PATH = POINTS.map(([x, y], i) => `${i ? 'L' : 'M'}${x},${y.toFixed(1)}`).join(' ');

const ARTICLE = {
  outlet: 'CNBC',
  date: '31 Aug 2016',
  byline: 'Justina Crabtree, special to CNBC.com',
  headline: 'Nigeria’s economy drops into recession',
  url: 'https://www.cnbc.com/2016/08/31/nigerias-economy-drops-into-recession.html',
};

// Claims in the explanation, in order of appearance, with the verdict from
// checking each against the excerpt.
const CLAIMS = [
  {
    id: 'recession',
    text: 'its first recession in 25 years',
    status: 'none',
    verdict:
      'Not in the excerpt. It reports that the country dropped into recession but does not say how long since the last one. A reviewer would look for another source.',
  },
  {
    id: 'price',
    text: 'collapse in global oil prices',
    status: 'supported',
    verdict:
      'Supported. The excerpt says the oil industry “has suffered under weak global prices.”',
  },
  {
    id: 'attacks',
    text: 'militant attacks on oil infrastructure in the Niger Delta',
    status: 'partial',
    verdict:
      'Partly supported. The excerpt cites losses of crude oil production from “vandalism” and “sabotage.” It does not name militants or the Niger Delta.',
  },
];

const STATUS_LABEL = {
  supported: 'Supported',
  partial: 'Partly supported',
  none: 'Not in excerpt',
};

// Phases: 0 empty, 1 line drawn, 2 window flagged, 3 classification shown,
// 4 evidence shown.
const PHASE_MS = [300, 1100, 900, 900];

export default function AnomalyEvidence({active = true}) {
  const [phase, setPhase] = useState(0);
  const [run, setRun] = useState(0);
  const [claim, setClaim] = useState('price');
  const [hover, setHover] = useState(null);

  useEffect(() => {
    setClaim('price');
    if (!active) {
      setPhase(0);
      return undefined;
    }
    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      setPhase(4);
      return undefined;
    }
    setPhase(0);
    const timers = [];
    let t = 0;
    PHASE_MS.forEach((ms, i) => {
      t += ms;
      timers.push(setTimeout(() => setPhase(i + 1), t));
    });
    return () => timers.forEach(clearTimeout);
  }, [active, run]);

  const current = CLAIMS.find((c) => c.id === claim);
  const mark = (id, children) => (
    <mark
      className={clsx(styles.anMark, claim === id && styles.anMarkOn)}
      onClick={() => setClaim(id)}>
      {children}
    </mark>
  );

  return (
    <div className={styles.evidenceBox}>
      <div className={styles.anHead}>
        <span className={styles.chartLabel}>GDP growth (annual %), Nigeria</span>
        <button
          type="button"
          className={styles.pcnReplay}
          onClick={() => setRun((r) => r + 1)}>
          ↻ Replay
        </button>
      </div>

      <svg
        className={styles.anChart}
        viewBox="0 0 480 138"
        role="img"
        aria-label="Nigeria GDP growth 2013 to 2017, with 2015 to 2016 flagged as an anomaly window">
        <line x1="20" x2="465" y1={yFor(0)} y2={yFor(0)} className={styles.anZero} />
        <text x="20" y={yFor(0) - 4} className={styles.anZeroLabel}>
          0%
        </text>
        <rect
          x="222"
          y="6"
          width="131"
          height="108"
          rx="6"
          className={clsx(styles.anWindow, phase >= 2 && styles.anWindowOn)}
        />
        <text
          x="287"
          y="17"
          textAnchor="middle"
          className={clsx(styles.anWindowLabel, phase >= 2 && styles.anWindowLabelOn)}>
          Anomaly window
        </text>
        <path
          d={PATH}
          pathLength="1"
          fill="none"
          className={clsx(styles.anLine, phase >= 1 && styles.anLineOn)}
        />
        {POINTS.map(([x, y], i) => {
          const flagged = phase >= 2 && (i === 2 || i === 3);
          return (
            <g
              key={YEARS[i]}
              className={clsx(styles.anPoint, phase >= 1 && styles.anPointOn)}
              style={{transitionDelay: `${0.2 + i * 0.12}s`}}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}>
              <circle cx={x} cy={y} r="14" fill="transparent" />
              <circle
                cx={x}
                cy={y}
                r={hover === i ? 6 : i === 3 && flagged ? 5.5 : 3.5}
                className={flagged ? styles.anDotFlag : styles.anDot}
              />
              <text
                x={x}
                y={y + (i === 3 ? 17 : -9)}
                textAnchor="middle"
                className={clsx(
                  styles.anValue,
                  (hover === i || flagged) && styles.anValueOn,
                )}>
                {VALUES[i]}
              </text>
              <text x={x} y="132" textAnchor="middle" className={styles.axisLabel}>
                {YEARS[i]}
              </text>
            </g>
          );
        })}
      </svg>

      <div className={clsx(styles.anOutput, phase >= 3 && styles.anShow)}>
        <div>
          <span className={styles.anKey}>Classification</span>
          <span className={styles.badge}>external_driver</span>
        </div>
        <div>
          <span className={styles.anKey}>Confidence</span>
          <span className={styles.anConf}>
            <span className={styles.anConfBar}>
              <span
                className={styles.anConfFill}
                style={{width: phase >= 3 ? '92%' : '0%'}}
              />
            </span>
            0.92
          </span>
        </div>
        <div>
          <span className={styles.anKey}>Evidence strength</span>
          <span className={styles.outputValue}>strong_direct</span>
        </div>
      </div>

      <div className={clsx(styles.anExplain, phase >= 3 && styles.anShow)}>
        <span className={styles.anKey}>Explanation</span>
        <p>
          Nigeria experienced a sharp GDP contraction in 2016,{' '}
          <ClaimButton
            n={1}
            claim={CLAIMS[0]}
            selected={claim === 'recession'}
            onSelect={setClaim}
          />
          , driven by the{' '}
          <ClaimButton
            n={2}
            claim={CLAIMS[1]}
            selected={claim === 'price'}
            onSelect={setClaim}
          />{' '}
          beginning in 2014–2015 and{' '}
          <ClaimButton
            n={3}
            claim={CLAIMS[2]}
            selected={claim === 'attacks'}
            onSelect={setClaim}
          />
          .
        </p>
      </div>

      <div className={clsx(styles.anNews, phase >= 4 && styles.anShow)}>
        <div className={styles.anNewsMeta}>
          <span className={styles.anOutlet}>{ARTICLE.outlet}</span>
          <span>{ARTICLE.date}</span>
          <span>{ARTICLE.byline}</span>
        </div>
        <a
          className={styles.anHeadline}
          href={ARTICLE.url}
          target="_blank"
          rel="noopener noreferrer">
          {ARTICLE.headline} ↗
        </a>
        <blockquote className={styles.anQuote}>
          Nigeria’s statistics office said Wednesday that the country has dropped
          into recession as its all-important oil industry has suffered under{' '}
          {mark('price', 'weak global prices')}. […] An economic adviser to
          President Muhammadu Buhari, Adeyemi Dipeolu, told the Associated Press
          that the bleak data was largely attributed to “a sharp contraction in
          the oil sector due to huge losses of crude oil production,” resulting
          from {mark('attacks', 'vandalism and “sabotage.”')}
        </blockquote>
        <p className={clsx(styles.anVerdict, styles[`anVerdict_${current.status}`])}>
          <strong>{STATUS_LABEL[current.status]}.</strong>{' '}
          {current.verdict.replace(/^[^.]*\.\s*/, '')}
        </p>
      </div>
      <span className={clsx(styles.pcnFine, styles.anNote, phase >= 4 && styles.anShow)}>
        The CNBC excerpt was added to illustrate checking a cited event.
      </span>
    </div>
  );
}

function ClaimButton({n, claim, selected, onSelect}) {
  return (
    <button
      type="button"
      className={clsx(
        styles.anClaim,
        styles[`anClaim_${claim.status}`],
        selected && styles.anClaimSel,
      )}
      aria-pressed={selected}
      onClick={() => onSelect(claim.id)}>
      {claim.text}
      <sup className={styles.anClaimN}>{n}</sup>
    </button>
  );
}
