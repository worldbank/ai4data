import {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Simulated chatbot for the PCN example (arXiv:2509.06902): Philippines GDP
// growth, 2024, World Development Indicators. The official record is what the
// application compares the model's tagged number against.
const OFFICIAL = 5.69201612823412;
const RECORD_ID = '0328';
const PAYLOAD = {
  claim_id: RECORD_ID,
  indicator: 'GDP growth (annual %)',
  country: 'Philippines',
  date: '2024',
  value: OFFICIAL,
  unit: 'Percentage',
  source: 'World Development Indicators',
};
const QUESTION = 'How fast did the Philippines’ economy grow in 2024?';

const Cases = [
  {key: 'match', label: 'Correct number', value: 5.7, tag: RECORD_ID},
  {key: 'mismatch', label: 'Wrong number', value: 6.0, tag: RECORD_ID},
  {key: 'untagged', label: 'No source tag', value: 6.0, tag: null},
];

const WORDS = ['The', 'Philippines’', 'economy', 'grew', 'by', '{NUM}', 'in', '2024.'];
const NUM_INDEX = WORDS.indexOf('{NUM}');

const round1 = (x) => Math.round(x * 10) / 10;

function outcomeFor(c) {
  if (!c.tag) {
    return {
      status: 'none',
      text: 'No check mark. The model attached no source tag, so the number was not checked.',
    };
  }
  return round1(c.value) === round1(OFFICIAL)
    ? {
        status: 'verified',
        text: `Verified. ${c.value.toFixed(1)} matches record ${c.tag} (${round1(OFFICIAL).toFixed(1)}).`,
      }
    : {
        status: 'flagged',
        text: `Flagged. ${c.value.toFixed(1)} does not match record ${c.tag} (${round1(OFFICIAL).toFixed(1)}).`,
      };
}

// phase: idle -> typing -> streaming -> done. Verification is instant, so the
// verdict shows as soon as the number appears.
export default function PcnEvidence({active = true}) {
  const [caseKey, setCaseKey] = useState('match');
  const [run, setRun] = useState(0);
  const [phase, setPhase] = useState('idle');
  const [shown, setShown] = useState(0);

  const c = Cases.find((x) => x.key === caseKey);
  const outcome = outcomeFor(c);

  useEffect(() => {
    if (!active) {
      setPhase('idle');
      setShown(0);
      return undefined;
    }
    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      setShown(WORDS.length);
      setPhase('done');
      return undefined;
    }
    const timers = [];
    const at = (ms, fn) => timers.push(setTimeout(fn, ms));
    setShown(0);
    setPhase('typing');
    at(800, () => setPhase('streaming'));
    WORDS.forEach((_, i) => at(800 + 110 * (i + 1), () => setShown(i + 1)));
    at(800 + 110 * WORDS.length, () => setPhase('done'));
    return () => timers.forEach(clearTimeout);
  }, [active, caseKey, run]);

  const numValue = `${c.value.toFixed(1)}%`;
  // The verdict applies from the moment the number is on screen.
  const numVisible = shown > NUM_INDEX;

  return (
    <div className={styles.evidenceBox}>
      <div className={styles.pcnTop}>
        <div className={styles.pcnCases} role="group" aria-label="Model behaviour">
          {Cases.map((x) => (
            <button
              type="button"
              key={x.key}
              aria-pressed={x.key === caseKey}
              className={clsx(
                styles.pcnCase,
                x.key === caseKey && styles.pcnCaseActive,
              )}
              onClick={() => {
                setCaseKey(x.key);
                setRun((r) => r + 1);
              }}>
              {x.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          className={styles.pcnReplay}
          onClick={() => setRun((r) => r + 1)}
          aria-label="Replay the conversation">
          ↻ Replay
        </button>
      </div>

      <div className={styles.chat}>
        <div className={clsx(styles.msg, styles.msgUser)}>{QUESTION}</div>

        {phase !== 'idle' && (
          <div className={clsx(styles.msg, styles.msgBot)}>
            {phase === 'typing' ? (
              <span className={styles.typing} aria-label="Assistant is typing">
                <i />
                <i />
                <i />
              </span>
            ) : (
              <p className={styles.botText}>
                {WORDS.slice(0, shown).map((w, i) => {
                  if (i !== NUM_INDEX) {
                    return <span key={i}>{w} </span>;
                  }
                  const verified = outcome.status === 'verified';
                  const num = (
                    <span
                      className={clsx(
                        styles.pcnNum,
                        styles[`pcn_${outcome.status}`],
                      )}>
                      {numValue}
                    </span>
                  );
                  return (
                    <span key={i}>
                      {verified ? (
                        <span
                          className={styles.pcnProof}
                          tabIndex={0}
                          aria-describedby="pcn-proof-card">
                          {num}
                          <span
                            className={styles.pcnBadgeOk}
                            role="img"
                            aria-label="Verified">
                            ✓
                          </span>
                          <span
                            className={styles.pcnPopover}
                            role="tooltip"
                            id="pcn-proof-card">
                            <strong>Verified data</strong>
                            <dl>
                              <dt>Indicator</dt>
                              <dd>GDP growth (annual %)</dd>
                              <dt>Country</dt>
                              <dd>Philippines</dd>
                              <dt>Date</dt>
                              <dd>2024</dd>
                              <dt>Source value</dt>
                              <dd>{OFFICIAL.toFixed(3)}</dd>
                              <dt>Unit</dt>
                              <dd>Percentage</dd>
                              <dt>Display rule</dt>
                              <dd>Match at 1 decimal place</dd>
                            </dl>
                          </span>
                        </span>
                      ) : (
                        <>
                          {num}
                          {outcome.status === 'flagged' && (
                            <span
                              className={styles.pcnBadgeWarn}
                              role="img"
                              aria-label="Flagged">
                              !
                            </span>
                          )}
                        </>
                      )}{' '}
                    </span>
                  );
                })}
              </p>
            )}
          </div>
        )}

      </div>

      <div
        className={clsx(
          styles.pcnStatusLine,
          numVisible && styles[`pcn_${outcome.status}`],
        )}
        aria-live="polite">
        {numVisible && (
          <>
            <span>{outcome.text}</span>
            {outcome.status === 'verified' && (
              <span className={styles.pcnHint}>
                {' '}
                Hover over the number to see the source record.
              </span>
            )}
          </>
        )}
      </div>

      <details className={styles.pcnDetails}>
          <summary>Show the tagged model output</summary>
          <code className={styles.pcnCode}>
            The Philippines&apos; economy grew by{' '}
            {c.tag ? (
              <>
                <span className={styles.pcnTag}>{`<claim id="${c.tag}">`}</span>
                {c.value.toFixed(1)}
                <span className={styles.pcnTag}>{'</claim>'}</span>
              </>
            ) : (
              c.value.toFixed(1)
            )}
            % in 2024.
          </code>
      </details>

      <details className={styles.pcnDetails}>
        <summary>Show the claim payload</summary>
        <code className={styles.pcnCode}>
          <pre>{JSON.stringify(PAYLOAD, null, 2)}</pre>
        </code>
        <span className={styles.pcnFine}>
          Simplified for illustration. The application passes this record to
          the model, which cites it by claim ID.
        </span>
      </details>
    </div>
  );
}
