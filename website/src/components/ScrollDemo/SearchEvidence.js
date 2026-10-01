import {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Semantic search example over real WDI data. The ranking was computed with
// avsolatorio/GIST-all-MiniLM-L6-v2 (cosine similarity between the full query
// and the names of the 1,498 WDI indicators), and the Philippines values come
// from the WDI API (latest year available, fetched 2026-10-01). Tool names
// follow docs/mcp/mcp.md.
const QUERY = 'how many people go hungry in the Philippines';

const RESULTS = [
  {
    code: 'SN.ITK.SVFI.ZS',
    name: 'Prevalence of severe food insecurity in the population (%)',
    score: 0.796,
    year: 2023,
    value: '3%',
  },
  {
    code: 'SN.ITK.MSFI.ZS',
    name: 'Prevalence of moderate or severe food insecurity in the population (%)',
    score: 0.786,
    year: 2023,
    value: '32.9%',
  },
  {
    code: 'SI.SPR.PCAP',
    name: 'Survey mean consumption or income per capita, total population (2021 PPP $ per day)',
    score: 0.783,
    year: 2023,
    value: '$9.93 per day',
  },
  {
    code: 'SI.SPR.PC40',
    name: 'Survey mean consumption or income per capita, bottom 40% of population (2021 PPP $ per day)',
    score: 0.781,
    year: 2023,
    value: '$4.37 per day',
  },
];

const Modes = [
  {key: 'person', label: 'Person: natural language search'},
  {key: 'agent', label: 'AI agent: MCP tool calls'},
];

const STOPWORDS = new Set(['how', 'many', 'go', 'in', 'the']);
const queryWords = new Set(
  QUERY.toLowerCase()
    .split(/\s+/)
    .filter((w) => !STOPWORDS.has(w)),
);
const tokenize = (name) => name.split(/(\s+)/);
const isHit = (token) =>
  queryWords.has(token.toLowerCase().replace(/[^a-z]/g, ''));

const REQ1 = `→ tools/call  search_indicators\n  { "query": "${QUERY}" }`;
const RESP1 = `← [ ${RESULTS.map(
  (r) => `{ "indicator": "${r.code}", "score": ${r.score} }`,
).join(',\n    ')} ]`;
const REQ2 = `→ tools/call  get_indicator\n  { "indicator_code": "${RESULTS[0].code}",\n    "country_code": "PHL",\n    "start_year": 2023, "end_year": 2023 }`;
const RESP2 = `← { "country": "PHL", "year": 2023,\n    "value": 3 }`;

export default function SearchEvidence({active = true}) {
  const [mode, setMode] = useState('person');
  const [run, setRun] = useState(0);
  const [typed, setTyped] = useState(0); // characters of the current input
  const [step, setStep] = useState(0);
  const [shown, setShown] = useState(0); // person view: results revealed
  const [sel, setSel] = useState(0);
  const [copied, setCopied] = useState(false);

  // Person steps: 0 typing, 1 waiting, 2 results. Agent steps: 0 typing
  // request 1, 1 waiting, 2 response 1, 3 typing request 2, 4 waiting, 5 done.
  useEffect(() => {
    setSel(0);
    if (!active) {
      setTyped(0);
      setStep(0);
      setShown(0);
      return undefined;
    }
    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      setStep(mode === 'person' ? 2 : 5);
      setShown(RESULTS.length);
      setTyped(1000);
      return undefined;
    }
    const timers = [];
    const at = (ms, fn) => timers.push(setTimeout(fn, ms));
    setStep(0);
    setTyped(0);
    setShown(0);
    if (mode === 'person') {
      const per = 32;
      for (let i = 1; i <= QUERY.length; i += 1) {
        at(350 + per * i, () => setTyped(i));
      }
      const t = 350 + per * QUERY.length;
      at(t + 80, () => setStep(1));
      at(t + 650, () => setStep(2));
      RESULTS.forEach((_, i) => at(t + 650 + 170 * (i + 1), () => setShown(i + 1)));
    } else {
      const per = 13;
      for (let i = 1; i <= REQ1.length; i += 1) {
        at(350 + per * i, () => setTyped(i));
      }
      let t = 350 + per * REQ1.length;
      at(t + 80, () => setStep(1));
      t += 650;
      at(t, () => setStep(2));
      t += 900;
      at(t, () => {
        setStep(3);
        setTyped(0);
      });
      for (let i = 1; i <= REQ2.length; i += 1) {
        at(t + 40 + per * i, () => setTyped(i));
      }
      t += 40 + per * REQ2.length;
      at(t + 80, () => setStep(4));
      at(t + 600, () => setStep(5));
    }
    return () => timers.forEach(clearTimeout);
  }, [active, mode, run]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(
        [REQ1, RESP1, REQ2, RESP2].join('\n\n'),
      );
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable.
    }
  };

  const personDone = step >= 2;
  const agentDone = step === 5;
  const done = mode === 'person' ? personDone && shown === RESULTS.length : agentDone;
  const sink = (full, upTo) => full.slice(0, upTo);

  return (
    <div className={styles.evidenceBox}>
      <div className={styles.srTabs} role="tablist" aria-label="Who is asking">
        {Modes.map((m) => (
          <button
            type="button"
            role="tab"
            key={m.key}
            aria-selected={m.key === mode}
            className={clsx(styles.srTab, m.key === mode && styles.srTabActive)}
            onClick={() => setMode(m.key)}>
            {m.label}
          </button>
        ))}
      </div>

      <div className={styles.srStage}>
        {mode === 'person' ? (
          <>
            <div className={styles.srSearchBar}>
              <span className={styles.srIcon} aria-hidden="true">
                &#8981;
              </span>
              <span>
                {sink(QUERY, typed)}
                {step === 0 && (
                  <span className={styles.srCaret} aria-hidden="true" />
                )}
              </span>
            </div>
            {step === 1 && (
              <div className={styles.srStatus}>Matching on meaning…</div>
            )}
            {personDone && (
              <ul className={styles.srList}>
                {RESULTS.slice(0, shown).map((r, i) => {
                  return (
                    <li key={r.code}>
                      <button
                        type="button"
                        aria-expanded={sel === i}
                        className={clsx(
                          styles.srRow,
                          sel === i && styles.srRowSel,
                        )}
                        onClick={() => setSel(i)}>
                        <span className={styles.srRowMain}>
                          <span className={styles.srRowCode}>{r.code}</span>
                          <span className={styles.srRowName}>
                            {tokenize(r.name).map((t, k) =>
                              isHit(t) ? (
                                <mark className={styles.srMark} key={k}>
                                  {t}
                                </mark>
                              ) : (
                                <span key={k}>{t}</span>
                              ),
                            )}
                          </span>
                        </span>
                        <span className={styles.srScore}>
                          <span className={styles.srScoreNum}>
                            {r.score.toFixed(3)}
                          </span>
                          <span className={styles.srBar}>
                            <span
                              className={styles.srBarFill}
                              style={{width: `${r.score * 100}%`}}
                            />
                          </span>
                        </span>
                      </button>
                      {sel === i && (
                        <div className={styles.srDetail}>
                          Philippines, {r.year}: <strong>{r.value}</strong>
                          <span className={styles.srDetailSrc}>
                            {' '}
                            (World Development Indicators)
                          </span>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </>
        ) : (
          <>
            <pre className={styles.srTerminal}>
              {step === 0 ? sink(REQ1, typed) : REQ1}
              {step === 0 && <span className={styles.srCaret} aria-hidden="true" />}
            </pre>
            {step === 1 && (
              <div className={styles.srStatus}>Waiting for the server…</div>
            )}
            {step >= 2 && (
              <pre className={clsx(styles.srTerminal, styles.srResponse)}>
                {RESP1}
              </pre>
            )}
            {step >= 3 && (
              <pre className={styles.srTerminal}>
                {step === 3 ? sink(REQ2, typed) : REQ2}
                {step === 3 && (
                  <span className={styles.srCaret} aria-hidden="true" />
                )}
              </pre>
            )}
            {step === 4 && (
              <div className={styles.srStatus}>Waiting for the server…</div>
            )}
            {step === 5 && (
              <pre className={clsx(styles.srTerminal, styles.srResponse)}>
                {RESP2}
              </pre>
            )}
          </>
        )}
      </div>

      <div className={styles.srFoot}>
        <p className={styles.srNote}>
          {done
            ? mode === 'person'
              ? 'No title contains a word from the query. The top two are food insecurity indicators and the next two are related consumption measures. Select a result to see its Philippines value.'
              : 'The agent searches, then retrieves the value for the top result. Any MCP client can make these calls.'
            : ' '}
        </p>
        <div className={styles.srActions}>
          {mode === 'agent' && done && (
            <button type="button" className={styles.pcnReplay} onClick={handleCopy}>
              {copied ? 'Copied' : 'Copy'}
            </button>
          )}
          <button
            type="button"
            className={styles.pcnReplay}
            onClick={() => setRun((r) => r + 1)}>
            ↻ Replay
          </button>
        </div>
      </div>
      <span className={styles.pcnFine}>
        Ranking computed with avsolatorio/GIST-all-MiniLM-L6-v2 over the names
        of the 1,498 WDI indicators. Values are the latest available for the
        Philippines.
      </span>
    </div>
  );
}
