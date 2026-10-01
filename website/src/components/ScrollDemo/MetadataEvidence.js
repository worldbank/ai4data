import {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Worked example from docs/metadata-reviewer/user-manual/quick-start.md. The
// record and the final output are taken from that page; the candidate lists
// at the early stages are a simplified illustration of the same run.
const Stages = [
  {
    name: 'Primary',
    role: 'Detect',
    note: 'Independent first pass over the raw record. Returns candidate issues with the current and suggested values.',
  },
  {
    name: 'Secondary',
    role: 'Re-scan',
    note: 'Independent re-scan that does not see the primary’s output. It adds a candidate the first pass missed.',
  },
  {
    name: 'Critic',
    role: 'Filter',
    note: 'Removes issues that match the exclusion rules: excluded fields and style-only differences.',
  },
  {
    name: 'Categorizer',
    role: 'Classify',
    note: 'Assigns one of six categories to each remaining issue. It adds and removes nothing.',
  },
  {
    name: 'Severity Scorer',
    role: 'Rank',
    note: 'Scores each issue by impact from 1 to 5 and ranks the list.',
  },
];

const RECORD = [
  {key: 'name', value: 'GDP per capita (curent US$)'},
  {key: 'measurement_unit', value: 'Constant 2015 US$'},
  {key: 'time_coverage', value: '1960-2023'},
  {key: 'idno', value: 'WB_WDI_NY.GDP.PCAP.CD'},
];

// `from`: first stage at which the candidate exists. `removedAt`: stage at
// which the critic drops it.
const ISSUES = [
  {
    id: 'typo',
    field: 'name',
    from: 1,
    text: 'Typo in series name: “curent” should be “current”.',
    category: 'Typo / Language',
    severity: 2,
  },
  {
    id: 'unit',
    field: 'measurement_unit',
    from: 1,
    text: 'Measurement unit contradicts the name and definition, which both state current US dollars.',
    category: 'Inconsistency / Conflict',
    severity: 4,
  },
  {
    id: 'date',
    field: 'time_coverage',
    from: 1,
    removedAt: 3,
    reason: 'Formatting or style only',
    text: 'Date format in time_coverage is unconventional.',
  },
  {
    id: 'idno',
    field: 'idno',
    from: 2,
    removedAt: 3,
    reason: 'Excluded field: idno',
    text: 'The identifier may be malformed.',
  },
];

const FINAL_JSON = `[
  {
    "detected_issue": "Measurement unit contradicts the name and definition...",
    "issue_category": "Inconsistency / Conflict",
    "issue_severity": 4,
    "current_metadata": {"series_description.measurement_unit": "Constant 2015 US$"},
    "suggested_metadata": {"series_description.measurement_unit": "Current US$"}
  },
  {
    "detected_issue": "Typo in series name: 'curent' should be 'current'.",
    "issue_category": "Typo / Language",
    "issue_severity": 2,
    "current_metadata": {"series_description.name": "GDP per capita (curent US$)"},
    "suggested_metadata": {"series_description.name": "GDP per capita (current US$)"}
  }
]`;

const STEP_MS = 1700;

export default function MetadataEvidence({active = true}) {
  const [stage, setStage] = useState(0); // 0 = not started, 1..5 = agent done
  const [playing, setPlaying] = useState(false);
  const [run, setRun] = useState(0);
  const [showJson, setShowJson] = useState(false);

  // Autoplay when the example becomes active or Replay is pressed.
  useEffect(() => {
    if (!active) {
      setStage(0);
      setPlaying(false);
      return undefined;
    }
    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      setStage(Stages.length);
      setPlaying(false);
      return undefined;
    }
    setStage(0);
    setShowJson(false);
    setPlaying(true);
    return undefined;
  }, [active, run]);

  useEffect(() => {
    if (!playing) {
      return undefined;
    }
    const t = setTimeout(() => {
      setStage((s) => {
        if (s + 1 >= Stages.length) {
          setPlaying(false);
        }
        return Math.min(s + 1, Stages.length);
      });
    }, stage === 0 ? 600 : STEP_MS);
    return () => clearTimeout(t);
  }, [playing, stage]);

  const goTo = (s) => {
    setPlaying(false);
    setStage(s);
  };

  const isRemoved = (i) => i.removedAt && stage >= i.removedAt;
  const visible = ISSUES.filter((i) => i.from <= stage);
  const ordered =
    stage >= 5
      ? [...visible].sort(
          (a, b) =>
            Number(!!isRemoved(a)) - Number(!!isRemoved(b)) ||
            (b.severity || 0) - (a.severity || 0),
        )
      : visible;
  const kept = visible.filter((i) => !isRemoved(i)).length;
  const fieldState = (key) => {
    const hits = visible.filter((i) => i.field === key);
    if (!hits.length) return null;
    return hits.every(isRemoved) ? 'dropped' : 'flagged';
  };

  return (
    <div className={clsx(styles.evidenceBox, styles.mrRoot)}>
      <div className={styles.mrSteps} role="group" aria-label="Pipeline stage">
        {Stages.map((s, i) => (
          <button
            type="button"
            key={s.name}
            aria-pressed={stage === i + 1}
            className={clsx(
              styles.mrStep,
              stage >= i + 1 && styles.mrStepDone,
              stage === i + 1 && styles.mrStepActive,
            )}
            title={`${s.role}: ${s.name}`}
            onClick={() => goTo(i + 1)}>
            <span className={styles.mrStepIndex}>{i + 1}</span>
            <span className={styles.mrStepText}>
              <span className={styles.mrStepName}>{s.name}</span>
              <span className={styles.mrStepRole}>{s.role}</span>
            </span>
          </button>
        ))}
      </div>

      <div className={styles.mrNote} aria-live="polite">
        {stage === 0
          ? 'A metadata record enters the pipeline. Select a stage or press Run.'
          : Stages[stage - 1].note}
      </div>

      <div className={styles.mrColumns}>
        <div>
          <span className={styles.mrLabel}>Metadata record</span>
          <dl className={styles.mrRecord}>
            {RECORD.map((r) => (
              <div
                key={r.key}
                className={clsx(
                  styles.mrField,
                  fieldState(r.key) && styles[`mrField_${fieldState(r.key)}`],
                )}>
                <dt>{r.key}</dt>
                <dd>{r.value}</dd>
              </div>
            ))}
          </dl>
          <div className={styles.mrFoot}>
            <button
              type="button"
              className={styles.pcnReplay}
              onClick={() => setRun((r) => r + 1)}>
              {stage === 0 && !playing ? '▶ Run' : '↻ Replay'}
            </button>
            {stage >= 5 && (
              <button
                type="button"
                className={styles.pcnReplay}
                onClick={() => setShowJson(true)}>
                {'{ }'} JSON output
              </button>
            )}
          </div>
          <span className={styles.pcnFine}>
            Illustrative run on the record from the quick-start guide.
          </span>
        </div>

        <div>
          <span className={styles.mrLabel}>
            Issues{' '}
            <span className={styles.mrCount}>
              {stage === 0 ? '' : `${visible.length} found, ${kept} kept`}
            </span>
          </span>
          <ul className={styles.mrIssues}>
            {stage === 0 && (
              <li className={styles.mrEmpty}>Waiting for the first agent.</li>
            )}
            {ordered.map((i) => {
              const removed = isRemoved(i);
              return (
                <li
                  key={i.id}
                  className={clsx(styles.mrIssue, removed && styles.mrIssueGone)}>
                  <span className={styles.mrIssueText}>{i.text}</span>
                  <span className={styles.mrChips}>
                    {removed && (
                      <span className={styles.mrChipOut}>{i.reason}</span>
                    )}
                    {!removed && stage >= 4 && (
                      <span className={styles.mrChip}>{i.category}</span>
                    )}
                    {!removed && stage >= 5 && (
                      <span
                        className={clsx(
                          styles.mrSeverity,
                          i.severity >= 4 && styles.mrSeverityHigh,
                        )}>
                        Severity {i.severity}/5
                      </span>
                    )}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      </div>

      {showJson && (
        <div className={styles.mrJson} role="dialog" aria-label="JSON output">
          <div className={styles.mrJsonHead}>
            <span className={styles.mrLabel}>JSON output</span>
            <button
              type="button"
              className={styles.pcnReplay}
              onClick={() => setShowJson(false)}>
              Close
            </button>
          </div>
          <pre className={styles.mrJsonBody}>{FINAL_JSON}</pre>
        </div>
      )}
    </div>
  );
}
