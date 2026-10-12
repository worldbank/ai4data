import {useEffect, useMemo, useRef, useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {LEVELS, LEVEL_INTRO, SIZES, dimensionsByUnit, releaseDay, resources, units} from '@site/src/content/modernNso';
import {isHiddenPath} from '@site/src/content/visibility';
import styles from './modern-nso.module.css';

const LEVEL_INDEX = Object.fromEntries(LEVELS.map((l, i) => [l.id, i]));

function levelFromHash() {
  if (typeof window === 'undefined') {
    return null;
  }
  const m = window.location.hash.match(/^#(foundational|ai-ready|ai-native)$/);
  return m ? m[1] : null;
}

function textAt(step, level) {
  const order = ['ai-native', 'ai-ready', 'foundational'];
  const from = order.indexOf(level);
  for (let i = from; i < order.length; i += 1) {
    if (step.text[order[i]]) {
      return step.text[order[i]];
    }
  }
  return '';
}

/* ---------- Controls ---------- */

function LevelSwitch({level, setLevel, compact}) {
  return (
    <div className={clsx(styles.switch, compact && styles.switchCompact)} role="group" aria-label="Maturity level">
      {LEVELS.map((l) => (
        <button
          key={l.id}
          type="button"
          className={clsx(styles.switchItem, level === l.id && styles.switchOn)}
          aria-pressed={level === l.id}
          onClick={() => setLevel(l.id)}>
          {compact ? l.short : l.label}
        </button>
      ))}
    </div>
  );
}

function AiToggle({aiOff, setAiOff}) {
  return (
    <label className={styles.toggle}>
      <input type="checkbox" checked={aiOff} onChange={(e) => setAiOff(e.target.checked)} />
      <span className={styles.toggleTrack} aria-hidden="true">
        <span className={styles.toggleKnob} />
      </span>
      <span>Switch the AI layer off</span>
    </label>
  );
}

/* ---------- Hero ---------- */

function Hero({level, setLevel, aiOff, setAiOff}) {
  const counts = useMemo(() => {
    const ai = units.reduce((n, u) => n + u.levels[level].ai.length, 0);
    const steps = releaseDay.filter((s) => LEVEL_INDEX[s.min] <= LEVEL_INDEX[level]).length;
    return {ai, steps};
  }, [level]);
  return (
    <header className={styles.hero}>
      <div className="container">
        <span className="eyebrow">A composite portrait · fictional office · illustrative figures</span>
        <Heading as="h1" className={styles.heroTitle}>
          A modern statistical office
        </Heading>
        <p className={styles.heroLede}>
          The assessment says where an office stands and the cookbooks say
          what to do next. This page shows the destination: one fictional
          national statistical organization, the example organization of
          the cookbooks, drawn at three maturity levels. Choose a level to
          redraw the office, open a unit to see who works there and with
          what, follow a release day hour by hour, and switch the AI layer
          off to see what still stands.
        </p>
        <div className={styles.heroControls}>
          <LevelSwitch level={level} setLevel={setLevel} />
          <AiToggle aiOff={aiOff} setAiOff={setAiOff} />
        </div>
        <p className={styles.levelIntro}>{LEVEL_INTRO[level]}</p>
        <dl className={styles.facts}>
          <div>
            <dt>Units</dt>
            <dd>{units.length}</dd>
          </div>
          <div>
            <dt>AI components at this level</dt>
            <dd>{aiOff ? 0 : counts.ai}</dd>
          </div>
          <div>
            <dt>Steps in a release day</dt>
            <dd>{counts.steps}</dd>
          </div>
          <div>
            <dt>Cookbooks behind it</dt>
            <dd>11</dd>
          </div>
        </dl>
      </div>
    </header>
  );
}

/* ---------- The office ---------- */

function UnitCard({unit, level, aiOff, selected, onSelect}) {
  const at = unit.levels[level];
  return (
    <button
      type="button"
      className={clsx(styles.unit, styles[`pos_${unit.position}`], selected && styles.unitOn)}
      aria-pressed={selected}
      onClick={() => onSelect(unit.id)}>
      <span className={styles.unitName}>{unit.name}</span>
      <span className={styles.unitTag}>{unit.tagline}</span>
      <span className={styles.chips}>
        {unit.people.slice(0, 3).map((p) => (
          <span key={p} className={styles.chip}>
            {p}
          </span>
        ))}
      </span>
      <span className={styles.chips}>
        {at.ai.length === 0 && <span className={styles.chipMuted}>No AI component at this level</span>}
        {at.ai.map((a) => (
          <span key={a.name} className={clsx(styles.chipAi, aiOff && styles.chipOff)}>
            {a.name}
          </span>
        ))}
      </span>
    </button>
  );
}

function UnitDetail({unit, level, aiOff, setLevel}) {
  const at = unit.levels[level];
  const nextLevel = LEVELS[LEVEL_INDEX[level] + 1];
  return (
    <div className={styles.detail} id="unit-detail">
      <div className={styles.detailHead}>
        <span className="eyebrow">{LEVELS[LEVEL_INDEX[level]].label} level</span>
        <Heading as="h3" className={styles.detailTitle}>
          {unit.name}
        </Heading>
        <p className={styles.detailDay}>{at.day}</p>
      </div>
      <div className={styles.detailGrid}>
        <div>
          <h4 className={styles.h4}>People and skills</h4>
          <ul className={styles.list}>
            {unit.people.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
          <h4 className={styles.h4}>On their screens</h4>
          <ul className={styles.list}>
            {at.tools.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className={styles.h4}>AI components and who decides</h4>
          {at.ai.length === 0 ? (
            <p className={styles.fine}>None at this level. The unit's work is documented, measured, and ready for a model to assist.</p>
          ) : (
            <ul className={styles.aiList}>
              {at.ai.map((a) => (
                <li key={a.name} className={clsx(aiOff && styles.aiOffItem)}>
                  <strong>{a.name}</strong> {a.role}.
                  <br />
                  <span className={styles.decision}>Decision: {a.decision}.</span>
                  {aiOff && (
                    <span className={styles.offNote}>
                      Switched off: {a.off}.
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
          <h4 className={styles.h4}>Assessment dimensions this unit answers for</h4>
          <p className={styles.dims}>
            {unit.dims.map((d) => (
              <Link key={d} to={`/ai-readiness-in-practice#dim-${d}`} className={styles.dimLink}>
                {d} {dimensionsByUnit[d]}
              </Link>
            ))}
          </p>
        </div>
      </div>
      <div className={styles.detailFoot}>
        <div className={styles.links}>
          {at.links.filter((l) => !isHiddenPath(l.to)).map((l) => (
            <Link key={l.to} to={l.to}>
              {l.label} →
            </Link>
          ))}
        </div>
        {at.next && nextLevel && (
          <button type="button" className={styles.nextLevel} onClick={() => setLevel(nextLevel.id)}>
            <span className={styles.nextLabel}>What the {nextLevel.label} level adds</span>
            <span>{at.next}</span>
          </button>
        )}
      </div>
    </div>
  );
}

function Office({level, setLevel, aiOff}) {
  const [selected, setSelected] = useState('production');
  const unit = units.find((u) => u.id === selected);
  return (
    <section className={styles.section} id="office">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">The office</span>
          <Heading as="h2" className={styles.title}>
            The five units of the office
          </Heading>
          <p className={styles.lede}>
            Each unit opens to its people, its tools, and its AI components
            at the chosen level. Every component names the person who
            decides on its output.
          </p>
        </div>
        <div className={styles.floor}>
          {units.map((u) => (
            <UnitCard key={u.id} unit={u} level={level} aiOff={aiOff} selected={u.id === selected} onSelect={setSelected} />
          ))}
        </div>
        <UnitDetail unit={unit} level={level} aiOff={aiOff} setLevel={setLevel} />
      </div>
    </section>
  );
}

/* ---------- A release day ---------- */

const KINDS = {
  decision: 'A person decides',
  check: 'A check runs',
  ai: 'A model works',
  release: 'The release goes out',
};

function ReleaseDay({level, aiOff}) {
  const steps = useMemo(() => releaseDay.filter((s) => LEVEL_INDEX[s.min] <= LEVEL_INDEX[level]), [level]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [filter, setFilter] = useState('all');
  const timer = useRef(null);

  useEffect(() => {
    setIndex(0);
    setPlaying(false);
  }, [level]);

  useEffect(() => {
    if (!playing) {
      return undefined;
    }
    timer.current = setInterval(() => {
      setIndex((i) => {
        if (i >= steps.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 3200);
    return () => clearInterval(timer.current);
  }, [playing, steps.length]);

  const visible = steps.map((s, i) => ({...s, i})).filter((s) => filter === 'all' || s.kind === filter);
  const current = steps[index];
  const unitName = (id) => units.find((u) => u.id === id).short;

  return (
    <section className={clsx(styles.section, styles.band)} id="release-day">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">A release day</span>
          <Heading as="h2" className={styles.title}>
            One release day of the labour force survey
          </Heading>
          <p className={styles.lede}>
            One quarterly release passes through the five units. Step
            through it, or play it, and watch where a model works, where a
            check runs, and where a person decides.
          </p>
        </div>
        <div className={styles.dayControls}>
          <div className={styles.playGroup}>
            <button type="button" className={styles.ctrl} onClick={() => setIndex((i) => Math.max(0, i - 1))} disabled={index === 0}>
              ← Earlier
            </button>
            <button type="button" className={clsx(styles.ctrl, styles.ctrlPrimary)} onClick={() => setPlaying((p) => !p)}>
              {playing ? 'Pause' : index >= steps.length - 1 ? 'Replay' : 'Play the day'}
            </button>
            <button type="button" className={styles.ctrl} onClick={() => setIndex((i) => Math.min(steps.length - 1, i + 1))} disabled={index >= steps.length - 1}>
              Later →
            </button>
          </div>
          <div className={styles.filters} role="group" aria-label="Show">
            {[['all', 'All steps'], ...Object.entries(KINDS)].map(([k, label]) => (
              <button key={k} type="button" className={clsx(styles.filter, filter === k && styles.filterOn)} aria-pressed={filter === k} onClick={() => setFilter(k)}>
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className={styles.day}>
          <ol className={styles.timeline}>
            {visible.map((s) => (
              <li key={s.time} className={clsx(styles.tick, s.i === index && styles.tickOn, s.i < index && styles.tickDone)}>
                <button type="button" className={styles.tickButton} onClick={() => { setIndex(s.i); setPlaying(false); }}>
                  <span className={styles.tickTime}>{s.time}</span>
                  <span className={styles.tickUnit}>{unitName(s.unit)}</span>
                  <span className={clsx(styles.kind, styles[`kind_${s.kind}`])}>{KINDS[s.kind]}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className={styles.now} aria-live="polite">
            <span className={styles.nowTime}>{current.time}</span>
            <span className={styles.nowMeta}>
              {unitName(current.unit)} · {current.actor}
            </span>
            <p className={styles.nowText}>{aiOff && (current.kind === 'ai' || current.off) ? current.off : textAt(current, level)}</p>
            {aiOff && <p className={styles.fine}>The AI layer is switched off. This is the fallback path the switch-off test proves.</p>}
            <Link to={current.link.to} className={styles.nowLink}>
              {current.link.label} →
            </Link>
            <div className={styles.progress} aria-hidden="true">
              <span style={{width: `${((index + 1) / steps.length) * 100}%`}} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------- Resources ---------- */

function Resources() {
  const [size, setSize] = useState('medium');
  const [tab, setTab] = useState('people');
  const table = resources[tab];
  const sizeIndex = SIZES.findIndex((s) => s.id === size);
  const hasSizes = table.columns.length === 5;
  return (
    <section className={styles.section} id="resources">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">What it takes</span>
          <Heading as="h2" className={styles.title}>
            People, compute, and tools at the AI-native level
          </Heading>
          <p className={styles.lede}>
            Illustrative ranges for three sizes of office. The skills column
            matters more than the counts, and the compute figures come from
            the sizing recipe of the small models cookbook.
          </p>
        </div>
        <div className={styles.resControls}>
          <div className={styles.tabs} role="tablist">
            {[['people', 'People and skills'], ['compute', 'Compute'], ['tools', 'Tool stack']].map(([k, label]) => (
              <button key={k} type="button" role="tab" aria-selected={tab === k} className={clsx(styles.tab, tab === k && styles.tabOn)} onClick={() => setTab(k)}>
                {label}
              </button>
            ))}
          </div>
          {hasSizes && (
            <div className={styles.switch} role="group" aria-label="Office size">
              {SIZES.map((s) => (
                <button key={s.id} type="button" className={clsx(styles.switchItem, size === s.id && styles.switchOn)} aria-pressed={size === s.id} onClick={() => setSize(s.id)}>
                  {s.label}
                  <span className={styles.switchSub}>{s.staff}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                {hasSizes ? (
                  <>
                    <th scope="col">{table.columns[0]}</th>
                    <th scope="col">{table.columns[1]}</th>
                    <th scope="col">{SIZES[sizeIndex].label}</th>
                  </>
                ) : (
                  table.columns.map((c) => (
                    <th scope="col" key={c}>
                      {c}
                    </th>
                  ))
                )}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((r) => (
                <tr key={r[0]}>
                  <th scope="row">{r[0]}</th>
                  <td>{r[1]}</td>
                  {hasSizes ? <td className={styles.figure}>{r[2 + sizeIndex]}</td> : <td>{r[2]}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={styles.note}>{table.note}</p>
      </div>
    </section>
  );
}

/* ---------- Pathway ---------- */

function Pathway() {
  return (
    <section className={clsx(styles.section, styles.band)} id="pathway">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">From here to there</span>
          <Heading as="h2" className={styles.title}>
            How an office gets to this picture
          </Heading>
        </div>
        <ol className={styles.steps}>
          <li className={styles.step}>
            <span className={styles.stepNum}>1</span>
            <Heading as="h3" className={styles.stepTitle}>
              Find the office in the picture
            </Heading>
            <p>Set the level switch to where the office is today. Each unit's entry names the assessment dimensions it answers for; the readiness profile says which gaps are largest.</p>
            <Link to="/ai-readiness-assessment">The assessment framework →</Link>
          </li>
          <li className={styles.step}>
            <span className={styles.stepNum}>2</span>
            <Heading as="h3" className={styles.stepTitle}>
              Read what the next level adds
            </Heading>
            <p>Every unit entry ends with what the next level adds. The operational companion lists, per dimension, the tools, recipes, and evidence that get there.</p>
            <Link to="/ai-readiness-in-practice">Operationalizing the assessment →</Link>
          </li>
          <li className={styles.step}>
            <span className={styles.stepNum}>3</span>
            <Heading as="h3" className={styles.stepTitle}>
              Build it recipe by recipe
            </Heading>
            <p>Each AI component on this page is a chapter of a cookbook, with scripts, a running example, and the check that gates it. The cookbooks are also the training path for the roles in the resources table.</p>
            <Link to="/cookbook/">The cookbooks →</Link>
          </li>
        </ol>
        <p className={styles.note}>
          The office on this page is a composite. No single organization
          runs every component shown, the figures are ranges for planning,
          and the people are roles. What is exact is the set of controls:
          every AI component has a decision point, a suite, a version, and a
          fallback, and the data layer stands without the AI layer.
        </p>
      </div>
    </section>
  );
}

/* ---------- Page ---------- */

export default function ModernNso() {
  const [level, setLevelState] = useState('ai-native');
  const [aiOff, setAiOff] = useState(false);

  useEffect(() => {
    const fromHash = levelFromHash();
    if (fromHash) {
      setLevelState(fromHash);
    }
  }, []);

  const setLevel = (id) => {
    setLevelState(id);
    if (typeof window !== 'undefined') {
      window.history.replaceState(null, '', `#${id}`);
    }
  };

  return (
    <Layout
      title="A modern statistical office"
      description="A composite portrait of an AI-ready national statistical organization at three maturity levels: its units, people, tools, AI components, a release day hour by hour, and the people, compute, and tools it takes.">
      <main className={clsx(styles.page, aiOff && styles.pageOff)}>
        <Hero level={level} setLevel={setLevel} aiOff={aiOff} setAiOff={setAiOff} />
        <div className={styles.sticky}>
          <div className="container">
            <LevelSwitch level={level} setLevel={setLevel} compact />
            <AiToggle aiOff={aiOff} setAiOff={setAiOff} />
          </div>
        </div>
        <Office level={level} setLevel={setLevel} aiOff={aiOff} />
        <ReleaseDay level={level} aiOff={aiOff} />
        <Resources />
        <Pathway />
      </main>
    </Layout>
  );
}
