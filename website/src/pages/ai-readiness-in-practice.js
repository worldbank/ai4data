import {Fragment, useEffect, useMemo, useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {
  FRAMEWORK_VERSION,
  coverageLabels,
  dimensions,
  resources,
  supportKinds,
} from '@site/src/content/readinessMap';
import investment from '@site/src/content/investment.json';
import {LEVEL_TO_GRADE, investmentNotes} from '@site/src/content/investmentNotes';
import {cookbooks} from '@site/src/content/cookbooks';
import {isHiddenPath} from '@site/src/content/visibility';
import styles from './ai-readiness-in-practice.module.css';

const CHAPTER_COUNT = cookbooks.reduce((n, c) => n + c.chapters, 0);

const PILLARS = {
  1: {label: 'Pillar I', name: 'Institutional readiness for AI adoption', tag: 'AI for Data'},
  2: {label: 'Pillar II', name: 'Readiness of data products for AI', tag: 'Data for AI'},
};

function hashToId() {
  if (typeof window === 'undefined') {
    return null;
  }
  const m = window.location.hash.match(/^#dim-(\d\.\d)$/);
  return m ? m[1] : null;
}

function Hero({counts}) {
  return (
    <header className={styles.hero}>
      <div className="container">
        <span className="eyebrow">AI-readiness assessment framework · operational companion</span>
        <Heading as="h1" className={styles.heroTitle}>
          Operationalizing the AI-readiness assessment
        </Heading>
        <p className={styles.heroLede}>
          The AI-readiness assessment framework scores a national statistical
          organization on twelve dimensions across two pillars. This page sets
          out, for each dimension, the program workstreams, open-source tools,
          reference implementations, and cookbook recipes that support
          progress to a higher maturity level. It also identifies the evidence
          that each resource produces for the assessment form.
        </p>
        <div className={styles.heroActions}>
          <Link className={clsx('button button--md', styles.primary)} to="/ai-readiness-assessment">
            Assessment framework
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/cookbook/ai-ready-dissemination/start-here">
            Cookbook self-assessment
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/modern-nso">
            A modern statistical office
          </Link>
        </div>
        <dl className={styles.facts}>
          <div>
            <dt>Dimensions</dt>
            <dd>12</dd>
          </div>
          <div>
            <dt>Dimension questions</dt>
            <dd>{counts.questions}</dd>
          </div>
          <div>
            <dt>Addressed by a program component</dt>
            <dd>{counts.covered}</dd>
          </div>
          <div>
            <dt>Cookbooks, chapters</dt>
            <dd>{cookbooks.length}, {CHAPTER_COUNT}</dd>
          </div>
        </dl>
      </div>
    </header>
  );
}

function HowItWorks() {
  const steps = [
    {
      n: '1',
      title: 'Assessment',
      body: 'Complete the assessment framework to obtain a maturity level (A to D) for each dimension. The cookbook provides a short self-assessment for a first orientation.',
      to: '/ai-readiness-assessment',
      cta: 'Assessment framework',
    },
    {
      n: '2',
      title: 'Mapping',
      body: 'For each dimension where the gap between the current and target levels is largest, consult its entry below. The entry lists the program resources that apply and the questions that remain the organization\'s own.',
      to: '#explorer',
      cta: 'Dimension entries',
    },
    {
      n: '3',
      title: 'Implementation',
      body: 'Apply the cookbook recipes and tools identified for the dimension. Each produces a document, dataset, or measurement that can be attached to the assessment form as evidence.',
      to: '/cookbook/ai-ready-dissemination/',
      cta: 'Cookbook',
    },
  ];
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Purpose and structure</span>
          <Heading as="h2" className={styles.title}>
            Using the assessment results
          </Heading>
        </div>
        <ol className={styles.steps}>
          {steps.map((s) => (
            <li key={s.n} className={styles.step}>
              <span className={styles.stepNum}>{s.n}</span>
              <Heading as="h3" className={styles.stepTitle}>
                {s.title}
              </Heading>
              <p>{s.body}</p>
              {s.to.startsWith('#') ? (
                <a href={s.to}>{s.cta} →</a>
              ) : (
                <Link to={s.to}>{s.cta} →</Link>
              )}
            </li>
          ))}
        </ol>
        <p className={styles.note}>
          The program's contribution is concentrated in Pillar II, the
          readiness of data products. In Pillar I it addresses specific
          questions: governance instruments (a policy, a register, a map of
          model-assisted tasks, a statement of model use), licences and
          terms, skills through the cookbooks and their reading paths,
          computing and security for model servers, privacy-preserving
          techniques, evaluation, costs, and partnerships. Strategy,
          oversight bodies, legal frameworks, connectivity, and funding
          structure remain the organization's own decisions. The matrix
          below records this.
        </p>
      </div>
    </section>
  );
}

function Matrix({onPick}) {
  return (
    <section className={clsx(styles.section, styles.band)}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Coverage by dimension</span>
          <Heading as="h2" className={styles.title}>
            Program support by assessment dimension
          </Heading>
          <p className={styles.lede}>
            Each cell indicates the extent of one kind of support for a
            dimension. Selecting a dimension opens its entry below.
          </p>
        </div>
        <div className={styles.matrixWrap}>
          <table className={styles.matrix}>
            <thead>
              <tr>
                <th scope="col">Dimension</th>
                {supportKinds.map((k) => (
                  <th scope="col" key={k.id}>
                    <span className={styles.kindFull}>{k.label}</span>
                    <span className={styles.kindShort}>{k.short}</span>
                  </th>
                ))}
                <th scope="col">Coverage</th>
              </tr>
            </thead>
            <tbody>
              {[1, 2].map((p) => (
                <Fragment key={p}>
                  <tr className={styles.matrixGroup}>
                    <th scope="rowgroup" colSpan={supportKinds.length + 2}>
                      {PILLARS[p].label} · {PILLARS[p].name}
                    </th>
                  </tr>
                  {dimensions
                    .filter((d) => d.pillar === p)
                    .map((d) => (
                      <tr key={d.id}>
                        <th scope="row">
                          <button type="button" className={styles.rowButton} onClick={() => onPick(d.id)}>
                            <span className={styles.dimId}>{d.id}</span> {d.name}
                          </button>
                        </th>
                        {supportKinds.map((k) => {
                          const v = d.support[k.id];
                          return (
                            <td key={k.id} className={styles.cell}>
                              <span
                                className={clsx(styles.dot, v === 2 && styles.dotFull, v === 1 && styles.dotHalf)}
                                aria-label={v === 2 ? 'substantial' : v === 1 ? 'some' : 'none'}
                                title={`${k.label}: ${v === 2 ? 'substantial' : v === 1 ? 'some' : 'none'}`}
                              />
                            </td>
                          );
                        })}
                        <td>
                          <span className={clsx(styles.badge, styles[`cov_${d.coverage}`])}>{d.coverage}</span>
                        </td>
                      </tr>
                    ))}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
        <p className={styles.legend}>
          <span className={clsx(styles.dot, styles.dotFull)} /> substantial&nbsp;&nbsp;
          <span className={clsx(styles.dot, styles.dotHalf)} /> some&nbsp;&nbsp;
          <span className={styles.dot} /> none
        </p>
      </div>
    </section>
  );
}

const SIZES = [
  {id: 'small', label: 'Small office'},
  {id: 'medium', label: 'Medium office'},
  {id: 'large', label: 'Large office'},
];

function effortLabel([lo, hi]) {
  if (hi === 0) {
    return 'under a day';
  }
  if (lo === hi) {
    return `about ${lo} person-day${lo === 1 ? '' : 's'}`;
  }
  return `${lo} to ${hi} person-days`;
}

function Investment({dim}) {
  const [size, setSize] = useState('medium');
  const [open, setOpen] = useState(null);
  const data = investment[dim];
  const notes = investmentNotes[dim];
  if (!data || !notes) {
    return null;
  }
  const levels = Object.keys(data);
  return (
    <div className={styles.block}>
      <Heading as="h4" className={styles.blockTitle}>
        What it takes{' '}
        <span className={styles.blockMeta}>roles and effort summed from the linked recipes; systems and hosting written per dimension</span>
      </Heading>
      <div className={styles.invControls}>
        <span className={styles.invHint}>Hosting shown for</span>
        <div className={styles.sizeSwitch} role="group" aria-label="Office size">
          {SIZES.map((s) => (
            <button key={s.id} type="button" className={clsx(styles.sizeItem, size === s.id && styles.sizeOn)} aria-pressed={size === s.id} onClick={() => setSize(s.id)}>
              {s.label}
            </button>
          ))}
        </div>
      </div>
      <div className={styles.invWrap}>
        <table className={styles.inv}>
          <thead>
            <tr>
              <th scope="col">Step</th>
              <th scope="col">Who</th>
              <th scope="col">One-off effort</th>
              <th scope="col">Work that recurs</th>
              <th scope="col">Systems that have to exist</th>
            </tr>
          </thead>
          <tbody>
            {levels.map((lv) => {
              const d = data[lv];
              const isOpen = open === lv;
              return (
                <Fragment key={lv}>
                  <tr>
                    <th scope="row">
                      <span className={styles.invGrade}>{LEVEL_TO_GRADE[lv]}</span>
                      <span className={styles.invLevel}>{lv} recipes</span>
                    </th>
                    <td>{d.roles.slice(0, 4).join(', ')}{d.roles.length > 4 ? `, and ${d.roles.length - 4} more` : ''}</td>
                    <td>
                      {effortLabel(d.effort_days)}
                      <button type="button" className={styles.invMore} onClick={() => setOpen(isOpen ? null : lv)} aria-expanded={isOpen}>
                        {isOpen ? 'hide' : 'show'} the {d.recipes.length} recipes
                      </button>
                    </td>
                    <td>
                      {d.per_item.length > 0 && <span className={styles.invLine}>Per item: {d.per_item.slice(0, 3).join('; ').toLowerCase()}</span>}
                      {d.per_period.length > 0 && <span className={styles.invLine}>Per period: {d.per_period.slice(0, 3).join('; ').toLowerCase()}</span>}
                      {d.per_item.length === 0 && d.per_period.length === 0 && '—'}
                    </td>
                    <td>
                      <ul className={styles.invList}>
                        {(notes.systems[lv] || []).map((x) => (
                          <li key={x}>{x}</li>
                        ))}
                      </ul>
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className={styles.invRecipes}>
                      <td colSpan={5}>
                        <ul className={styles.invList}>
                          {d.recipes.filter((r) => !isHiddenPath(`/cookbook/${r.chapter}`)).map((r) => (
                            <li key={r.chapter + r.title}>
                              <Link to={`/cookbook/${r.chapter}`}>{r.title}</Link>
                              <span className={styles.resNote}>{r.skills} · {r.time}</span>
                            </li>
                          ))}
                        </ul>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <dl className={styles.invFacts}>
        <div>
          <dt>Hardware and hosting, {SIZES.find((s) => s.id === size).label.toLowerCase()}</dt>
          <dd>{notes.hosting[size]}</dd>
        </div>
        <div>
          <dt>Kinds of cost</dt>
          <dd>
            <ul className={styles.invList}>
              {notes.money.map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
          </dd>
        </div>
        <div>
          <dt>Depends on</dt>
          <dd>
            {notes.dependsOn.length === 0
              ? 'No other dimension; this is where the data side starts.'
              : notes.dependsOn.map((x, i) => (
                  <Fragment key={x}>
                    {i > 0 && ', '}
                    <a href={`#dim-${x}`}>{x} {dimensions.find((q) => q.id === x).name}</a>
                  </Fragment>
                ))}
          </dd>
        </div>
      </dl>
      <p className={styles.fine}>
        Effort is the sum of the time lines of the recipes the dimension links, written for a first implementation on the running examples; a large catalog multiplies the per-item work. Money is given as kinds of cost because prices differ too much between countries and years for a figure to hold. Pillar I has no such table: its investments are institutional decisions the program does not cost.
      </p>
    </div>
  );
}

function Explorer({selected, setSelected}) {
  const d = dimensions.find((x) => x.id === selected);
  const pillar = PILLARS[d.pillar];
  const byKind = useMemo(() => {
    const groups = {};
    d.resources.forEach((key) => {
      const r = resources[key];
      if (isHiddenPath(r.to)) {
        return;
      }
      (groups[r.kind] = groups[r.kind] || []).push(r);
    });
    return groups;
  }, [d]);
  const covered = d.questions.filter((q) => q.covered).length;

  return (
    <section className={styles.section} id="explorer">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Dimension entries</span>
          <Heading as="h2" className={styles.title}>
            Program resources and evidence for each dimension
          </Heading>
        </div>
        <div className={styles.explorer}>
          <nav className={styles.rail} aria-label="Dimensions">
            {[1, 2].map((p) => (
              <div key={p} className={styles.railGroup}>
                <span className={styles.railLabel}>
                  {PILLARS[p].label} · {PILLARS[p].tag}
                </span>
                {dimensions
                  .filter((x) => x.pillar === p)
                  .map((x) => (
                    <button
                      type="button"
                      key={x.id}
                      className={clsx(styles.railItem, x.id === selected && styles.railItemOn)}
                      aria-current={x.id === selected ? 'true' : undefined}
                      onClick={() => setSelected(x.id)}>
                      <span className={styles.dimId}>{x.id}</span>
                      <span>{x.short}</span>
                      <span className={clsx(styles.miniDot, styles[`cov_${x.coverage}`])} aria-hidden="true" />
                    </button>
                  ))}
              </div>
            ))}
          </nav>

          <article className={styles.panel} key={d.id}>
            <header className={styles.panelHead}>
              <span className={styles.panelKicker}>
                {pillar.label} · {pillar.name}
              </span>
              <Heading as="h3" className={styles.panelTitle}>
                <span className={styles.dimIdBig}>{d.id}</span> {d.name}
              </Heading>
              <span className={clsx(styles.badge, styles[`cov_${d.coverage}`])}>{coverageLabels[d.coverage]}</span>
            </header>

            <p className={styles.panelSummary}>{d.summary}</p>

            <div className={styles.block}>
              <Heading as="h4" className={styles.blockTitle}>
                Questions in this dimension{' '}
                <span className={styles.blockMeta}>
                  {covered} of {d.questions.length} addressed by a program component
                </span>
              </Heading>
              <ul className={styles.questions}>
                {d.questions.map((q) => (
                  <li key={q.id} className={clsx(styles.question, q.covered && styles.questionOn)}>
                    <span className={styles.qId}>{q.id}</span>
                    <span>{q.topic}</span>
                  </li>
                ))}
              </ul>
            </div>

            {d.resources.length > 0 && (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Program resources
                </Heading>
                <div className={styles.kinds}>
                  {supportKinds
                    .filter((k) => byKind[k.id])
                    .map((k) => (
                      <div key={k.id} className={styles.kind}>
                        <span className={styles.kindLabel}>{k.label}</span>
                        <ul className={styles.resList}>
                          {byKind[k.id].map((r) => (
                            <li key={r.title}>
                              {r.to ? <Link to={r.to}>{r.title}</Link> : <strong>{r.title}</strong>}
                              <span className={styles.resNote}>{r.note}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {d.pillar === 2 && <Investment dim={d.id} />}

            {d.cookbook.some((c) => !isHiddenPath(c.to)) && (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Relevant cookbook chapters
                </Heading>
                <ul className={styles.cbList}>
                  {d.cookbook.filter((c) => !isHiddenPath(c.to)).map((c) => (
                    <li key={c.to + c.label}>
                      <Link to={c.to}>{c.label}</Link>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {d.steps ? (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Progression between maturity levels
                </Heading>
                <ol className={styles.levels}>
                  {d.steps.map((s) => (
                    <li key={s.from} className={styles.level}>
                      <span className={styles.levelTag}>
                        {s.from} <span aria-hidden="true">→</span> {s.to}
                      </span>
                      <p>{s.text}</p>
                    </li>
                  ))}
                </ol>
                <p className={styles.fine}>
                  Levels A to D are the maturity scale of the framework. The
                  actions listed are those supported by program resources;
                  the framework's suggested actions for each transition also
                  apply.
                </p>
              </div>
            ) : (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Progression between maturity levels
                </Heading>
                <p className={styles.fine}>
                  The program has no component for this dimension. The
                  framework's suggested actions and the organization's own
                  planning apply.
                </p>
              </div>
            )}

            {d.evidence.length > 0 && (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Evidence for the assessment form
                </Heading>
                <ul className={styles.evidence}>
                  {d.evidence.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              </div>
            )}

            <footer className={styles.panelFoot}>
              <Link to={`/ai-readiness-assessment`}>Full dimension in the assessment framework →</Link>
              <span className={styles.permalink}>
                Link to this dimension: <code>#dim-{d.id}</code>
              </span>
            </footer>
          </article>
        </div>
      </div>
    </section>
  );
}

function Closing() {
  return (
    <section className={clsx(styles.section, styles.closing)}>
      <div className="container">
        <div className={styles.closingRow}>
          <div>
            <Heading as="h2" className={styles.closingTitle}>
              Recommended sequence
            </Heading>
            <p className={styles.closingText}>
              The framework's pathway for early-stage organizations places
              data quality control, metadata standards, a searchable catalog,
              and API access under a clear licence first. These correspond to
              dimensions 2.2, 2.1, 2.3, 2.4, and 2.6 on this page and to
              chapters 1 to 3 and 5 of the cookbook. Agentic and generative AI
              access (dimension 2.5) follows.
            </p>
          </div>
          <div className={styles.closingActions}>
            <Link className={clsx('button button--md', styles.primary)} to="/cookbook/ai-ready-dissemination/">
              Cookbook
            </Link>
            <Link className={clsx('button button--md', styles.secondaryOnDark)} to="mailto:ai4data@worldbank.org">
              Contact the program
            </Link>
          </div>
        </div>
        <p className={styles.version}>
          Question numbering follows the {FRAMEWORK_VERSION}. The mapping is
          maintained with the site; corrections are welcome as{' '}
          <Link to="https://github.com/worldbank/ai4data/issues">GitHub issues</Link>.
        </p>
      </div>
    </section>
  );
}


export default function InPractice() {
  const [selected, setSelected] = useState('2.1');
  const counts = useMemo(() => {
    const all = dimensions.flatMap((d) => d.questions);
    return {questions: all.length, covered: all.filter((q) => q.covered).length};
  }, []);

  useEffect(() => {
    const fromHash = hashToId();
    if (fromHash && dimensions.some((d) => d.id === fromHash)) {
      setSelected(fromHash);
      document.getElementById('explorer')?.scrollIntoView({block: 'start'});
    }
  }, []);

  const pick = (id) => {
    setSelected(id);
    if (typeof window !== 'undefined') {
      window.history.replaceState(null, '', `#dim-${id}`);
      document.getElementById('explorer')?.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  };

  return (
    <Layout
      title="Operationalizing the AI-readiness assessment"
      description="The program workstreams, tools, reference implementations, and cookbook recipes that support each dimension of the AI-readiness assessment framework, with the evidence each produces.">
      <Hero counts={counts} />
      <main>
        <HowItWorks />
        <Matrix onPick={pick} />
        <Explorer selected={selected} setSelected={pick} />
        <Closing />
      </main>
    </Layout>
  );
}
