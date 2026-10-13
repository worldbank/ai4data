import {Fragment, useEffect, useMemo, useState} from 'react';
import clsx from 'clsx';
import Head from '@docusaurus/Head';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {AREAS, CATEGORIES, EXCLUDED, GATES, LEVELS, PHASES, RISKS, ROLES} from '@site/src/content/nsoRoadmap';
import styles from './nso-roadmap.module.css';

const LEVEL_BY_ID = Object.fromEntries(LEVELS.map((l) => [l.id, l]));
const AREA_BY_ID = Object.fromEntries(AREAS.map((a) => [a.id, a]));
const CATEGORY_BY_ID = Object.fromEntries(CATEGORIES.map((c) => [c.id, c]));

function hashToArea() {
  if (typeof window === 'undefined') {
    return null;
  }
  const m = window.location.hash.match(/^#area-([a-z-]+)$/);
  return m && AREA_BY_ID[m[1]] ? m[1] : null;
}

function LevelBadge({level}) {
  const l = LEVEL_BY_ID[level];
  return <span className={clsx(styles.badge, styles[`lv_${level}`])}>{l.short}</span>;
}

function MaybeLink({to, children}) {
  if (!to) {
    return <span>{children}</span>;
  }
  return <Link to={to}>{children}</Link>;
}

/* ---------- Hero ---------- */

function Hero({counts}) {
  return (
    <header className={styles.hero}>
      <div className="container">
        <span className="eyebrow">Planning draft · unlisted page</span>
        <Heading as="h1" className={styles.heroTitle}>
          A roadmap for AI-readiness in a national statistical office
        </Heading>
        <p className={styles.heroLede}>
          This page sets out the components of an operational project that
          would support one national statistical office through the full
          implementation of AI-readiness: governance, people, infrastructure,
          metadata, access for AI systems, AI in production, and measurement.
          For each component it states what the office needs, what the project
          would finance, what the program provides today, and where the program
          could expand. The sequence follows the dependencies between the
          dimensions of the assessment framework.
        </p>
        <div className={styles.heroActions}>
          <Link className={clsx('button button--md', styles.primary)} to="#areas">
            Investment areas
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="#expansions">
            Expansion opportunities
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="#gates">
            Readiness gates
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/ai-readiness-in-practice">
            Operational companion
          </Link>
        </div>
        <dl className={styles.facts}>
          <div>
            <dt>Phases</dt>
            <dd>{PHASES.length}</dd>
          </div>
          <div>
            <dt>Investment areas</dt>
            <dd>{AREAS.length}</dd>
          </div>
          <div>
            <dt>Program contributions</dt>
            <dd>{counts.contributions}</dd>
          </div>
          <div>
            <dt>Expansion opportunities</dt>
            <dd>{counts.expansions}</dd>
          </div>
        </dl>
      </div>
    </header>
  );
}

/* ---------- Scope ---------- */

function Scope() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Scope</span>
          <Heading as="h2" className={styles.title}>
            The project and the program
          </Heading>
          <p className={styles.lede}>
            Two actors appear on this page. The project is the operational
            engagement that finances the office. The program is AI for Data -
            Data for AI, which provides the tools, methods, cookbooks, and
            advisory work that the project applies.
          </p>
        </div>
        <div className={styles.twoCol}>
          <div className={styles.panel}>
            <Heading as="h3" className={styles.panelTitle}>
              The project finances
            </Heading>
            <ul className={styles.list}>
              <li>Staff: recruitment, secondment, and retention of the roles the components need.</li>
              <li>Assets: servers, hosting, gateways, and the development environment.</li>
              <li>Training: three cycles that follow the components as they arrive.</li>
              <li>Legal work: review of the statistics act and drafting of licences and terms of use.</li>
              <li>Adaptation: the work of fitting program components to the office's languages, classifications, and systems.</li>
            </ul>
          </div>
          <div className={styles.panel}>
            <Heading as="h3" className={styles.panelTitle}>
              The program provides
            </Heading>
            <ul className={styles.list}>
              <li>The assessment framework that gives the project its components and its results indicators.</li>
              <li>Open-source tools at the status stated on the <Link to="/tools">tools page</Link>: the Metadata Editor, NADA, the Data360 MCP server, the chat application, Proof-Carrying Numbers, anomaly detection, the PI-FT toolkit, and the end-to-end reference implementation.</li>
              <li>The cookbook series as the method and the training material of each component.</li>
              <li>Advisory work during assessment, design, and the phase gates.</li>
              <li>Shared resources that the office joins at the end: the Global Question Bank, evaluation suites, and a peer network of offices.</li>
            </ul>
          </div>
        </div>
        <div className={styles.legendGrid}>
          {LEVELS.map((l) => (
            <div key={l.id} className={styles.legendItem}>
              <LevelBadge level={l.id} />
              <div>
                <strong>{l.label}</strong>
                <p>{l.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------- Phases ---------- */

function Phases() {
  return (
    <section className={clsx(styles.section, styles.band)} id="phases">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Sequence</span>
          <Heading as="h2" className={styles.title}>
            Four phases with entry gates
          </Heading>
          <p className={styles.lede}>
            The phases are ordered by what each one depends on. A phase opens
            when its gate is met, so the duration of each depends on the office.
            The framework's pathway for early-stage organizations gives the
            same order: quality control, metadata standards, a searchable
            catalog, and API access under a clear licence come before agentic
            and generative access.
          </p>
        </div>
        <ol className={styles.phases}>
          {PHASES.map((p) => (
            <li key={p.id} className={styles.phase} id={`phase-${p.n}`}>
              <div className={styles.phaseHead}>
                <span className={styles.phaseN}>{p.n}</span>
                <Heading as="h3" className={styles.phaseTitle}>
                  {p.title}
                </Heading>
              </div>
              <p className={styles.gate}>{p.gate}</p>
              <p className={styles.phaseSummary}>{p.summary}</p>
              <Heading as="h4" className={styles.blockTitle}>
                The project finances
              </Heading>
              <ul className={styles.list}>
                {p.project.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
              <Heading as="h4" className={styles.blockTitle}>
                The program provides
              </Heading>
              <ul className={styles.list}>
                {p.program.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

/* ---------- Readiness gates ---------- */

function Gates() {
  return (
    <section className={styles.section} id="gates">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">External adoption</span>
          <Heading as="h2" className={styles.title}>
            Readiness gates for inviting other institutions
          </Heading>
          <p className={styles.lede}>
            Three gates separate showing the tools from asking another
            institution to adopt them, and adoption from presenting an
            implementation. Each gate is a set of conditions with the
            evidence that shows it is met. The gates carry no dates; the
            program passes one when its conditions hold.
          </p>
        </div>
        <ol className={styles.phases}>
          {GATES.map((g) => (
            <li key={g.id} className={styles.phase} id={`gate-${g.n}`}>
              <div className={styles.phaseHead}>
                <span className={styles.phaseN}>{g.n}</span>
                <Heading as="h3" className={styles.phaseTitle}>
                  {g.title}
                </Heading>
              </div>
              <p className={styles.gate}>{g.sufficient}</p>
              <p className={styles.phaseSummary}>{g.summary}</p>
              <Heading as="h4" className={styles.blockTitle}>
                Conditions
              </Heading>
              <ul className={styles.list}>
                {g.conditions.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
              <Heading as="h4" className={styles.blockTitle}>
                Evidence
              </Heading>
              <ul className={styles.list}>
                {g.evidence.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
        <p className={styles.note}>
          A tool can pass the second gate while its country adoption is still
          at a pilot stage, and a pilot can succeed in one country without
          showing that another institution can deploy the tool alone. The
          gates keep the two claims apart. The status of each tool against
          the first two gates is on the <Link to="/tools">tools page</Link>.
        </p>
      </div>
    </section>
  );
}

/* ---------- Areas ---------- */

function Matrix({selected, onPick}) {
  return (
    <div className={styles.matrixWrap}>
      <table className={styles.matrix}>
        <thead>
          <tr>
            <th scope="col">Investment area</th>
            <th scope="col" className={styles.dimCol}>
              Dimensions
            </th>
            {PHASES.map((p) => (
              <th key={p.id} scope="col" className={styles.cell}>
                <span className={styles.phaseFull}>Phase {p.n}</span>
                <span className={styles.phaseShort}>{p.n}</span>
              </th>
            ))}
            <th scope="col" className={styles.levelCol}>
              Program contribution
            </th>
          </tr>
        </thead>
        <tbody>
          {CATEGORIES.map((c) => (
            <Fragment key={c.id}>
              <tr className={styles.matrixGroup}>
                <th scope="rowgroup" colSpan={PHASES.length + 3}>
                  {c.label}
                </th>
              </tr>
              {AREAS.filter((a) => a.category === c.id).map((a) => {
                const levels = Array.from(new Set(a.contributes.map((x) => x.level)));
                if (a.expansions.length) {
                  levels.push('expand');
                }
                return (
                  <tr key={a.id} className={clsx(selected === a.id && styles.rowOn)}>
                    <th scope="row">
                      <button type="button" className={styles.rowButton} onClick={() => onPick(a.id)}>
                        {a.title}
                      </button>
                    </th>
                    <td className={styles.dimCol}>
                      {a.dimensions.map((d) => (
                        <span key={d} className={styles.dimId}>
                          {d}
                        </span>
                      ))}
                    </td>
                    {PHASES.map((p) => (
                      <td key={p.id} className={styles.cell}>
                        <span
                          className={clsx(styles.dot, a.phases.includes(p.id) && styles.dotFull)}
                          aria-label={a.phases.includes(p.id) ? `Active in phase ${p.n}` : `Not active in phase ${p.n}`}
                        />
                      </td>
                    ))}
                    <td className={styles.levelCol}>
                      {Array.from(new Set(levels)).map((l) => (
                        <LevelBadge key={l} level={l} />
                      ))}
                    </td>
                  </tr>
                );
              })}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AreaDetail({area, onPick}) {
  const category = CATEGORY_BY_ID[area.category];
  return (
    <article className={styles.detail} id="area-detail" aria-live="polite">
      <div className={styles.detailHead}>
        <span className="eyebrow">{category.label}</span>
        <Heading as="h3" className={styles.detailTitle}>
          {area.title}
        </Heading>
        <p className={styles.detailMeta}>
          <span>Phases {area.phases.map((p) => p.replace('p', '')).join(', ')}</span>
          <span>
            Assessment dimensions{' '}
            {area.dimensions.map((d, i) => (
              <Fragment key={d}>
                {i > 0 && ', '}
                <Link to={`/ai-readiness-in-practice#dim-${d}`}>{d}</Link>
              </Fragment>
            ))}
          </span>
          {area.dependsOn.length > 0 && (
            <span>
              Depends on{' '}
              {area.dependsOn.map((id, i) => (
                <Fragment key={id}>
                  {i > 0 && ', '}
                  <button type="button" className={styles.inlineButton} onClick={() => onPick(id)}>
                    {AREA_BY_ID[id].title.toLowerCase()}
                  </button>
                </Fragment>
              ))}
            </span>
          )}
        </p>
      </div>
      <p className={styles.need}>{area.need}</p>
      <div className={styles.detailGrid}>
        <div>
          <Heading as="h4" className={styles.blockTitle}>
            The project finances
          </Heading>
          <ul className={styles.list}>
            {area.finances.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
        <div>
          <Heading as="h4" className={styles.blockTitle}>
            The program contributes
          </Heading>
          <ul className={styles.contribList}>
            {area.contributes.map((c) => (
              <li key={c.text}>
                <LevelBadge level={c.level} />
                <MaybeLink to={c.to}>{c.text}</MaybeLink>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <Heading as="h4" className={styles.blockTitle}>
            Expansion opportunities
          </Heading>
          <ul className={styles.list}>
            {area.expansions.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
        <div>
          <Heading as="h4" className={styles.blockTitle}>
            Evidence for the assessment
          </Heading>
          <ul className={styles.list}>
            {area.evidence.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
      </div>
    </article>
  );
}

function Areas({selected, onPick}) {
  const area = AREA_BY_ID[selected];
  return (
    <section className={styles.section} id="areas">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Components</span>
          <Heading as="h2" className={styles.title}>
            Investment areas by phase
          </Heading>
          <p className={styles.lede}>
            Each row is one component of the project. The dots show the phases
            in which it is active, and the badges show the kind of contribution
            the program makes. Select a row to read what the office needs, what
            the project finances, what the program contributes, where the
            program could expand, and the evidence the area produces for the
            assessment form.
          </p>
        </div>
        <Matrix selected={selected} onPick={onPick} />
        <AreaDetail area={area} onPick={onPick} />
      </div>
    </section>
  );
}

/* ---------- Expansions ---------- */

function Expansions() {
  return (
    <section className={clsx(styles.section, styles.band)} id="expansions">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Where the program could grow</span>
          <Heading as="h2" className={styles.title}>
            Expansion opportunities
          </Heading>
          <p className={styles.lede}>
            These are the components the program does not provide today and
            that a project of this kind would need. Each one is written so that
            it is built once for the first office and reused in later
            engagements. They are grouped by the investment area that needs
            them.
          </p>
        </div>
        <div className={styles.expGrid}>
          {CATEGORIES.map((c) => {
            const areas = AREAS.filter((a) => a.category === c.id && a.expansions.length);
            if (!areas.length) {
              return null;
            }
            return (
              <div key={c.id} className={styles.expGroup}>
                <Heading as="h3" className={styles.expGroupTitle}>
                  {c.label}
                </Heading>
                {areas.map((a) => (
                  <div key={a.id} className={styles.expArea}>
                    <Link className={styles.expAreaTitle} to={`#area-${a.id}`}>
                      {a.title}
                    </Link>
                    <ul className={styles.list}>
                      {a.expansions.map((t) => (
                        <li key={t}>{t}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ---------- Roles ---------- */

function Roles() {
  return (
    <section className={styles.section} id="roles">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">People</span>
          <Heading as="h2" className={styles.title}>
            Roles the project staffs
          </Heading>
          <p className={styles.lede}>
            The components need these roles inside the office. Curators,
            methodologists, and subject-matter staff usually exist. The
            engineering and data science roles are the ones the project most
            often has to recruit or second.
          </p>
        </div>
        <div className={styles.matrixWrap}>
          <table className={clsx(styles.matrix, styles.rolesTable)}>
            <thead>
              <tr>
                <th scope="col">Role</th>
                <th scope="col">Where it sits</th>
                <th scope="col">What it owns</th>
                <th scope="col">What the program gives it</th>
              </tr>
            </thead>
            <tbody>
              {ROLES.map((r) => (
                <tr key={r.role}>
                  <th scope="row">{r.role}</th>
                  <td>{r.where}</td>
                  <td>{r.work}</td>
                  <td>{r.program}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

/* ---------- Risks and exclusions ---------- */

function Risks() {
  return (
    <section className={clsx(styles.section, styles.band)} id="risks">
      <div className="container">
        <div className={styles.twoCol}>
          <div>
            <div className={styles.head}>
              <span className="eyebrow">Risks</span>
              <Heading as="h2" className={styles.title}>
                Risks and responses
              </Heading>
            </div>
            <dl className={styles.risks}>
              {RISKS.map((r) => (
                <div key={r.risk}>
                  <dt>{r.risk}</dt>
                  <dd>{r.response}</dd>
                </div>
              ))}
            </dl>
          </div>
          <div>
            <div className={styles.head}>
              <span className="eyebrow">Boundaries</span>
              <Heading as="h2" className={styles.title}>
                What this page leaves out
              </Heading>
            </div>
            <ul className={styles.list}>
              {EXCLUDED.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
            <p className={styles.note}>
              This page is a planning draft and is unlisted. It is linked from
              the site footer only, and it is excluded from search engines and
              the sitemap. The assessment framework and the operational
              companion remain the public reference for the dimensions it
              cites.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function NsoRoadmap() {
  const [selected, setSelected] = useState(AREAS[0].id);
  const counts = useMemo(
    () => ({
      contributions: AREAS.reduce((n, a) => n + a.contributes.length, 0),
      expansions: AREAS.reduce((n, a) => n + a.expansions.length, 0),
    }),
    [],
  );

  useEffect(() => {
    const fromHash = hashToArea();
    if (fromHash) {
      setSelected(fromHash);
      document.getElementById('areas')?.scrollIntoView({block: 'start'});
    }
    const onHash = () => {
      const id = hashToArea();
      if (id) {
        setSelected(id);
        document.getElementById('areas')?.scrollIntoView({behavior: 'smooth', block: 'start'});
      }
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const pick = (id) => {
    setSelected(id);
    if (typeof window !== 'undefined') {
      window.history.replaceState(null, '', `#area-${id}`);
      document.getElementById('area-detail')?.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  };

  return (
    <Layout
      title="Roadmap for AI-readiness in a statistical office"
      description="Planning draft of the components of an operational project that supports the full implementation of AI-readiness in a national statistical office, with the program's contribution in each.">
      <Head>
        <meta name="robots" content="noindex, nofollow" />
      </Head>
      <Hero counts={counts} />
      <main>
        <Scope />
        <Phases />
        <Gates />
        <Areas selected={selected} onPick={pick} />
        <Expansions />
        <Roles />
        <Risks />
      </main>
    </Layout>
  );
}
