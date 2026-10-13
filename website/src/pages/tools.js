import {useMemo, useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {AREAS, IN_DEVELOPMENT, STATUSES, STATUS_ORDER, TOOLS} from '@site/src/content/tools';
import styles from './tools.module.css';

const AREA_BY_ID = Object.fromEntries(AREAS.map((a) => [a.id, a]));
const CHECKED = TOOLS.reduce((latest, t) => (t.checked > latest ? t.checked : latest), '');

function StatusChip({status}) {
  return <span className={clsx(styles.chip, styles[`st_${status}`])}>{STATUSES[status].short}</span>;
}

function Hero() {
  const counts = Object.fromEntries(STATUS_ORDER.map((s) => [s, TOOLS.filter((t) => t.status === s).length]));
  return (
    <header className={styles.hero}>
      <div className="container">
        <span className="eyebrow">Implementation hub · tools</span>
        <Heading as="h1" className={styles.heroTitle}>
          Tools catalogue
        </Heading>
        <p className={styles.heroLede}>
          This page lists the software and the methods of the program. Each
          entry records what the tool does, where the code is, who operates
          it, its status, the installation guide, the licence, and the known
          limits. The status of each entry was checked against its
          repository on{' '}
          {CHECKED}.
        </p>
        <div className={styles.heroActions}>
          <Link className={clsx('button button--md', styles.primary)} to="/hub">
            Implementation hub
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="#statuses">
            What the statuses mean
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/hub#adopt">
            Installation test
          </Link>
        </div>
        <dl className={styles.facts}>
          {STATUS_ORDER.filter((s) => counts[s] > 0).map((s) => (
            <div key={s}>
              <dt>{STATUSES[s].label}</dt>
              <dd>{counts[s]}</dd>
            </div>
          ))}
        </dl>
      </div>
    </header>
  );
}

function Statuses() {
  return (
    <section className={clsx(styles.section, styles.band)} id="statuses">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Definitions</span>
          <Heading as="h2" className={styles.title}>
            What the statuses mean
          </Heading>
          <p className={styles.lede}>
            A status describes the software and makes no statement about
            its adoption. A deployable tool may have no deployment outside
            the World Bank, and a method may be in daily use. A status
            changes when the evidence changes: a release, a running
            endpoint, a test suite, or a published package.
          </p>
        </div>
        <dl className={styles.statusGrid}>
          {STATUS_ORDER.map((s) => (
            <div key={s} className={styles.statusItem}>
              <dt>
                <StatusChip status={s} />
              </dt>
              <dd>{STATUSES[s].description}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

function Filters({area, status, onArea, onStatus}) {
  return (
    <div className={styles.filters}>
      <div className={styles.filterGroup} role="group" aria-label="Filter by area">
        <button type="button" className={clsx(styles.filter, area === 'all' && styles.filterOn)} onClick={() => onArea('all')}>
          All areas
        </button>
        {AREAS.filter((a) => TOOLS.some((t) => t.area === a.id)).map((a) => (
          <button
            key={a.id}
            type="button"
            className={clsx(styles.filter, area === a.id && styles.filterOn)}
            onClick={() => onArea(a.id)}>
            {a.label}
          </button>
        ))}
      </div>
      <div className={styles.filterGroup} role="group" aria-label="Filter by status">
        <button type="button" className={clsx(styles.filter, status === 'all' && styles.filterOn)} onClick={() => onStatus('all')}>
          All statuses
        </button>
        {STATUS_ORDER.filter((s) => TOOLS.some((t) => t.status === s)).map((s) => (
          <button
            key={s}
            type="button"
            className={clsx(styles.filter, status === s && styles.filterOn)}
            onClick={() => onStatus(s)}>
            {STATUSES[s].label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Summary({tools}) {
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Tool</th>
            <th scope="col">Area</th>
            <th scope="col">Status</th>
            <th scope="col">Kind</th>
            <th scope="col">Operator</th>
            <th scope="col">Links</th>
          </tr>
        </thead>
        <tbody>
          {tools.map((t) => (
            <tr key={t.id}>
              <th scope="row">
                <a href={`#${t.id}`}>{t.name}</a>
              </th>
              <td>{AREA_BY_ID[t.area].label}</td>
              <td>
                <StatusChip status={t.status} />
              </td>
              <td>{t.kind}</td>
              <td className={styles.operatorCell}>{t.operator ? t.operator.split('.')[0] : 'Not run as a service'}</td>
              <td className={styles.linksCell}>
                {t.docs && <Link to={t.docs}>Docs</Link>}
                {t.install && <Link to={t.install}>Install</Link>}
                {t.repo && <Link to={t.repo}>Code</Link>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Field({label, children}) {
  if (!children) {
    return null;
  }
  return (
    <div className={styles.field}>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function ToolCard({tool}) {
  return (
    <article className={styles.card} id={tool.id}>
      <div className={styles.cardHead}>
        <div>
          <span className="eyebrow">{AREA_BY_ID[tool.area].label} · {tool.kind}</span>
          <Heading as="h3" className={styles.cardTitle}>
            {tool.name}
          </Heading>
        </div>
        <StatusChip status={tool.status} />
      </div>
      <p className={styles.cardSummary}>{tool.summary}</p>
      <dl className={styles.fields}>
        <Field label="Stack">{tool.stack}</Field>
        <Field label="Licence">{tool.licence}</Field>
        <Field label="Operator">{tool.operator}</Field>
        {tool.endpoint && (
          <Field label="Public endpoint">
            <code>{tool.endpoint}</code>
          </Field>
        )}
        <Field label="Status evidence">{tool.verifies.join('; ')}.</Field>
        <Field label="Checked">{tool.checked}</Field>
      </dl>
      <div className={styles.cardLinks}>
        {tool.docs && (
          <Link className={clsx('button button--sm', styles.primary)} to={tool.docs}>
            Documentation
          </Link>
        )}
        {tool.install && (
          <Link className={clsx('button button--sm', styles.secondary)} to={tool.install}>
            Installation
          </Link>
        )}
        {tool.repo && (
          <Link className={clsx('button button--sm', styles.secondary)} to={tool.repo}>
            Repository
          </Link>
        )}
        {tool.releases && (
          <Link className={clsx('button button--sm', styles.secondary)} to={tool.releases}>
            Releases
          </Link>
        )}
      </div>
      <Heading as="h4" className={styles.limitsTitle}>
        Known limits
      </Heading>
      <ul className={styles.limits}>
        {tool.limits.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </article>
  );
}

function Catalogue() {
  const [area, setArea] = useState('all');
  const [status, setStatus] = useState('all');
  const tools = useMemo(
    () => TOOLS.filter((t) => (area === 'all' || t.area === area) && (status === 'all' || t.status === status)),
    [area, status],
  );
  return (
    <section className={styles.section} id="catalogue">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Catalogue</span>
          <Heading as="h2" className={styles.title}>
            Tools by area and status
          </Heading>
          <p className={styles.lede}>
            The table lists the entries. The cards below it hold the full
            record of each. The filters apply to both.
          </p>
        </div>
        <Filters area={area} status={status} onArea={setArea} onStatus={setStatus} />
        <Summary tools={tools} />
        <div className={styles.cards}>
          {tools.map((t) => (
            <ToolCard key={t.id} tool={t} />
          ))}
        </div>
        {tools.length === 0 && <p className={styles.empty}>No tool matches both filters.</p>}
      </div>
    </section>
  );
}

function InDevelopment() {
  return (
    <section className={clsx(styles.section, styles.band)} id="in-development">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Not yet available</span>
          <Heading as="h2" className={styles.title}>
            Work in development
          </Heading>
          <p className={styles.lede}>
            Workstreams the program has announced for which nothing can be
            installed or read yet. Each entry names the material that exists
            in its place.
          </p>
        </div>
        <ul className={styles.devList}>
          {IN_DEVELOPMENT.map((d) => (
            <li key={d.name}>
              <div className={styles.devTop}>
                <strong>{d.name}</strong>
                <span className={styles.devArea}>{AREA_BY_ID[d.area].label}</span>
                <StatusChip status="development" />
              </div>
              <p>{d.note}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

export default function Tools() {
  return (
    <Layout
      title="Tools catalogue"
      description="Every tool and method of the AI for Data – Data for AI program with its status, stack, licence, operator, installation guide, and known limits.">
      <Hero />
      <main>
        <Statuses />
        <Catalogue />
        <InDevelopment />
      </main>
    </Layout>
  );
}
