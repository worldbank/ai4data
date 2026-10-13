import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import HubGrid, {AreaIcon} from '@site/src/components/HubGrid';
import {AREAS, STATUSES, TOOLS, toolById} from '@site/src/content/tools';
import {HUB, INSTALL_TEST, RELEASE_PRACTICE, SUPPORT} from '@site/src/content/hub';
import {cookbooks} from '@site/src/content/cookbooks';
import styles from './hub.module.css';

const COOKBOOK_BY_ID = Object.fromEntries(cookbooks.map((c) => [c.id, c]));

function StatusChip({status}) {
  const s = STATUSES[status];
  return <span className={clsx(styles.chip, styles[`st_${status}`])}>{s.short}</span>;
}

function Hero() {
  const counts = {
    tools: TOOLS.length,
    operation: TOOLS.filter((t) => t.status === 'operation').length,
    deployable: TOOLS.filter((t) => t.status === 'reference').length,
    cookbooks: cookbooks.length,
  };
  return (
    <header className={styles.hero}>
      <div className="container">
        <span className="eyebrow">Implementation hub</span>
        <Heading as="h1" className={styles.heroTitle}>
          AI readiness in practice
        </Heading>
        <p className={styles.heroLede}>
          One place for the program’s tools, guides, and cookbooks, organized
          by what an organization does with them: assess its readiness,
          document its data, connect it to software and AI agents, verify
          what is generated from it, implement the recipes, and adopt the
          tools. Every tool carries a status that says what the repository
          shows today.
        </p>
        <div className={styles.heroActions}>
          <Link className={clsx('button button--md', styles.primary)} to="/tools">
            Tools catalogue
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/docs/reference-implementation/">
            Reference implementation
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/ai-readiness-assessment">
            Assessment framework
          </Link>
        </div>
        <dl className={styles.facts}>
          <div>
            <dt>Tools and methods</dt>
            <dd>{counts.tools}</dd>
          </div>
          <div>
            <dt>In operation</dt>
            <dd>{counts.operation}</dd>
          </div>
          <div>
            <dt>Deployable</dt>
            <dd>{counts.deployable}</dd>
          </div>
          <div>
            <dt>Cookbooks</dt>
            <dd>{counts.cookbooks}</dd>
          </div>
        </dl>
      </div>
    </header>
  );
}

function Overview() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Six areas</span>
          <Heading as="h2" className={styles.title}>
            What the hub covers
          </Heading>
          <p className={styles.lede}>
            The areas follow the order of the work. An assessment names the
            gaps. Documentation, access, and verification close them. The
            cookbooks and the reference implementation show the steps, and
            the adoption area says what to expect from each tool.
          </p>
        </div>
        <HubGrid hrefFor={(id) => `#${id}`} />
      </div>
    </section>
  );
}

function ToolRow({tool}) {
  return (
    <li className={styles.toolRow}>
      <div className={styles.toolTop}>
        <Link className={styles.toolName} to={`/tools#${tool.id}`}>
          {tool.name}
        </Link>
        <StatusChip status={tool.status} />
      </div>
      <p className={styles.toolSummary}>{tool.summary}</p>
      <p className={styles.toolLinks}>
        {tool.docs && <Link to={tool.docs}>Documentation</Link>}
        {tool.install && <Link to={tool.install}>Installation</Link>}
        {tool.repo && <Link to={tool.repo}>Repository</Link>}
      </p>
    </li>
  );
}

function AreaSection({area, index}) {
  const content = HUB[area.id];
  const tools = content.tools.map(toolById).filter(Boolean);
  const books = content.cookbooks.map((id) => COOKBOOK_BY_ID[id]).filter(Boolean);
  return (
    <section className={clsx(styles.section, index % 2 === 1 && styles.band)} id={area.id}>
      <div className="container">
        <div className={styles.areaHead}>
          <AreaIcon id={area.id} className={styles.areaIcon} />
          <div>
            <span className="eyebrow">{area.label}</span>
            <Heading as="h2" className={styles.title}>
              {area.title}
            </Heading>
            <p className={styles.lede}>{area.summary}</p>
          </div>
        </div>
        <div className={styles.start}>
          <Heading as="h3" className={styles.blockTitle}>
            Where to start
          </Heading>
          <p>{content.start}</p>
          {content.dimensions.length > 0 && (
            <p className={styles.dims}>
              Assessment dimensions:{' '}
              {content.dimensions.map((d, i) => (
                <span key={d}>
                  {i > 0 && ', '}
                  <Link to={`/ai-readiness-in-practice#dim-${d}`}>{d}</Link>
                </span>
              ))}
            </p>
          )}
        </div>
        <div className={styles.columns}>
          {tools.length > 0 && (
            <div className={styles.column}>
              <Heading as="h3" className={styles.blockTitle}>
                Tools
              </Heading>
              <ul className={styles.toolList}>
                {tools.map((t) => (
                  <ToolRow key={t.id} tool={t} />
                ))}
              </ul>
            </div>
          )}
          <div className={styles.columnNarrow}>
            <Heading as="h3" className={styles.blockTitle}>
              Guides
            </Heading>
            <ul className={styles.guideList}>
              {content.guides.map((g) => (
                <li key={g.to}>
                  <Link to={g.to}>{g.title}</Link>
                  <span>{g.note}</span>
                </li>
              ))}
            </ul>
            {books.length > 0 && (
              <>
                <Heading as="h3" className={styles.blockTitle}>
                  Cookbooks
                </Heading>
                <ul className={styles.guideList}>
                  {books.map((c) => (
                    <li key={c.id}>
                      <Link to={c.to}>{c.title.replace('Practical Guide to ', '').replace('Practical Guide from ', 'From ')}</Link>
                      <span>{c.chapters} chapters</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function AdoptSection({area, index}) {
  const content = HUB[area.id];
  const books = content.cookbooks.map((id) => COOKBOOK_BY_ID[id]).filter(Boolean);
  return (
    <section className={clsx(styles.section, index % 2 === 1 && styles.band)} id={area.id}>
      <div className="container">
        <div className={styles.areaHead}>
          <AreaIcon id={area.id} className={styles.areaIcon} />
          <div>
            <span className="eyebrow">{area.label}</span>
            <Heading as="h2" className={styles.title}>
              {area.title}
            </Heading>
            <p className={styles.lede}>{area.summary}</p>
          </div>
        </div>
        <div className={styles.start}>
          <Heading as="h3" className={styles.blockTitle}>
            Where to start
          </Heading>
          <p>{content.start}</p>
        </div>
        <div className={styles.adoptGrid}>
          <div className={styles.panel}>
            <Heading as="h3" className={styles.panelTitle}>
              The installation test
            </Heading>
            <p className={styles.panelLede}>
              The test of adoption readiness is that a team outside the
              program can install a tool from its documentation. The record
              of each test, passed or failed, improves the guide for the
              next team.
            </p>
            <ol className={styles.steps}>
              {INSTALL_TEST.map((s) => (
                <li key={s.step}>{s.step}</li>
              ))}
            </ol>
          </div>
          <div className={styles.panel}>
            <Heading as="h3" className={styles.panelTitle}>
              Support boundaries
            </Heading>
            <Heading as="h4" className={styles.subTitle}>
              The program offers
            </Heading>
            <ul className={styles.list}>
              {SUPPORT.offers.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
            <Heading as="h4" className={styles.subTitle}>
              The program does not offer
            </Heading>
            <ul className={styles.list}>
              {SUPPORT.excludes.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </div>
        </div>
        <div className={styles.releaseWrap}>
          <Heading as="h3" className={styles.blockTitle}>
            Releases and versioning by repository
          </Heading>
          <table className={styles.releaseTable}>
            <thead>
              <tr>
                <th scope="col">Repository</th>
                <th scope="col">Practice</th>
              </tr>
            </thead>
            <tbody>
              {RELEASE_PRACTICE.map((r) => (
                <tr key={r.repo}>
                  <th scope="row">
                    <Link to={r.to}>{r.repo}</Link>
                  </th>
                  <td>{r.practice}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className={styles.columns}>
          <div className={styles.columnNarrow}>
            <Heading as="h3" className={styles.blockTitle}>
              Guides
            </Heading>
            <ul className={styles.guideList}>
              {content.guides.map((g) => (
                <li key={g.to}>
                  <Link to={g.to}>{g.title}</Link>
                  <span>{g.note}</span>
                </li>
              ))}
            </ul>
          </div>
          {books.length > 0 && (
            <div className={styles.columnNarrow}>
              <Heading as="h3" className={styles.blockTitle}>
                Cookbooks on governance and everyday use
              </Heading>
              <ul className={styles.guideList}>
                {books.map((c) => (
                  <li key={c.id}>
                    <Link to={c.to}>{c.title.replace('Practical Guide to ', '')}</Link>
                    <span>{c.chapters} chapters</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <p className={styles.note}>
          Organizations planning an adoption or a joint implementation can
          write to the Development Data Group at{' '}
          <a href="mailto:ai4data@worldbank.org">ai4data@worldbank.org</a>.
          Reports of installation tests and bugs go to the issue tracker of
          the tool’s repository, where the maintainers read them.
        </p>
      </div>
    </section>
  );
}

export default function Hub() {
  return (
    <Layout
      title="AI readiness in practice"
      description="The implementation hub of the AI for Data – Data for AI program: tools, guides, and cookbooks organized by what an organization does with them, each tool with a stated status.">
      <Hero />
      <main>
        <Overview />
        {AREAS.map((a, i) =>
          a.id === 'adopt' ? (
            <AdoptSection key={a.id} area={a} index={i} />
          ) : (
            <AreaSection key={a.id} area={a} index={i} />
          ),
        )}
      </main>
    </Layout>
  );
}
