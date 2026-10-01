import {useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import {dimensions, pillars, workstreams} from '@site/src/content/program';
import styles from './styles.module.css';

function Row({w, showGsbpm}) {
  const body = (
    <>
      <span className={styles.rowTop}>
        <span className={styles.rowTitle}>{w.title}</span>
        {showGsbpm && w.gsbpm && (
          <span className={styles.gsbpm}>
            {/^\d/.test(w.gsbpm) ? `GSBPM ${w.gsbpm}` : w.gsbpm}
          </span>
        )}
      </span>
      <span className={styles.rowBody}>{w.description}</span>
    </>
  );
  return w.to ? (
    <Link className={clsx(styles.row, styles.rowLink)} to={w.to}>
      {body}
    </Link>
  ) : (
    <div className={styles.row}>{body}</div>
  );
}

const views = [
  {id: 'pillar', label: 'By pillar', tabs: pillars},
  {id: 'fair', label: 'By AI-ready dimension', tabs: dimensions},
];

export default function Program() {
  const [view, setView] = useState('pillar');
  const [tabs, setTabs] = useState({pillar: pillars[0].id, fair: dimensions[0].id});
  const [showGsbpm, setShowGsbpm] = useState(false);
  const current = views.find((v) => v.id === view);
  const tab = tabs[view];

  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">The Program</span>
          <Heading as="h2" className={styles.title}>
            AI for Data and Data for AI
          </Heading>
          <p className={styles.lede}>
            AI for Data applies AI to the production, curation, and
            dissemination of data. Data for AI prepares data for use by AI
            systems. Both aim at development data that AI can find, interpret,
            and verify. All methods, software, and guidance are developed as
            open resources.
          </p>
          <p className={styles.lede}>
            The program builds on the FAIR principles and extends them for AI
            systems as consumers of data. Browse the workstreams by pillar or
            by the AI-ready dimension they improve. The program also develops
            an{' '}
            <Link to="/ai-readiness-assessment">
              AI-readiness assessment framework
            </Link>{' '}
            for national statistical organizations.
          </p>
        </div>

        <div className={styles.viewBar}>
          <div className={styles.views} role="group" aria-label="Browse the workstreams">
            {views.map((v) => (
              <button
                type="button"
                key={v.id}
                aria-pressed={view === v.id}
                className={clsx(styles.view, view === v.id && styles.viewActive)}
                onClick={() => setView(v.id)}>
                {v.label}
              </button>
            ))}
          </div>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={showGsbpm}
              onChange={(e) => setShowGsbpm(e.target.checked)}
            />
            Show GSBPM tags
          </label>
        </div>

        <div className={styles.bar}>
          <div className={styles.tabs} role="tablist" aria-label={current.label}>
            {current.tabs.map((t) => (
              <button
                type="button"
                role="tab"
                key={t.id}
                aria-selected={tab === t.id}
                className={clsx(styles.tab, tab === t.id && styles.tabActive)}
                onClick={() => setTabs((prev) => ({...prev, [view]: t.id}))}>
                {t.label}
                {t.plus && <span className={styles.plus}>Added for AI</span>}
              </button>
            ))}
          </div>
        </div>

        {pillars.map((pillar) => (
          <div
            className={styles.pillar}
            key={pillar.id}
            role="tabpanel"
            hidden={view !== 'pillar' || tab !== pillar.id}>
            <p className={styles.pillarLede}>{pillar.lede}</p>
            <div className={styles.groups}>
              {pillar.groups.map((group) => (
                <div className={styles.group} key={group.label}>
                  <Heading as="h3" className={styles.groupLabel}>
                    {group.label}
                  </Heading>
                  <div className={styles.rows}>
                    {group.items.map((id) => (
                      <Row key={id} w={workstreams[id]} showGsbpm={showGsbpm} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}

        {dimensions.map((d) => (
          <div
            className={styles.pillar}
            key={d.id}
            role="tabpanel"
            hidden={view !== 'fair' || tab !== d.id}>
            <p className={styles.pillarLede}>
              <strong>{d.question}</strong> Attributes covered:{' '}
              {d.attributes.join(', ').toLowerCase()}.
            </p>
            <div className={clsx(styles.group, styles.groupWide)}>
              <div className={clsx(styles.rows, styles.rowsCols)}>
                {d.items.map((id) => (
                  <Row key={id} w={workstreams[id]} showGsbpm={showGsbpm} />
                ))}
              </div>
            </div>
          </div>
        ))}

        <p className={styles.note}>
          GSBPM tags show where a workstream applies in the Generic Statistical
          Business Process Model.{' '}
          <Link to="/docs/gsbpm-mapping">See the full mapping →</Link>
        </p>
      </div>
    </section>
  );
}
