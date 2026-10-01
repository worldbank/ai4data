import {useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import {pillars, workstreams} from '@site/src/content/program';
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

export default function Program() {
  const [tab, setTab] = useState(pillars[0].id);
  const [showGsbpm, setShowGsbpm] = useState(false);
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
        </div>

        <div className={styles.bar}>
          <div className={styles.tabs} role="tablist" aria-label="Program pillar">
            {pillars.map((pillar) => (
              <button
                type="button"
                role="tab"
                key={pillar.id}
                id={`tab-${pillar.id}`}
                aria-selected={tab === pillar.id}
                aria-controls={`panel-${pillar.id}`}
                className={clsx(styles.tab, tab === pillar.id && styles.tabActive)}
                onClick={() => setTab(pillar.id)}>
                {pillar.label}
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

        {pillars.map((pillar) => (
          <div
            className={styles.pillar}
            key={pillar.id}
            role="tabpanel"
            id={`panel-${pillar.id}`}
            aria-labelledby={`tab-${pillar.id}`}
            hidden={tab !== pillar.id}>
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

        <p className={styles.note}>
          GSBPM tags show where a workstream applies in the Generic Statistical
          Business Process Model.{' '}
          <Link to="/docs/gsbpm-mapping">See the full mapping →</Link>
        </p>
      </div>
    </section>
  );
}
