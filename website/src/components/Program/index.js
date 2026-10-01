import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import {pillars, workstreams} from '@site/src/content/program';
import styles from './styles.module.css';

function Row({w}) {
  const body = (
    <>
      <span className={styles.rowTop}>
        <span className={styles.rowTitle}>{w.title}</span>
        {w.status === 'development' && (
          <span className={styles.dev}>In development</span>
        )}
        {w.gsbpm && (
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

        {pillars.map((pillar) => (
          <div className={styles.pillar} key={pillar.id}>
            <div className={styles.pillarHead}>
              <Heading as="h3" className={styles.pillarTitle}>
                {pillar.label}
              </Heading>
              <p className={styles.pillarLede}>{pillar.lede}</p>
            </div>
            <div className={styles.groups}>
              {pillar.groups.map((group) => (
                <div className={styles.group} key={group.label}>
                  <Heading as="h4" className={styles.groupLabel}>
                    {group.label}
                  </Heading>
                  <div className={styles.rows}>
                    {group.items.map((id) => (
                      <Row key={id} w={workstreams[id]} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}

        <p className={styles.note}>
          Items marked In development are planned or underway and not yet
          released. Tags such as GSBPM 6.3 show where a workstream applies in
          the Generic Statistical Business Process Model.{' '}
          <Link to="/docs/gsbpm-mapping">See the full mapping →</Link>
        </p>
      </div>
    </section>
  );
}
