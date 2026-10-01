import clsx from 'clsx';
import Heading from '@theme/Heading';
import {dimensions, workstreams} from '@site/src/content/program';
import styles from './styles.module.css';

export default function AiReady() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">AI-Ready Data</span>
          <Heading as="h2" className={styles.title}>
            What AI-ready means
          </Heading>
          <p className={styles.lede}>
            The program builds on the FAIR principles and extends them for AI
            systems as consumers of data. Each dimension below lists the
            attributes it covers and the workstreams that improve it.
          </p>
        </div>

        <div className={styles.grid}>
          {dimensions.map((d) => (
            <div className={styles.card} key={d.id}>
              <div className={styles.cardHead}>
                <Heading as="h3" className={styles.cardTitle}>
                  {d.label}
                </Heading>
                {d.plus && <span className={styles.plus}>Added for AI</span>}
              </div>
              <p className={styles.question}>{d.question}</p>
              <p className={styles.attributes}>
                {d.attributes.join(' · ')}
              </p>
              <ul className={styles.chips}>
                {d.items.map((id) => {
                  const w = workstreams[id];
                  return (
                    <li
                      key={id}
                      className={clsx(
                        styles.chip,
                        w.status === 'development' && styles.chipDev,
                      )}>
                      {w.title}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
        <p className={styles.legend}>
          Dashed items are in development and not yet released.
        </p>
      </div>
    </section>
  );
}
