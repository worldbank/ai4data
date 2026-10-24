import React, {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// The worked-example documents of a training session, grouped by their role in the
// session. The list selects a document; the sheet shows its metadata and its chunks,
// each labelled with the citation the application writes for it. The selection is
// kept in the URL hash so that a document can be linked to.
export default function DocumentViewer({data}) {
  const ids = data.groups.flatMap((g) => g.documents.map((d) => d.id));
  const [current, setCurrent] = useState(ids[0]);

  useEffect(() => {
    const fromHash = () => {
      const id = window.location.hash.replace('#', '');
      if (ids.includes(id)) setCurrent(id);
    };
    fromHash();
    window.addEventListener('hashchange', fromHash);
    return () => window.removeEventListener('hashchange', fromHash);
  }, []);

  const select = (id) => {
    setCurrent(id);
    try {
      window.history.replaceState(null, '', `#${id}`);
    } catch (e) {}
  };

  return (
    <div className={styles.viewer}>
      <nav className={styles.list} aria-label="Documents">
        {data.groups.map((g) => (
          <section key={g.key}>
            <p className={styles.groupHeading}>
              <span className={clsx(styles.chip, styles[g.key])}>{g.label}</span>
            </p>
            <p className={styles.groupText}>{g.description}</p>
            <ul>
              {g.documents.map((d) => (
                <li key={d.id}>
                  <button
                    type="button"
                    className={styles.doc}
                    aria-current={d.id === current ? 'true' : 'false'}
                    onClick={() => select(d.id)}>
                    <span className={styles.docTitle}>{d.title}</span>
                    <span className={styles.docId}>
                      {d.id} · {d.year}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </nav>
      <div className={styles.sheets}>
        {data.groups.flatMap((g) =>
          g.documents.map((d) => (
            <article key={d.id} className={styles.sheet} hidden={d.id !== current}>
              <p className={styles.role}>
                <span className={clsx(styles.chip, styles[g.key])}>{g.label}</span> {d.note}
              </p>
              <dl className={styles.meta}>
                {d.metadata.map(([k, v]) => (
                  <div key={k} className={styles.row}>
                    <dt>{k}</dt>
                    <dd>{v}</dd>
                  </div>
                ))}
              </dl>
              <p className={styles.chunkNote}>
                Notebook 02 splits this document at its headings into {d.chunks.length}{' '}
                {d.chunks.length === 1 ? 'chunk' : 'chunks'}.
              </p>
              {d.chunks.map((c) => (
                <section key={c.citation} className={styles.chunk}>
                  <div className={styles.cite}>
                    <code>{c.citation}</code>
                    <span>{c.characters.toLocaleString('en-US')} characters</span>
                  </div>
                  <div className={styles.prose} dangerouslySetInnerHTML={{__html: c.html}} />
                </section>
              ))}
            </article>
          )),
        )}
      </div>
    </div>
  );
}
