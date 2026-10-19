import {useEffect, useState} from 'react';
import styles from './styles.module.css';

// A checklist that remembers its ticks in the reader's browser. Nothing is
// sent anywhere; the state lives in localStorage under the given id.
export default function Checklist({id, items}) {
  const key = `cookbook-checklist-${id}`;
  const [done, setDone] = useState(() => new Set());
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw) {
        setDone(new Set(JSON.parse(raw)));
      }
    } catch (e) {
      // Storage may be unavailable; the list still works for the session.
    }
  }, [key]);

  const save = (next) => {
    setDone(next);
    try {
      window.localStorage.setItem(key, JSON.stringify([...next]));
    } catch (e) {
      // ignore
    }
  };

  const toggle = (item) => {
    const next = new Set(done);
    if (next.has(item)) {
      next.delete(item);
    } else {
      next.add(item);
    }
    save(next);
  };

  const copy = async () => {
    const text = items
      .map((item) => `- [${done.has(item) ? 'x' : ' '}] ${item}`)
      .join('\n');
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      // ignore
    }
  };

  const count = items.filter((i) => done.has(i)).length;

  return (
    <div className={styles.checklist}>
      <div className={styles.checklistBar}>
        <span className={styles.checklistCount}>
          {count} of {items.length} done
        </span>
        <span className={styles.checklistActions}>
          <button type="button" className={styles.smallButton} onClick={copy}>
            {copied ? 'Copied' : 'Copy as Markdown'}
          </button>
          <button
            type="button"
            className={styles.smallButton}
            onClick={() => save(new Set())}>
            Reset
          </button>
        </span>
      </div>
      <ul className={styles.checklistItems}>
        {items.map((item) => (
          <li key={item}>
            <label className={styles.checklistItem}>
              <input
                type="checkbox"
                checked={done.has(item)}
                onChange={() => toggle(item)}
              />
              <span className={done.has(item) ? styles.checklistDone : undefined}>
                {item}
              </span>
            </label>
          </li>
        ))}
      </ul>
      <p className={styles.checklistNote}>
        Ticks are saved in this browser only.
      </p>
    </div>
  );
}
