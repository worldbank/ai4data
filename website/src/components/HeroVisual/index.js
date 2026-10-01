import {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Each stage shows a real artifact from the examples further down the page:
// the quick-start metadata record, a WDI semantic search result, and a
// verified number from the PCN example.
const STEP_MS = 3200;

export default function HeroVisual() {
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (paused || reduce) {
      return undefined;
    }
    const t = setInterval(() => setActive((a) => (a + 1) % 3), STEP_MS);
    return () => clearInterval(t);
  }, [paused]);

  const fixed = active === 0;

  return (
    <div
      className={styles.wrap}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      aria-label="Three stages of the data lifecycle with example outputs">
      <div className={styles.rail} aria-hidden="true">
        <span
          className={styles.railFill}
          style={{height: `${(active / 2) * 100}%`}}
        />
      </div>

      <div className={styles.cards}>
        <button
          type="button"
          className={clsx(styles.card, active === 0 && styles.cardActive)}
          onClick={() => setActive(0)}>
          <span className={styles.step}>01</span>
          <span className={styles.kicker}>Metadata quality</span>
          <span className={styles.line}>
            <span className={styles.field}>name</span>
            GDP per capita (
            {fixed ? (
              <>
                <del className={styles.del}>curent</del>
                <ins className={styles.ins}>current</ins>
              </>
            ) : (
              <span className={styles.warn}>curent</span>
            )}{' '}
            US$)
          </span>
          <span className={clsx(styles.chip, fixed && styles.chipOn)}>
            {fixed ? 'Typo / Language fixed' : 'Typo / Language, severity 2/5'}
          </span>
        </button>

        <button
          type="button"
          className={clsx(styles.card, active === 1 && styles.cardActive)}
          onClick={() => setActive(1)}>
          <span className={styles.step}>02</span>
          <span className={styles.kicker}>Discovery</span>
          <span className={styles.search}>
            <span aria-hidden="true">&#8981;</span> how many people go hungry in the Philippines
          </span>
          <span className={styles.result}>
            <span className={styles.code}>SN.ITK.SVFI.ZS</span>
            <span className={styles.resultName}>
              Prevalence of severe food insecurity
            </span>
            <span className={styles.bar}>
              <span
                className={styles.barFill}
                style={{width: active === 1 ? '80%' : '0%'}}
              />
            </span>
          </span>
        </button>

        <button
          type="button"
          className={clsx(styles.card, active === 2 && styles.cardActive)}
          onClick={() => setActive(2)}>
          <span className={styles.step}>03</span>
          <span className={styles.kicker}>Trust</span>
          <span className={styles.line}>
            Philippines, 2023:{' '}
            <span className={clsx(styles.num, active === 2 && styles.numOk)}>
              3%
            </span>
            <span
              className={clsx(styles.check, active === 2 && styles.checkOn)}
              aria-hidden="true">
              ✓
            </span>
          </span>
          <span className={clsx(styles.chip, active === 2 && styles.chipOn)}>
            {active === 2
              ? 'Verified against the WDI record'
              : 'Awaiting verification'}
          </span>
        </button>
      </div>
    </div>
  );
}
