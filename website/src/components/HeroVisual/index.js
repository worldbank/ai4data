import {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// A simple example that follows one indicator through three properties. The indicator name and the
// Philippines value come from the WDI API (SN.ITK.SVFI.ZS, 2023). The typo in
// the first step is an illustrative metadata issue.
const STEP_MS = [3400, 3400, 5200];

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
    const t = setTimeout(() => setActive((a) => (a + 1) % 3), STEP_MS[active]);
    return () => clearTimeout(t);
  }, [paused, active]);

  const verified = active === 2;

  return (
    <div
      className={styles.wrap}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      aria-label="A simple example with one indicator: clean metadata, findable by meaning, verifiable number">
      <div className={styles.header}>
        A simple example with one indicator
      </div>

      <div className={styles.body}>
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
            <span className={styles.title}>
              <span className={styles.step}>1</span>
              Clean metadata
            </span>
            <span className={styles.field}>name</span>
            <span className={styles.line}>
              Prevalence of severe food{' '}
              {active === 0 ? (
                <>
                  <del className={styles.del}>insecurty</del>
                  <ins className={styles.ins}>insecurity</ins>
                </>
              ) : (
                <span className={styles.plain}>insecurity</span>
              )}{' '}
              (%)
            </span>
            <span className={clsx(styles.chip, active === 0 && styles.chipOn)}>
              Typo corrected
            </span>
          </button>

          <button
            type="button"
            className={clsx(styles.card, active === 1 && styles.cardActive)}
            onClick={() => setActive(1)}>
            <span className={styles.title}>
              <span className={styles.step}>2</span>
              Findable by meaning
            </span>
            <span className={styles.search}>
              <span aria-hidden="true">&#8981;</span> how many people go hungry
              in the Philippines
            </span>
            <span className={styles.line}>
              Prevalence of severe food insecurity (%)
            </span>
            <span className={styles.code}>SN.ITK.SVFI.ZS</span>
          </button>

          <button
            type="button"
            className={clsx(styles.card, active === 2 && styles.cardActive)}
            onClick={() => setActive(2)}>
            <span className={styles.title}>
              <span className={styles.step}>3</span>
              Verifiable number
            </span>
            <span className={styles.line}>
              Philippines, 2023:{' '}
              <span className={clsx(styles.proof, verified && styles.proofOn)}>
                <span className={clsx(styles.num, verified && styles.numOk)}>
                  3%
                </span>
                <span
                  className={clsx(styles.check, verified && styles.checkOn)}
                  aria-hidden="true">
                  ✓
                </span>
                <span className={styles.popover} role="tooltip">
                  <span className={styles.popTitle}>Verified data</span>
                  <span className={styles.popGrid}>
                    <span>Indicator</span>
                    <span>Prevalence of severe food insecurity (%)</span>
                    <span>Country</span>
                    <span>Philippines</span>
                    <span>Date</span>
                    <span>2023</span>
                    <span>Source value</span>
                    <span>3</span>
                    <span>Source</span>
                    <span>World Development Indicators</span>
                    <span>Display rule</span>
                    <span>Match at 1 decimal place</span>
                  </span>
                </span>
              </span>
            </span>
            <span className={clsx(styles.chip, verified && styles.chipOn)}>
              {verified ? 'Verified against WDI' : 'Source: WDI'}
            </span>
            <span
              className={clsx(styles.hint, !verified && styles.hintHidden)}
              aria-hidden={!verified}>
              Hover over 3% to see the proof
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
