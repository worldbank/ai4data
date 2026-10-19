import {useEffect, useRef, useState} from 'react';

// Shared stage driver for the example panels.
//
// A panel shows a sequence of stages, 0 (nothing has happened) to `count`
// (finished). The stage comes from one of three sources:
//
// - the scroll position inside the pinned stop (`scrub`, 0 to 1), when the
//   container supplies one; stages are spread over the first part of the stop
//   and the finished state holds for the rest;
// - timed playback, on Replay or, where there is no scroll driver (small
//   screens), when the panel becomes active;
// - a manual choice from the panel's own controls, which holds until the
//   reader scrolls on.
//
// Reduced motion shows the finished state at once.

const LEAD = 0.06; // fraction of the stop before the first stage
const SPAN = 0.74; // fraction of the stop over which the stages are spread
const MOVE = 0.04; // scroll movement that cancels playback or a manual choice

export function stageFromScrub(u, count) {
  if (u < LEAD) {
    return 0;
  }
  return Math.min(count, 1 + Math.floor((u - LEAD) / (SPAN / count)));
}

export function useReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) {
      return undefined;
    }
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const apply = () => setReduce(mq.matches);
    apply();
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, []);
  return reduce;
}

// Runs `fn(i)` for i in 1..n at the given delays (ms, cumulative from
// `offset`) and clears the timers on cleanup. `delays` is an array of n
// per-step delays or a single number used for every step.
export function schedule(n, delays, fn, offset = 0) {
  const timers = [];
  let t = offset;
  for (let i = 1; i <= n; i += 1) {
    t += Array.isArray(delays) ? delays[i - 1] : delays;
    timers.push(setTimeout(() => fn(i), t));
  }
  return () => timers.forEach(clearTimeout);
}

export function useStage({active, scrub, count, timings, run = 0, deps = []}) {
  const reduce = useReducedMotion();
  const scrubbing = typeof scrub === 'number';
  const [timed, setTimed] = useState(null); // stage during timed playback
  const [manual, setManual] = useState(null); // {stage, at}
  const timingsRef = useRef(timings);
  timingsRef.current = timings;
  const lastRun = useRef(run);
  const startedAt = useRef(0);

  // Timed playback: on Replay in every mode, and on activation when there
  // is no scroll driver.
  useEffect(() => {
    const runChanged = run !== lastRun.current;
    lastRun.current = run;
    setManual(null);
    if (!active || reduce) {
      setTimed(null);
      return undefined;
    }
    if (scrubbing && !runChanged) {
      setTimed(null);
      return undefined;
    }
    startedAt.current = scrubbing ? scrub : 0;
    setTimed(0);
    return schedule(count, timingsRef.current, setTimed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, run, scrubbing, reduce, count, ...deps]);

  // Scrolling on cancels playback and a manual choice.
  useEffect(() => {
    if (!scrubbing) {
      return;
    }
    if (timed !== null && Math.abs(scrub - startedAt.current) > MOVE) {
      setTimed(null);
    }
    if (manual && Math.abs(scrub - manual.at) > MOVE) {
      setManual(null);
    }
  }, [scrub, scrubbing, timed, manual]);

  let stage;
  if (!active) {
    stage = 0;
  } else if (reduce) {
    stage = count;
  } else if (timed !== null) {
    stage = timed;
  } else if (manual) {
    stage = manual.stage;
  } else if (scrubbing) {
    stage = stageFromScrub(scrub, count);
  } else {
    stage = 0;
  }
  stage = Math.min(stage, count);

  const setStage = (s) => {
    setTimed(null);
    setManual({stage: s, at: scrubbing ? scrub : 0});
  };

  return {
    stage,
    setStage,
    playing: timed !== null && timed < count,
    reduce,
    scrubbing,
  };
}
