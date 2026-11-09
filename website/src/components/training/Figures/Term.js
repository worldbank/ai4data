import React, {useEffect, useId, useRef, useState} from 'react';
import styles from './styles.module.css';

// A term in the text of a training page with its definition in a pop-up. The page
// generator writes <Term name="..." definition="...">text</Term> for each link to the
// page's "Terms" section; the definitions come from that section. The pop-up opens on
// hover, on keyboard focus, and on a tap, and moves left when it would leave the window.
export default function Term({name, definition, children}) {
  const [open, setOpen] = useState(false);
  const [shift, setShift] = useState(0);
  const root = useRef(null);
  const pop = useRef(null);
  const id = useId();

  useEffect(() => {
    if (!open) {
      setShift(0);
      return undefined;
    }
    if (pop.current) {
      const over = pop.current.getBoundingClientRect().right - (window.innerWidth - 12);
      if (over > 0) setShift(-over);
    }
    const close = (e) => {
      if (e.type === 'keydown' ? e.key === 'Escape' : !root.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('pointerdown', close);
    document.addEventListener('keydown', close);
    return () => {
      document.removeEventListener('pointerdown', close);
      document.removeEventListener('keydown', close);
    };
  }, [open]);

  return (
    <span ref={root} className={styles.term} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button type="button" className={styles.termButton} aria-expanded={open} aria-describedby={open ? id : undefined}
        onClick={() => setOpen(true)} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}>
        {children}
      </button>
      {open && (
        <span ref={pop} id={id} role="tooltip" className={styles.termPop} style={{transform: `translateX(${shift}px)`}}>
          <strong>{name}</strong>
          {definition}
        </span>
      )}
    </span>
  );
}
