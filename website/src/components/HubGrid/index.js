import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import {AREAS} from '@site/src/content/tools';
import styles from './styles.module.css';

const ICONS = {
  assess: (
    <>
      <rect x="5" y="3.5" width="14" height="17" rx="2" />
      <path d="M9 3.5V2.5h6v1" />
      <path d="M8.5 12.5l2.5 2.5 4.5-5" />
    </>
  ),
  document: (
    <>
      <ellipse cx="12" cy="6" rx="7" ry="2.8" />
      <path d="M5 6v12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8V6" />
      <path d="M5 12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8" />
    </>
  ),
  connect: (
    <>
      <circle cx="12" cy="5" r="2.2" />
      <circle cx="5" cy="18" r="2.2" />
      <circle cx="19" cy="18" r="2.2" />
      <path d="M12 7.2V12M12 12l-5.5 4M12 12l5.5 4" />
    </>
  ),
  verify: (
    <>
      <path d="M12 3l7 2.5v5.5c0 4.5-3 8.3-7 9.5-4-1.2-7-5-7-9.5V5.5L12 3z" />
      <path d="M9 12l2.2 2.2L15.5 10" />
    </>
  ),
  implement: (
    <>
      <path d="M3 5.5h6.5a2.5 2.5 0 0 1 2.5 2.5v11a2 2 0 0 0-2-2H3z" />
      <path d="M21 5.5h-6.5A2.5 2.5 0 0 0 12 8v11a2 2 0 0 1 2-2h7z" />
    </>
  ),
  adopt: (
    <>
      <path d="M12 3v11" />
      <path d="M8 10l4 4 4-4" />
      <path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15" />
    </>
  ),
};

export function AreaIcon({id, className}) {
  return (
    <svg
      className={clsx(styles.icon, className)}
      viewBox="0 0 24 24"
      width="22"
      height="22"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true">
      {ICONS[id]}
    </svg>
  );
}

// `hrefFor` maps an area id to a link target; on the hub page the tiles jump
// to the sections, elsewhere they open the hub page at the section.
export default function HubGrid({hrefFor = (id) => `/hub#${id}`, compact = false, headingLevel = 'h3'}) {
  return (
    <div className={clsx(styles.grid, compact && styles.gridCompact)}>
      {AREAS.map((a) => (
        <Link key={a.id} className={styles.tile} to={hrefFor(a.id)}>
          <AreaIcon id={a.id} />
          <Heading as={headingLevel} className={styles.tileTitle}>
            {a.label}
          </Heading>
          <span className={styles.tileText}>{a.title}</span>
        </Link>
      ))}
    </div>
  );
}
