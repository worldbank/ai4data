import styles from './styles.module.css';

// Shown at the top of every chapter of a cookbook that has not cleared
// review. The index page carries the same statement in its own note.
export default function DraftBanner({entry}) {
  return (
    <aside className={styles.draftBanner} role="note" aria-label="Working draft">
      <strong>Working draft.</strong> Version {entry.version}, {entry.date}. This
      guide is working material of the AI for Data – Data for AI program. It
      does not represent official World Bank Group guidance or policy and is
      not legal advice; an organization applies it under its own statistical
      law, data protection rules, and institutional policies. Corrections are
      welcome as{' '}
      <a href="https://github.com/worldbank/ai4data/issues">GitHub issues</a>.
    </aside>
  );
}
