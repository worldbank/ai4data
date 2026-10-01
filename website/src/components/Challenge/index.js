import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Gaps = [
  {
    label: 'Find',
    title: 'Metadata is incomplete or inconsistent',
    detail:
      'AI cannot reliably find relevant data when descriptions are missing, vague, or contradictory.',
  },
  {
    label: 'Interpret',
    title: 'Context is limited',
    detail:
      'Semantic context is thin and data models differ across organizations, so AI can misread what a number means.',
  },
  {
    label: 'Access',
    title: 'Data is locked in documents',
    detail:
      'Statistics sit in PDFs and spreadsheets, and are rarely available through machine-readable or AI-native interfaces.',
  },
  {
    label: 'Verify',
    title: 'Answers cannot be traced',
    detail:
      'AI answers rarely link back to the authoritative figure, so outdated or wrong values go unnoticed.',
  },
];

export default function Challenge() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">The Challenge</span>
          <Heading as="h2" className={styles.title}>
            AI is becoming how people reach development data
          </Heading>
          <p className={styles.lede}>
            People increasingly put questions about poverty, employment,
            climate, and food security to AI systems. Whether the answers draw
            on official statistics depends on whether AI can find, interpret,
            and verify the data. When it cannot, AI relies on secondary
            sources that may be outdated or wrong. Official statistics remain
            the most trusted evidence, and they lose influence when AI cannot
            reach them.
          </p>
        </div>
        <div className={styles.grid}>
          {Gaps.map((gap) => (
            <div className={styles.tile} key={gap.label}>
              <div className={styles.label}>{gap.label}</div>
              <Heading as="h3" className={styles.tileTitle}>
                {gap.title}
              </Heading>
              <p className={styles.detail}>{gap.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
