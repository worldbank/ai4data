import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Cards = [
  {
    title: 'Issues',
    body: 'Report bugs or request features on GitHub.',
    to: 'https://github.com/worldbank/ai4data/issues',
  },
  {
    title: 'Documentation',
    body: 'Read the methodology, pipeline code, and API reference.',
    to: '/docs/introduction',
  },
  {
    title: 'Adoption',
    body: 'Tool statuses, the installation test, and support boundaries.',
    to: '/hub#adopt',
  },
  {
    title: 'Contact',
    body: 'Contact the Development Data Group.',
    to: 'mailto:ai4data@worldbank.org',
  },
];

export default function GetInvolved() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Participation</span>
          <Heading as="h2" className={styles.title}>
            Contributions and contact
          </Heading>
        </div>
        <div className={styles.grid}>
          {Cards.map((card) => (
            <Link className={styles.card} to={card.to} key={card.title}>
              <Heading as="h3" className={styles.cardTitle}>
                {card.title}
              </Heading>
              <p className={styles.cardBody}>{card.body}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
