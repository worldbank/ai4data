import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const WorkstreamList = [
  {
    title: 'Generative AI for Metadata Quality',
    to: '/docs/metadata-quality/generative-ai-for-metadata-quality',
    description:
      'Scores metadata across four dimensions — completeness, semantic alignment, specificity, and consistency — with structured, auditable LLM output.',
  },
  {
    title: 'Metadata Augmentation',
    to: '/docs/metadata-augmentation/',
    description:
      'Organizes hundreds of survey variables into DDI-style thematic groups via semantic clustering — no manual labeling required.',
  },
  {
    title: 'Anomaly Detection and Explanation',
    to: '/docs/anomaly-detection/',
    description:
      'Classifies flagged anomalies as an external driver, data error, measurement update, or insufficient data — with cited evidence.',
  },
  {
    title: 'Data Discoverability',
    to: '/docs/data-discoverability/',
    description:
      'Conceptual search surfaces "Gini index" when you search "income inequality" — no exact keyword match required.',
  },
  {
    title: 'Model Context Protocol',
    to: '/docs/mcp/',
    description:
      'One MCP server, built once, callable by any compatible AI client — Claude, ChatGPT, or a custom agent — with no bespoke integration.',
  },
  {
    title: 'Inclusive AI Applications',
    to: '/docs/inclusive-ai/',
    description:
      'Multilingual embedding models spanning 50+ languages, and batch inference at roughly half the synchronous-call cost.',
  },
];

function Workstream({title, to, description, idx}) {
  return (
    <Link to={to} className={styles.row}>
      <span className={styles.rowIndex}>{String(idx + 1).padStart(2, '0')}</span>
      <div className={styles.rowBody}>
        <Heading as="h3" className={styles.rowTitle}>
          {title}
        </Heading>
        <p className={styles.rowDescription}>{description}</p>
      </div>
      <span className={styles.rowArrow} aria-hidden="true">
        →
      </span>
    </Link>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.head}>
          <span className="kicker">What We Build</span>
          <Heading as="h2" className={styles.title}>
            Flagship Workstreams
          </Heading>
        </div>
        <div className={styles.list}>
          {WorkstreamList.map((props, idx) => (
            <Workstream key={props.title} {...props} idx={idx} />
          ))}
        </div>
      </div>
    </section>
  );
}
