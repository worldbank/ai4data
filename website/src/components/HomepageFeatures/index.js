import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

function SparklesIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path
        d="M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6L12 3z"
        strokeLinejoin="round"
      />
      <path d="M19 15l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7.7-2z" strokeLinejoin="round" />
    </svg>
  );
}

function LayersIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M12 3l8 4.5-8 4.5-8-4.5L12 3z" strokeLinejoin="round" />
      <path d="M4 12l8 4.5 8-4.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 16.5L12 21l8-4.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PulseIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path
        d="M3 12h4l2 6 4-14 2 8h6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="11" cy="11" r="6.5" />
      <path d="M20 20l-4.5-4.5" strokeLinecap="round" />
    </svg>
  );
}

function PlugIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M9 3v5M15 3v5" strokeLinecap="round" />
      <path d="M6 8h12v3a6 6 0 01-12 0V8z" strokeLinejoin="round" />
      <path d="M12 17v4" strokeLinecap="round" />
    </svg>
  );
}

function GlobeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="12" cy="12" r="8.5" />
      <path d="M3.5 12h17M12 3.5c2.4 2.4 3.6 5.2 3.6 8.5s-1.2 6.1-3.6 8.5c-2.4-2.4-3.6-5.2-3.6-8.5S9.6 5.9 12 3.5z" />
    </svg>
  );
}

const WorkstreamList = [
  {
    title: 'Generative AI for Metadata Quality',
    to: '/docs/metadata-quality/generative-ai-for-metadata-quality',
    description:
      'Scores metadata across four dimensions — completeness, semantic alignment, specificity, and consistency — with structured, auditable LLM output.',
    Icon: SparklesIcon,
  },
  {
    title: 'Metadata Augmentation',
    to: '/docs/metadata-augmentation/',
    description:
      'Organizes hundreds of survey variables into DDI-style thematic groups via semantic clustering — no manual labeling required.',
    Icon: LayersIcon,
  },
  {
    title: 'Anomaly Detection and Explanation',
    to: '/docs/anomaly-detection/',
    description:
      'Classifies flagged anomalies as an external driver, data error, measurement update, or insufficient data — with cited evidence.',
    Icon: PulseIcon,
  },
  {
    title: 'Data Discoverability',
    to: '/docs/data-discoverability/',
    description:
      'Conceptual search surfaces "Gini index" when you search "income inequality" — no exact keyword match required.',
    Icon: SearchIcon,
  },
  {
    title: 'Model Context Protocol',
    to: '/docs/mcp/',
    description:
      'One MCP server, built once, callable by any compatible AI client — Claude, ChatGPT, or a custom agent — with no bespoke integration.',
    Icon: PlugIcon,
  },
  {
    title: 'Inclusive AI Applications',
    to: '/docs/inclusive-ai/',
    description:
      'Multilingual embedding models spanning 50+ languages, and batch inference at roughly half the synchronous-call cost.',
    Icon: GlobeIcon,
  },
];

function Workstream({title, to, description, Icon}) {
  return (
    <div className={clsx('col col--4')}>
      <Link to={to} className={styles.card}>
        <span className={styles.cardIcon}>
          <Icon />
        </span>
        <Heading as="h3" className={styles.cardTitle}>
          {title}
        </Heading>
        <p className={styles.cardDescription}>{description}</p>
        <span className={styles.cardLink}>
          Read more
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className={styles.cardLinkArrow}>
            <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </Link>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <span className={styles.sectionEyebrow}>What we build</span>
        <Heading as="h2" className={styles.sectionTitle}>
          Flagship Workstreams
        </Heading>
        <div className="row">
          {WorkstreamList.map((props, idx) => (
            <Workstream key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
