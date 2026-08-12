import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import McpDiagram from '@site/src/components/McpDiagram';
import styles from './styles.module.css';

const Groups = [
  {
    label: 'Metadata Quality & Enrichment',
    description: 'Making the data itself more accurate and complete.',
    items: [
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
    ],
  },
  {
    label: 'Discovery & Access',
    description: 'Getting the data to whoever — or whatever — needs it.',
    items: [
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
        visual: <McpDiagram />,
      },
    ],
  },
  {
    label: 'Monitoring & Trust',
    description: 'Keeping the data trustworthy after it ships.',
    items: [
      {
        title: 'Anomaly Detection and Explanation',
        to: '/docs/anomaly-detection/',
        description:
          'Classifies flagged anomalies as an external driver, data error, measurement update, or insufficient data — with cited evidence.',
      },
      {
        title: 'Monitoring of Data Use',
        to: '/docs/data_use/',
        description:
          'Extracts dataset mentions from reports and papers via zero-shot NER, even when names vary — "WDI" vs. "World Development Indicators."',
      },
    ],
  },
  {
    label: 'Enabling Capabilities',
    description: 'Cross-cutting infrastructure the workstreams above depend on.',
    items: [
      {
        title: 'Inclusive AI Applications',
        to: '/docs/inclusive-ai/',
        description:
          'Multilingual embedding models spanning 50+ languages, and batch inference at roughly half the synchronous-call cost.',
      },
    ],
  },
];

function Workstream({title, to, description, idx, visual}) {
  return (
    <div className={styles.item}>
      <Link to={to} className={styles.row}>
        <span className={styles.rowIndex}>{String(idx + 1).padStart(2, '0')}</span>
        <div className={styles.rowBody}>
          <Heading as="h4" className={styles.rowTitle}>
            {title}
          </Heading>
          <p className={styles.rowDescription}>{description}</p>
        </div>
        <span className={styles.rowArrow} aria-hidden="true">
          →
        </span>
      </Link>
      {visual && <div className={styles.rowVisual}>{visual}</div>}
    </div>
  );
}

export default function HomepageFeatures() {
  let counter = 0;
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">What We Build</span>
          <Heading as="h2" className={styles.title}>
            Flagship Workstreams
          </Heading>
        </div>
        {Groups.map((group) => (
          <div className={styles.group} key={group.label}>
            <div className={styles.groupHead}>
              <Heading as="h3" className={styles.groupLabel}>
                {group.label}
              </Heading>
              <span className={styles.groupDescription}>
                {group.description}
              </span>
            </div>
            <div className={styles.list}>
              {group.items.map((props) => {
                const idx = counter;
                counter += 1;
                return <Workstream key={props.title} {...props} idx={idx} />;
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
