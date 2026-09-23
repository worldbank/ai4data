import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Stages = [
  {
    step: '01',
    moment: 'Data and metadata are created',
    label: 'Metadata Quality & Enrichment',
    description:
      'LLMs score metadata completeness and consistency, and cluster survey variables into thematic groups — cutting manual tagging from days to minutes.',
    workstreams: [
      {
        title: 'Generative AI for Metadata Quality',
        to: '/docs/metadata-quality/generative-ai-for-metadata-quality',
      },
      {title: 'Metadata Augmentation', to: '/docs/metadata-augmentation/'},
    ],
  },
  {
    step: '02',
    moment: 'Data is published and searched',
    label: 'Discovery & Access',
    description:
      'Semantic search understands intent beyond keywords, and the Model Context Protocol lets any AI client query catalogs directly.',
    workstreams: [
      {title: 'Data Discoverability', to: '/docs/data-discoverability/'},
      {title: 'Model Context Protocol', to: '/docs/mcp/'},
    ],
  },
  {
    step: '03',
    moment: 'Data is used, revised, and tracked over time',
    label: 'Monitoring & Trust',
    description:
      'Statistical detectors flag anomalies and LLMs explain them with cited evidence; NER pipelines trace where datasets get cited in research and policy.',
    workstreams: [
      {
        title: 'Anomaly Detection and Explanation',
        to: '/docs/anomaly-detection/',
      },
      {title: 'Monitoring of Data Use', to: '/docs/data_use/'},
    ],
  },
];

export default function DataLifecycle() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">The Data Lifecycle</span>
          <Heading as="h2" className={styles.title}>
            Where AI makes data AI-ready
          </Heading>
          <p className={styles.lede}>
            Data becomes AI-ready gradually — as it&apos;s documented, made
            discoverable, and proven trustworthy over time. Each stage below
            is where a workstream applies AI to get it there.
          </p>
        </div>

        <div className={styles.flow}>
          {Stages.map((stage) => (
            <div className={styles.stage} key={stage.label}>
              <div className={styles.stageHead}>
                <span className={styles.stageStep}>{stage.step}</span>
                <span className={styles.stageMoment}>{stage.moment}</span>
              </div>
              <Heading as="h3" className={styles.stageLabel}>
                {stage.label}
              </Heading>
              <p className={styles.stageDescription}>{stage.description}</p>
              <div className={styles.stageTags}>
                {stage.workstreams.map((w) => (
                  <Link className={styles.tag} to={w.to} key={w.title}>
                    {w.title}
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className={styles.foundation}>
          <span className={styles.foundationLabel}>
            Enabling Capabilities — underpins every stage
          </span>
          <Link className={styles.foundationLink} to="/docs/inclusive-ai/">
            Inclusive AI Applications — multilingual models across 50+
            languages →
          </Link>
        </div>
      </div>
    </section>
  );
}
