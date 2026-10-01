import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Stages = [
  {
    step: '01',
    moment: 'Data and metadata are created',
    label: 'Metadata Quality & Enrichment',
    workstreams: [
      {
        title: 'Generative AI for Metadata Quality',
        gsbpm: 'Metadata management',
        to: '/docs/metadata-quality/generative-ai-for-metadata-quality',
        description:
          'Scores metadata on four dimensions: completeness, semantic alignment, specificity, and consistency. The LLM output is structured and auditable.',
      },
      {
        title: 'Metadata Augmentation',
        gsbpm: '5.2',
        to: '/docs/metadata-augmentation/',
        description:
          'Organizes hundreds of survey variables into DDI-style thematic groups using semantic clustering.',
      },
    ],
  },
  {
    step: '02',
    moment: 'Data is published and searched',
    label: 'Discovery & Access',
    workstreams: [
      {
        title: 'Data Discoverability',
        gsbpm: '7.2, 7.5',
        to: '/docs/data-discoverability/',
        description:
          'Conceptual search returns food insecurity indicators for the query “how many people go hungry”, although none of their titles contains those words.',
      },
      {
        title: 'Model Context Protocol',
        gsbpm: '7.2, 7.3',
        to: '/docs/mcp/',
        description:
          'An MCP server that any compatible AI client, such as Claude, ChatGPT, or a custom agent, can call.',
      },
    ],
  },
  {
    step: '03',
    moment: 'Data is used, revised, and tracked over time',
    label: 'Monitoring & Trust',
    workstreams: [
      {
        title: 'Anomaly Detection and Explanation',
        gsbpm: '5.3, 6.2, 6.3',
        to: '/docs/anomaly-detection/',
        description:
          'Classifies flagged anomalies as an external driver, data error, measurement update, or insufficient data, and cites the evidence.',
      },
      {
        title: 'Monitoring of Data Use',
        gsbpm: '8.1, 8.2',
        to: '/docs/data_use/',
        description:
          'Extracts dataset mentions from reports and papers with zero-shot NER, including name variants such as “WDI” and “World Development Indicators.”',
      },
      {
        title: 'Proof-Carrying Numbers',
        gsbpm: '6.2, 7.2',
        to: 'https://arxiv.org/abs/2509.06902',
        description:
          'Checks each number in a chatbot answer against the official record and marks it as verified or flagged.',
      },
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
            AI across the data lifecycle
          </Heading>
          <p className={styles.lede}>
            Data becomes AI-ready gradually, as it is documented, made
            discoverable, and shown to be trustworthy over time. Each stage
            below lists the workstreams that apply AI at that point. Each
            workstream has its own methodology, pipeline code, and
            documentation.
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
              <ul className={styles.workstreams}>
                {stage.workstreams.map((w) => (
                  <li key={w.title}>
                    <Link className={styles.workstream} to={w.to}>
                      <span className={styles.workstreamTitle}>{w.title}</span>
                      {w.gsbpm && (
                        <span className={styles.gsbpm}>
                          {/^\d/.test(w.gsbpm) ? `GSBPM ${w.gsbpm}` : w.gsbpm}
                        </span>
                      )}
                      <span className={styles.workstreamBody}>
                        {w.description}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className={styles.foundation}>
          <span className={styles.foundationLabel}>
            Enabling Capabilities used at every stage
          </span>
          <Link className={styles.foundationLink} to="/docs/inclusive-ai/">
            <span className={styles.foundationTitle}>
              Inclusive AI Applications →
            </span>
            <span className={styles.foundationBody}>
              Multilingual embedding models covering 50+ languages, and batch
              inference at roughly half the cost of synchronous calls.
            </span>
          </Link>
        </div>

        <p className={styles.gsbpmNote}>
          Tags such as GSBPM 6.3 show where a workstream applies in the Generic
          Statistical Business Process Model.{' '}
          <Link to="/docs/gsbpm-mapping">See the full mapping →</Link>
        </p>
      </div>
    </section>
  );
}
