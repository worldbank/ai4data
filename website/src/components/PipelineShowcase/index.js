import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const Stages = [
  {
    name: 'Primary',
    role: 'Detect',
    description: 'Scans metadata for typos, inconsistencies, and missing fields.',
  },
  {
    name: 'Secondary',
    role: 'Re-scan',
    description: 'Independently re-scans the same record to catch what Primary missed.',
  },
  {
    name: 'Critic',
    role: 'Filter',
    description: 'Removes false positives by applying exclusion rules.',
  },
  {
    name: 'Categorizer',
    role: 'Classify',
    description: 'Assigns each confirmed issue one of six categories.',
  },
  {
    name: 'Severity Scorer',
    role: 'Rank',
    description: 'Scores impact from 1 (trivial) to 5 (critical).',
  },
];

const Categories = [
  'Typo / Language',
  'Formatting / Structure',
  'Missing / Redundant Info',
  'Inconsistency / Conflict',
  'Incorrect / Invalid Content',
  'Ambiguity / Unclear',
];

export default function PipelineShowcase() {
  return (
    <section className={styles.section}>
      <div className="container">
        <span className={styles.eyebrow}>How it works</span>
        <Heading as="h2" className={styles.title}>
          A five-agent pipeline, not a single prompt
        </Heading>
        <p className={styles.lede}>
          The Metadata Reviewer decomposes quality review into specialized,
          sequential agents — so detection, filtering, classification, and
          scoring stay consistent and auditable across thousands of records.
        </p>

        <div className={styles.pipeline}>
          {Stages.map((stage, idx) => (
            <div className={styles.stageWrap} key={stage.name}>
              <div className={styles.stage}>
                <span className={styles.stageIndex}>{idx + 1}</span>
                <span className={styles.stageRole}>{stage.role}</span>
                <div className={styles.stageName}>{stage.name}</div>
                <p className={styles.stageDescription}>{stage.description}</p>
              </div>
              {idx < Stages.length - 1 && (
                <svg
                  className={styles.arrow}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2">
                  <path
                    d="M5 12h14M13 6l6 6-6 6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </div>
          ))}
        </div>

        <div className={styles.footRow}>
          <div className={styles.categoriesBlock}>
            <span className={styles.footLabel}>Output: 6 issue categories</span>
            <div className={styles.chips}>
              {Categories.map((c) => (
                <span className={styles.chip} key={c}>
                  {c}
                </span>
              ))}
            </div>
          </div>
          <Link
            className={styles.footLink}
            to="/docs/metadata-reviewer/overview">
            See the full pipeline reference →
          </Link>
        </div>
      </div>
    </section>
  );
}
