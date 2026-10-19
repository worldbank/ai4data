import {useEffect, useState} from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const BASE = '/cookbook/ai-ready-dissemination/';

// One question per chapter. Each option describes a state; the first option
// means none of the levels is reached yet.
const QUESTIONS = [
  {
    id: 'find',
    title: '1. Find',
    question: 'How complete and visible is your catalog metadata?',
    options: [
      'Many series lack a description, unit, or coverage, or have no page of their own.',
      'Every published series has a page with a permanent URL and a complete record.',
      'Catalog records are also published as DCAT or schema.org markup, and a catalog API or feed exists.',
      'Search works by meaning as well as by keyword, and metadata gaps are found with AI and fixed under review.',
    ],
  },
  {
    id: 'retrieve',
    title: '2. Retrieve',
    question: 'How can software obtain your data?',
    options: [
      'Mostly from PDFs, Excel files, or pages that need manual steps.',
      'Open files (CSV or similar) with stable download URLs.',
      'A documented API with stable identifiers that returns unit, period, and source with the values.',
      'An agent interface such as an MCP server, returning provenance with every response.',
    ],
  },
  {
    id: 'understand',
    title: '3. Understand',
    question: 'How is the meaning of each series documented?',
    options: [
      'Definitions and units are in footnotes or missing for many series.',
      'Each indicator has a definition, unit, method note, and standard geography and date codes.',
      'Structure and code lists are published in SDMX or DDI, with crosswalks to national classifications.',
      'Concepts are linked to shared vocabularies or an ontology.',
    ],
  },
  {
    id: 'ask',
    title: '4. Ask',
    question: 'How do users ask for statistics?',
    options: [
      'Browsing menus or a keyword search without filters.',
      'Search with filters and example queries; failed searches are reviewed.',
      'Semantic search with structured results and support for the main user languages.',
      'A conversational interface that answers only from retrieved records, with citations.',
    ],
  },
  {
    id: 'trust',
    title: '5. Trust',
    question: 'How are source and freshness shown?',
    options: [
      'Source and release date are not consistently shown.',
      'Every dataset page shows source, release date, and a suggested citation; a revisions log exists.',
      'Every API response carries source, identifier, and release date; superseded series point to replacements.',
      'Numbers in generated answers are checked against the record and answers are logged for audit.',
    ],
  },
  {
    id: 'evaluate',
    title: '6. Evaluate',
    question: 'How do you measure whether search and answers work?',
    options: [
      'There is no fixed test set.',
      'A set of known-item questions is run by hand on a schedule.',
      'Retrieval measures are automated and reported per language.',
      'Answer-level tests (numeric accuracy, citations, refusals) run before every change.',
    ],
  },
  {
    id: 'monitor-use',
    title: '7. Monitor use',
    question: 'How do you track use of your data?',
    options: [
      'Download counts only.',
      'A recommended citation and identifier per dataset; automated clients separated in logs.',
      'Publication sources searched for mentions on a schedule, with a table of name variants.',
      'Mentions extracted automatically and reported; AI systems checked for how they cite the organization.',
    ],
  },
  {
    id: 'govern',
    title: '8. Govern',
    question: 'How is AI use governed?',
    options: [
      'No written policy on AI use.',
      'A short policy on permitted and prohibited uses; confidential data kept out of external tools.',
      'A register of AI components with owners; human review for generated content; read-only agent access.',
      'Inputs and outputs logged; prompt-injection tests; an exit plan per provider.',
    ],
  },
  {
    id: 'sustain',
    title: '9. Sustain',
    question: 'How maintainable is the system?',
    options: [
      'Key tasks depend on one person and are not documented.',
      'The production process is documented with backups for each task.',
      'Open standards, version control, scheduled reviews, and a cost estimate per query.',
      'Cost, latency, and quality are monitored together; small open models are used where they suffice.',
    ],
  },
];

const LEVELS = ['Not yet', 'Foundational', 'AI-ready', 'AI-native'];

// `questions` has one entry per chapter: {id, title, question, options[4]}
// with the options ordered from the lowest level to the highest. `base` is
// the cookbook's URL prefix. Both default to the dissemination cookbook.
export default function SelfAssessment({questions = QUESTIONS, base = BASE, storageKey}) {
  const key = storageKey || `cookbook-self-assessment-${base.replace(/\W+/g, '-')}`;
  const [answers, setAnswers] = useState({});
  const [showResult, setShowResult] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw) {
        setAnswers(JSON.parse(raw));
      }
    } catch (e) {
      // ignore
    }
  }, []);

  const choose = (id, value) => {
    const next = {...answers, [id]: value};
    setAnswers(next);
    try {
      window.localStorage.setItem(key, JSON.stringify(next));
    } catch (e) {
      // ignore
    }
  };

  const answered = questions.filter((q) => answers[q.id] !== undefined);
  const complete = answered.length === questions.length;
  const lowest = complete
    ? Math.min(...questions.map((q) => answers[q.id]))
    : null;
  const next = complete ? questions.filter((q) => answers[q.id] === lowest) : [];

  return (
    <div className={styles.assessment}>
      {questions.map((q) => (
        <fieldset className={styles.aq} key={q.id}>
          <legend className={styles.aqTitle}>
            <span>{q.title}</span> {q.question}
          </legend>
          {q.options.map((opt, i) => (
            <label className={styles.aqOption} key={i}>
              <input
                type="radio"
                name={q.id}
                checked={answers[q.id] === i}
                onChange={() => choose(q.id, i)}
              />
              <span className={styles.aqLevel}>{LEVELS[i]}</span>
              <span>{opt}</span>
            </label>
          ))}
        </fieldset>
      ))}

      <div className={styles.aBar}>
        <span>
          {answered.length} of {questions.length} answered
        </span>
        <button
          type="button"
          className={styles.smallButton}
          disabled={!complete}
          onClick={() => setShowResult(true)}>
          Show reading path
        </button>
        <button
          type="button"
          className={styles.smallButton}
          onClick={() => {
            setAnswers({});
            setShowResult(false);
            try {
              window.localStorage.removeItem(key);
            } catch (e) {
              // ignore
            }
          }}>
          Reset
        </button>
      </div>

      {showResult && complete && (
        <div className={styles.aResult}>
          <table>
            <thead>
              <tr>
                <th>Chapter</th>
                <th>Current level</th>
                <th>Next step</th>
              </tr>
            </thead>
            <tbody>
              {questions.map((q) => {
                const lvl = answers[q.id];
                return (
                  <tr key={q.id}>
                    <td>
                      <Link to={`${base}${q.id}`}>{q.title}</Link>
                    </td>
                    <td>{LEVELS[lvl]}</td>
                    <td>
                      {lvl === 3
                        ? 'Maintain and re-test'
                        : `Reach ${LEVELS[lvl + 1]}`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p>
            Start with the chapters at the lowest level:{' '}
            {next.map((q, i) => (
              <span key={q.id}>
                {i > 0 && ', '}
                <Link to={`${base}${q.id}`}>{q.title}</Link>
              </span>
            ))}
            . Chapters 1 to 3 come first when they are at the same level as
            others, because the later chapters depend on them.
          </p>
        </div>
      )}
    </div>
  );
}
