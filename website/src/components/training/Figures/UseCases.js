import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// Tasks from development work that suit a small model trained on synthetic data. Each card
// opens a panel with the task, the reasons why no training data exist, one practice case,
// and where the model runs. The practice cases were written for this page as illustrations.
const REASONS = {
  private: ['Private', 'The real records hold personal data and cannot be sent to an outside service.'],
  rare: ['Rare', 'The important cases are a small share of all cases.'],
  missing: ['Not yet written', 'The cases do not exist until the event happens.'],
  language: ['Local language', 'The cases are written in a local language, with abbreviations and mixed languages.'],
  few: ['Few labelled', 'Few past cases have the right answer recorded.'],
  wording: ['Varied wording', 'People describe the same thing in many ways that no existing list covers.'],
};

const CASES = [
  {
    key: 'jobs',
    name: 'Job matching',
    who: 'A public employment office',
    task: 'Reads how a job seeker describes their work and suggests the occupations and vacancies that match.',
    reasons: ['private', 'wording', 'few'],
    input: 'I fix motorbikes and sell spare parts at the market, 6 years. I keep the accounts myself.',
    inputKind: 'Job seeker',
    answer: [['Occupation', 'Motor vehicle mechanic'], ['Matching vacancies', 'Motorcycle mechanic at a garage; sales assistant in a spare parts shop']],
    runs: 'On the employment office\'s own server. The descriptions of job seekers stay in the office.',
    icon: (
      <g>
        <circle cx="22" cy="22" r="8" className={styles.iconLine} fill="none" />
        <path d="M10 44 C10 34 34 34 34 44" className={styles.iconLine} fill="none" />
        <path d="M40 30 H56" className={styles.iconArrow} markerEnd="url(#uc-head)" />
        <rect x="60" y="16" width="30" height="24" rx="3" className={styles.iconPaper} />
        <path d="M70 16 V12 H80 V16" className={styles.iconLine} fill="none" />
      </g>
    ),
  },
  {
    key: 'grievance',
    name: 'Complaints by text message',
    who: 'The grievance mechanism of a project',
    task: 'Sorts each complaint by issue and urgency and sends it to the office that handles it.',
    reasons: ['few', 'private', 'language'],
    input: 'pls help, maji hayajatoka tangu jumatatu. contractor broke the pipe near the school',
    inputKind: 'Text message',
    answer: [['Issue', 'Water supply'], ['Urgency', 'High'], ['Send to', 'Site engineer']],
    runs: 'On a server of the project team. The messages and phone numbers stay with the team.',
    icon: (
      <g>
        <rect x="8" y="10" width="40" height="28" rx="6" className={styles.iconPaper} />
        <path d="M16 38 L14 46 L24 38" className={styles.iconPaper} />
        <path d="M16 20 H40 M16 28 H32" className={styles.iconLine} />
        <path d="M54 26 H66" className={styles.iconArrow} markerEnd="url(#uc-head)" />
        <rect x="70" y="14" width="22" height="7" rx="3.5" className={styles.iconTag} />
        <rect x="70" y="26" width="16" height="7" rx="3.5" className={styles.iconTag} />
      </g>
    ),
  },
  {
    key: 'disaster',
    name: 'Messages after a flood',
    who: 'A disaster response center',
    task: 'Picks out requests for rescue, water, or shelter from incoming messages, with the place and the number of people.',
    reasons: ['missing', 'rare', 'language'],
    input: 'Water rising in Kalanga village, 6 people on the roof of the clinic, we need a boat',
    inputKind: 'Message',
    answer: [['Request', 'Rescue'], ['Place', 'Kalanga village, clinic'], ['People', '6']],
    runs: 'On a laptop at the response center. It keeps working when the internet connection is down.',
    icon: (
      <g>
        <path d="M14 30 L30 16 L46 30 V44 H14 Z" className={styles.iconPaper} />
        <path d="M4 40 Q14 34 24 40 T44 40 T64 40 T84 40 T96 40" className={styles.iconStrokeSmall} fill="none" />
        <path d="M4 48 Q14 42 24 48 T44 48 T64 48 T84 48 T96 48" className={styles.iconStrokeSmall} fill="none" />
        <circle cx="74" cy="20" r="9" className={styles.iconTag} />
        <path d="M74 14 V21 M74 24 V25" className={styles.iconLine} />
      </g>
    ),
  },
  {
    key: 'farmers',
    name: 'Farmers\' questions',
    who: 'An agricultural extension service',
    task: 'Answers questions sent by text message in a local language, from the approved extension guide only, and passes the rest to an extension officer.',
    reasons: ['language', 'few'],
    input: 'My maize leaves have white and yellow stripes and the plants stay short. What do I do?',
    inputKind: 'Question, translated',
    answer: [['Answer', 'The guide\'s section on leaf streaks: pull out plants with streaks and plant a tolerant variety next season.'], ['Also', 'Sent to the extension officer of the area']],
    runs: 'On the extension service\'s own server, reached by text message from a basic phone.',
    icon: (
      <g>
        <path d="M30 48 V18" className={styles.iconStrokeSmall} />
        <path d="M30 30 C20 26 16 18 18 12 C26 14 30 22 30 30 Z" className={styles.iconSmall} />
        <path d="M30 38 C40 34 44 26 42 20 C34 22 30 30 30 38 Z" className={styles.iconSmall} />
        <rect x="56" y="10" width="34" height="22" rx="6" className={styles.iconPaper} />
        <path d="M64 18 H82 M64 24 H76" className={styles.iconLine} />
        <path d="M66 32 L64 38 L72 32" className={styles.iconPaper} />
      </g>
    ),
  },
  {
    key: 'benefits',
    name: 'Benefit applications',
    who: 'A social protection agency',
    task: 'Flags missing or inconsistent entries in an application before a caseworker reviews it.',
    reasons: ['private', 'rare'],
    input: 'Household members: 3 · Children under 5: 4 · Monthly income: (blank)',
    inputKind: 'Application form',
    answer: [['Flag', 'More children under 5 than household members'], ['Flag', 'Monthly income missing']],
    runs: 'On the agency\'s own server, next to the register of applicants.',
    icon: (
      <g>
        <rect x="12" y="6" width="34" height="44" rx="3" className={styles.iconPaper} />
        <path d="M18 16 H40 M18 24 H40 M18 32 H32" className={styles.iconLine} />
        <path d="M54 28 H66" className={styles.iconArrow} markerEnd="url(#uc-head)" />
        <path d="M80 12 L92 40 H68 Z" className={styles.iconTag} />
        <path d="M80 22 V31 M80 35 V36" className={styles.iconLine} />
      </g>
    ),
  },
  {
    key: 'survey',
    name: 'Coding survey answers',
    who: 'A national statistical office',
    task: 'Gives each occupation written in a household survey its code in the international classification of occupations.',
    reasons: ['few', 'wording'],
    input: 'Sells grilled maize by the roadside',
    inputKind: 'Survey answer',
    answer: [['Occupation', 'Street food salesperson'], ['Code (ISCO-08)', '5212']],
    runs: 'On the laptops of the office\'s survey team, without an internet connection in the field.',
    tested: true,
    icon: (
      <g>
        <rect x="10" y="8" width="38" height="40" rx="3" className={styles.iconPaper} />
        <path d="M16 18 H20 M24 18 H42 M16 28 H20 M24 28 H38 M16 38 H20 M24 38 H40" className={styles.iconLine} />
        <path d="M54 28 H66" className={styles.iconArrow} markerEnd="url(#uc-head)" />
        <rect x="70" y="20" width="24" height="16" rx="3" className={styles.iconTag} />
      </g>
    ),
  },
];

export default function UseCases() {
  const [open, setOpen] = useState('grievance');
  const c = CASES.find((x) => x.key === open);
  return (
    <figure className={styles.figure}>
      <p className={styles.title}>Tasks for a small model</p>
      <p className={styles.caption}>
        The figure shows six tasks from development work. Select a task to see what the model does, why no training data exist, one
        practice case, and where the model runs.
      </p>
      <svg width="0" height="0" style={{position: 'absolute'}} aria-hidden="true">
        <defs>
          <marker id="uc-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--faint)" />
          </marker>
        </defs>
      </svg>
      <div className={styles.caseGrid} role="group" aria-label="Task">
        {CASES.map((x) => (
          <button key={x.key} type="button" aria-pressed={open === x.key}
            className={clsx(styles.caseCard, open === x.key && styles.caseCardOn)} onClick={() => setOpen(x.key)}>
            <svg viewBox="0 0 100 56" aria-hidden="true">{x.icon}</svg>
            <span>{x.name}</span>
          </button>
        ))}
      </div>

      <div className={styles.casePanel} aria-live="polite">
        <div>
          <span className={styles.stepNumber}>{c.who}</span>
          <p className={styles.caseTask}>{c.task}</p>
          <p style={{margin: '12px 0 6px'}}><strong>Why no training data exist</strong></p>
          <ul className={styles.reasonList}>
            {c.reasons.map((r) => (
              <li key={r}><span className={clsx(styles.chip, styles.reasonChip)}>{REASONS[r][0]}</span> {REASONS[r][1]}</li>
            ))}
          </ul>
          <p style={{margin: '12px 0 4px'}}><strong>Where the model runs</strong></p>
          <p className={styles.stepText}>{c.runs}</p>
          {c.tested && (
            <p className={styles.source}>
              This task was tested in the experiments behind this session. The small model scored 0.15 below the best
              large model on job titles from a survey (section on step 3).
            </p>
          )}
        </div>
        <div>
          <p style={{margin: '0 0 6px'}}><strong>One practice case</strong></p>
          <div className={styles.practice}>
            <span className={styles.practiceKind}>{c.inputKind}</span>
            <p className={styles.practiceText}>{c.input}</p>
            <span className={styles.practiceKind}>Right answer</span>
            <dl className={styles.practiceAnswer}>
              {c.answer.map(([k, v], i) => (
                <React.Fragment key={i}>
                  <dt>{k}</dt>
                  <dd>{v}</dd>
                </React.Fragment>
              ))}
            </dl>
          </div>
          <p className={styles.source}>
            A large model writes thousands of cases like this one, each with its answer. The case shown was written for
            this page as an illustration.
          </p>
        </div>
      </div>
      <p className={styles.source}>
        The tasks illustrate where the method could apply. They do not describe systems in use.
      </p>
    </figure>
  );
}
