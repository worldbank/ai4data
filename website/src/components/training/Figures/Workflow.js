import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// The process of building a small model from synthetic data in nine steps. Each step has an
// office comparison, what happens, why it is needed, whether the worked example used it, and
// a small demonstration. The demonstrations follow the complaints example with cases written
// for this page; steps 3 and 7 also show measured results from the session's experiments.

const ISSUES = [
  ['Water supply', 'Water cut, damaged pipes, or a dry well caused by the works.'],
  ['Land and compensation', 'Payment for land, crops, or buildings affected by the project.'],
  ['Safety', 'Danger to people from traffic, machines, or open trenches.'],
  ['Dust and noise', 'Dust, noise, or vibration from the works.'],
  ['Workers\' conduct', 'Behaviour of the contractor\'s staff.'],
  ['Other', 'Any complaint that fits none of the issues above.'],
];

function TaskDemo() {
  const [on, setOn] = useState(ISSUES[0][0]);
  return (
    <div>
      <p className={styles.demoLabel}>The list of answers for the complaints example</p>
      <ul className={styles.chips}>
        {ISSUES.map(([name]) => (
          <li key={name}>
            <button type="button" className={styles.chip} aria-pressed={on === name} onClick={() => setOn(name)}>{name}</button>
          </li>
        ))}
      </ul>
      <p className={styles.definition} aria-live="polite"><strong>{on}.</strong> {ISSUES.find(([n]) => n === on)[1]}</p>
      <p className={styles.source}>Select an answer to see its definition. The large model and the people who check its cases read the same definitions.</p>
    </div>
  );
}

const SEEDS = ['no water 3 days now. who do we call', 'ur trucks almost hit my son today near the school!!',
  'Land money still not paid since march, chief said wait'];
const UNSEEDED = [
  'I would like to report that the water supply in our community has been interrupted for several days.',
  'I am writing to express my concern about the safety of children near the construction site.',
  'I wish to inform you that I have not yet received compensation for my land.',
];
const SEEDED = [
  'tap dry since sunday, women walking to the river. pls send the engineer',
  'big lorry from the site drives too fast by the market, someone will die',
  'they measured my farm in jan, no payment yet. who to see?',
];

function SeedDemo() {
  const [seeded, setSeeded] = useState(false);
  return (
    <div>
      <div className={styles.segmented} role="group" aria-label="Request">
        <button type="button" aria-pressed={!seeded} onClick={() => setSeeded(false)}>No samples</button>
        <button type="button" aria-pressed={seeded} onClick={() => setSeeded(true)}>Three real samples</button>
      </div>
      {seeded && (
        <div className={styles.seedBox}>
          <p className={styles.demoLabel}>Real messages added to the request, names removed</p>
          <ul className={styles.msgList}>
            {SEEDS.map((s) => <li key={s} className={clsx(styles.msg, styles.msgReal)}>{s}</li>)}
          </ul>
        </div>
      )}
      <p className={styles.demoLabel}>Cases written by the large model</p>
      <ul className={styles.msgList} aria-live="polite">
        {(seeded ? SEEDED : UNSEEDED).map((s) => <li key={s} className={styles.msg}>{s}</li>)}
      </ul>
      <p className={styles.source}>
        Without samples, the large model writes tidy, formal text. Real text messages are short and informal. All
        messages here were written for this page as an illustration.
      </p>
    </div>
  );
}

const ATTRS = [
  ['Issue', ISSUES.map(([n]) => n)],
  ['Length', ['one line', 'two lines', 'a short paragraph']],
  ['Style', ['text-message shorthand', 'formal', 'angry', 'polite']],
  ['Language', ['English', 'Swahili', 'English and Swahili mixed']],
  ['Urgency', ['low', 'high']],
  ['Spelling', ['correct', 'with errors']],
];

function VaryDemo({dup}) {
  const [onAttrs, setOnAttrs] = useState(['Issue']);
  const [pick, setPick] = useState(() => ATTRS.map(() => 0));
  const draw = () => setPick(ATTRS.map(([, vals]) => Math.floor(Math.random() * vals.length)));
  const toggle = (a) => setOnAttrs(onAttrs.includes(a) ? onAttrs.filter((x) => x !== a) : [...onAttrs, a]);
  const combos = ATTRS.filter(([a]) => onAttrs.includes(a)).reduce((n, [, v]) => n * v.length, 1);
  const val = (a) => {
    const i = ATTRS.findIndex(([x]) => x === a);
    return onAttrs.includes(a) ? ATTRS[i][1][pick[i]] : null;
  };
  const share = dup.repeats / dup.written;
  return (
    <div>
      <p className={styles.demoLabel}>Attributes that change from one request to the next</p>
      <ul className={styles.chips}>
        {ATTRS.map(([a]) => (
          <li key={a}>
            <button type="button" className={styles.chip} aria-pressed={onAttrs.includes(a)} onClick={() => toggle(a)}>
              {onAttrs.includes(a) ? '✓ ' : ''}{a}
            </button>
          </li>
        ))}
      </ul>
      <div className={styles.prompt} style={{marginTop: 10}} aria-live="polite">
        Write a complaint by text message{val('Issue') ? ` about ${val('Issue').toLowerCase()}` : ''}.
        {val('Length') ? ` Length: ${val('Length')}.` : ''}
        {val('Style') ? ` Style: ${val('Style')}.` : ''}
        {val('Language') ? ` Language: ${val('Language')}.` : ''}
        {val('Urgency') ? ` Urgency: ${val('Urgency')}.` : ''}
        {val('Spelling') ? ` Spelling: ${val('Spelling')}.` : ''}
      </div>
      <div className={styles.varyRow}>
        <button type="button" className={styles.primaryButton} onClick={draw}>Draw a new request</button>
        <span><strong>{combos.toLocaleString('en')}</strong> different requests possible</span>
      </div>
      <div className={styles.evidence}>
        <p className={styles.demoLabel}>Measured in the experiments</p>
        <div className={styles.hundred} role="img" aria-label={`${Math.round(share * 100)} of 100 titles repeat an earlier title.`}>
          {Array.from({length: 100}, (_, i) => <span key={i} className={clsx(i < Math.round(share * 100) && styles.hundredOn)} />)}
        </div>
        <p className={styles.stepText}>
          Asked many times for titles of the same occupation group, {dup.writer} wrote the same common titles again:{' '}
          {dup.repeats.toLocaleString('en')} of {dup.written.toLocaleString('en')} job titles ({Math.round(share * 100)}%)
          repeated an earlier one. A repeated case teaches the small model nothing new, and the checks remove it.
        </p>
      </div>
    </div>
  );
}

const CHECKED = [
  {text: 'no water at the borehole since tuesday, pls send someone', writer: 'Water supply', checker: 'Water supply'},
  {text: 'no water at the borehole since tuesday, pls send someone', writer: 'Water supply', checker: 'Water supply', drop: 'Repeats an earlier case'},
  {text: 'trucks drive fast past the school every morning', writer: 'Dust and noise', checker: 'Safety', drop: 'The two models disagree'},
  {text: 'compensation for my land not paid, 3 months now', writer: 'Land and compensation', checker: 'Land and compensation'},
  {text: 'my electricity bill is too high this month', writer: 'Electricity', checker: 'Other', drop: 'Answer not in the list'},
  {text: 'dust from the works covers our market stalls every day', writer: 'Dust and noise', checker: 'Dust and noise'},
  {text: 'pls help', writer: '', checker: 'Other', drop: 'No answer given'},
  {text: 'the workers from the camp shout at women at the well', writer: 'Workers\' conduct', checker: 'Workers\' conduct'},
];

function CheckDemo() {
  const [run, setRun] = useState(false);
  const kept = CHECKED.filter((c) => !c.drop).length;
  return (
    <div>
      <button type="button" className={styles.primaryButton} onClick={() => setRun(!run)}>
        {run ? 'Show the cases before the checks' : 'Run the checks'}
      </button>
      <div className={styles.checkTable} role="table" aria-label="Written cases and their checks">
        <div role="row" className={styles.checkHead}>
          <span role="columnheader">Case</span><span role="columnheader">Writer's answer</span><span role="columnheader">Second model's answer</span>
        </div>
        {CHECKED.map((c, i) => (
          <div role="row" key={i} className={clsx(styles.checkRow, run && c.drop && styles.checkDropped)}>
            <span role="cell"><span className={clsx(run && c.drop && styles.struck)}>{c.text}</span>
              {run && c.drop && <em className={styles.dropReason}><span className={styles.dropMark} aria-hidden="true">✕ </span>{c.drop}</em>}</span>
            <span role="cell">{c.writer || '(none)'}</span>
            <span role="cell">{c.checker}</span>
          </div>
        ))}
      </div>
      <p className={styles.stepText} aria-live="polite">
        {run ? `${kept} of ${CHECKED.length} cases are kept.` : 'A second model labels each case without seeing the writer\'s answer.'}
      </p>
    </div>
  );
}

function TrainDemo() {
  return (
    <div>
      <svg viewBox="0 0 260 90" className={styles.svg} aria-hidden="true">
        {[0, 1, 2, 3].map((i) => (
          <rect key={i} x={14 + i * 6} y={14 + i * 6} width="70" height="50" rx="4" className={styles.iconPaper} />
        ))}
        <path d="M40 50 H76 M40 58 H66" className={styles.iconLine} />
        <path d="M112 50 H150" className={styles.iconArrow} markerEnd="url(#wf-head)" />
        <rect x="166" y="26" width="48" height="48" rx="10" className={styles.iconSmall} />
        <circle cx="182" cy="44" r="4" className={styles.iconDotSmall} />
        <circle cx="198" cy="56" r="4" className={styles.iconDotSmall} />
        <path d="M182 44 L198 56" className={styles.iconStrokeSmall} />
      </svg>
      <p className={styles.stepText}>
        The figure in step 2 of the worked example trains a simple model in the browser. It shows how the number of
        practice cases changes the result.
      </p>
      <a className={styles.stepLink} href="#step-2-the-small-model-learns">Go to step 2 of the worked example</a>
    </div>
  );
}

function TestDemo() {
  const [wrong, setWrong] = useState(false);
  return (
    <div>
      <div className={styles.piles}>
        <div className={clsx(styles.pile, wrong && styles.pileTest)}>
          <strong>Practice cases</strong>
          <span>Thousands, written by the large model</span>
          <span className={styles.pileUse}>{wrong ? 'Used for training and testing' : 'Used for training'}</span>
        </div>
        <div className={clsx(styles.pile, !wrong && styles.pileTest)}>
          <strong>Real cases</strong>
          <span>A few hundred, answers given by people</span>
          <span className={styles.pileUse}>{wrong ? 'Not used' : 'Used only for testing'}</span>
        </div>
      </div>
      <label className={styles.toggle}>
        <input type="checkbox" checked={wrong} onChange={(e) => setWrong(e.target.checked)} />
        Test on practice cases
      </label>
      <p className={styles.stepText} aria-live="polite">
        {wrong
          ? 'The practice cases used for testing share the writing habits of the training cases. A high score then says little about real messages.'
          : 'The real cases are set aside before any practice case is written, and no model is trained on them.'}
      </p>
    </div>
  );
}

const ARMS = {first: 'First set of cases', control: 'Plus rounds of evenly spread cases', targeted: 'Plus rounds targeted at mistakes'};

function RoundsDemo({data}) {
  const [rep, setRep] = useState(0);
  const r = data.rounds[rep];
  const top = Math.max(...data.large.map((m) => m.f1));
  const low = Math.min(...data.large.map((m) => m.f1));
  const MAX = 0.6;
  return (
    <div>
      <div className={styles.loop} aria-hidden="true">
        {['Test on a separate set of real cases', 'Find the topics with the most mistakes', 'Write more cases on those topics', 'Train again'].map((s, i) => (
          <span key={s} className={styles.loopStep}><b>{i + 1}</b>{s}</span>
        ))}
      </div>
      <div className={styles.segmented} role="group" aria-label="Repeat of the experiment" style={{marginTop: 12}}>
        {data.rounds.map((x, i) => (
          <button key={x.replicate} type="button" aria-pressed={rep === i} onClick={() => setRep(i)}>Repeat {x.replicate}</button>
        ))}
      </div>
      <p className={styles.demoLabel}>Score of the small model on 300 real report abstracts (F1, 0 to 1)</p>
      <div className={styles.roundBars}>
        {r.arms.map((a) => (
          <div key={a.arm} className={styles.roundBar}>
            <span className={styles.roundLabel}>{ARMS[a.arm]} <small>{a.examples.toLocaleString('en')} cases</small></span>
            <span className={styles.roundTrack} title={`${ARMS[a.arm]}: ${a.f1.toFixed(3)}`}>
              <span className={styles.barFill} style={{width: `${(a.f1 / MAX) * 100}%`,
                background: a.arm === 'targeted' ? 'var(--small)' : 'var(--faint)'}} />
              <span className={styles.largeBand} style={{left: `${(low / MAX) * 100}%`, width: `${((top - low) / MAX) * 100}%`}} />
              <span className={styles.roundValue}>{a.f1.toFixed(3)}</span>
            </span>
          </div>
        ))}
        <div className={styles.roundAxis}>
          {[0, 0.2, 0.4, 0.6].map((v) => <span key={v} style={{left: `${(v / MAX) * 100}%`}}>{v.toFixed(1)}</span>)}
        </div>
      </div>
      <p className={styles.source}>
        The orange band marks the three large models asked directly ({low.toFixed(3)} to {top.toFixed(3)}). Targeted
        rounds minus evenly spread rounds: +{r.difference.toFixed(3)} (95% interval {r.interval[0].toFixed(3)} to{' '}
        {r.interval[1].toFixed(3)}). Each score is the mean of three training runs.
      </p>
    </div>
  );
}

const PAIRS = [
  {
    kind: 'Which written complaint is more realistic?',
    a: 'Dear Sir or Madam, I hope this message finds you well. I am writing to formally express my concern regarding the interruption of water supply in our community.',
    b: 'no water at the borehole since tuesday. kids walking 2km. pls help',
    best: 'b',
    note: 'Complaints by text message are short and informal. The first case reads like a letter.',
  },
  {
    kind: 'Which case has the right answer?',
    a: '"Thank you for the new road, it is very good." Answer: Safety, high urgency.',
    b: '"trucks for the road work go fast past the school, a child was almost hit today." Answer: Safety, high urgency.',
    best: 'b',
    note: 'The first message is not a complaint. A case with a wrong answer teaches the small model a wrong answer.',
  },
  {
    kind: 'Which answer to a farmer is better?',
    a: 'Streaks on maize leaves can have several causes, including viral infection transmitted by leafhoppers, nutrient deficiency, or genetic factors. Consider a systemic insecticide and a foliar feed.',
    b: 'This looks like maize streak. Pull out the plants with streaks. Next season, plant a tolerant variety. The extension officer will call you.',
    best: 'b',
    note: 'The second answer stays within the approved guide, is short enough for a text message, and passes the case to a person.',
  },
];

function PreferDemo() {
  const [picks, setPicks] = useState({});
  const [i, setI] = useState(0);
  const p = PAIRS[i];
  const pick = picks[i];
  const agreed = Object.entries(picks).filter(([k, v]) => PAIRS[k].best === v).length;
  return (
    <div>
      <p className={styles.demoLabel}>Pair {i + 1} of {PAIRS.length}: {p.kind}</p>
      <div className={styles.pair}>
        {['a', 'b'].map((side) => (
          <button key={side} type="button" aria-pressed={pick === side}
            className={clsx(styles.pairCard, pick && side === p.best && styles.pairBest)}
            onClick={() => setPicks({...picks, [i]: side})}>
            <span className={styles.practiceKind}>{side.toUpperCase()}</span>
            {side === 'a' ? p.a : p.b}
          </button>
        ))}
      </div>
      {pick && (
        <p className={styles.stepText} aria-live="polite">
          <strong>{pick === p.best ? 'An experienced staff member chose the same.' : 'An experienced staff member chose the other one.'}</strong>{' '}
          {p.note}
        </p>
      )}
      <div className={styles.varyRow}>
        <button type="button" className={styles.linkButton} disabled={i === 0} onClick={() => setI(i - 1)}>Previous pair</button>
        <button type="button" className={styles.linkButton} disabled={i === PAIRS.length - 1} onClick={() => setI(i + 1)}>Next pair</button>
        <span>Same choice as the staff member: {agreed} of {Object.keys(picks).length}</span>
      </div>
      <p className={styles.source}>
        The cases were written for this page as an illustration. The farmer&apos;s answer follows the illustration in
        the tasks figure above.
      </p>
    </div>
  );
}

function WatchDemo() {
  return (
    <div>
      <div className={styles.practice}>
        <span className={styles.practiceKind}>World Development Report 2026, spotlight 1</span>
        <p className={styles.stepText} style={{margin: '4px 0 0'}}>
          A screening tool for diabetic eye disease rejected more than one-fifth of the images that nurses took,
          because the lighting in the clinic differed from that in its training images. The report advises testing a
          tool where it will be used and watching it as conditions change.
        </p>
      </div>
      <p className={styles.stepText} style={{marginTop: 10}}>
        For a small model trained on practice cases, a change can come from a new issue on the list, a new language,
        or a new kind of message after an event. Regular reviews of a sample of real cases show when new practice
        cases and a new round of training are needed.
      </p>
    </div>
  );
}

const STEPS = [
  {
    short: 'Describe the task',
    office: 'The manager writes down the decisions the new staff member will make.',
    what: 'The team writes the list of possible answers, with a definition of each, or the format of the answer.',
    why: 'The large model that writes the cases, the people who check them, and the test all need the same definitions.',
    used: ['yes', 'A list of 86 topics.'],
    Demo: TaskDemo,
  },
  {
    short: 'Show a few real cases',
    office: 'The colleague who writes the practice cases first reads a few real files.',
    what: 'A few dozen real examples go into the request to the large model as samples of the style, language, and errors of real cases. Personal details are removed first.',
    why: 'Without samples, a large model writes tidy, generic text that differs from real messages. The samples must not leak into the output: a check looks for copies of them.',
    used: ['no', 'The requests named the topics only.'],
    Demo: SeedDemo,
  },
  {
    short: 'Vary the cases',
    office: 'The colleague writes cases of every kind: short and long, calm and angry, easy and hard.',
    what: 'Each request to the large model sets a different combination of attributes: the answer, the length, the style, the language, the difficulty.',
    why: 'Real cases vary in all these ways, and a small model learns only from the kinds of cases in its training set. Varying the request also reduces repeated cases.',
    used: ['partly', 'Each request named a different set of eight topics and asked for sentences of varied length and style, with two or three topics in about a third of them.'],
    Demo: VaryDemo,
  },
  {
    short: 'Write and check',
    office: 'A second colleague reads each practice case and its answer before it is used.',
    what: 'The large model writes the cases with their answers. A second model labels each case on its own, and scripts remove repeats, answers outside the list, and cases where the two models disagree.',
    why: 'The model that writes a case can be wrong about its answer. A wrong answer in training becomes a wrong answer in use.',
    used: ['yes', 'Scripts checked every set. In the later rounds, a second model also labelled each sentence.'],
    Demo: CheckDemo,
  },
  {
    short: 'Train the small model',
    office: 'The new staff member studies the practice cases.',
    what: 'The small model is trained on the checked cases until its answers match theirs.',
    why: 'Training turns the cases into a model that answers on its own, on the organization\'s own computer.',
    used: ['yes', 'About 20,000 sentences in the first set.'],
    Demo: TrainDemo,
  },
  {
    short: 'Test on real cases',
    office: 'Before the new staff member works alone, a supervisor checks their decisions on real past files.',
    what: 'A few hundred real cases, with answers given by people, are set aside for testing only.',
    why: 'Only real cases show how the model will do in use. Practice cases share the writing habits of the large model, so a high score on them can fail to hold on real cases.',
    used: ['yes', '300 real report abstracts with recorded topics.'],
    Demo: TestDemo,
  },
  {
    short: 'Practise the weak spots',
    office: 'The colleague writes more practice cases of the kinds the new staff member gets wrong.',
    what: 'The model is tested on a second set of real cases. A program finds where it makes most mistakes, asks the large model for more cases of those kinds, and the model is trained again. The rounds repeat while the score improves.',
    why: 'Extra cases help most where the model is weakest. Evenly spread extra cases add many that the model already handles.',
    used: ['yes', 'Up to five rounds, compared with rounds of the same size spread evenly over all topics.'],
    Demo: RoundsDemo,
  },
  {
    short: 'Ask people which is better',
    office: 'Experienced staff compare two practice cases, or two answers, and say which one is better.',
    what: 'People choose between pairs. The choices improve the request to the large model and filter the cases. Where the small model writes text, such as an answer to a farmer, the choices can also train it towards the preferred answers.',
    why: 'Experienced staff know what real cases and good answers look like. Their choices bring that knowledge into the process, and choosing between two is quicker than writing a case.',
    used: ['no', 'Not part of the experiments.'],
    Demo: PreferDemo,
  },
  {
    short: 'Watch and refresh',
    office: 'The supervisor keeps reviewing a sample of the staff member\'s decisions.',
    what: 'People review a sample of the model\'s answers at regular intervals. When the list of answers, the language, or the situation changes, new practice cases are written and the model is trained again.',
    why: 'A model does worse on cases that differ from its training cases, as the example from the World Development Report shows.',
    used: ['no', 'Outside the experiments.'],
    Demo: WatchDemo,
  },
];

const USED = {yes: 'Used in the worked example', partly: 'Partly used in the worked example', no: 'Not used in the worked example'};

export default function Workflow({data}) {
  const [i, setI] = useState(0);
  const s = STEPS[i];
  const {Demo} = s;
  return (
    <figure className={styles.figure}>
      <p className={styles.title}>From a task to a small model in nine steps</p>
      <p className={styles.caption}>
        Select a step. Each step has an office comparison, what happens, why it is needed, and a demonstration based
        on the complaints example. The marks under the steps show which ones the worked example of this page used.
      </p>
      <svg width="0" height="0" style={{position: 'absolute'}} aria-hidden="true">
        <defs>
          <marker id="wf-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--faint)" />
          </marker>
        </defs>
      </svg>
      <ol className={styles.rail}>
        {STEPS.map((x, k) => (
          <li key={x.short}>
            <button type="button" aria-pressed={i === k} aria-label={`Step ${k + 1}: ${x.short}. ${USED[x.used[0]]}.`}
              className={clsx(styles.railStep, i === k && styles.railOn)} onClick={() => setI(k)}>
              <span className={styles.railNum}>{k + 1}</span>
              <span className={styles.railText}>{x.short}</span>
              <span className={clsx(styles.usedMark, styles[`used_${x.used[0]}`])} aria-hidden="true" />
            </button>
          </li>
        ))}
      </ol>
      <div className={styles.legend} style={{margin: '4px 0 14px'}}>
        <span><span className={clsx(styles.usedMark, styles.used_yes)} /> used in the worked example</span>
        <span><span className={clsx(styles.usedMark, styles.used_partly)} /> partly</span>
        <span><span className={clsx(styles.usedMark, styles.used_no)} /> not used</span>
      </div>

      <div className={styles.stepPanel}>
        <div>
          <span className={styles.stepNumber}>Step {i + 1} of {STEPS.length}</span>
          <p className={styles.stepPanelTitle}>{s.short}</p>
          <p className={styles.analogyLine}>{s.office}</p>
          <dl className={styles.whatWhy}>
            <dt>What happens</dt>
            <dd>{s.what}</dd>
            <dt>Why it is needed</dt>
            <dd>{s.why}</dd>
            <dt>{USED[s.used[0]]}</dt>
            <dd>{s.used[1]}</dd>
          </dl>
          <div className={styles.varyRow}>
            <button type="button" className={styles.linkButton} disabled={i === 0} onClick={() => setI(i - 1)}>Previous step</button>
            <button type="button" className={styles.linkButton} disabled={i === STEPS.length - 1} onClick={() => setI(i + 1)}>Next step</button>
          </div>
        </div>
        <div className={styles.demo}>
          <Demo dup={data.duplicates} data={data} />
        </div>
      </div>
    </figure>
  );
}
