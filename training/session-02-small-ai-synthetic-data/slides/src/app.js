(() => {
'use strict';
const DATA = /*__DATA__*/null;

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmt = (n) => n.toLocaleString('en');
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const readingMQ = matchMedia('(max-width: 820px), (max-height: 460px)');

const seg = (a, items, cur, label) =>
  `<div class="seg" role="group" aria-label="${esc(label)}">${items.map(([v, t]) =>
    `<button type="button" data-a="${a}" data-v="${esc(v)}" aria-pressed="${String(v) === String(cur)}">${esc(t)}</button>`).join('')}</div>`;
const chips = (items, cls = '') => `<div class="chips">${items.map((t) => `<span class="chip ${cls}">${esc(t)}</span>`).join('')}</div>`;
const bar = (share, color, extra = '') =>
  `<div class="track">${extra}<i class="fillb" style="width:${(share * 100).toFixed(2)}%;background:${color}"></i></div>`;

/* ------------------------------------------------------------------ terms */
const TERMS = {
  llm: ['Large language model', 'A model trained on a very large amount of text. It can write, translate, summarize, and answer questions. Chat assistants run on such models.'],
  parameter: ['Parameter', 'One of the internal numbers of a model, adjusted during training. The number of parameters measures the size of a model.'],
  smallai: ['Small AI', 'The term of the World Development Report 2026 for AI that is small in size or narrow in scope and is built to run where power, internet connectivity, and digital devices are scarce.'],
  gpu: ['GPU', 'A graphics processor, now also used to run AI models. Large models run on GPUs or similar chips, which cost more and use more power than an ordinary processor.'],
  training: ['Training', 'The step in which a model adjusts its parameters until its answers match a set of labelled examples.'],
  labelled: ['Labelled example', 'An example together with the right answer. For occupation coding, a survey response together with its code.'],
  ml: ['Machine learning', 'The methods that build a model from examples. A team gives the model labelled examples, and training adjusts the model until its answers match them.'],
  token: ['Token', 'A unit of text for a language model, about three quarters of an English word. Hosted services charge for the tokens that a model reads and writes.'],
  synthetic: ['Synthetic data', 'Examples written by a computer program. In this session a large language model writes them.'],
};

/* ------------------------------------------------------------------ demos */
const DEMOS = {};

// A response and the five codes suggested for it (s02 and s31).
const ORDINAL = ['first', 'second', 'third', 'fourth', 'fifth'];
const suggestDemo = (list, o = {}) => {
  const ref = o.ref || 'coder\'s code';
  return {
    state: () => ({i: 0, shown: !!o.open}),
    on(st, a, v) {
      if (a === 'show') { st.shown = true; return; }
      st.i = Math.max(0, Math.min(list().length - 1, st.i + Number(v)));
      st.shown = !!o.open;
    },
    render(st) {
      const all = list(), r = all[st.i];
      const rank = r.suggestions.findIndex((x) => x.code === r.expected);
      const verdict = !r.expected ? 'The response does not say what the person does. The model still returns five codes, and none of them can be checked.'
        : rank < 0 ? `The ${ref} is missing from the five suggestions.`
        : `The ${ref} is the ${ORDINAL[rank]} suggestion.`;
      const rows = r.suggestions.map((x) => `<div class="sgrow${x.code === r.expected ? ' hit' : ''}"><b class="code">${x.code}</b><span>${esc(x.name)}${x.code === r.expected ? ` <em>✓ ${ref}</em>` : ''}</span>${bar(x.score, 'var(--small)')}<span class="v">${x.score.toFixed(2)}</span></div>`).join('');
      return `<div class="card sgleft">
        <span class="lbl">Household survey · question on the main job</span>
        <p class="sgq">What kind of work do you do in your main job?</p>
        <span class="lbl mt">Response</span>
        <p class="msg real">${esc(r.text)}</p>
        <div class="sgcoder"${st.shown ? '' : ' style="visibility:hidden"'}>
          <span class="lbl">${o.refLabel || 'Code given by a coder'}</span>
          <p>${r.expected ? `<b class="code">${r.expected}</b> ${esc(r.expectedName)}` : 'No code. The coder asks for more detail.'}</p>
        </div>
      </div>
      <div class="card small-k sgright" aria-live="polite">
        <span class="lbl">Five codes suggested by the small model and their scores</span>
        ${st.shown ? `<div class="sg">${rows}</div><p class="def">${verdict}</p>`
          : `<div class="sgempty">Which of the ${DATA.occ.groups} groups fits this response?</div>`}
      </div>
      <div class="row">
        ${o.open ? '' : `<button type="button" class="btn" data-a="show" data-v="1"${st.shown ? ' disabled' : ''}>Suggest five codes</button>`}
        <button type="button" class="link" data-a="go" data-v="-1"${st.i === 0 ? ' disabled' : ''}>Previous response</button>
        <button type="button" class="link" data-a="go" data-v="1"${st.i === all.length - 1 ? ' disabled' : ''}>Next response</button>
        <span class="txt-s" style="margin-left:auto">Response ${st.i + 1} of ${all.length}</span>
      </div>`;
    },
  };
};
DEMOS.suggest = suggestDemo(() => DATA.occ.responses);
DEMOS.atwork = suggestDemo(() => DATA.occ.atWork, {open: true, ref: 'reference code', refLabel: 'Reference code'});

// Tasks for a small model (s03).
const REASONS = {
  private: ['Private', 'The real records hold personal data and cannot be sent to an outside service.'],
  rare: ['Rare', 'The important cases are a small share of all cases.'],
  missing: ['Not yet written', 'The cases do not exist until the event happens.'],
  language: ['Local language', 'The cases are written in a local language, with abbreviations and mixed languages.'],
  few: ['Few labelled', 'Few past cases have the right answer recorded.'],
  wording: ['Varied wording', 'People describe the same thing in many ways that no existing list covers.'],
};
const ARROW = (d) => `<path d="${d}" class="iA" marker-end="url(#ah)"/>`;
const CASES = [
  {key: 'survey', name: 'Occupation codes (ISCO)', who: 'A national statistical office',
    task: 'Suggests codes of the international classification of occupations for each occupation written in a household survey. A coder chooses among them.',
    reasons: ['private', 'few', 'wording'],
    input: 'sells roasted maize by the roadside', inputKind: 'Survey response',
    answer: [['Occupation group', 'Street food salespersons'], ['Code (ISCO-08)', '5212']],
    runs: 'On the laptops of the office\'s survey team, without an internet connection in the field.',
    note: 'The session follows this task through the nine steps and through the build of a small model.',
    synth: 'A large model writes survey responses in the respondents\' own words and gives the code of each.',
    icon: `<rect x="10" y="8" width="38" height="40" rx="3" class="iP"/><path d="M16 18 H20 M24 18 H42 M16 28 H20 M24 28 H38 M16 38 H20 M24 38 H40" class="iL"/>${ARROW('M54 28 H66')}<rect x="70" y="20" width="24" height="16" rx="3" class="iT"/>`},
  {key: 'isic', name: 'Industry codes (ISIC)', who: 'A national statistical office',
    task: 'Suggests codes of the international classification of industries for the activity that a business or a worker describes. A coder chooses among them.',
    reasons: ['private', 'few', 'wording'],
    input: 'we grind maize and sell the flour to shops in town', inputKind: 'Survey response',
    answer: [['Industry class', 'Manufacture of grain mill products'], ['Code (ISIC Rev. 4)', '1061']],
    runs: 'On the office\'s own computers. The business records stay in the office.',
    synth: 'A large model writes descriptions of business activities in everyday words and gives the code of each.',
    icon: `<path d="M10 46 V26 L22 33 V26 L34 33 V14 H44 V46 Z" class="iP"/><path d="M16 40 H20 M26 40 H30 M36 40 H40" class="iL"/>${ARROW('M52 28 H66')}<rect x="70" y="20" width="24" height="16" rx="3" class="iT"/>`},
  {key: 'docs', name: 'Document classification', who: 'An organization with thousands of reports',
    task: 'Tags each report with its topics from a fixed list of 86, such as Education or Transport.',
    reasons: ['few'],
    input: 'The project paved 120 km of rural roads. Travel time to the nearest secondary school fell, and attendance rose.', inputKind: 'Report abstract',
    answer: [['Topic', 'Transport'], ['Topic', 'Education']],
    runs: 'On a laptop or a server of the organization. No report leaves the organization.',
    note: 'Part 6 of the session shows a small model for this task and its test on 300 real reports.',
    synth: 'A large model writes sentences in the style of a report and gives the topics of each.',
    icon: `<rect x="16" y="6" width="30" height="38" rx="3" class="iP"/><rect x="10" y="12" width="30" height="38" rx="3" class="iP"/><path d="M16 22 H34 M16 30 H34 M16 38 H28" class="iL"/>${ARROW('M52 28 H64')}<rect x="68" y="14" width="24" height="8" rx="4" class="iT"/><rect x="68" y="27" width="18" height="8" rx="4" class="iT"/><rect x="68" y="40" width="22" height="8" rx="4" class="iT"/>`},
  {key: 'metadata', name: 'Metadata quality', who: 'The team of a data catalogue',
    task: 'Checks the description of each dataset in a catalogue and flags fields that are empty, vague, or inconsistent before the dataset is published.',
    reasons: ['few'],
    input: 'Title: Survey 2019 · Description: Data from the survey · Country: (blank)', inputKind: 'Dataset description',
    answer: [['Flag', 'The title does not name the survey'], ['Flag', 'The description does not describe the data'], ['Flag', 'Country missing']],
    runs: 'On the catalogue\'s own server, as a check before a dataset is published.',
    synth: 'A large model writes dataset descriptions with known faults and lists the faults of each.',
    icon: `<rect x="10" y="6" width="38" height="44" rx="3" class="iP"/><path d="M16 16 H24 M28 16 H42 M16 28 H24 M16 40 H24 M28 40 H38" class="iL"/><rect x="27" y="23" width="16" height="10" rx="2" class="iT"/>${ARROW('M54 28 H66')}<path d="M82 12 L94 40 H70 Z" class="iT"/><path d="M82 22 V31 M82 35 V36" class="iL"/>`},
  {key: 'search', name: 'Semantic search', who: 'A data catalogue',
    task: 'Finds the datasets and documents that answer a question when the question and the text share no words.',
    reasons: ['few', 'wording'],
    input: 'how many people go hungry in the Philippines', inputKind: 'Search query',
    answer: [['Best match', 'Prevalence of severe food insecurity in the population (%)']],
    runs: 'On the catalogue\'s own server, on an ordinary processor.',
    note: 'We built GISTEmbed for this task: a large model guides the training of a small search model. The match shown is the first result of a GISTEmbed model with 23 million parameters.',
    synth: 'A large model reads each document and writes questions that the document answers. The pairs of question and document train the search model.',
    icon: `<circle cx="24" cy="24" r="12" class="iP"/><path d="M33 33 L44 46" class="iSS"/><rect x="54" y="8" width="42" height="12" rx="3" class="iS"/><path d="M58 30 H90 M58 42 H84" class="iL"/>`},
  {key: 'entities', name: 'Entity extraction', who: 'A research library',
    task: 'Finds the names of datasets, places, and organizations in documents. Each document is then linked to its data sources.',
    reasons: ['few', 'wording'],
    input: 'We use the 2018 Demographic and Health Survey of Zambia to estimate child stunting by province.', inputKind: 'Sentence in a report',
    answer: [['Dataset', 'Demographic and Health Survey'], ['Year', '2018'], ['Country', 'Zambia']],
    runs: 'On the library\'s own server, over its whole collection of documents.',
    synth: 'A large model writes sentences that mention datasets, places, and organizations and marks each name.',
    icon: `<rect x="6" y="6" width="88" height="44" rx="3" class="iP"/><path d="M14 18 H30 M56 18 H86 M14 28 H44 M72 28 H86 M14 38 H62" class="iL"/><rect x="34" y="13" width="18" height="10" rx="2" class="iS"/><rect x="48" y="23" width="20" height="10" rx="2" class="iT"/>`},
  {key: 'jobs', name: 'Job matching', who: 'A public employment office',
    task: 'Reads how a job seeker describes their work and suggests the occupations and vacancies that match.',
    reasons: ['private', 'wording', 'few'],
    input: 'I fix motorbikes and sell spare parts at the market, 6 years. I keep the accounts myself.', inputKind: 'Job seeker',
    answer: [['Occupation', 'Motor vehicle mechanic'], ['Matching vacancies', 'Motorcycle mechanic at a garage; sales assistant in a spare parts shop']],
    runs: 'On the employment office\'s own server. The descriptions of job seekers stay in the office.',
    synth: 'A large model writes descriptions of work in the words of job seekers and gives the occupation of each.',
    icon: `<circle cx="22" cy="22" r="8" class="iL"/><path d="M10 44 C10 34 34 34 34 44" class="iL"/>${ARROW('M40 30 H56')}<rect x="60" y="16" width="30" height="24" rx="3" class="iP"/><path d="M70 16 V12 H80 V16" class="iL"/>`},
  {key: 'grievance', name: 'Complaints by text message', who: 'The grievance mechanism of a project',
    task: 'Sorts each complaint by issue and urgency and sends it to the office that handles it.',
    reasons: ['few', 'private', 'language'],
    input: 'pls help, maji hayajatoka tangu jumatatu. contractor broke the pipe near the school', inputKind: 'Text message',
    answer: [['Issue', 'Water supply'], ['Urgency', 'High'], ['Send to', 'Site engineer']],
    runs: 'On a server of the project team. The messages and phone numbers stay with the team.',
    synth: 'A large model writes complaints in the style of text messages and gives the issue and urgency of each.',
    icon: `<rect x="8" y="10" width="40" height="28" rx="6" class="iP"/><path d="M16 38 L14 46 L24 38" class="iP"/><path d="M16 20 H40 M16 28 H32" class="iL"/>${ARROW('M54 26 H66')}<rect x="70" y="14" width="22" height="7" rx="3.5" class="iT"/><rect x="70" y="26" width="16" height="7" rx="3.5" class="iT"/>`},
  {key: 'disaster', name: 'Messages after a flood', who: 'A disaster response center',
    task: 'Picks out requests for rescue, water, or shelter from incoming messages, with the place and the number of people.',
    reasons: ['missing', 'rare', 'language'],
    input: 'Water rising in Kalanga village, 6 people on the roof of the clinic, we need a boat', inputKind: 'Message',
    answer: [['Request', 'Rescue'], ['Place', 'Kalanga village, clinic'], ['People', '6']],
    runs: 'On a laptop at the response center. It keeps working when the internet connection is down.',
    synth: 'A large model writes messages for floods that have not happened and gives the request, the place, and the number of people.',
    icon: `<path d="M14 30 L30 16 L46 30 V44 H14 Z" class="iP"/><path d="M4 40 Q14 34 24 40 T44 40 T64 40 T84 40 T96 40" class="iSS"/><path d="M4 48 Q14 42 24 48 T44 48 T64 48 T84 48 T96 48" class="iSS"/><circle cx="74" cy="20" r="9" class="iT"/><path d="M74 14 V21 M74 24 V25" class="iL"/>`},
  {key: 'farmers', name: 'Farmers\' questions', who: 'An agricultural extension service',
    task: 'Answers questions sent by text message in a local language, from the approved extension guide only, and passes the rest to an extension officer.',
    reasons: ['language', 'few'],
    input: 'My maize leaves have white and yellow stripes and the plants stay short. What do I do?', inputKind: 'Question, translated',
    answer: [['Answer', 'The guide\'s section on leaf streaks: pull out plants with streaks and plant a tolerant variety next season.'], ['Also', 'Sent to the extension officer of the area']],
    runs: 'On the extension service\'s own server, reached by text message from a basic phone.',
    synth: 'A large model writes farmers\' questions in the local language and gives the section of the guide that answers each.',
    icon: `<path d="M30 48 V18" class="iSS"/><path d="M30 30 C20 26 16 18 18 12 C26 14 30 22 30 30 Z" class="iS"/><path d="M30 38 C40 34 44 26 42 20 C34 22 30 30 30 38 Z" class="iS"/><rect x="56" y="10" width="34" height="22" rx="6" class="iP"/><path d="M64 18 H82 M64 24 H76" class="iL"/><path d="M66 32 L64 38 L72 32" class="iP"/>`},
  {key: 'benefits', name: 'Benefit applications', who: 'A social protection agency',
    task: 'Flags missing or inconsistent entries in an application before a caseworker reviews it.',
    reasons: ['private', 'rare'],
    input: 'Household members: 3 · Children under 5: 4 · Monthly income: (blank)', inputKind: 'Application form',
    answer: [['Flag', 'More children under 5 than household members'], ['Flag', 'Monthly income missing']],
    runs: 'On the agency\'s own server, next to the register of applicants.',
    synth: 'A large model writes applications with known faults and lists the faults of each.',
    icon: `<rect x="12" y="6" width="34" height="44" rx="3" class="iP"/><path d="M18 16 H40 M18 24 H40 M18 32 H32" class="iL"/>${ARROW('M54 28 H66')}<path d="M80 12 L92 40 H68 Z" class="iT"/><path d="M80 22 V31 M80 35 V36" class="iL"/>`},
];

const CASE_ROWS = [['Data and documents', 6], ['Services', 5]];

DEMOS.cases = {
  state: () => ({open: 'survey'}),
  on(st, a, v) { st.open = v; },
  render(st) {
    const c = CASES.find((x) => x.key === st.open);
    let from = 0;
    const rows = CASE_ROWS.map(([label, n]) => {
      const row = CASES.slice(from, from + n); from += n;
      return `<span class="lbl">${label}</span>` + row.map((x) =>
        `<button type="button" class="casebtn" data-a="open" data-v="${x.key}" aria-pressed="${x.key === st.open}"><svg viewBox="0 0 100 56" aria-hidden="true">${x.icon}</svg><span>${esc(x.name)}</span></button>`).join('');
    });
    return `<div class="caselist" role="group" aria-label="Task">${rows.join('')}</div>
    <div class="card casepanel" aria-live="polite">
      <div>
        <span class="lbl">${esc(c.who)}</span>
        <p class="big mt">${esc(c.task)}</p>
        <span class="lbl mt2">Why no training data exist</span>
        <ul class="reasons">${c.reasons.map((r) => `<li><span class="chip warn">${REASONS[r][0]}</span>${REASONS[r][1]}</li>`).join('')}</ul>
        <span class="lbl mt2">Where the model runs</span>
        <p class="txt-s mt">${esc(c.runs)}</p>
        ${c.note ? `<p class="txt-s mt">${esc(c.note)}</p>` : ''}
      </div>
      <div>
        <span class="lbl">One practice case · ${esc(c.inputKind)}</span>
        <p class="msg mt">${esc(c.input)}</p>
        <span class="lbl mt2">Right answer</span>
        <dl class="dl mt">${c.answer.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>
        <span class="lbl mt2">Synthetic examples</span>
        <p class="txt-s mt">${esc(c.synth)}</p>
      </div>
    </div>`;
  },
};

// Where a model can run (s08).
const MACHINE_ICONS = {
  device: '<rect x="4" y="8" width="26" height="18" rx="2"/><path d="M1 29 H33"/><rect x="34" y="6" width="11" height="22" rx="2"/>',
  server: '<rect x="8" y="3" width="30" height="8" rx="1.5"/><rect x="8" y="14" width="30" height="8" rx="1.5"/><rect x="8" y="25" width="30" height="8" rx="1.5"/><path d="M12 7 H14 M12 18 H14 M12 29 H14"/>',
  datacenter: '<rect x="3" y="10" width="40" height="23" rx="1.5"/><path d="M3 10 L23 2 L43 10"/><path d="M10 16 V28 M17 16 V28 M24 16 V28 M31 16 V28 M38 16 V28"/>',
};
DEMOS.machines = {
  state: () => ({m: 0}),
  on(st, a, v) { st.m = Number(v); },
  render(st) {
    const {models, computers} = DATA.machines;
    const m = models[st.m];
    const color = `var(--${m.kind})`;
    const squares = (n, cols) => {
      let h = `<div class="squares" style="grid-template-columns:repeat(${cols},minmax(0,.98em));--fillc:${color}" aria-hidden="true">`;
      for (let i = 0; i < n; i++) {
        const f = Math.max(0, Math.min(1, m.gb - i));
        h += `<span>${f > 0 ? `<i style="width:${f * 100}%"></i>` : ''}</span>`;
      }
      return h + '</div>';
    };
    const card = (c) => {
      const fits = c.gb === null || m.gb <= c.gb;
      const mem = c.key === 'device' ? squares(20, 10) : c.key === 'server' ? squares(200, 25)
        : `<div class="bigmem${m.gb > 200 ? ' on' : ''}" style="--fillc:${color}">Hundreds of GB or more</div>`;
      return `<div class="card machine">
        <svg viewBox="0 0 46 36" aria-hidden="true">${MACHINE_ICONS[c.key]}</svg>
        <div><p class="name">${esc(c.label)}</p><p class="sub">${esc(c.memory)} of memory</p></div>
        ${mem}
        <p class="verdict ${fits ? 'yes' : 'no'}">${fits ? '✓ The model fits' : '✗ Too large for this computer'}</p>
        <dl class="dl" style="font-size:.8em"><dt>Internet</dt><dd>${esc(c.internet)}</dd><dt>The text</dt><dd>${esc(c.text)}</dd></dl>
      </div>`;
    };
    return `<div class="row" style="flex-wrap:nowrap;align-items:flex-start">
      ${seg('m', models.map((x, i) => [i, x.name]), st.m, 'Model')}
      <p class="modelline" aria-live="polite"><span class="dot ${m.kind}"></span><b>${esc(m.params)}</b>, ${esc(m.memory)}. ${esc(m.role)}</p>
    </div>
    <div class="machines">${computers.map(card).join('')}</div>`;
  },
};

// Rare cases (s12).
DEMOS.rare = {
  state: () => ({on: false}),
  on(st) { st.on = !st.on; },
  render(st) {
    const o = DATA.occ, real = new Set([37, 141, 236, 318]);
    let dots = '';
    for (let i = 0; i < 400; i++) dots += `<span${real.has(i) ? ' class="real"' : ''}></span>`;
    if (st.on) for (let i = 0; i < 120; i++) dots += `<span class="written" style="animation-delay:${i * 6}ms"></span>`;
    return `<div>
      <div class="raregrid" aria-hidden="true">${dots}</div>
      <p class="legend mt2"><span><i style="background:var(--line)"></i>real response</span><span><i style="background:var(--bad)"></i>real response from the rare group</span>${st.on ? '<span><i style="background:var(--large)"></i>written response for the group</span>' : ''}</p>
    </div>
    <div aria-live="polite">
      <p class="stat">${st.on ? '120' : '4'}<small>${st.on ? 'written responses for the group' : 'real responses from one group among 400'}</small></p>
      <p class="txt mt2">${st.on
        ? 'The model now learns from 120 written responses of varied wording for the group. The 4 real ones are kept for testing.'
        : 'Among 400 real responses, 4 come from one occupation group. A model trained on them sees too few to learn how people describe that work.'}</p>
      <p class="mt2"><button type="button" class="btn" data-a="t" data-v="1">${st.on ? 'Remove the written responses' : 'Add 120 written responses for the group'}</button></p>
      <p class="txt-s mt2">Among ${fmt(o.testTitles)} real job titles from a job board, ${o.groups - o.groupsInTest} of the ${o.groups} groups of ISCO-08 do not occur.</p>
    </div>`;
  },
};

// A labelled job title rewritten as survey answers (s15).
DEMOS.rewrite = {
  on() { return false; },
  render() {
    const r = DATA.occ.rewrite;
    return `<div class="card large-k">
      <span class="lbl">A labelled job title</span>
      <div class="rwtitle mt"><b>${esc(r.title)}</b><span class="ans">${esc(r.code)} ${esc(r.name)}</span></div>
      <span class="lbl mt2" style="display:block">Three answers written by a large model</span>
      <ul class="rwlist">${r.answers.map((x) => `<li><small>${esc(x.style)}</small><span class="msg">${esc(x.text)}</span></li>`).join('')}</ul>
    </div>
    <div class="card small-k" style="display:flex;flex-direction:column">
      <span class="lbl">From a job title to survey answers</span>
      <ul class="ticks">
        <li data-step="1">The wording changes from an advertisement to the words of a respondent.</li>
        <li data-step="2">The code of the job title stays with each answer.</li>
        <li data-step="3">A second model checks each answer against the code.</li>
      </ul>
      <p class="txt-s" style="margin-top:auto"><button type="button" class="link" data-go="p5" style="font-size:1em">Part 5 shows this build and its test</button></p>
    </div>`;
  },
};

// Step 1: describe the task.
const UNITS = [
  ['5212', 'Street food salespersons', 'Service and sales workers', 'Prepare food and drinks that are ready to eat and sell them in streets and public places.'],
  ['5211', 'Stall and market salespersons', 'Service and sales workers', 'Sell goods from a stall at a market or on a street.'],
  ['9520', 'Street vendors (excluding food)', 'Elementary occupations', 'Sell a small range of goods other than ready-to-eat food in streets and public places.'],
  ['5221', 'Shopkeepers', 'Service and sales workers', 'Run a small shop, alone or with a few helpers.'],
  ['6310', 'Subsistence crop farmers', 'Skilled agricultural, forestry and fishery workers', 'Grow crops mainly to feed their own household.'],
  ['9211', 'Crop farm labourers', 'Elementary occupations', 'Do routine tasks on a farm for the farmer, such as weeding and harvesting.'],
];
DEMOS.task = {
  state: () => ({i: 0}),
  on(st, a, v) { st.i = Number(v); },
  render(st) {
    const [code, name, major, text] = UNITS[st.i];
    return `<span class="lbl">Six of the ${DATA.occ.groups} groups in the list of answers (ISCO-08)</span>
    <div class="chips" role="group" aria-label="Occupation group">${UNITS.map(([c, n], i) =>
      `<button type="button" class="chip off" style="font-size:.86em" data-a="i" data-v="${i}" aria-pressed="${i === st.i}">${c} ${esc(n)}</button>`).join('')}</div>
    <p class="def" aria-live="polite"><b>${code} ${esc(name)}.</b> ${esc(text)}<span class="major">Major group ${code[0]}: ${esc(major)}</span></p>
    <p class="hint" style="margin-top:auto">Select a group to see its description, shortened for this slide. The large model and the people who check its cases read the same descriptions.</p>`;
  },
};

// Step 2: show a few real cases.
const SEEDS = ['sell tea and mandazi at the stage', 'helping my uncle farm, we plant maize', 'fundi, i repair phones and radios'];
const UNSEEDED = ['Street Food Vendor', 'Agricultural Labourer', 'Mobile Phone Repair Technician'];
const SEEDED = ['i fry fish and sell at the landing site', 'dig and weed for a neighbour, paid by the day', 'fix bicycles and punctures under the tree by the road'];
DEMOS.seeds = {
  state: () => ({on: 0}),
  on(st, a, v) { st.on = Number(v); },
  render(st) {
    const list = (items, cls = '') => `<div class="msgs">${items.map((t) => `<p class="msg ${cls}">${esc(t)}</p>`).join('')}</div>`;
    return `${seg('on', [[0, 'No samples'], [1, 'Three real samples']], st.on, 'Request to the large model')}
    ${st.on ? `<span class="lbl">Real responses added to the request</span>${list(SEEDS, 'real')}` : ''}
    <span class="lbl">Cases written by the large model</span>
    ${list(st.on ? SEEDED : UNSEEDED)}
    <p class="hint" style="margin-top:auto">The responses on this slide were written for this session as an illustration.</p>`;
  },
};

// Step 3: vary the cases.
const ATTRS = [
  ['Group', UNITS.map(([, n]) => n)],
  ['Length', ['two or three words', 'one short sentence', 'two sentences']],
  ['Style', ['a job title', 'the tasks done', 'the place of work', 'the product or service']],
  ['Language', ['English', 'Swahili', 'English and Swahili mixed']],
  ['Detail', ['vague', 'specific']],
  ['Spelling', ['correct', 'with errors']],
];
DEMOS.vary = {
  state: () => ({on: new Set(['Group']), pick: Object.fromEntries(ATTRS.map(([a]) => [a, 0]))}),
  on(st, a, v) {
    if (a === 'attr') { st.on.has(v) ? st.on.delete(v) : st.on.add(v); }
    else for (const [name, vals] of ATTRS) st.pick[name] = Math.floor(Math.random() * vals.length);
  },
  render(st) {
    const val = (name) => (st.on.has(name) ? ATTRS.find(([a]) => a === name)[1][st.pick[name]] : null);
    const combos = ATTRS.reduce((n, [a, vals]) => n * (st.on.has(a) ? vals.length : 1), 1);
    const dup = DATA.workflow.duplicates;
    const share = Math.round(dup.repeats / dup.written * 100);
    let request = `Write a survey response on the main job${val('Group') ? ` of a person in the group "${val('Group')}"` : ''}.`;
    for (const name of ['Length', 'Style', 'Language', 'Detail', 'Spelling']) if (val(name)) request += ` ${name}: ${val(name)}.`;
    let grid = '';
    for (let i = 0; i < 100; i++) grid += `<span${i < share ? ' class="on"' : ''}></span>`;
    return `<span class="lbl">Attributes that change from one request to the next</span>
    <div class="chips" role="group" aria-label="Attributes">${ATTRS.map(([a]) =>
      `<button type="button" class="chip off" style="font-size:.86em" data-a="attr" data-v="${a}" aria-pressed="${st.on.has(a)}">${st.on.has(a) ? '✓ ' : ''}${a}</button>`).join('')}</div>
    <p class="prompt" aria-live="polite" style="min-height:4.6em">${esc(request)}</p>
    <div class="row"><button type="button" class="btn" data-a="draw" data-v="1">Draw a new request</button><span class="txt"><b>${fmt(combos)}</b> different requests possible</span></div>
    <div class="evidence">
      <div class="hundred" aria-hidden="true">${grid}</div>
      <div><span class="lbl">A measured result</span>
      <p class="txt-s mt">Asked many times for titles of the same occupation group, ${esc(dup.writer)} wrote the same common titles again: ${fmt(dup.repeats)} of ${fmt(dup.written)} job titles (${share}%) repeated an earlier one. A repeated case teaches the small model nothing new, and the checks remove it.</p></div>
    </div>`;
  },
};

// Step 4: write and check.
const CHECKED = [
  ['sells boiled eggs and tea outside the bus park', '5212 Street food', '5212 Street food'],
  ['sells boiled eggs and tea outside the bus park', '5212 Street food', '5212 Street food', 'Repeats an earlier case'],
  ['sells second-hand clothes at the market', '5212 Street food', '5211 Market stall', 'The two models disagree'],
  ['plants rice on the family plot, for home use', '6310 Subsistence crops', '6310 Subsistence crops'],
  ['sells airtime and phone cards from a kiosk', '5215', '5211 Market stall', 'Code not in the list'],
  ['weeds and harvests on a tea estate, paid by the day', '9211 Farm labour', '9211 Farm labour'],
  ['business', '', '5221 Shopkeeper', 'No code given'],
];
DEMOS.check = {
  state: () => ({run: false}),
  on(st) { st.run = !st.run; },
  render(st) {
    const kept = CHECKED.filter((r) => !r[3]).length;
    return `<div class="checkt" role="table" aria-label="Cases and the codes of two models">
      <div class="head" role="row"><span role="columnheader">Case</span><span role="columnheader">Writer's code</span><span role="columnheader">Second model's code</span></div>
      ${CHECKED.map(([text, writer, checker, drop]) => {
        const out = st.run && drop;
        return `<div role="row"${out ? ' class="drop"' : ''}><span role="cell" class="txtc">${out ? `<s>${esc(text)}</s><em>✕ ${esc(drop)}</em>` : esc(text)}</span><span role="cell">${esc(writer || '(none)')}</span><span role="cell">${esc(checker)}</span></div>`;
      }).join('')}
    </div>
    <div class="row" style="margin-top:auto"><button type="button" class="btn" data-a="run" data-v="1">${st.run ? 'Show the cases before the checks' : 'Run the checks'}</button>
    <span class="txt" aria-live="polite">${st.run ? `<b>${kept} of ${CHECKED.length}</b> cases are kept.` : 'A second model codes each case without seeing the writer\'s code.'}</span></div>`;
  },
};

// Step 5: train the small model.
DEMOS.train = {
  on() { return false; },
  render() {
    let pages = '';
    for (let i = 0; i < 4; i++) pages += `<rect x="${14 + i * 6}" y="${14 + i * 6}" width="70" height="50" rx="4" class="iP"/>`;
    return `<svg viewBox="0 0 260 104" style="width:100%;max-height:14em" role="img" aria-label="A stack of checked cases goes into training, and a small model comes out.">
      ${pages}<path d="M40 50 H76 M40 58 H66" class="iL"/><path d="M112 50 H150" class="iA" marker-end="url(#ah)"/>
      <rect x="166" y="26" width="48" height="48" rx="10" class="iS"/><circle cx="182" cy="44" r="4" class="iD"/><circle cx="198" cy="56" r="4" class="iD"/><path d="M182 44 L198 56" class="iSS"/>
      <text x="58" y="98" text-anchor="middle" font-size="8">checked cases</text><text x="131" y="42" text-anchor="middle" font-size="8">training</text><text x="190" y="98" text-anchor="middle" font-size="8">small model</text>
    </svg>
    <p class="def">Training turns the cases into a model that answers on its own, on the organization's own computer.</p>
    <p class="txt-s" style="margin-top:auto">Part 5 shows the scores of the occupation model at 13 points of its training. <button type="button" class="link" data-go="s29" style="font-size:1em">Go to that slide</button></p>`;
  },
};

// Step 6: test on real cases.
DEMOS.test = {
  state: () => ({wrong: false}),
  on(st, a, v) { st.wrong = !!v; },
  render(st) {
    return `<div class="piles">
      <div class="pile${st.wrong ? ' bad' : ''}"><b>Practice cases</b><span>Thousands of cases written by the large model</span><span class="use">${st.wrong ? 'Used for training and testing' : 'Used for training'}</span></div>
      <div class="pile${st.wrong ? '' : ' test'}"><b>Real cases</b><span>A few hundred or more with answers given by people</span><span class="use">${st.wrong ? 'Not used' : 'Used only for testing'}</span></div>
    </div>
    <label class="toggle"><input type="checkbox" data-a="wrong" data-v="1"${st.wrong ? ' checked' : ''}>Test on practice cases</label>
    <p class="def" aria-live="polite">${st.wrong
      ? 'The practice cases used for testing share the writing habits of the training cases. A high score then says little about real responses.'
      : 'The real cases are set aside before any practice case is written, and no model is trained on them.'}</p>
    <div class="scores"><span class="lbl">Two scores from one test</span>
      <p><b>First code is right</b>The share of test cases where the model's first code is the code given by people.</p>
      <p><b>Right code among five</b>The share of test cases where that code is among the model's first five codes.</p></div>`;
  },
};

// Step 7: practise the weak spots.
const ROUND = ['Test on separate real cases', 'Find the weakest groups', 'Write more cases on them', 'Train again'];
const ROUND_TEST = [
  ['teacher at the primary school', '2341 Primary school', '2341 Primary school'],
  ['sells roasted maize by the roadside', '5212 Street food', '5223 Shop sales'],
  ['fries chips and sells them outside the school', '5212 Street food', '5223 Shop sales'],
  ['weeds and harvests on a tea estate, paid by the day', '9211 Farm labour', '9211 Farm labour'],
];
const ROUND_NEW = ['fry cassava and sell it by the road in the evening', 'cook rice and beans and sell to builders at the site', 'sell chapati and tea near the bus park'];
DEMOS.rounds = {
  state: () => ({k: 0}),
  on(st) { st.k = (st.k + 1) % ROUND.length; },
  render(st) {
    const stage = [
      `<div class="checkt" role="table" aria-label="Test responses with the coder's code and the model's first code">
        <div class="head" role="row"><span role="columnheader">Real response</span><span role="columnheader">Coder's code</span><span role="columnheader">Model's first code</span></div>
        ${ROUND_TEST.map(([text, coder, model]) => `<div role="row"${coder === model ? '' : ' class="drop"'}><span role="cell" class="txtc">${esc(text)}</span><span role="cell">${esc(coder)}</span><span role="cell">${esc(model)}${coder === model ? '' : '<em>✕ wrong</em>'}</span></div>`).join('')}
      </div>`,
      `<p class="def">The mistakes are counted for each group. The weakest group in this test is <b>5212 Street food salespersons</b>: the model gives these responses the code of shop sales assistants.</p>`,
      `<span class="lbl">Cases written by the large model for group 5212</span>
      <div class="msgs">${ROUND_NEW.map((t) => `<p class="msg">${esc(t)}</p>`).join('')}</div>`,
      `<p class="def">The model is trained on the earlier cases and the new ones. It takes the same test again, and the test cases stay out of training.</p>`,
    ][st.k];
    return `<div class="loop">${ROUND.map((t, i) => `<span${i === st.k ? ' class="on" aria-current="step"' : ''}><b>${i + 1}</b>${t}</span>`).join('')}</div>
    <div class="roundstage" aria-live="polite">${stage}</div>
    <div class="row" style="margin-top:auto"><button type="button" class="btn" data-a="next" data-v="1">${st.k === ROUND.length - 1 ? 'Start the next round' : 'Next stage'}</button>
    <span class="hint">The responses on this slide were written for this session as an illustration.</span></div>`;
  },
};

// Step 8: ask people which is better.
const PAIRS = [
  {kind: 'Which written response is more realistic?',
    a: 'I am employed as a Street Food Vendor, responsible for the preparation and sale of meals to customers in public areas.',
    b: 'sell chips and eggs outside the school gate',
    note: 'A respondent answers an interviewer in a few words. The first case reads like a job advertisement.'},
  {kind: 'Which case has the right code?',
    a: '"sells second-hand clothes at the market." Code: 5212 Street food salespersons.',
    b: '"sells boiled maize and groundnuts at the taxi park." Code: 5212 Street food salespersons.',
    note: 'Second-hand clothes are not food. A case with a wrong code teaches the small model a wrong code.'},
  {kind: 'Which case is more useful for training?',
    a: '"business." Code: 5221 Shopkeepers.',
    b: '"has a small shop at home, sells soap, sugar and paraffin." Code: 5221 Shopkeepers.',
    note: 'The word "business" does not say what the person does, so nobody can check its code. The second case names the shop and the goods.'},
];
DEMOS.pairs = {
  state: () => ({i: 0, picks: PAIRS.map(() => null)}),
  on(st, a, v) {
    if (a === 'pick') { if (st.picks[st.i]) return false; st.picks[st.i] = v; }
    else st.i = Math.max(0, Math.min(PAIRS.length - 1, st.i + Number(v)));
  },
  render(st) {
    const p = PAIRS[st.i], pick = st.picks[st.i];
    const answered = st.picks.filter(Boolean).length, agreed = st.picks.filter((x) => x === 'b').length;
    const side = (k) => `<button type="button" class="pairbtn${pick && k === 'b' ? ' best' : ''}" data-a="pick" data-v="${k}" aria-pressed="${pick === k}"><span class="lbl">${k.toUpperCase()}</span>${esc(p[k])}</button>`;
    return `<span class="lbl">Pair ${st.i + 1} of ${PAIRS.length}: ${esc(p.kind)}</span>
    <div class="pairc">${side('a')}${side('b')}</div>
    <p class="def" aria-live="polite" style="font-size:.92em">${pick
      ? `<b>An experienced coder chose ${pick === 'b' ? 'the same' : 'the other one'}.</b> ${esc(p.note)}`
      : 'Select the better one of the two.'}</p>
    <div class="row" style="margin-top:auto">
      <button type="button" class="link" data-a="go" data-v="-1"${st.i === 0 ? ' disabled' : ''}>Previous pair</button>
      <button type="button" class="link" data-a="go" data-v="1"${st.i === PAIRS.length - 1 ? ' disabled' : ''}>Next pair</button>
      <span class="txt-s" style="margin-left:auto">Same choice as the coder: ${agreed} of ${answered}</span>
    </div>`;
  },
};

// Step 9: watch and refresh.
DEMOS.watch = {
  on() { return false; },
  render() {
    const loop = ['People review a sample of real cases', 'The review shows a change', 'New cases are written and the model is trained again'];
    return `<span class="lbl">World Development Report 2026, spotlight 1</span>
    <p class="def">A screening tool for diabetic eye disease rejected more than one-fifth of the images that nurses took, because the lighting in the clinic differed from that in its training images. The report advises testing a tool where it will be used and watching it as conditions change.</p>
    <p class="txt">For a small model trained on practice cases, a change can come from new kinds of work, a survey in another language, or a revision of the classification. Regular reviews of a sample of real cases show when new practice cases and a new round of training are needed.</p>
    <div class="loop" style="grid-template-columns:repeat(3,1fr);margin-top:auto">${loop.map((t, i) => `<span><b>${i + 1}</b>${t}</span>`).join('')}</div>`;
  },
};

// The report on s36.
DEMOS.report = {
  on() { return false; },
  render() {
    const d = DATA.doc;
    return `<span class="lbl">A report · ${d.year}</span>
    <h3>${esc(d.title)}</h3>
    <p class="abs">${esc(d.text)}</p>
    <span class="lbl mt">Topics recorded for the report</span>
    ${chips(d.reference)}`;
  },
};

// The method in three steps (s42). The process on s05 uses the same icons.
const METHOD = [
  {term: 'Synthetic data', kind: 'large',
    title: 'A large model writes practice examples',
    method: 'A large model writes thousands of examples, each with its answer.',
    icon: '<rect x="6" y="14" width="44" height="44" rx="8" class="iLg"/><path d="M18 30 H38 M18 38 H38 M18 46 H30" class="iLS"/><path d="M56 36 H66" class="iA" marker-end="url(#ah)"/><rect x="72" y="10" width="42" height="52" rx="4" class="iP"/><path d="M79 20 H107 M79 26 H101" class="iL"/><rect x="79" y="31" width="18" height="6" rx="3" class="iT"/><path d="M79 44 H107 M79 50 H97" class="iL"/><rect x="79" y="54" width="14" height="5" rx="2.5" class="iT"/>'},
  {term: 'Distillation', kind: 'small',
    title: 'A small model learns from the examples',
    method: 'A small model is trained on the examples until it gives the same answers.',
    icon: '<rect x="6" y="10" width="42" height="52" rx="4" class="iP"/><path d="M13 20 H41 M13 26 H35" class="iL"/><rect x="13" y="31" width="18" height="6" rx="3" class="iT"/><path d="M13 44 H41 M13 50 H31" class="iL"/><rect x="13" y="54" width="14" height="5" rx="2.5" class="iT"/><path d="M54 36 H68" class="iA" marker-end="url(#ah)"/><rect x="76" y="22" width="28" height="28" rx="6" class="iS"/><circle cx="86" cy="33" r="2.5" class="iD"/><circle cx="95" cy="40" r="2.5" class="iD"/><path d="M86 33 L95 40" class="iSS"/>'},
  {term: 'Small AI', kind: 'small',
    title: 'The small model works alone',
    method: 'The small model does the task on the organization\'s own computer, without the large model and without an internet connection.',
    icon: '<rect x="20" y="12" width="80" height="44" rx="4" class="iP"/><path d="M10 62 H110" class="iL" stroke-width="4"/><rect x="30" y="22" width="22" height="22" rx="5" class="iS"/><circle cx="38" cy="31" r="2" class="iD"/><circle cx="45" cy="36" r="2" class="iD"/><path d="M38 31 L45 36" class="iSS"/><path d="M60 26 H90 M60 32 H84" class="iL"/><rect x="60" y="38" width="18" height="6" rx="3" class="iT"/>'},
];
const methodCards = () => METHOD.map((s, i) =>
  `<div class="card ${s.kind}-k"><svg viewBox="0 0 120 72" aria-hidden="true">${s.icon}</svg>
    <span class="lbl">Step ${i + 1} · ${s.term}</span><h3>${s.title}</h3><p class="txt">${esc(s.method)}</p></div>`).join('');
DEMOS.sum3 = {on() { return false; }, render: methodCards};

// The process and the parts of the session (s05).
const CYCLE = [
  {k: 'plain', lbl: 'The task', t: 'A response needs one code from a fixed list',
    icon: '<rect x="6" y="19" width="52" height="32" rx="9" class="iP"/><path d="M16 31 H48 M16 39 H38" class="iL"/><path d="M63 35 H75" class="iA" marker-end="url(#ah)"/><rect x="80" y="23" width="36" height="24" rx="12" class="iT"/><text x="98" y="39.5" text-anchor="middle" font-size="11.5" font-weight="650">5212</text>'},
  {k: 'large', lbl: 'Practice cases', t: 'A large model writes thousands of responses with their codes', icon: METHOD[0].icon},
  {k: 'small', lbl: 'Training', t: 'A small model learns from the practice cases', icon: METHOD[1].icon},
  {k: 'teal', lbl: 'Test', t: 'Real responses coded by people test the small model',
    icon: '<rect x="26" y="10" width="42" height="52" rx="4" class="iP"/><path d="M33 20 H61 M33 26 H55" class="iL"/><rect x="33" y="31" width="18" height="6" rx="3" class="iT"/><path d="M33 44 H61 M33 50 H51" class="iL"/><circle cx="80" cy="46" r="13" class="iR"/><path d="M74 46.5 l4.5 4.5 l8 -9.5" class="iRS"/>'},
  {k: 'small', lbl: 'Use', t: 'The small model suggests codes on the office\'s own computer', icon: METHOD[2].icon},
];
const PARTS = [
  {cls: 'here', name: 'Tasks', sub: 'This part'},
  {cls: 'large', n: 3, go: 'p3', name: 'Synthetic data', sub: 'Where the practice cases come from'},
  {cls: 'small', n: 2, go: 'p2', name: 'Small model', sub: 'Size, memory, and cost'},
  {cls: '', n: 6, go: 'p6', name: 'Report tagging', sub: 'A second model and its test on real reports'},
  {cls: '', n: 7, go: 'p7', name: 'Small or large', sub: 'The tasks that fit a small model'},
  {cls: 'wide', n: 4, go: 'p4', name: 'Nine steps', sub: 'The whole process in nine steps'},
  {cls: 'wide', n: 5, go: 'p5', name: 'Occupation coding', sub: 'The build of one model from the examples to the test'},
];
DEMOS.cycle = {
  on() { return false; },
  render() {
    const part = (x) => (x.go
      ? `<button type="button" class="cypart ${x.cls}" style="--o:${x.n}" data-go="${x.go}"><i>${x.n}</i><b>${x.name}</b><span>${x.sub}</span></button>`
      : `<div class="cypart ${x.cls}"><b>${x.name}</b><span>${x.sub}</span></div>`);
    return `<ol class="cygrid cyflow">${CYCLE.map((x) => `<li class="card ${x.k}-k"><svg viewBox="0 0 120 72" aria-hidden="true">${x.icon}</svg>
      <span class="lbl">${x.lbl}</span><b>${esc(x.t)}</b></li>`).join('')}</ol>
    <div class="cygrid cyloop" data-step="1"><span class="cap">More practice cases for the kinds of cases that the small model gets wrong</span><span class="arc"></span></div>
    <div class="cyparts" data-step="2"><span class="lbl">Parts of the session</span>
      <div class="cygrid">${PARTS.map(part).join('')}</div></div>`;
  },
};

// The three sets of training texts (s26).
DEMOS.sets = {
  on() { return false; },
  render() {
    const kinds = ['large-k', 'teal-k', 'large-k'], units = ['job titles', 'job titles', 'answers'];
    return DATA.occ.sets.items.map((x, i) => `<div class="card ${kinds[i]}"${i ? ` data-step="${i}"` : ''}>
      <span class="lbl">${esc(x.name)}</span>
      <p class="stat mt">${fmt(x.n)}<small>${units[i]}</small></p>
      <p class="txt-s">${esc(x.how)}</p>
      <span class="lbl">Example</span>
      <p class="msg">${esc(x.example.text)}</p>
      <p class="txt-s"><b class="code">${esc(x.example.code)}</b> ${esc(x.example.name)}</p>
    </div>`).join('');
  },
};

// One request and the answers written from it (s27).
const answerList = (answers) => `<ul class="rwlist">${answers.map((a) =>
  `<li><small>${esc(a.style)}</small><span class="msg">${esc(a.text)}</span></li>`).join('')}</ul>`;
DEMOS.write = {
  mount(el) {
    const list = DATA.occ.rewritten;
    let timer = null;
    const request = (x) => esc(DATA.occ.request).replace('{title}', () => `<b>${esc(x.title)}</b>`).replace('{label}', () => `<b>${esc(x.name)}</b>`);
    el.innerHTML = `<div class="card storyleft">
      <span class="lbl">Shortened request to the large model</span>
      <p class="prompt request">${request(list[0])}</p>
      <div class="row" style="margin-top:auto"><button type="button" class="btn">Write the answers</button><span class="txt-s" aria-live="polite"></span></div>
    </div>
    <div style="display:flex;flex-direction:column;min-height:0">
      <div class="empty">The answers written by the large model appear here.</div>
      <div class="sents" hidden tabindex="0" aria-label="Answers written by the large model"></div>
    </div>`;
    const btn = $('.btn', el), count = $('.row .txt-s', el), empty = $('.empty', el), box = $('.sents', el), prompt = $('.prompt', el);
    const add = (i) => {
      const x = list[i];
      prompt.innerHTML = request(x);
      box.insertAdjacentHTML('beforeend', `<div class="sent"><div class="rwtitle"><b>${esc(x.title)}</b><span class="ans">${esc(x.code)} ${esc(x.name)}</span></div>${answerList(x.answers)}</div>`);
      box.scrollTop = box.scrollHeight;
      count.textContent = `${i + 1} of ${list.length} job titles`;
    };
    btn.addEventListener('click', () => {
      clearTimeout(timer);
      box.innerHTML = '';
      empty.hidden = true;
      box.hidden = false;
      btn.textContent = 'Write again';
      if (reduced) { list.forEach((_, i) => add(i)); box.scrollTop = 0; return; }
      let i = 0;
      const tick = () => { add(i++); if (i < list.length) timer = setTimeout(tick, 700); };
      timer = setTimeout(tick, 300);
    });
  },
};

// The check of the rewritten answers (s28).
DEMOS.checked = {
  state: () => ({i: 0}),
  on(st, a, v) { st.i = Math.max(0, Math.min(DATA.occ.rewritten.length - 1, st.i + Number(v))); },
  render(st) {
    const c = DATA.occ.check, all = DATA.occ.rewritten, x = all[st.i];
    const styles = c.byStyle.map((y) => {
      const share = y.kept / (y.kept + y.removed);
      return `<b>${esc(y.style)}</b>${bar(share, 'var(--large)')}<span class="v">${Math.round(share * 100)}%</span>`;
    }).join('');
    const rows = x.answers.map((a) => {
      const chosen = a.kept ? `${x.code} ${x.name}` : a.second ? `${a.second} ${a.secondName}` : 'None of the listed groups';
      return `<div role="row"${a.kept ? '' : ' class="drop"'}><span role="cell"><small>${esc(a.style)}</small>${esc(a.text)}</span><span role="cell">${esc(chosen)}</span><span role="cell"><b>${a.kept ? '✓ Kept' : '✕ Removed'}</b></span></div>`;
    }).join('');
    return `<div class="card large-k">
      <p class="stat">${fmt(c.written - c.kept)}<small>of ${fmt(c.written)} answers removed by the check</small></p>
      <span class="lbl mt2" style="display:block">Share of answers kept for each kind of answer</span>
      <div class="hb ckbars mt">${styles}</div>
      <span class="lbl mt2" style="display:block">Reply of the second model for the removed answers</span>
      <dl class="facts" style="margin-top:.5em;border-top:0;padding-top:0"><dt>None of the listed groups</dt><dd>${fmt(c.none)}</dd><dt>Another group</dt><dd>${fmt(c.other)}</dd></dl>
    </div>
    <div class="card" style="display:flex;flex-direction:column;gap:.7em">
      <span class="lbl">One job title and its three answers</span>
      <div class="rwtitle"><b>${esc(x.title)}</b><span class="ans">${esc(x.code)} ${esc(x.name)}</span></div>
      <div class="checkt ck3" role="table" aria-label="Three answers and the group chosen by the second model" aria-live="polite">
        <div class="head" role="row"><span role="columnheader">Answer</span><span role="columnheader">Group chosen by the second model</span><span role="columnheader">Result</span></div>
        ${rows}
      </div>
      <div class="row" style="margin-top:auto">
        <button type="button" class="link" data-a="go" data-v="-1"${st.i === 0 ? ' disabled' : ''}>Previous job title</button>
        <button type="button" class="link" data-a="go" data-v="1"${st.i === all.length - 1 ? ' disabled' : ''}>Next job title</button>
        <span class="txt-s" style="margin-left:auto">Job title ${st.i + 1} of ${all.length}</span>
      </div>
    </div>`;
  },
};

// The training run of the occupation model (s29): its scores on the test answers at recorded points of training.
DEMOS.learn = {
  mount(el) {
    const d = DATA.learn, pts = d.points, N = pts.length, last = pts[N - 1].passes;   // stop N is the saved model
    const W = 600, H = 330, L = 46, R = 14, T = 18, B = 46, SAVED = last + 0.8;
    const x = (p) => L + p / (last + 1.15) * (W - L - R), y = (v) => T + (1 - v) * (H - T - B);
    const path = (k, n) => pts.slice(0, n + 1).map((p, i) => `${i ? 'L' : 'M'}${x(p.passes).toFixed(1)} ${y(p[k]).toFixed(1)}`).join(' ');
    const pc = (v) => `${(v * 100).toFixed(1)}%`;
    const code = (c) => `<b>${c.code}</b><small title="${esc(c.name)}">${esc(c.name)}</small>`;
    let i = N, timer = null;
    el.innerHTML = `<div class="card lcard">
      <div data-read aria-live="polite"></div>
      <svg class="lchart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of the ${fmt(d.testAnswers)} test answers with the right code, at ${N} points of training and for the saved model" data-chart></svg>
      <div class="row lctl"><button type="button" class="btn" data-play></button>
        <input type="range" min="0" max="${N}" step="1" value="${N}" aria-label="Point of the training run" data-at></div>
    </div>
    <div class="card lcard">
      <span class="lbl">Six test answers in one of the runs</span>
      <div class="checkt learnt" role="table" aria-label="Six test answers, their reference code and the model's first code" aria-live="polite" data-rows></div>
      <p class="txt-s" style="margin-top:auto" aria-live="polite" data-match></p>
    </div>`;
    const svg = $('[data-chart]', el), range = $('[data-at]', el), playBtn = $('[data-play]', el);
    const draw = () => {
      const saved = i === N, p = saved ? d.saved : pts[i], n = Math.min(i, N - 1), px = x(saved ? SAVED : p.passes);
      const when = saved ? 'Saved model' : i === 0 ? 'Before training'
        : `${fmt(p.texts)} texts read · ${p.passes < 1 ? `${Math.round(p.passes * 100)}% of one pass` : p.passes === 1 ? 'one pass' : `${p.passes} passes`}`;
      $('[data-read]', el).innerHTML = `<span class="lbl">${when}</span>
        <div class="lstats"><p><i></i><b>${pc(p.top1)}</b>first code is right</p><p><i class="k5"></i><b>${pc(p.top5)}</b>right code among five</p></div>`;
      const dot = (k, q, cx, cls = '') => `<circle class="ld${cls}" cx="${cx.toFixed(1)}" cy="${y(q[k]).toFixed(1)}" r="${cls === ' sel' ? 6 : 3}"/>`;
      svg.innerHTML = `${[0, 0.25, 0.5, 0.75, 1].map((t) => `<path class="lg" d="M${L} ${y(t)} H${W - R}"/><text class="lt" x="${L - 8}" y="${y(t) + 4}" text-anchor="end">${t * 100}%</text>`).join('')}
        ${Array.from({length: last + 1}, (_, t) => `<text class="lt" x="${x(t)}" y="${H - B + 18}" text-anchor="middle">${t}</text>`).join('')}
        <text class="lt" x="${x(SAVED)}" y="${H - B + 18}" text-anchor="middle">saved</text>
        <text class="lt" x="${(L + x(last)) / 2}" y="${H - 6}" text-anchor="middle">Passes over the training texts</text>
        <path class="ll ghost" d="${path('top5', N - 1)}"/><path class="ll ghost" d="${path('top1', N - 1)}"/>
        <path class="lsel" d="M${px.toFixed(1)} ${T} V${H - B}"/>
        <path class="ll k5" d="${path('top5', n)}"/><path class="ll" d="${path('top1', n)}"/>
        ${pts.slice(0, n + 1).map((q) => dot('top5', q, x(q.passes)) + dot('top1', q, x(q.passes))).join('')}
        ${['top5', 'top1'].map((k) => dot(k, d.saved, x(SAVED), saved ? '' : ' off')).join('')}
        ${dot('top5', p, px, ' sel')}${dot('top1', p, px, ' sel')}
        <text class="lname" x="${x(3.5)}" y="${y(d.saved.top5) + 24}" text-anchor="middle">right code among five</text>
        <text class="lname" x="${x(3.5)}" y="${y(d.saved.top1) + 30}" text-anchor="middle">first code is right</text>`;
      const firsts = d.answers.map((a) => a.first[i]);
      $('[data-rows]', el).innerHTML = `<div class="head" role="row"><span role="columnheader">Test answer</span><span role="columnheader">Reference code</span><span role="columnheader">Model's first code</span></div>
        ${d.answers.map((a, k) => `<div role="row"${firsts[k].code === a.code ? ' class="hit"' : ''}><span role="cell">${esc(a.text)}</span><span role="cell">${code(a)}</span><span role="cell">${firsts[k].code === a.code ? '<em>✓</em>' : ''}${code(firsts[k])}</span></div>`).join('')}`;
      const right = firsts.filter((f, k) => f.code === d.answers[k].code).length;
      $('[data-match]', el).innerHTML = new Set(firsts.map((f) => f.code)).size === 1
        ? 'The model gives the six answers the same code.'
        : `The first code is the reference code for <b>${right} of ${firsts.length}</b> answers.`;
      range.value = i;
      playBtn.textContent = timer ? 'Pause' : 'Play the training run';
    };
    const stop = () => { clearInterval(timer); timer = null; };
    playBtn.addEventListener('click', () => {
      if (timer) { stop(); draw(); return; }
      i = 0;
      timer = setInterval(() => { i++; if (i >= N) stop(); draw(); }, 800);
      draw();
    });
    range.addEventListener('input', () => { stop(); i = Number(range.value); draw(); });
    svg.addEventListener('click', (ev) => {
      const m = svg.getScreenCTM(), at = m ? (ev.clientX - m.e) / m.a : 0;
      const xs = [...pts.map((p) => x(p.passes)), x(SAVED)];
      stop(); i = xs.reduce((best, v, k) => Math.abs(v - at) < Math.abs(xs[best] - at) ? k : best, 0); draw();
    });
    draw();
  },
};

// Scores of the small model by training set (s32).
const BT_TESTS = {
  survey: ['Survey-style answers', 'The test holds 1,000 survey-style answers. A large model wrote them from job titles set aside before training.'],
  titles: ['Real job titles', 'The test holds 1,000 job titles of the job board, set aside before training. 90 of them have the same wording as a training title.'],
};
DEMOS.bytraining = {
  state: () => ({k: 'top1', t: 'survey'}),
  on(st, a, v) { st[a] = v; },
  render(st) {
    const rows = DATA.occ.byTraining.map((x) => {
      const v = x[st.t][st.k], plain = x.arm === 'defs' || x.arm === 'titles';
      return `<b>${esc(x.label)}<small>${fmt(x.examples)} texts</small></b>${bar(v, plain ? 'var(--faint)' : 'var(--small)')}<span class="v">${(v * 100).toFixed(1)}%</span>`;
    }).join('');
    return `<div class="row">${seg('k', [['top1', 'First code is right'], ['top5', 'Right code among 5']], st.k, 'Score')}
      ${seg('t', Object.entries(BT_TESTS).map(([k, x]) => [k, x[0]]), st.t, 'Test')}</div>
    <div class="hb btbars" aria-live="polite">${rows}</div>
    <p class="legend"><span><i style="background:var(--faint)"></i>trained without the survey-style answers</span><span><i style="background:var(--small)"></i>trained with the survey-style answers</span></p>
    <p class="hint">${BT_TESTS[st.t][1]}</p>`;
  },
};

// The small model next to three large models (s33).
const VS_VIEWS = {
  survey1: ['Survey-style answers · first code', 'Gemini 3.8 Flash wrote the test answers. Its difference from the small model is within the margin of error.'],
  survey5: ['Survey-style answers · among five codes', 'Each large model was asked for five codes in order.'],
  titles1: ['Real job titles · first code', 'The small model trained on other job titles of the same job board. 90 of the 1,000 test titles have the same wording as a training title.'],
};
DEMOS.versus = {
  state: () => ({v: 'survey1'}),
  on(st, a, v) { st.v = v; },
  render(st) {
    const MAX = 1.1;
    const row = (x) => {
      const w = x.score / MAX * 100;
      return `<div class="name">${esc(x.model)}<small>${x.kind === 'small' ? 'trained on the three sets' : 'large model asked directly'}</small></div>
      <div class="track" title="${esc(x.model)}: ${(x.score * 100).toFixed(1)}%"><i class="fillb" style="width:${w}%;background:var(--${x.kind})"></i><span class="val" style="left:${w}%">${(x.score * 100).toFixed(1)}%</span></div>`;
    };
    return `${seg('v', Object.entries(VS_VIEWS).map(([k, x]) => [k, x[0]]), st.v, 'Test and score')}
    <div class="res" aria-live="polite">${DATA.occ.versus[st.v].map(row).join('')}
      <span></span><div class="axis">${[0, 0.25, 0.5, 0.75, 1].map((t) => `<span style="left:${t / MAX * 100}%">${t * 100}%</span>`).join('')}</div>
      <span></span><p class="legend"><span><i style="background:var(--small)"></i>small model</span><span><i style="background:var(--large)"></i>large model</span></p>
    </div>
    <p class="hint">${VS_VIEWS[st.v][1]}</p>`;
  },
};

// Results on 300 real reports (s37).
DEMOS.results = {
  on() { return false; },
  render() {
    const MAX = 0.6, sc = DATA.dist.scores;
    const order = [...sc.filter((x) => x.kind === 'large'), ...sc.filter((x) => x.kind === 'small')];
    let step = 0;
    const row = (x) => {
      const [name, detail] = x.model.split(', ');
      const st = x.kind === 'small' ? ` data-step="${++step}"` : '';
      const w = x.f1 / MAX * 100;
      return `<div class="name"${st}>${esc(name)}<small>${esc(detail || 'large model asked directly')}</small></div>
      <div class="track"${st} title="${esc(x.model)}: ${x.f1.toFixed(3)}"><i class="fillb" style="width:${w}%;background:var(--${x.kind})"></i><span class="val" style="left:${w}%">${x.f1.toFixed(3)}</span></div>`;
    };
    return `<div class="res">${order.map(row).join('')}
      <span></span><div class="axis">${[0, 0.2, 0.4, 0.6].map((t) => `<span style="left:${t / MAX * 100}%">${t.toFixed(1)}</span>`).join('')}</div>
      <span></span><p class="legend"><span><i style="background:var(--large)"></i>large model</span><span><i style="background:var(--small)"></i>small model</span></p>
    </div>`;
  },
};

// Speed (s34).
DEMOS.speed = {
  on() { return false; },
  render() {
    const {rows, gb} = DATA.occ.speed, W = 600, L = 215, R = 70, top = 36, gap = 50;
    const lo = Math.log10(0.003), hi = Math.log10(10);
    const x = (s) => L + (Math.log10(s) - lo) / (hi - lo) * (W - L - R);
    const H = top + rows.length * gap + 26;
    const ticks = [0.01, 0.1, 1, 10];
    const cpu = rows.find((r) => /CPU/.test(r.where)), large = rows.filter((r) => r.kind === 'large').map((r) => r.seconds);
    return `<div class="card"><span class="lbl">Seconds per answer (logarithmic scale)</span>
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Time per answer for the small model and three large models" style="width:100%;margin-top:.6em">
        ${ticks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 14}" y2="${H - 30}" stroke="var(--line)"/><text x="${x(t)}" y="${H - 12}" text-anchor="middle" font-size="15">${t} s</text>`).join('')}
        ${rows.map((r, i) => {
          const y = top + i * gap + 8, hot = r === cpu;
          return `<g><title>${esc(r.model)}, ${esc(r.where)}: ${r.seconds} seconds</title>
            <text x="0" y="${y - 3}" font-size="17" class="ink" font-weight="${hot ? 700 : 500}">${esc(r.model)}</text>
            <text x="0" y="${y + 15}" font-size="14">${esc(r.where)}</text>
            <line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-dasharray="2 4"/>
            <circle cx="${x(r.seconds)}" cy="${y}" r="${hot ? 10 : 8}" fill="var(--${r.kind})" stroke="var(--card)" stroke-width="2"/>
            <text x="${x(r.seconds) + 16}" y="${y + 6}" font-size="17" class="ink" font-weight="${hot ? 700 : 600}">${r.seconds} s</text></g>`;
        }).join('')}
      </svg>
      <p class="legend"><span><i style="background:var(--small)"></i>small model</span><span><i style="background:var(--large)"></i>large model</span></p></div>
    <div style="display:flex;flex-direction:column;gap:1.3em">
      <div class="card small-k"><p class="stat">${gb} GB<small>of memory is enough to run the small model on a laptop</small></p></div>
      <div class="card small-k"><p class="stat">${Math.floor(Math.min(...large) / cpu.seconds)}×<small>faster on an ordinary processor than the fastest of the three large models</small></p></div>
    </div>`;
  },
};

// Cost of building (s35).
DEMOS.cost = {
  state: () => ({p: '2'}),
  on(st, a, v) { st.p = v; },
  render(st) {
    const c = DATA.cost, price = Number(st.p);
    const usd = (tokens) => { const d = tokens / 1e6 * price; return d < 10 ? `$${d.toFixed(2)}` : `$${Math.round(d)}`; };
    return `<div class="g3">${c.builds.map((b) => `<div class="card">
      <span class="lbl">${esc(b.name)}</span>
      <p class="stat mt">${b.minutes}<small>minutes from the first request to the trained model</small></p>
      <dl class="facts">
        <dt>Requests to the large models</dt><dd>${fmt(b.requests)}</dd>
        <dt>Tokens read and written</dt><dd>${(b.tokens / 1e6).toFixed(1)} million</dd>
        <dt>Examples kept</dt><dd>${esc(b.kept)}</dd>
        <dt>Training</dt><dd>${esc(b.trainings)}</dd>
      </dl>
      <p class="price" aria-live="polite"><b>${usd(b.tokens)}</b> for the tokens at the example price</p>
    </div>`).join('')}</div>
    <div class="row" style="margin-top:1.1em"><span class="lbl">Example price for one million <button class="term" data-term="token">tokens</button></span>
      ${seg('p', c.prices.map((x) => [x, `$${x.toFixed(2)}`]), st.p, 'Example price for one million tokens')}</div>`;
  },
};

// Small model or large model (s39).
DEMOS.quiz = {
  state: () => ({ans: DATA.tasks.map(() => null), all: false}),
  on(st, a, v) {
    if (a === 'all') st.all = !st.all;
    else { const [i, k] = v.split(':'); st.ans[Number(i)] = k; }
  },
  render(st) {
    const answered = st.ans.filter(Boolean).length, matched = st.ans.filter((k, i) => k && k === DATA.tasks[i].fit).length;
    return `<div class="quiz">${DATA.tasks.map((t, i) => {
      const pick = st.ans[i], open = pick || st.all;
      const mark = pick ? (pick === t.fit ? '✓ ' : '✗ ') : '';
      return `<div class="q${open ? ' ' + t.fit : ''}"><p class="t">${esc(t.task)}</p>${open
        ? `<p class="why"><b>${mark}Suggested: ${t.fit} model.</b> ${esc(t.why)}</p>`
        : `<div class="seg" role="group" aria-label="Kind of model" style="align-self:flex-start"><button type="button" data-a="pick" data-v="${i}:small">Small model</button><button type="button" data-a="pick" data-v="${i}:large">Large model</button></div>`}</div>`;
    }).join('')}</div>
    <div class="row"><button type="button" class="link" data-a="all" data-v="1">${st.all ? 'Hide the answers' : 'Show all the answers'}</button>
      <span class="txt-s" style="margin-left:auto" aria-live="polite">${answered ? `Same as the suggestion: ${matched} of ${answered}` : ''}</span></div>`;
  },
};

function mountDemo(el) {
  const def = DEMOS[el.dataset.demo];
  if (!def) return;
  if (def.mount) { def.mount(el); return; }
  const st = def.state ? def.state() : {};
  const draw = () => {
    const act = document.activeElement;
    const key = act && el.contains(act) && act.dataset.a ? [act.dataset.a, act.dataset.v] : null;
    el.innerHTML = def.render(st);
    if (key) {
      const again = $$('[data-a]', el).find((n) => n.dataset.a === key[0] && n.dataset.v === key[1]);
      if (again) again.focus({preventScroll: true});
    }
  };
  el.addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-a]');
    if (!b || !el.contains(b) || b.tagName === 'INPUT' || b.disabled) return;
    if (def.on(st, b.dataset.a, b.dataset.v) !== false) draw();
  });
  el.addEventListener('change', (ev) => {
    const b = ev.target.closest('input[data-a]');
    if (b && def.on(st, b.dataset.a, b.type === 'checkbox' ? b.checked : b.value) !== false) draw();
  });
  draw();
}

/* ------------------------------------------------------------- navigation */
const SECS = ['', 'Tasks', 'Small model', 'Synthetic data', 'Nine steps', 'Occupation coding', 'Report tagging', 'Small or large'];
const stage = $('#stage'), slides = $$('.slide', stage), toc = $('#toc'), notesPanel = $('#notespanel');
const gotoBox = $('#goto'), tip = $('#scrubtip');
let idx = 0, step = 0;

const headline = (s) => ($('h1, h2', s) || {textContent: ''}).textContent.trim();
const kicker = (s) => { const k = $('.kicker', s); return k ? k.firstChild.textContent.trim() : ''; };
const maxStep = (s) => Math.max(0, ...$$('[data-step]', s).map((n) => Number(n.dataset.step)));
const applySteps = () => $$('[data-step]', slides[idx]).forEach((n) => n.classList.toggle('pending', Number(n.dataset.step) > step));

$$('[data-demo]').forEach(mountDemo);
slides.forEach((s) => $$('[data-step]', s).forEach((n) => n.classList.add('pending')));

// The row of nine numbered buttons on the step slides.
const stepSlides = slides.filter((s) => s.dataset.stepno);
stepSlides.forEach((s) => {
  const no = Number(s.dataset.stepno);
  s.insertAdjacentHTML('beforeend', `<div class="minirail" role="group" aria-label="The nine steps">${stepSlides.map((t, i) =>
    `<button type="button" data-go="${t.id}" class="${i + 1 === no ? 'on' : i + 1 < no ? 'past' : ''}" title="${esc(kicker(t).split(' · ')[1] || '')}" aria-label="Step ${i + 1}: ${esc(kicker(t).split(' · ')[1] || '')}"${i + 1 === no ? ' aria-current="step"' : ''}>${i + 1}</button>`).join('')}</div>`);
});

$('#secnav').innerHTML = SECS.map((name, n) => (n ? `<button type="button" data-sec="${n}">${name}</button>` : '')).join('') + '<span class="count" id="count"></span>';
$('#secnav').addEventListener('click', (ev) => {
  const b = ev.target.closest('[data-sec]');
  if (b) show(slides.findIndex((s) => s.dataset.sec === b.dataset.sec));
});

// The strip of slides in the bar: one mark for each slide, and a taller mark for the slide that opens a part.
const isDivider = (s) => s.classList.contains('divider');
$('#total').textContent = `/ ${slides.length}`;
$('#scrub').innerHTML = slides.map((s, i) =>
  `<button type="button" tabindex="-1" data-go="${s.id}" class="${isDivider(s) || i === slides.length - 1 ? 'part' : ''}" aria-label="Slide ${i + 1}: ${esc(headline(s))}"></button>`).join('');
const ticks = $$('#scrub button');
// A click on a mark leaves the focus where it was, so that Space and Enter keep stepping through the slides.
$('#scrub').addEventListener('mousedown', (ev) => ev.preventDefault());
$('#scrub').addEventListener('pointerover', (ev) => {
  const b = ev.target.closest('button');
  if (!b) return;
  const i = ticks.indexOf(b), s = slides[i], sec = s.dataset.sec;
  const part = isDivider(s) ? kicker(s) : sec === '0' ? '' : SECS[Number(sec)];
  tip.innerHTML = `<small>Slide ${i + 1}${part ? ' · ' + esc(part) : ''}</small>${esc(headline(s))}`;
  tip.hidden = false;
  const bar = tip.parentElement.getBoundingClientRect(), r = b.getBoundingClientRect(), w = tip.offsetWidth;
  tip.style.left = `${Math.max(0, Math.min(r.left + r.width / 2 - bar.left - w / 2, bar.width - w))}px`;
});
$('#scrub').addEventListener('pointerleave', () => { tip.hidden = true; });

function show(i, dir = 0) {
  i = Math.max(0, Math.min(slides.length - 1, i));
  closePop();
  slides[idx].classList.remove('on');
  idx = i;
  const s = slides[idx];
  s.classList.add('on');
  // A step forward starts the reveals of the next slide. Every other move shows the whole slide.
  step = dir > 0 ? 0 : maxStep(s);
  applySteps();
  const sec = s.dataset.sec;
  stage.classList.toggle('plain', sec === '0' || isDivider(s));
  $$('#secnav [data-sec]').forEach((b) => {
    b.classList.toggle('on', b.dataset.sec === sec);
    b.classList.toggle('done', sec !== '0' && Number(b.dataset.sec) < Number(sec));
    b.toggleAttribute('aria-current', b.dataset.sec === sec);
  });
  $('#count').textContent = `${idx + 1} / ${slides.length}`;
  gotoBox.value = idx + 1;
  ticks.forEach((b, k) => { b.classList.toggle('on', k === idx); b.classList.toggle('done', k < idx); });
  tip.hidden = true;
  $('#notestext').textContent = ($('.notes', s) || {textContent: ''}).textContent;
  $$('#toclist button').forEach((b, k) => b.classList.toggle('on', k === idx));
}
const next = () => { if (step < maxStep(slides[idx])) { step++; applySteps(); } else if (idx < slides.length - 1) show(idx + 1, 1); };
const prev = () => { if (step > 0) { step--; applySteps(); } else if (idx > 0) show(idx - 1, -1); };
const skip = (d) => show(idx + d);
const go = (id) => {
  const i = slides.findIndex((s) => s.id === id);
  if (i < 0) return;
  if (readingMQ.matches) slides[i].scrollIntoView({behavior: reduced ? 'auto' : 'smooth', block: 'start'});
  else show(i);
};

// Contents.
(() => {
  const groups = [];
  slides.forEach((s, i) => {
    const sec = s.dataset.sec, name = sec === '0' ? (i === 0 ? 'Opening' : 'Close') : SECS[Number(sec)];
    if (!groups.length || groups[groups.length - 1].name !== name) groups.push({name, items: []});
    groups[groups.length - 1].items.push(`<li><button type="button" data-go="${s.id}"${isDivider(s) ? ' class="part"' : ''}><i>${i + 1}</i><span>${esc(headline(s))}</span><small>${esc(kicker(s))}</small></button></li>`);
  });
  $('#toclist').innerHTML = groups.map((g) => `<h3>${g.name}</h3><ol>${g.items.join('')}</ol>`).join('');
})();
let tocReturn = null;
const openToc = () => { tocReturn = document.activeElement; toc.hidden = false; const cur = $$('#toclist button')[idx]; if (cur) cur.focus(); };
const closeToc = () => { if (toc.hidden) return; toc.hidden = true; if (tocReturn && tocReturn.focus) tocReturn.focus(); };
$('#bToc').addEventListener('click', openToc);
$('#bTocClose').addEventListener('click', closeToc);

const toggleNotes = () => {
  notesPanel.hidden = !notesPanel.hidden;
  $('#bNotes').setAttribute('aria-pressed', String(!notesPanel.hidden));
};
$('#bNotes').addEventListener('click', toggleNotes);

const toggleFull = () => {
  if (document.fullscreenElement) { if (document.exitFullscreen) document.exitFullscreen(); return; }
  const root = document.documentElement;
  const p = root.requestFullscreen && root.requestFullscreen();
  if (p && p.catch) p.catch(() => {});
};
$('#bFull').addEventListener('click', toggleFull);
$('#bPrev').addEventListener('click', prev);
$('#bNext').addEventListener('click', next);
$('#bSkipPrev').addEventListener('click', () => skip(-1));
$('#bSkipNext').addEventListener('click', () => skip(1));

// The slide number in the bar. A number and Enter open that slide.
gotoBox.addEventListener('focus', () => gotoBox.select());
gotoBox.addEventListener('blur', () => { gotoBox.value = idx + 1; });
gotoBox.addEventListener('keydown', (ev) => {
  if (ev.key !== 'Enter') return;
  const n = parseInt(gotoBox.value, 10);
  if (n >= 1 && n <= slides.length) show(n - 1);
  gotoBox.blur();
  ev.preventDefault();
});

// Definitions of terms.
let pop = null, popFor = null;
function closePop() {
  if (!pop) return;
  pop.remove(); popFor.setAttribute('aria-expanded', 'false'); pop = popFor = null;
}
function openPop(btn) {
  const t = TERMS[btn.dataset.term];
  if (!t) return;
  pop = document.createElement('div');
  pop.className = 'pop';
  pop.setAttribute('role', 'note');
  pop.innerHTML = `<b>${esc(t[0])}</b>${esc(t[1])}`;
  stage.appendChild(pop);
  popFor = btn;
  btn.setAttribute('aria-expanded', 'true');
  const sr = stage.getBoundingClientRect(), br = btn.getBoundingClientRect(), pr = pop.getBoundingClientRect(), pad = 8;
  const left = Math.max(pad, Math.min(br.left - sr.left, sr.width - pr.width - pad));
  let topPx = br.bottom - sr.top + 6;
  if (!readingMQ.matches && topPx + pr.height > sr.height - pad) topPx = br.top - sr.top - pr.height - 6;
  pop.style.left = `${left}px`;
  pop.style.top = `${topPx}px`;
}

document.addEventListener('click', (ev) => {
  const term = ev.target.closest('.term');
  if (term) { const same = popFor === term; closePop(); if (!same) openPop(term); return; }
  closePop();
  const g = ev.target.closest('[data-go]');
  if (g) { closeToc(); go(g.dataset.go); }
});

document.addEventListener('keydown', (ev) => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const t = ev.target, typing = t.matches && t.matches('input:not([type="checkbox"]), textarea, select');
  if (ev.key === 'Escape') {
    if (pop) closePop(); else if (!toc.hidden) closeToc(); else if (!notesPanel.hidden) toggleNotes(); else if (typing) t.blur();
    return;
  }
  if (typing) return;
  if (!toc.hidden) { if (ev.key === 'o' || ev.key === 'O') { closeToc(); ev.preventDefault(); } return; }
  if (ev.key === 'o' || ev.key === 'O') { openToc(); ev.preventDefault(); return; }
  if (readingMQ.matches) return;
  // A digit moves the focus to the slide number in the bar, and the digit is typed there.
  if (ev.key >= '0' && ev.key <= '9' && ev.key.length === 1) { gotoBox.focus(); gotoBox.value = ''; return; }
  const onControl = t.closest && t.closest('button, a, label, input, [tabindex]');
  switch (ev.key) {
    case 'ArrowRight': case 'PageDown': next(); break;
    case ' ': case 'Enter': if (onControl) return; next(); break;
    case 'ArrowLeft': case 'PageUp': prev(); break;
    case 'ArrowDown': skip(1); break;
    case 'ArrowUp': skip(-1); break;
    case 'r': case 'R': step = 0; applySteps(); break;
    case 'Home': show(0); break;
    case 'End': show(slides.length - 1); break;
    case 'n': case 'N': toggleNotes(); break;
    case 'f': case 'F': toggleFull(); break;
    default: return;
  }
  ev.preventDefault();
});

const start = slides.findIndex((s) => '#' + s.id === location.hash);
show(start < 0 ? 0 : start);
window.addEventListener('hashchange', () => go(location.hash.slice(1)));
})();
