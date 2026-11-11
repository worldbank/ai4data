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
  labelled: ['Labelled example', 'An example together with the right answer. In this session, a sentence together with its topics.'],
  synthetic: ['Synthetic data', 'Examples written by a computer program. In this session a large language model writes them.'],
};

/* ------------------------------------------------------------------ demos */
const DEMOS = {};

// Tasks for a small model (slide 3).
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
  {key: 'jobs', name: 'Job matching', who: 'A public employment office',
    task: 'Reads how a job seeker describes their work and suggests the occupations and vacancies that match.',
    reasons: ['private', 'wording', 'few'],
    input: 'I fix motorbikes and sell spare parts at the market, 6 years. I keep the accounts myself.', inputKind: 'Job seeker',
    answer: [['Occupation', 'Motor vehicle mechanic'], ['Matching vacancies', 'Motorcycle mechanic at a garage; sales assistant in a spare parts shop']],
    runs: 'On the employment office\'s own server. The descriptions of job seekers stay in the office.',
    icon: `<circle cx="22" cy="22" r="8" class="iL"/><path d="M10 44 C10 34 34 34 34 44" class="iL"/>${ARROW('M40 30 H56')}<rect x="60" y="16" width="30" height="24" rx="3" class="iP"/><path d="M70 16 V12 H80 V16" class="iL"/>`},
  {key: 'grievance', name: 'Complaints by text message', who: 'The grievance mechanism of a project',
    task: 'Sorts each complaint by issue and urgency and sends it to the office that handles it.',
    reasons: ['few', 'private', 'language'],
    input: 'pls help, maji hayajatoka tangu jumatatu. contractor broke the pipe near the school', inputKind: 'Text message',
    answer: [['Issue', 'Water supply'], ['Urgency', 'High'], ['Send to', 'Site engineer']],
    runs: 'On a server of the project team. The messages and phone numbers stay with the team.',
    icon: `<rect x="8" y="10" width="40" height="28" rx="6" class="iP"/><path d="M16 38 L14 46 L24 38" class="iP"/><path d="M16 20 H40 M16 28 H32" class="iL"/>${ARROW('M54 26 H66')}<rect x="70" y="14" width="22" height="7" rx="3.5" class="iT"/><rect x="70" y="26" width="16" height="7" rx="3.5" class="iT"/>`},
  {key: 'disaster', name: 'Messages after a flood', who: 'A disaster response center',
    task: 'Picks out requests for rescue, water, or shelter from incoming messages, with the place and the number of people.',
    reasons: ['missing', 'rare', 'language'],
    input: 'Water rising in Kalanga village, 6 people on the roof of the clinic, we need a boat', inputKind: 'Message',
    answer: [['Request', 'Rescue'], ['Place', 'Kalanga village, clinic'], ['People', '6']],
    runs: 'On a laptop at the response center. It keeps working when the internet connection is down.',
    icon: `<path d="M14 30 L30 16 L46 30 V44 H14 Z" class="iP"/><path d="M4 40 Q14 34 24 40 T44 40 T64 40 T84 40 T96 40" class="iSS"/><path d="M4 48 Q14 42 24 48 T44 48 T64 48 T84 48 T96 48" class="iSS"/><circle cx="74" cy="20" r="9" class="iT"/><path d="M74 14 V21 M74 24 V25" class="iL"/>`},
  {key: 'farmers', name: 'Farmers\' questions', who: 'An agricultural extension service',
    task: 'Answers questions sent by text message in a local language, from the approved extension guide only, and passes the rest to an extension officer.',
    reasons: ['language', 'few'],
    input: 'My maize leaves have white and yellow stripes and the plants stay short. What do I do?', inputKind: 'Question, translated',
    answer: [['Answer', 'The guide\'s section on leaf streaks: pull out plants with streaks and plant a tolerant variety next season.'], ['Also', 'Sent to the extension officer of the area']],
    runs: 'On the extension service\'s own server, reached by text message from a basic phone.',
    icon: `<path d="M30 48 V18" class="iSS"/><path d="M30 30 C20 26 16 18 18 12 C26 14 30 22 30 30 Z" class="iS"/><path d="M30 38 C40 34 44 26 42 20 C34 22 30 30 30 38 Z" class="iS"/><rect x="56" y="10" width="34" height="22" rx="6" class="iP"/><path d="M64 18 H82 M64 24 H76" class="iL"/><path d="M66 32 L64 38 L72 32" class="iP"/>`},
  {key: 'benefits', name: 'Benefit applications', who: 'A social protection agency',
    task: 'Flags missing or inconsistent entries in an application before a caseworker reviews it.',
    reasons: ['private', 'rare'],
    input: 'Household members: 3 · Children under 5: 4 · Monthly income: (blank)', inputKind: 'Application form',
    answer: [['Flag', 'More children under 5 than household members'], ['Flag', 'Monthly income missing']],
    runs: 'On the agency\'s own server, next to the register of applicants.',
    icon: `<rect x="12" y="6" width="34" height="44" rx="3" class="iP"/><path d="M18 16 H40 M18 24 H40 M18 32 H32" class="iL"/>${ARROW('M54 28 H66')}<path d="M80 12 L92 40 H68 Z" class="iT"/><path d="M80 22 V31 M80 35 V36" class="iL"/>`},
  {key: 'survey', name: 'Coding survey answers', who: 'A national statistical office',
    task: 'Gives each occupation written in a household survey its code in the international classification of occupations.',
    reasons: ['few', 'wording'],
    input: 'Sells grilled maize by the roadside', inputKind: 'Survey answer',
    answer: [['Occupation', 'Street food salesperson'], ['Code (ISCO-08)', '5212']],
    runs: 'On the laptops of the office\'s survey team, without an internet connection in the field.',
    tested: true,
    icon: `<rect x="10" y="8" width="38" height="40" rx="3" class="iP"/><path d="M16 18 H20 M24 18 H42 M16 28 H20 M24 28 H38 M16 38 H20 M24 38 H40" class="iL"/>${ARROW('M54 28 H66')}<rect x="70" y="20" width="24" height="16" rx="3" class="iT"/>`},
];

DEMOS.cases = {
  state: () => ({open: 'grievance'}),
  on(st, a, v) { st.open = v; },
  render(st) {
    const c = CASES.find((x) => x.key === st.open);
    return `<div class="caselist" role="group" aria-label="Task">${CASES.map((x) =>
      `<button type="button" class="casebtn" data-a="open" data-v="${x.key}" aria-pressed="${x.key === st.open}"><svg viewBox="0 0 100 56" aria-hidden="true">${x.icon}</svg><span>${esc(x.name)}</span></button>`).join('')}</div>
    <div class="card casepanel" aria-live="polite">
      <div>
        <span class="lbl">${esc(c.who)}</span>
        <p class="big mt">${esc(c.task)}</p>
        <span class="lbl mt2">Why no training data exist</span>
        <ul class="reasons">${c.reasons.map((r) => `<li><span class="chip warn">${REASONS[r][0]}</span>${REASONS[r][1]}</li>`).join('')}</ul>
        <span class="lbl mt2">Where the model runs</span>
        <p class="txt-s mt">${esc(c.runs)}</p>
        ${c.tested ? '<p class="txt-s mt">A small model was also built for this task. It scored 0.15 below the best large model on job titles from a job board.</p>' : ''}
      </div>
      <div>
        <span class="lbl">One practice case · ${esc(c.inputKind)}</span>
        <p class="msg mt">${esc(c.input)}</p>
        <span class="lbl mt2">Right answer</span>
        <dl class="dl mt">${c.answer.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>
      </div>
    </div>`;
  },
};

// Where a model can run (slide 8).
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

// Rare cases (slide 12).
DEMOS.rare = {
  state: () => ({on: false}),
  on(st) { st.on = !st.on; },
  render(st) {
    const real = new Set([37, 141, 236, 318]);
    let dots = '';
    for (let i = 0; i < 400; i++) dots += `<span${real.has(i) ? ' class="real"' : ''}></span>`;
    if (st.on) for (let i = 0; i < 120; i++) dots += `<span class="written" style="animation-delay:${i * 6}ms"></span>`;
    return `<div>
      <div class="raregrid" aria-hidden="true">${dots}</div>
      <p class="legend mt2"><span><i style="background:var(--line)"></i>routine message</span><span><i style="background:var(--bad)"></i>real emergency</span>${st.on ? '<span><i style="background:var(--large)"></i>written emergency case</span>' : ''}</p>
    </div>
    <div aria-live="polite">
      <p class="stat">${st.on ? '120' : '4'}<small>${st.on ? 'written emergency cases' : 'real emergencies among 400 messages'}</small></p>
      <p class="txt mt2">${st.on
        ? 'The model now learns from 120 written emergencies of many kinds. The 4 real ones are kept for testing.'
        : 'Among 400 real messages, 4 are emergencies. A model trained on them sees too few to learn what an emergency looks like.'}</p>
      <p class="mt2"><button type="button" class="btn" data-a="t" data-v="1">${st.on ? 'Remove the written cases' : 'Add 120 written emergency cases'}</button></p>
    </div>`;
  },
};

// Step 1: describe the task.
const ISSUES = [
  ['Water supply', 'Water cut, damaged pipes, or a dry well caused by the works.'],
  ['Land and compensation', 'Payment for land, crops, or buildings affected by the project.'],
  ['Safety', 'Danger to people from traffic, machines, or open trenches.'],
  ['Dust and noise', 'Dust, noise, or vibration from the works.'],
  ['Workers\' conduct', 'Behaviour of the contractor\'s staff.'],
  ['Other', 'Any complaint that fits none of the issues above.'],
];
DEMOS.task = {
  state: () => ({i: 0}),
  on(st, a, v) { st.i = Number(v); },
  render(st) {
    return `<span class="lbl">The list of answers for the complaints example</span>
    <div class="chips" role="group" aria-label="Issue">${ISSUES.map(([n], i) =>
      `<button type="button" class="chip off" style="font-size:.9em" data-a="i" data-v="${i}" aria-pressed="${i === st.i}">${esc(n)}</button>`).join('')}</div>
    <p class="def" aria-live="polite"><b>${esc(ISSUES[st.i][0])}.</b> ${esc(ISSUES[st.i][1])}</p>
    <p class="hint" style="margin-top:auto">Select an answer to see its definition. The large model and the people who check its cases read the same definitions.</p>`;
  },
};

// Step 2: show a few real cases.
const SEEDS = ['no water 3 days now. who do we call', 'ur trucks almost hit my son today near the school!!', 'Land money still not paid since march, chief said wait'];
const UNSEEDED = ['I would like to report that the water supply in our community has been interrupted for several days.',
  'I am writing to express my concern about the safety of children near the construction site.',
  'I wish to inform you that I have not yet received compensation for my land.'];
const SEEDED = ['tap dry since sunday, women walking to the river',
  'big lorry drives too fast by the market, someone will die',
  'they measured my farm in jan, no payment yet. who to see?'];
DEMOS.seeds = {
  state: () => ({on: 0}),
  on(st, a, v) { st.on = Number(v); },
  render(st) {
    const list = (items, cls = '') => `<div class="msgs">${items.map((t) => `<p class="msg ${cls}">${esc(t)}</p>`).join('')}</div>`;
    return `${seg('on', [[0, 'No samples'], [1, 'Three real samples']], st.on, 'Request to the large model')}
    ${st.on ? `<span class="lbl">Real messages added to the request, names removed</span>${list(SEEDS, 'real')}` : ''}
    <span class="lbl">Cases written by the large model</span>
    ${list(st.on ? SEEDED : UNSEEDED)}
    <p class="hint" style="margin-top:auto">The messages on this slide were written for this session as an illustration.</p>`;
  },
};

// Step 3: vary the cases.
const ATTRS = [
  ['Issue', ISSUES.map(([n]) => n)],
  ['Length', ['one line', 'two lines', 'a short paragraph']],
  ['Style', ['text-message shorthand', 'formal', 'angry', 'polite']],
  ['Language', ['English', 'Swahili', 'English and Swahili mixed']],
  ['Urgency', ['low', 'high']],
  ['Spelling', ['correct', 'with errors']],
];
DEMOS.vary = {
  state: () => ({on: new Set(['Issue']), pick: Object.fromEntries(ATTRS.map(([a]) => [a, 0]))}),
  on(st, a, v) {
    if (a === 'attr') { st.on.has(v) ? st.on.delete(v) : st.on.add(v); }
    else for (const [name, vals] of ATTRS) st.pick[name] = Math.floor(Math.random() * vals.length);
  },
  render(st) {
    const val = (name) => (st.on.has(name) ? ATTRS.find(([a]) => a === name)[1][st.pick[name]] : null);
    const combos = ATTRS.reduce((n, [a, vals]) => n * (st.on.has(a) ? vals.length : 1), 1);
    const dup = DATA.workflow.duplicates;
    const share = Math.round(dup.repeats / dup.written * 100);
    let request = `Write a complaint by text message${val('Issue') ? ` about ${val('Issue').toLowerCase()}` : ''}.`;
    for (const name of ['Length', 'Style', 'Language', 'Urgency', 'Spelling']) if (val(name)) request += ` ${name}: ${val(name)}.`;
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
  ['no water at the borehole since tuesday', 'Water supply', 'Water supply'],
  ['no water at the borehole since tuesday', 'Water supply', 'Water supply', 'Repeats an earlier case'],
  ['trucks drive fast past the school every morning', 'Dust and noise', 'Safety', 'The two models disagree'],
  ['my land is not paid, 3 months now', 'Land and compensation', 'Land and compensation'],
  ['my electricity bill is too high this month', 'Electricity', 'Other', 'Answer not in the list'],
  ['dust from the works covers our market stalls', 'Dust and noise', 'Dust and noise'],
  ['pls help', '', 'Other', 'No answer given'],
];
DEMOS.check = {
  state: () => ({run: false}),
  on(st) { st.run = !st.run; },
  render(st) {
    const kept = CHECKED.filter((r) => !r[3]).length;
    return `<div class="checkt" role="table" aria-label="Cases and the answers of two models">
      <div class="head" role="row"><span role="columnheader">Case</span><span role="columnheader">Writer's answer</span><span role="columnheader">Second model's answer</span></div>
      ${CHECKED.map(([text, writer, checker, drop]) => {
        const out = st.run && drop;
        return `<div role="row"${out ? ' class="drop"' : ''}><span role="cell" class="txtc">${out ? `<s>${esc(text)}</s><em>✕ ${esc(drop)}</em>` : esc(text)}</span><span role="cell">${esc(writer || '(none)')}</span><span role="cell">${esc(checker)}</span></div>`;
      }).join('')}
    </div>
    <div class="row" style="margin-top:auto"><button type="button" class="btn" data-a="run" data-v="1">${st.run ? 'Show the cases before the checks' : 'Run the checks'}</button>
    <span class="txt" aria-live="polite">${st.run ? `<b>${kept} of ${CHECKED.length}</b> cases are kept.` : 'A second model labels each case without seeing the writer\'s answer.'}</span></div>`;
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
    <p class="txt-s" style="margin-top:auto">A later slide trains a simple model in the page, so we can watch a model learn from examples. <button type="button" class="link" data-go="s30" style="font-size:1em">Go to that slide</button></p>`;
  },
};

// Step 6: test on real cases.
DEMOS.test = {
  state: () => ({wrong: false}),
  on(st, a, v) { st.wrong = !!v; },
  render(st) {
    return `<div class="piles">
      <div class="pile${st.wrong ? ' bad' : ''}"><b>Practice cases</b><span>Thousands, written by the large model</span><span class="use">${st.wrong ? 'Used for training and testing' : 'Used for training'}</span></div>
      <div class="pile${st.wrong ? '' : ' test'}"><b>Real cases</b><span>A few hundred, answers given by people</span><span class="use">${st.wrong ? 'Not used' : 'Used only for testing'}</span></div>
    </div>
    <label class="toggle"><input type="checkbox" data-a="wrong" data-v="1"${st.wrong ? ' checked' : ''}>Test on practice cases</label>
    <p class="def" aria-live="polite">${st.wrong
      ? 'The practice cases used for testing share the writing habits of the training cases. A high score then says little about real messages.'
      : 'The real cases are set aside before any practice case is written, and no model is trained on them.'}</p>`;
  },
};

// Step 7: practise the weak spots.
const ARMS = {first: 'First set of cases', control: 'Plus evenly spread rounds', targeted: 'Plus targeted rounds'};
DEMOS.rounds = {
  state: () => ({r: 0}),
  on(st, a, v) { st.r = Number(v); },
  render(st) {
    const w = DATA.workflow, rep = w.rounds[st.r], MAX = 0.6;
    const lo = Math.min(...w.large.map((x) => x.f1)), hi = Math.max(...w.large.map((x) => x.f1));
    const band = `<i class="band" style="left:${lo / MAX * 100}%;width:${(hi - lo) / MAX * 100}%"></i>`;
    const loop = ['Test on separate real cases', 'Find the weakest topics', 'Write more cases on them', 'Train again'];
    return `<div class="loop">${loop.map((t, i) => `<span><b>${i + 1}</b>${t}</span>`).join('')}</div>
    <div class="row">${seg('r', w.rounds.map((x, i) => [i, `Repeat ${x.replicate}`]), st.r, 'Repeat')}
    <span class="lbl">Score of the small model on 300 real report abstracts (F1, 0 to 1)</span></div>
    <div class="rbars">
      <div class="rbar rbar3"><span></span><div class="bandlbl"><span style="right:${100 - hi / MAX * 100}%">three large models</span></div><span></span></div>
      ${rep.arms.map((a) => `<div class="rbar rbar3"><span>${ARMS[a.arm]}<small>${fmt(a.examples)} cases</small></span>
        ${bar(a.f1 / MAX, a.arm === 'targeted' ? 'var(--small)' : 'var(--faint)', band)}<b class="v">${a.f1.toFixed(3)}</b></div>`).join('')}
      <div class="rbar rbar3"><span></span><div class="axis">${[0, 0.2, 0.4, 0.6].map((t) => `<span style="left:${t / MAX * 100}%">${t.toFixed(1)}</span>`).join('')}</div><span></span></div>
    </div>
    <p class="hint" style="margin-top:auto">Orange band: three large models asked directly (${lo.toFixed(3)} to ${hi.toFixed(3)}). Targeted minus evenly spread: +${rep.difference.toFixed(3)} (95% interval ${rep.interval[0].toFixed(3)} to ${rep.interval[1].toFixed(3)}). Each score is the mean of three training runs.</p>`;
  },
};

// Step 8: ask people which is better.
const PAIRS = [
  {kind: 'Which written complaint is more realistic?',
    a: 'Dear Sir or Madam, I hope this message finds you well. I am writing to formally express my concern regarding the interruption of water supply in our community.',
    b: 'no water at the borehole since tuesday. kids walking 2km. pls help',
    note: 'Complaints by text message are short and informal. The first case reads like a letter.'},
  {kind: 'Which case has the right answer?',
    a: '"Thank you for the new road, it is very good." Answer: Safety, high urgency.',
    b: '"trucks for the road work go fast past the school, a child was almost hit today." Answer: Safety, high urgency.',
    note: 'The first message is not a complaint. A case with a wrong answer teaches the small model a wrong answer.'},
  {kind: 'Which answer to a farmer is better?',
    a: 'Streaks on maize leaves can have several causes, including viral infection transmitted by leafhoppers, nutrient deficiency, or genetic factors. Consider a systemic insecticide and a foliar feed.',
    b: 'This looks like maize streak. Pull out the plants with streaks. Next season, plant a tolerant variety. The extension officer will call you.',
    note: 'The second answer stays within the approved guide, is short enough for a text message, and passes the case to a person.'},
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
      ? `<b>An experienced staff member chose ${pick === 'b' ? 'the same' : 'the other one'}.</b> ${esc(p.note)}`
      : 'Select the better one of the two.'}</p>
    <div class="row" style="margin-top:auto">
      <button type="button" class="link" data-a="go" data-v="-1"${st.i === 0 ? ' disabled' : ''}>Previous pair</button>
      <button type="button" class="link" data-a="go" data-v="1"${st.i === PAIRS.length - 1 ? ' disabled' : ''}>Next pair</button>
      <span class="txt-s" style="margin-left:auto">Same choice as the staff member: ${agreed} of ${answered}</span>
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
    <p class="txt">For a small model trained on practice cases, a change can come from a new issue on the list, a new language, or a new kind of message after an event. Regular reviews of a sample of real cases show when new practice cases and a new round of training are needed.</p>
    <div class="loop" style="grid-template-columns:repeat(3,1fr);margin-top:auto">${loop.map((t, i) => `<span><b>${i + 1}</b>${t}</span>`).join('')}</div>`;
  },
};

// The report on slide 25.
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

// The method in three steps (slides 27 and 39).
const METHOD = [
  {term: 'Synthetic data', kind: 'large', go: 's28', link: 'See one request',
    title: 'A large model writes practice examples',
    method: 'A large AI model writes thousands of example sentences and lists the topics of each one.',
    office: 'An experienced officer writes practice cases, each with the correct answer.',
    icon: '<rect x="6" y="14" width="44" height="44" rx="8" class="iLg"/><path d="M18 30 H38 M18 38 H38 M18 46 H30" class="iLS"/><path d="M56 36 H66" class="iA" marker-end="url(#ah)"/><rect x="72" y="10" width="42" height="52" rx="4" class="iP"/><path d="M79 20 H107 M79 26 H101" class="iL"/><rect x="79" y="31" width="18" height="6" rx="3" class="iT"/><path d="M79 44 H107 M79 50 H97" class="iL"/><rect x="79" y="54" width="14" height="5" rx="2.5" class="iT"/>'},
  {term: 'Distillation', kind: 'small', go: 's30', link: 'Watch a model learn',
    title: 'A small model learns from the examples',
    method: 'A small model is trained on the examples until it gives the same topics to the same sentences.',
    office: 'A new staff member studies the practice cases and learns the answers.',
    icon: '<rect x="6" y="10" width="42" height="52" rx="4" class="iP"/><path d="M13 20 H41 M13 26 H35" class="iL"/><rect x="13" y="31" width="18" height="6" rx="3" class="iT"/><path d="M13 44 H41 M13 50 H31" class="iL"/><rect x="13" y="54" width="14" height="5" rx="2.5" class="iT"/><path d="M54 36 H68" class="iA" marker-end="url(#ah)"/><rect x="76" y="22" width="28" height="28" rx="6" class="iS"/><circle cx="86" cy="33" r="2.5" class="iD"/><circle cx="95" cy="40" r="2.5" class="iD"/><path d="M86 33 L95 40" class="iSS"/>'},
  {term: 'Small AI', kind: 'small', go: 's32', link: 'See the small model at work',
    title: 'The small model works alone',
    method: 'The small model tags the reports on the organization\'s own computer, without the large model and without an internet connection.',
    office: 'The new staff member handles the routine files alone, and the files stay in the office.',
    icon: '<rect x="20" y="12" width="80" height="44" rx="4" class="iP"/><path d="M10 62 H110" class="iL" stroke-width="4"/><rect x="30" y="22" width="22" height="22" rx="5" class="iS"/><circle cx="38" cy="31" r="2" class="iD"/><circle cx="45" cy="36" r="2" class="iD"/><path d="M38 31 L45 36" class="iSS"/><path d="M60 26 H90 M60 32 H84" class="iL"/><rect x="60" y="38" width="18" height="6" rx="3" class="iT"/>'},
];
const methodCards = (view, links) => METHOD.map((s, i) =>
  `<div class="card ${s.kind}-k"><svg viewBox="0 0 120 72" aria-hidden="true">${s.icon}</svg>
    <span class="lbl">Step ${i + 1} · ${s.term}</span><h3>${s.title}</h3><p class="txt">${esc(s[view])}</p>
    ${links ? `<button type="button" class="link" data-go="${s.go}" style="margin-top:auto;text-align:left">${s.link}</button>` : ''}</div>`).join('');
DEMOS.m3 = {
  state: () => ({v: 'method'}),
  on(st, a, v) { st.v = v; },
  render(st) {
    return `<div class="row">${seg('v', [['method', 'The method'], ['office', 'An office comparison']], st.v, 'View')}
      <p class="legend"><span><i style="background:var(--large)"></i>large model</span><span><i style="background:var(--small)"></i>small model</span></p></div>
    <div class="g3 fill" aria-live="polite">${methodCards(st.v, true)}</div>`;
  },
};
DEMOS.sum3 = {on() { return false; }, render: () => methodCards('method', false)};

// Step 1 of the method: one request and its sentences (slide 28).
DEMOS.write = {
  mount(el) {
    const s = DATA.synth;
    let timer = null;
    el.innerHTML = `<div class="card storyleft">
      <span class="lbl">Request ${esc(s.request)} to ${esc(s.writer)}</span>
      <p class="prompt" style="min-height:0">${esc(s.opening)}</p>
      <span class="lbl">The eight topics named in this request</span>
      ${chips(s.topics, 'off')}
      <div class="row" style="margin-top:auto"><button type="button" class="btn">Write the sentences</button><span class="txt-s" aria-live="polite"></span></div>
    </div>
    <div style="display:flex;flex-direction:column;min-height:0">
      <div class="empty">The sentences written by the large model appear here.</div>
      <div class="sents" hidden tabindex="0" aria-label="Sentences written by the large model"></div>
    </div>`;
    const btn = $('.btn', el), count = $('.row .txt-s', el), empty = $('.empty', el), list = $('.sents', el);
    const add = (i) => {
      const x = s.sentences[i];
      list.insertAdjacentHTML('beforeend', `<div class="sent">${esc(x.text)}${chips(x.labels)}</div>`);
      list.scrollTop = list.scrollHeight;
      count.textContent = `${i + 1} of ${s.sentences.length} sentences`;
    };
    btn.addEventListener('click', () => {
      clearTimeout(timer);
      list.innerHTML = '';
      empty.hidden = true;
      list.hidden = false;
      btn.textContent = 'Write again';
      if (reduced) { s.sentences.forEach((_, i) => add(i)); list.scrollTop = 0; return; }
      let i = 0;
      const tick = () => { add(i++); if (i < s.sentences.length) timer = setTimeout(tick, 650); };
      timer = setTimeout(tick, 300);
    });
  },
};

// The wording of the request (slide 29).
DEMOS.wording = {
  on() { return false; },
  render() {
    const {allow, avoid} = DATA.synth.sets;
    const rows = (max, f) => [['Topic names allowed', allow, 'var(--small)'], ['Topic names avoided', avoid, 'var(--faint)']].map(([name, set, color]) =>
      `<b>${name}</b>${bar(f(set)[0] / max, color)}<span class="v">${f(set)[1]}</span>`).join('');
    const axis = (a, b) => `<span></span><div class="axis"><span style="left:0">${a}</span><span style="left:100%">${b}</span></div><span></span>`;
    return `<div class="g2">
      <div class="card small-k"><span class="lbl">Version 1 · topic names allowed</span><p class="prompt mt" style="min-height:0">${esc(allow.rule)}</p></div>
      <div class="card plain-k"><span class="lbl">Version 2 · topic names avoided</span><p class="prompt mt" style="min-height:0">${esc(avoid.rule)}</p></div>
    </div>
    <div class="g2">
      <div class="card"><span class="lbl">Sentences that name one of their topics</span>
        <div class="hb mt2">${rows(1, (x) => [x.share, (x.share * 100).toFixed(1) + '%'])}${axis('0%', '100%')}</div></div>
      <div class="card"><span class="lbl">Score of a small model trained on each version (F1)</span>
        <div class="hb mt2">${rows(0.6, (x) => [x.f1, x.f1.toFixed(3)])}${axis('0.0', '0.6')}</div></div>
    </div>`;
  },
};

// A model that learns in the page (slide 30).
DEMOS.learn = {
  mount(el) {
    const d = DATA.learn, stop = new Set(d.stop);
    const tok = (t) => (t.toLowerCase().match(/[a-z]+/g) || []).filter((w) => w.length > 2 && !stop.has(w));
    const train = (n) => {
      const counts = {}, totals = {}, vocab = new Set();
      for (const t of d.topics) {
        counts[t] = new Map(); totals[t] = 0;
        for (const s of d.sentences[t].slice(0, n)) for (const w of tok(s)) { counts[t].set(w, (counts[t].get(w) || 0) + 1); totals[t]++; vocab.add(w); }
      }
      return {counts, totals, size: vocab.size + 1};
    };
    const rate = (m, t, w) => ((m.counts[t].get(w) || 0) + 1) / (m.totals[t] + m.size);
    const predict = (m, words) => {
      const logs = d.topics.map((t) => words.reduce((s, w) => s + Math.log(rate(m, t, w)), 0));
      const top = Math.max(...logs), e = logs.map((l) => Math.exp(l - top)), z = e.reduce((a, b) => a + b, 0);
      return e.map((x) => x / z);
    };
    const learned = (m, topic, k = 6) => [...m.counts[topic].entries()].filter(([, c]) => c >= 2)
      .map(([w]) => [w, rate(m, topic, w) / Math.max(...d.topics.filter((t) => t !== topic).map((t) => rate(m, t, w)))])
      .sort((a, b) => b[1] - a[1]).slice(0, k).map(([w]) => w);

    let n = 1, model = train(n);
    el.innerHTML = `<div class="card" style="display:flex;flex-direction:column;gap:.7em">
      <div class="row"><span class="lbl">Examples read per topic</span><div class="seg" role="group" aria-label="Examples read per topic">${[1, 10, 40].map((k) =>
        `<button type="button" data-n="${k}" aria-pressed="${k === n}" style="min-width:3.4em;font-size:.9em">${k}</button>`).join('')}</div></div>
      <span class="lbl mt">Words the model links to each topic</span>
      <div class="wordt" data-words aria-live="polite"></div>
    </div>
    <div style="display:flex;flex-direction:column;gap:1em;min-height:0">
      <div class="card"><span class="lbl">Test on 60 real reports</span>
        <p class="big mt" aria-live="polite"><span data-right></span> of ${d.tests.length} given the right topic</p>
        <div class="dotrows" data-dots></div>
        <p class="txt-s mt" style="font-size:.72em">Each square is one report, filled when the model gives it the right topic. A random guess among the four topics gets about 15 right.</p></div>
      <div class="card"><div class="try">
        <div><label class="lbl" for="tryin">Try a sentence</label><input id="tryin" type="text" autocomplete="off" spellcheck="false" value="The new road cuts the travel time from farms to the market."></div>
        <div class="pb" data-probs aria-live="polite"></div>
      </div></div>
    </div>`;
    const input = $('#tryin', el);
    const probs = () => {
      const p = predict(model, tok(input.value));
      $('[data-probs]', el).innerHTML = d.topics.map((t, i) =>
        `<span>${t}</span>${bar(p[i], 'var(--small)')}<span class="v">${Math.round(p[i] * 100)}%</span>`).join('');
    };
    const draw = () => {
      model = train(n);
      $$('[data-n]', el).forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.n) === n)));
      $('[data-words]', el).innerHTML = d.topics.map((t) => {
        const w = learned(model, t);
        return `<b>${t}</b>${chips(w.length ? w : ['none yet'], w.length ? '' : 'off')}`;
      }).join('');
      const guesses = d.tests.map((x) => {
        const p = predict(model, x.w), best = d.topics[p.indexOf(Math.max(...p))];
        return {...x, best, right: best === x.topic};
      });
      $('[data-right]', el).textContent = guesses.filter((g) => g.right).length;
      $('[data-dots]', el).innerHTML = d.topics.map((t) => `<span>${t}</span><div class="dotg">${guesses.filter((g) => g.topic === t).map((g) =>
        `<i${g.right ? ' class="r"' : ''} title="${esc(g.title)}\nRecorded topic: ${g.topic}\nModel's guess: ${g.best}"></i>`).join('')}</div>`).join('');
      probs();
    };
    el.addEventListener('click', (ev) => { const b = ev.target.closest('[data-n]'); if (b) { n = Number(b.dataset.n); draw(); } });
    input.addEventListener('input', probs);
    draw();
  },
};

// The student at work on one report (slide 32).
DEMOS.work = {
  on() { return false; },
  render() {
    const d = DATA.doc, {threshold, top} = DATA.dist;
    const rows = d.student.slice(0, 8), max = rows[0].score;
    const chosen = d.student.slice(0, top).filter((x) => x.score >= threshold).map((x) => x.label);
    const right = chosen.filter((l) => d.reference.includes(l)).length;
    const line = `<i class="thr" style="left:${threshold / max * 100}%"></i>`;
    return `<div class="card report"><span class="lbl">The report</span><h3>${esc(d.title)}</h3><p class="abs" style="-webkit-line-clamp:11">${esc(d.text)}</p></div>
    <div class="card"><span class="lbl">The student's scores, top eight of 86 topics</span>
      <div class="sb mt2">${rows.map((x) => `<span>${esc(x.label)}</span>${bar(x.score / max, chosen.includes(x.label) ? 'var(--small)' : 'var(--faint)', line)}<b class="v">${x.score.toFixed(3)}</b>`).join('')}</div>
      <p class="txt-s mt2" style="font-size:.72em"><span class="thrkey"></span>The dashed line marks the threshold of ${threshold}.</p></div>
    <div class="card" style="display:flex;flex-direction:column;gap:.6em">
      <span class="lbl">Topics recorded for the report</span>${chips(d.reference, 'off')}
      <span class="lbl mt">Topics the student assigned</span>
      <div class="chips">${chosen.map((l) => `<span class="chip ${d.reference.includes(l) ? 'ok' : 'warn'}">${d.reference.includes(l) ? '✓ ' : ''}${esc(l)}</span>`).join('')}</div>
      <p class="callout" style="margin-top:auto;font-size:.95em">The student assigned ${chosen.length} topics, and ${right === chosen.length ? `all ${right}` : `${right} of them`} are recorded for the report.</p>
    </div>`;
  },
};

// Results on 300 real reports (slide 33).
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
      <span></span><p class="legend"><span><i style="background:var(--large)"></i>large model</span><span><i style="background:var(--small)"></i>small model (the student)</span></p>
    </div>`;
  },
};

// Speed (slide 34).
DEMOS.speed = {
  on() { return false; },
  render() {
    const rows = DATA.dist.speed, W = 560, L = 190, R = 62, top = 36, gap = 54;
    const lo = Math.log10(0.003), hi = Math.log10(10);
    const x = (s) => L + (Math.log10(s) - lo) / (hi - lo) * (W - L - R);
    const H = top + rows.length * gap + 26;
    const ticks = [0.01, 0.1, 1, 10];
    const cpu = rows.find((r) => /CPU/.test(r.where)), large = rows.filter((r) => r.kind === 'large').map((r) => r.seconds);
    const times = [Math.min(...large) / cpu.seconds, Math.max(...large) / cpu.seconds];
    const small = DATA.machines.models[0];
    return `<div class="card"><span class="lbl">Seconds per report (logarithmic scale)</span>
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Time per report for the student and three large models" style="width:100%;margin-top:.6em">
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
      <p class="legend"><span><i style="background:var(--small)"></i>small model (the student)</span><span><i style="background:var(--large)"></i>large model</span></p></div>
    <div style="display:flex;flex-direction:column;gap:1.3em">
      <div class="card small-k"><p class="stat">${small.gb} GB<small>of memory for the student, which a laptop has</small></p></div>
      <div class="card small-k"><p class="stat">${Math.round(times[0])} to ${Math.round(times[1])}×<small>faster on an ordinary processor than the three large models</small></p></div>
    </div>`;
  },
};

// Topics covered by the test (slide 35).
DEMOS.g86 = {
  on() { return false; },
  render() { let h = ''; for (let i = 0; i < 86; i++) h += `<i${i < 44 ? ' class="on"' : ''}></i>`; return h; },
};

// Small model or large model (slide 37).
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
const SECS = ['', 'Tasks', 'Small model', 'Synthetic data', 'Nine steps', 'Report tagging', 'Small or large'];
const stage = $('#stage'), slides = $$('.slide', stage), toc = $('#toc'), notesPanel = $('#notespanel');
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

function show(i, dir = 0) {
  i = Math.max(0, Math.min(slides.length - 1, i));
  closePop();
  slides[idx].classList.remove('on');
  idx = i;
  const s = slides[idx];
  s.classList.add('on');
  step = dir < 0 ? maxStep(s) : 0;
  applySteps();
  const sec = s.dataset.sec;
  stage.classList.toggle('plain', sec === '0');
  $$('#secnav [data-sec]').forEach((b) => {
    b.classList.toggle('on', b.dataset.sec === sec);
    b.classList.toggle('done', sec !== '0' && Number(b.dataset.sec) < Number(sec));
    b.toggleAttribute('aria-current', b.dataset.sec === sec);
  });
  $('#count').textContent = `${idx + 1} / ${slides.length}`;
  $('#where').textContent = `${idx + 1} / ${slides.length}`;
  $('#prog').style.width = `${idx / (slides.length - 1) * 100}%`;
  $('#notestext').textContent = ($('.notes', s) || {textContent: ''}).textContent;
  $$('#toclist button').forEach((b, k) => b.classList.toggle('on', k === idx));
}
const next = () => { if (step < maxStep(slides[idx])) { step++; applySteps(); } else if (idx < slides.length - 1) show(idx + 1, 1); };
const prev = () => { if (step > 0) { step--; applySteps(); } else if (idx > 0) show(idx - 1, -1); };
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
    groups[groups.length - 1].items.push(`<li><button type="button" data-go="${s.id}"><i>${i + 1}</i><span>${esc(headline(s))}</span><small>${esc(kicker(s))}</small></button></li>`);
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
  const onControl = t.closest && t.closest('button, a, label, input, [tabindex]');
  switch (ev.key) {
    case 'ArrowRight': case 'PageDown': next(); break;
    case ' ': case 'Enter': if (onControl) return; next(); break;
    case 'ArrowLeft': case 'PageUp': prev(); break;
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
