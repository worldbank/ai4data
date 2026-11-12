"""Build the session 02 deck as one HTML file.

Reads the slide sources in src/, the recorded data of the overview page (../assets/overview.json),
and the program mark. The slides of the nine steps (15 to 24) and the slide of the steps used for the report-tagging model (26)
are generated from the STEPS table below.

    python3 build_deck.py              write deck.html and outline.md
    python3 build_deck.py --fit        also recompute the scale of each slide (src/fit.css); needs Playwright
    python3 build_deck.py --fragment PATH
                                       also write the page without the document skeleton, for a host
                                       that adds its own
"""
import base64
import html
import io
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'src'
REPO = HERE.parents[2]
DATA = HERE.parent / 'assets/overview.json'
LOGO = REPO / 'website/static/img/program-mark.png'
OUT = HERE / 'deck.html'
OUTLINE = HERE / 'outline.md'

# The head of deck.html. The page's own <title>, font link, and <style> follow it inside <head>.
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, '
        'initial-scale=1, viewport-fit=cover"><style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);'
        'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui}img{max-width:100%}'
        '[hidden]{display:none!important}</style>')

USED = {'yes': 'Used', 'partly': 'Partly used', 'no': 'Not used'}

# The captions of the model sizes on slide 8. They replace the captions written for the overview page.
MODELS = {
    'Small model': dict(role='Both small models of this session are this size.'),
    'Large model': dict(memory='about 31 GB of memory in its 8-bit version (about 56 GB at full 16-bit precision)',
                        role='Qwen3.8-27B is this size.'),
}

STEPS = [
    dict(short='Describe the task',
         h='The team writes down the possible answers and defines each one.',
         office='The manager writes down the decisions the new staff member will make.',
         model='The team writes the list of possible answers, with a definition of each.',
         used=('yes', 'A list of 86 topics.'), demo='task',
         notes='Start with the list of answers. Select two or three groups and read the descriptions aloud. The full '
               'list has 436 groups, and the first digit of a code gives its major group. A seller of cooked food '
               '(5212) and a street vendor of other goods (9520) sit in different major groups. Ask the room who '
               'keeps this list in their organization. The same descriptions go to the large model, to the people '
               'who check the cases, and to the test.'),
    dict(short='Show a few real cases',
         h='A few real cases show the large model how people describe their work.',
         office='The colleague who writes the practice cases first reads a few real files.',
         model='A few dozen real examples go into the request to the large model. Personal details are removed first.',
         used=('no', 'The requests named the topics only.'), demo='seeds',
         notes='Switch between the two requests. Without samples the large model writes official job titles. With '
               'three real responses in the request, the written cases take on their style: lower case, a place, a '
               'product.'),
    dict(short='Vary the cases',
         h='Each request to the large model asks for a different kind of case.',
         office='The colleague writes cases of every kind: short and long, calm and angry, easy and hard.',
         model='Each request sets a different combination of answer, length, style, language, and difficulty.',
         used=('partly', 'Each request named a different set of eight topics and asked for varied length and style.'),
         demo='vary',
         notes='Turn on the attributes one at a time and watch the number of possible requests grow to 864. Draw a '
               'few requests. Then point at the measured result below: asked for the same thing many times, a large '
               'model repeated itself in more than a third of the cases.'),
    dict(short='Write and check',
         h='A second model checks every case before it is used.',
         office='A second colleague reads each practice case and its answer before it is used.',
         model='A second model labels each case on its own. Scripts remove repeats, answers outside the list, and '
               'cases where the two models disagree.',
         used=('yes', 'Scripts checked every set. In the later rounds, a second model also labelled each sentence.'),
         demo='check',
         notes='Run the checks and count the cases that remain. Each removed case has its reason next to it. The '
               'third row matters most: the writer was wrong about its own case, and the second model caught it. A '
               'wrong answer in training becomes a wrong answer in use.'),
    dict(short='Train the small model',
         h='The small model studies the checked cases until its answers match.',
         office='The new staff member studies the practice cases.',
         model='The small model is trained on the checked cases until its answers match theirs.',
         used=('yes', 'About 20,000 sentences in the first set.'), demo='train',
         notes='This step is short on the slide and long in practice. The checked cases go in, and a model comes '
               'out. We will train a small model live on a later slide, so I will take questions about learning '
               'there.'),
    dict(short='Test on real cases',
         h='The test uses real cases that no model has trained on.',
         office='Before the new staff member works alone, a supervisor checks their decisions on real past files.',
         model='A few hundred real cases with answers given by people are set aside for testing only.',
         used=('yes', '300 real report abstracts with recorded topics.'), demo='test',
         notes='Turn the switch on and ask the room what is wrong with the picture. A test on practice cases '
               'measures how well the small model copies the writing habits of the large model. The real cases are '
               'set aside before anything is written, and no model trains on them. The chart shows the '
               'occupation-coding model on 1,000 real job titles: the first code is right for 35 percent, and the '
               'right code is among five suggestions for 56.5 percent. The titles come from the job board hahu.jobs, and '
               'the job board assigned their codes. A test on coded survey responses is still needed before use on '
               'a survey.'),
    dict(short='Practise the weak spots',
         h='Extra cases are aimed at the kinds of cases the model gets wrong.',
         office='The colleague writes more practice cases of the kinds the new staff member gets wrong.',
         model='A program finds where the model makes most mistakes and asks the large model for more cases of '
               'those kinds. The model is trained again.',
         used=('yes', 'Up to five rounds, compared with rounds of the same size spread evenly over all topics.'),
         demo='rounds',
         notes='Read the loop first. Then compare the three bars: the first set, extra cases spread evenly, and '
               'extra cases aimed at the mistakes. For occupation codes, both kinds of extra cases gave a higher '
               'score than the first set, the difference between them is within the margin of error, and the small '
               'model stays below the orange band. For report topics, the targeted rounds were better in both '
               'repeats and reached the band, which is the level of the large models.'),
    dict(short='Ask people which is better',
         h='Experienced staff choose the better of two cases.',
         office='Experienced staff compare two practice cases, or two answers, and say which one is better.',
         model='People choose between pairs. The choices improve the request to the large model and filter the cases.',
         used=('no', 'The cases were not compared by people.'), demo='pairs',
         notes='Run this as a vote. Read each pair and ask the room for A or B before you select one. People who '
               'know the work can tell within seconds which case is realistic, and choosing between two is quicker '
               'than writing a case.'),
    dict(short='Watch and refresh',
         h='A model in use needs regular review by people.',
         office="The supervisor keeps reviewing a sample of the staff member's decisions.",
         model="People review a sample of the model's answers at regular intervals. When the answers, the language, "
               'or the situation change, new cases are written and the model is trained again.',
         used=('no', 'The work ended at the test on real reports.'), demo='watch',
         notes='The example from the report is about images. The same thing happens with text when new kinds of '
               'work appear or a survey moves to another language. Ask the room who would review the sample in '
               'their organization after the project ends.'),
]

GROUPS = [('Write the cases', range(0, 4), None), ('Train and test', range(4, 6), 1),
          ('Improve and maintain', range(6, 9), 2)]

e = html.escape


def overview_slide():
    groups = []
    for name, idxs, step in GROUPS:
        items = ''.join(
            f'<li><button class="stepbtn" data-go="s{17 + i}"><span class="num">{i + 1}</span>{e(STEPS[i]["short"])}</button></li>'
            for i in idxs)
        attr = f' data-step="{step}"' if step else ''
        groups.append(f'<div class="group"{attr}><span class="lbl">{name}</span><ol>{items}</ol></div>')
    return f'''<section class="slide" id="s16" data-sec="4">
  <p class="kicker">The process</p>
  <h2>The process from a task to a small model has nine steps.</h2>
  <div class="body center">
    <div class="groups">{''.join(groups)}
      <p class="gloop" data-step="3"><span class="cap">The large model writes more cases where the small model fails</span><i class="a"></i><i class="b"></i><i class="c"></i></p>
      <p class="gkind k-syn" data-step="4"><span>Specific to synthetic data</span></p>
      <p class="gkind k-std" data-step="4"><span>Standard <button class="term" data-term="ml">machine learning</button> practice</span></p>
    </div>
    <p class="txt" data-step="5">We follow one example through the nine steps: coding the occupations written in a household survey.</p>
  </div>
  <aside class="notes">This is the process from the start of the session, now in nine steps. We look at all nine before we take them one at a time. The first four steps produce the cases, the next two build and test the model, and the last three keep improving it. The arrow above the steps shows that the process is a loop. Where the test shows mistakes, the large model writes more cases of those kinds, and the small model is trained and tested again. Step 7 runs this loop during the project. Steps 8 and 9 also end with new cases and another round of training. Steps 5 to 9 are standard machine learning practice. A team that has trained a model on labelled examples has done them before. Steps 1 to 4 are specific to synthetic data: a large model writes the examples that people would otherwise collect and label. In step 7 the large model also writes the extra cases. Each number opens its step. We use occupation coding for all nine steps because the room saw it on the second slide.</aside>
</section>'''


def step_slide(i, s):
    return f'''<section class="slide" id="s{17 + i}" data-sec="4" data-stepno="{i + 1}">
  <p class="kicker">Step {i + 1} of 9 · {e(s['short'])}</p>
  <h2>{e(s['h'])}</h2>
  <div class="body"><div class="stepgrid">
    <div class="stepside">
      <div class="pairb office"><span class="lbl">In an office</span><p>{e(s['office'])}</p></div>
      <div class="pairb model"><span class="lbl">With the models</span><p>{e(s['model'])}</p></div>
    </div>
    <div class="card demo" data-demo="{s['demo']}"></div>
  </div></div>
  <aside class="notes">{e(s['notes'])}</aside>
</section>'''


def used_slide():
    cards = ''.join(
        f'<li class="u9 {s["used"][0]}"><span class="num">{i + 1}</span><div><b>{e(s["short"])}</b>'
        f'<span class="usedrow"><span class="usedmark {s["used"][0]}" aria-hidden="true"></span>{USED[s["used"][0]]}</span>'
        f'<span class="usednote">{e(s["used"][1])}</span></div></li>'
        for i, s in enumerate(STEPS))
    return f'''<section class="slide" id="s27" data-sec="5">
  <p class="kicker">Steps used for this model</p>
  <h2>The small model for this task was built with five of the nine steps in full.</h2>
  <div class="body center">
    <ol class="used9">{cards}</ol>
  </div>
  <aside class="notes">Before the results, be clear about how this model was built. Five steps were used in full and one in part. Real samples, the choices of experienced staff, and review after deployment were not part of it. Keep that in mind when we read the scores.</aside>
</section>'''


def logo_uri():
    raw = LOGO.read_bytes()
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert('RGBA').resize((144, 144), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, 'PNG', optimize=True)
        raw = buf.getvalue()
    except ImportError:
        pass
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()


def data():
    d = json.loads(DATA.read_text())
    sy, di, le = d['synthetic'], d['distillation'], d['learn']
    stop = set(le['stop'])
    tok = lambda t: [w for w in re.findall(r'[a-z]+', t.lower()) if len(w) > 2 and w not in stop]
    rule = lambda p: next(l[2:] for l in p.split('\n') if l.startswith('- ') and 'topic names' in l)
    mean = lambda xs: sum(xs) / len(xs)
    return {
        'tasks': d['smallAI']['tasks'],
        'machines': {**d['smallAI']['machines'],
                     'models': [{**m, **MODELS.get(m['name'], {})} for m in d['smallAI']['machines']['models']]},
        'synth': {
            'request': sy['request'], 'writer': sy['writer'], 'topics': sy['topics'],
            'opening': sy['prompt']['allow'].split('\n')[0],
            'sentences': [{'text': s['text'], 'labels': s['labels']} for s in sy['sets']['allow']['sentences']],
            'sets': {k: {'rule': rule(sy['prompt'][k]), 'share': v['named'] / v['total'], 'f1': mean(v['report_f1'])}
                     for k, v in sy['sets'].items()},
        },
        'doc': {k: di['documents'][0][k] for k in ('title', 'year', 'text', 'reference', 'student')},
        'dist': {'threshold': di['threshold'], 'top': di['top'], 'scores': di['scores'], 'speed': di['speed']},
        'workflow': d['workflow'],
        'occ': json.loads((SRC / 'occupation.json').read_text()),
        'cost': json.loads((SRC / 'cost.json').read_text()),
        'learn': {'topics': le['topics'], 'stop': le['stop'], 'sentences': le['sentences'],
                  'tests': [{'title': t['title'], 'topic': t['topic'], 'w': tok(t['text'])} for t in le['tests']]},
    }


def page():
    slides = '\n\n'.join([
        (SRC / 'slides-01-15.html').read_text().replace('__LOGO__', logo_uri()),
        overview_slide(), *[step_slide(i, s) for i, s in enumerate(STEPS)],
        (SRC / 'slides-26-43.html').read_text().replace('<!--__USED__-->', used_slide()),
    ])
    payload = json.dumps(data(), ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    js = (SRC / 'app.js').read_text().replace('/*__DATA__*/null', payload)
    fit = SRC / 'fit.css'
    css = (SRC / 'style.css').read_text().replace('/*__FIT__*/', fit.read_text() if fit.exists() else '')
    out = (SRC / 'shell.html').read_text().replace('/*__CSS__*/', css)
    return out.replace('<!--__SLIDES__-->', slides).replace('/*__JS__*/', js), slides


def outline(slides):
    """The headline and the speaker notes of every slide, as Markdown for review."""
    parts = re.search(r"const SECS = \[(.*?)\];", (SRC / 'app.js').read_text()).group(1)
    parts = [p.strip().strip("'") for p in parts.split(',')]
    text = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', s))).strip()
    first = lambda pat, s: (lambda m: text(m.group(1)) if m else '')(re.search(pat, s, re.S))
    lines = ['# Session 02 deck outline', '',
             'Generated by `build_deck.py` from the slide sources. This file holds the headline and the speaker '
             'notes of every slide. To change a slide, edit the sources and regenerate.', '']
    part = None
    for n, sec in enumerate(re.findall(r'<section class="slide.*?</section>', slides, re.S), start=1):
        p = parts[int(re.search(r'data-sec="(\d)"', sec).group(1))]
        if p and p != part:
            lines += [f'## {p}', '']
            part = p
        kicker = first(r'<p class="kicker">(.*?)</p>', sec.replace(' <span class="deeper">', ' · <span>'))
        lines += [f'### {n}. {kicker}' if kicker else f'### {n}.', '', first(r'<h[12]>(.*?)</h[12]>', sec), '']
        if 'data-demo=' in sec:
            lines += ['The slide has an interactive demonstration.', '']
        notes = first(r'<aside class="notes">(.*?)</aside>', sec)
        if notes:
            lines += [f'Notes: {notes}', '']
    return '\n'.join(lines)


def build():
    out, slides = page()
    head, body = out.split('</style>', 1)
    OUT.write_text(HEAD + head + '</style></head><body>' + body + '</body></html>\n')
    OUTLINE.write_text(outline(slides))
    print(OUT.relative_to(REPO), f'{len(out) / 1024:.0f} KB', slides.count('<section'), 'slides')
    return out


def main():
    out = build()
    if '--fit' in sys.argv:
        subprocess.run([sys.executable, str(HERE / 'fit_slides.py')], check=True)
        out = build()
    if '--fragment' in sys.argv:
        Path(sys.argv[sys.argv.index('--fragment') + 1]).write_text(out)


if __name__ == '__main__':
    main()
