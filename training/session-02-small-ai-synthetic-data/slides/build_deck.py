"""Build the session 02 deck as one HTML file.

Reads the slide sources in src/, the recorded data of the overview page (../assets/overview.json),
the recorded files of the occupation model (../assets/occupation/), and the two logos of the cover. The slides of the
nine steps (s16 to s25) are generated from the STEPS table below. The divider that opens each part (p1 to p7) is
generated from the DIVIDERS table. The annex (src/slides-annex.html) follows the closing slide, and its divider (pa)
lists the slides of the annex. A slide keeps its id when slides are added before it or when it moves to the annex, so
the id is not the slide number.

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
OCC = HERE.parent / 'assets/occupation'
IMG = REPO / 'website/static/img'
# The logos of the cover, each with the height in pixels at which it is embedded.
LOGOS = {'__LOGO__': ('program-logo.png', 150), '__DDG__': ('ddg-logo.png', 100)}
OUT = HERE / 'deck.html'
OUTLINE = HERE / 'outline.md'

# The head of deck.html. The page's own <title>, font link, and <style> follow it inside <head>.
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, '
        'initial-scale=1, viewport-fit=cover"><style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);'
        'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui}img{max-width:100%}'
        '[hidden]{display:none!important}</style>')


# The captions of the model sizes on s08. They replace the captions written for the overview page.
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
         demo='task',
         notes='Start with the list of answers. Select two or three groups and read the descriptions aloud. The full '
               'list has 436 groups, and the first digit of a code gives its major group. A seller of cooked food '
               '(5212) and a street vendor of other goods (9520) sit in different major groups. Ask the room who '
               'keeps this list in their organization. The same descriptions go to the large model, to the people '
               'who check the cases, and to the test.'),
    dict(short='Show a few real cases',
         h='A few real cases show the large model how people describe their work.',
         office='The colleague who writes the practice cases first reads a few real files.',
         model='A few dozen real examples go into the request to the large model. Personal details are removed first.',
         demo='seeds',
         notes='Switch between the two requests. Without samples the large model writes official job titles. With '
               'three real responses in the request, the written cases take on their style: lower case, a place, a '
               'product.'),
    dict(short='Vary the cases',
         h='Each request to the large model asks for a different kind of case.',
         office='The colleague writes cases of every kind: short and long, calm and angry, easy and hard.',
         model='Each request sets a different combination of answer, length, style, language, and difficulty.',
         demo='vary',
         notes='Turn on the attributes one at a time and watch the number of possible requests grow to 864. Draw a '
               'few requests. Then point at the measured result below: asked for the same thing many times, a large '
               'model repeated itself in more than a third of the cases.'),
    dict(short='Write and check',
         h='A second model checks every case before it is used.',
         office='A second colleague reads each practice case and its answer before it is used.',
         model='A second model labels each case on its own. Scripts remove repeats, answers outside the list, and '
               'cases where the two models disagree.',
         demo='check',
         notes='Run the checks and count the cases that remain. Each removed case has its reason next to it. The '
               'third row matters most: the writer was wrong about its own case, and the second model caught it. A '
               'wrong answer in training becomes a wrong answer in use.'),
    dict(short='Train the small model',
         h='The small model studies the checked cases until its answers match.',
         office='The new staff member studies the practice cases.',
         model='The small model is trained on the checked cases until its answers match theirs.',
         demo='train',
         notes='This step is short on the slide and long in practice. The checked cases go in, and a model comes '
               'out. Part 5 shows the scores of the occupation model during its training, so I will take questions about '
               'learning there.'),
    dict(short='Test on real cases',
         h='The test uses real cases that no model has trained on.',
         office='Before the new staff member works alone, a supervisor checks their decisions on real past files.',
         model='A few hundred real cases with answers given by people are set aside for testing only.',
         demo='test',
         notes='Turn the switch on and ask the room what is wrong with the picture. A test on practice cases '
               'measures how well the small model copies the writing habits of the large model. The real cases are '
               'set aside before anything is written, and no model trains on them. We read two scores from the test: '
               'the share of cases where the first code of the model is right, and the share where the right code '
               'is among its first five. Part 5 shows the test of the occupation model.'),
    dict(short='Practise the weak spots',
         h='Extra cases are aimed at the kinds of cases the model gets wrong.',
         office='The colleague writes more practice cases of the kinds the new staff member gets wrong.',
         model='A program finds where the model makes most mistakes and asks the large model for more cases of '
               'those kinds. The model is trained again.',
         demo='rounds',
         notes='Go through one round with the button. The test shows where the model is wrong, the mistakes are '
               'counted for each group, the large model writes more cases for the weakest group, and the model is '
               'trained and tested again. The test cases never enter training. The annex has the scores of the '
               'report-tagging model before and after rounds like this one.'),
    dict(short='Ask people which is better',
         h='Experienced staff choose the better of two cases.',
         office='Experienced staff compare two practice cases, or two answers, and say which one is better.',
         model='People choose between pairs. The choices improve the request to the large model and filter the cases.',
         demo='pairs',
         notes='Run this as a vote. Read each pair and ask the room for A or B before you select one. People who '
               'know the work can tell within seconds which case is realistic, and choosing between two is quicker '
               'than writing a case.'),
    dict(short='Watch and refresh',
         h='A model in use needs regular review by people.',
         office="The supervisor keeps reviewing a sample of the staff member's decisions.",
         model="People review a sample of the model's answers at regular intervals. When the answers, the language, "
               'or the situation change, new cases are written and the model is trained again.',
         demo='watch',
         notes='The example from the report is about images. The same thing happens with text when new kinds of '
               'work appear or a survey moves to another language. Ask the room who would review the sample in '
               'their organization after the project ends.'),
]

GROUPS = [('Write the cases', range(0, 4), None), ('Train and test', range(4, 6), 1),
          ('Improve and maintain', range(6, 9), 2)]

# One divider opens each part. Each entry holds the line under the name of the part and the speaker notes.
# The names of the parts are the SECS constant of src/app.js. The last name of SECS is the annex.
DIVIDERS = [
    ('We start with the task: a survey response needs one code from a fixed list.',
     'Say that the session has six parts and name them from the strip at the bottom of the slide. The first part shows '
     'the task: one survey response, then other tasks of the same kind, then an outline of the process.'),
    ('We compare a small model with a large one and give the reasons to build one.',
     'The room has seen the task and the outline of the process. This part explains the small '
     'model: its size next to a large language model and the reasons to build one.'),
    ('A small model needs labelled examples. In this part a large model writes them.',
     'The room now knows what a small model is. This part explains the examples that the model '
     'learns from, why real ones are often missing, and how a large model writes them. It also names the method: distillation.'),
    ('We go through the process one step at a time, with examples from occupation coding.',
     'The room has now seen the task, the small model, and synthetic data. This part puts them '
     'together as a process of nine steps. Each step has a short example from occupation coding. The scores of the '
     'occupation model come in part 5.'),
    ('We apply the steps to one occupation model that was built and tested.',
     'Part 4 gave the process. This part applies it to one model: the training texts, the training run, and the '
     'test. The line above each headline names the step. The part ends with the scores next to three large models '
     'and the speed. This build did not include steps 8 and 9.'),
    ('We end with the choice between a small and a large model for a task.',
     'The last part is about the choice between a small and a large model. It has the criteria, '
     'the checks before relying on a small model, the requirements of a first attempt, and an exercise for the room.'),
]

# The divider of the annex: the line under the title and the speaker notes.
ANNEX = ('These slides add detail to the main part. Each one links back to its slide.',
         'The annex holds slides with more detail: where a model can run, the real cases of a rare group, the request '
         'and the check of the rewritten answers, the cost of building, and a second small model that tags reports '
         'with topics. Open one when a question calls for it. '
         'The button in the top line of each slide returns to its slide in the main part.')

e = html.escape


def part_names():
    names = re.search(r"const SECS = \[(.*?)\];", (SRC / 'app.js').read_text()).group(1)
    return [n.strip().strip("'") for n in names.split(',')]


def divider_slide(n):
    """The slide that opens part n: the name of the part, one line, and all the parts as a strip."""
    names = part_names()[1:-1]
    assert len(names) == len(DIVIDERS)
    line, notes = DIVIDERS[n - 1]
    cells = []
    for k, name in enumerate(names, start=1):
        if k == n:
            cells.append(f'<li class="here" aria-current="step"><span><i>{k}</i>{e(name)}</span></li>')
        else:
            past = ' class="past"' if k < n else ''
            cells.append(f'<li{past}><button type="button" data-go="p{k}"><i>{k}</i>{e(name)}</button></li>')
    return f'''<section class="slide divider" id="p{n}" data-sec="{n}">
  <p class="kicker">Part {n} of {len(names)}</p>
  <div class="dvtitle">
    <h2>{e(names[n - 1])}</h2>
    <p class="sub">{e(line)}</p>
  </div>
  <span class="dvnum" aria-hidden="true">{n}</span>
  <ol class="route" style="--cols:{len(names)}" aria-label="Parts of the session">{''.join(cells)}</ol>
  <aside class="notes">{e(notes)}</aside>
</section>'''


def annex_divider(annex):
    """The slide that opens the annex: the title, one line, and the slides of the annex as a strip."""
    name, n = part_names()[-1], len(DIVIDERS) + 1
    line, notes = ANNEX
    heads = re.findall(r'<section class="slide" id="(\w+)" data-sec="(\d)"[^>]*>\s*<p class="kicker">(.*?)</p>', annex)
    assert heads and all(sec == str(n) for _, sec, _ in heads)
    cells = [f'<li><button type="button" data-go="{sid}"><i>{k}</i>{kicker.split(" · ")[-1]}</button></li>'
             for k, (sid, _, kicker) in enumerate(heads, start=1)]
    return f'''<section class="slide divider" id="pa" data-sec="{n}">
  <p class="kicker">{e(name)}</p>
  <div class="dvtitle">
    <h2>Further detail</h2>
    <p class="sub">{e(line)}</p>
  </div>
  <span class="dvnum" aria-hidden="true">A</span>
  <ol class="route" style="--cols:{len(cells)}" aria-label="Slides of the annex">{''.join(cells)}</ol>
  <aside class="notes">{e(notes)}</aside>
</section>'''


def with_dividers(slides):
    """Put the divider of each part before the first slide of the part."""
    for n in range(1, len(DIVIDERS) + 1):
        tag = re.search(rf'<section class="slide[^>]*data-sec="{n}"', slides)
        slides = slides[:tag.start()] + divider_slide(n) + '\n\n' + slides[tag.start():]
    return slides


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
  <aside class="notes">This is the process from the start of the session, now in nine steps. We look at all nine before we take them one at a time. The first four steps produce the cases, the next two build and test the model, and the last three keep improving it. The arrow above the steps shows that the process is a loop. Where the test shows mistakes, the large model writes more cases of those kinds, and the small model is trained and tested again. Step 7 runs this loop during the project. Steps 8 and 9 also end with new cases and another round of training. Steps 5 to 9 are standard machine learning practice. A team that has trained a model on labelled examples has done them before. Steps 1 to 4 are specific to synthetic data: a large model writes the examples that people would otherwise collect and label. In step 7 the large model also writes the extra cases. Each number opens its step. We use occupation coding for all nine steps because the room saw it at the start of the session.</aside>
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


def logo_uri(name, height):
    raw = (IMG / name).read_bytes()
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert('RGBA')
        im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, 'PNG', optimize=True)
        raw = buf.getvalue()
    except ImportError:
        pass
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()


def data():
    d = json.loads(DATA.read_text())
    di = d['distillation']
    occ = json.loads((SRC / 'occupation.json').read_text())
    # The request of the rewriting run, without its opening paragraph and with the first of its four rules.
    paras = json.loads((OCC / 'results.json').read_text())['prompt'].split('\n\n')
    occ['request'] = '\n\n'.join([*paras[1:3], paras[3].split('\n')[1][2:]])
    return {
        'tasks': d['smallAI']['tasks'],
        'machines': {**d['smallAI']['machines'],
                     'models': [{**m, **MODELS.get(m['name'], {})} for m in d['smallAI']['machines']['models']]},
        'doc': {k: di['documents'][0][k] for k in ('title', 'year', 'text', 'reference')},
        'dist': {'scores': [{**x, 'model': x['model'].replace('Student', 'Small model')} for x in di['scores']]},
        'workflow': d['workflow'],
        'occ': occ,
        'cost': json.loads((SRC / 'cost.json').read_text()),
        'learn': json.loads((SRC / 'learn.json').read_text()),
    }


def page():
    first = (SRC / 'slides-01-15.html').read_text()
    for mark, (name, height) in LOGOS.items():
        first = first.replace(mark, logo_uri(name, height))
    slides = '\n\n'.join([
        first,
        overview_slide(), *[step_slide(i, s) for i, s in enumerate(STEPS)],
        (SRC / 'slides-26-43.html').read_text(),
    ])
    annex = (SRC / 'slides-annex.html').read_text()
    slides = '\n\n'.join([with_dividers(slides), annex_divider(annex), annex])
    payload = json.dumps(data(), ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    js = (SRC / 'app.js').read_text().replace('/*__DATA__*/null', payload)
    fit = SRC / 'fit.css'
    css = (SRC / 'style.css').read_text().replace('/*__FIT__*/', fit.read_text() if fit.exists() else '')
    out = (SRC / 'shell.html').read_text().replace('/*__CSS__*/', css)
    return out.replace('<!--__SLIDES__-->', slides).replace('/*__JS__*/', js), slides


def outline(slides):
    """The headline and the speaker notes of every slide, as Markdown for review."""
    parts = part_names()
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
