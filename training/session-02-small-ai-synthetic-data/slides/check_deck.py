"""Check deck.html in a browser: layout of every slide, the reading layout at phone width, and the controls.

    python3 check_deck.py                  run every check; the exit status is 1 if one fails
    python3 check_deck.py --shots DIR      also save a screenshot of every slide in DIR

Needs Playwright with Chromium. Run it after `build_deck.py`.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
URL = (HERE / 'deck.html').as_uri()
SHOTS = Path(sys.argv[sys.argv.index('--shots') + 1]) if '--shots' in sys.argv else None
FAILED = []


def ok(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name, extra if not cond or extra else '')
    if not cond:
        FAILED.append(name)


# Elements of the current slide that leave the stage, overflow their box, or are clipped.
LAYOUT = """() => {
  const s = document.querySelector('.slide.on'), st = document.querySelector('#stage').getBoundingClientRect();
  const out = [];
  const body = s.querySelector('.body');
  if (body && body.scrollHeight > body.clientHeight + 2) out.push(`body overflows by ${body.scrollHeight - body.clientHeight}px`);
  const pad = 4;
  for (const e of s.querySelectorAll('*')) {
    if (e.closest('.notes') || e.closest('svg') && e.tagName !== 'svg') continue;
    const r = e.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    if (r.bottom > st.bottom + pad || r.right > st.right + pad || r.left < st.left - pad)
      out.push(`${e.tagName}.${e.className} outside stage (b ${Math.round(r.bottom - st.bottom)}, r ${Math.round(r.right - st.right)})`);
    if (e.scrollWidth > e.clientWidth + 2 && getComputedStyle(e).overflowX === 'visible' && e.children.length === 0)
      out.push(`${e.tagName}.${e.className} text wider than box by ${e.scrollWidth - e.clientWidth}: ${e.textContent.slice(0, 30)}`);
  }
  for (const e of s.querySelectorAll('.card, .demo, .fill')) {
    const o = getComputedStyle(e).overflowY;
    if (e.scrollHeight > e.clientHeight + 3 && o !== 'visible' && !e.classList.contains('sents'))
      out.push(`${e.className} clips ${e.scrollHeight - e.clientHeight}px`);
  }
  return out.slice(0, 6);
}"""
WIDE = """() => { const out = []; for (const e of document.querySelectorAll('.slide *')) { const r = e.getBoundingClientRect();
  if (r.width && (r.right > innerWidth + 1 || r.left < -1)) out.push([e.closest('.slide').id, e.tagName, e.className.toString().slice(0, 30), Math.round(r.right)]); }
  return out.slice(0, 20); }"""
CLIPPED = """() => { const out = []; for (const e of document.querySelectorAll('.slide .body *')) { if (e.closest('svg') || e.tagName === 'INPUT') continue;
  if (e.scrollWidth > e.clientWidth + 2 && getComputedStyle(e).overflowX !== 'visible' && !e.classList.contains('abs')) out.push([e.closest('.slide').id, e.className.toString().slice(0, 30), e.scrollWidth, e.clientWidth]); }
  return out.slice(0, 20); }"""


def open_page(browser, w, h, theme='light'):
    pg = browser.new_page(viewport={'width': w, 'height': h}, color_scheme=theme)
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(URL)
    pg.wait_for_timeout(1500)
    return pg, errs


def layout(browser, w, h, theme):
    """Walk the deck with the arrow key and check each slide once all its reveals are shown."""
    pg, errs = open_page(browser, w, h, theme)
    total = pg.evaluate("() => document.querySelectorAll('.slide').length")
    seen, bad = [], []
    for _ in range(300):
        sid, pending = pg.evaluate("() => { const s = document.querySelector('.slide.on'); return [s.id, s.querySelectorAll('.pending').length]; }")
        if not pending and sid not in seen:
            seen.append(sid)
            pg.wait_for_timeout(450)
            found = pg.evaluate(LAYOUT)
            if found:
                bad.append((sid, found))
            if SHOTS:
                SHOTS.mkdir(parents=True, exist_ok=True)
                pg.screenshot(path=str(SHOTS / f'{theme}{w}_{sid}.png'))
            if len(seen) == total:
                break
        pg.keyboard.press('ArrowRight')
    ok(f'layout {w}x{h} {theme}: {len(seen)} of {total} slides fit the stage', len(seen) == total and not bad, bad)
    ok(f'layout {w}x{h} {theme}: no console or page errors', not errs, errs[:5])
    pg.close()


def reading(browser, theme):
    """Below 820 px the deck is a scrolling page. Nothing may be wider than a 400 px screen."""
    pg, errs = open_page(browser, 400, 800, theme)
    width = pg.evaluate("() => document.documentElement.scrollWidth")
    ok(f'reading {theme}: the page is 400 px wide', width == 400, width)
    wide = pg.evaluate(WIDE)
    ok(f'reading {theme}: no element is wider than the screen', not wide, wide)
    clipped = pg.evaluate(CLIPPED)
    ok(f'reading {theme}: no text is clipped', not clipped, clipped)
    ok(f'reading {theme}: no console or page errors', not errs, errs[:5])
    if SHOTS:
        SHOTS.mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=str(SHOTS / f'reading_{theme}.png'), full_page=True)
    pg.close()


def controls(browser):
    """Drive the deck with real keys and clicks."""
    pg, errs = open_page(browser, 1280, 720)
    cur = lambda: pg.evaluate("() => document.querySelector('.slide.on').id")
    pend = lambda: pg.evaluate("() => document.querySelectorAll('.slide.on .pending').length")
    body = lambda: pg.inner_text('.slide.on .body')

    def at(sid):
        pg.goto(URL + '#' + sid)
        pg.wait_for_timeout(400)
        while pend():
            pg.keyboard.press('ArrowRight')

    # Keys.
    presses, seen = 0, [cur()]
    while cur() != 's40' and presses < 200:
        pg.keyboard.press('ArrowRight')
        presses += 1
        if cur() != seen[-1]:
            seen.append(cur())
    ok('forward walk visits 40 slides in order', seen == [f's{i:02d}' for i in range(1, 41)], f'{presses} presses')
    ok('slide counter', pg.inner_text('#where') == '40 / 40', pg.inner_text('#where'))
    back = 0
    while cur() != 's01' and back < 200:
        pg.keyboard.press('ArrowLeft')
        back += 1
    ok('backward walk returns to the first slide in the same number of presses', back == presses, back)
    pg.keyboard.press('ArrowRight')
    n0 = pend()
    pg.keyboard.press('ArrowRight')
    ok('a key press shows the next reveal', cur() == 's02' and pend() < n0)
    pg.keyboard.press('End'); ok('End', cur() == 's40')
    pg.keyboard.press('Home'); ok('Home', cur() == 's01')
    pg.keyboard.press('PageDown'); ok('PageDown', cur() == 's02')
    pg.keyboard.press('PageUp'); ok('PageUp', cur() == 's01')
    pg.keyboard.press('Space'); ok('Space', cur() == 's02')

    # Contents, notes, and links between slides.
    pg.keyboard.press('o'); ok('O opens the contents', pg.is_visible('#toc'))
    ok('the contents list 40 slides', pg.locator('#toclist button').count() == 40)
    pg.locator('#toclist button').nth(21).click()
    ok('a contents item opens its slide', cur() == 's22' and not pg.is_visible('#toc'))
    pg.click('#bToc'); pg.keyboard.press('Escape'); ok('Esc closes the contents', not pg.is_visible('#toc'))
    pg.keyboard.press('n')
    ok('N opens the notes of the current slide', pg.is_visible('#notespanel') and pg.inner_text('#notestext').startswith('Read the loop first'))
    pg.keyboard.press('ArrowRight'); ok('the notes follow the slide', 'vote' in pg.inner_text('#notestext'), cur())
    pg.keyboard.press('n'); ok('N closes the notes', not pg.is_visible('#notespanel'))
    pg.locator('.slide.on .minirail button').nth(4).click(); ok('step number 5 opens slide 20', cur() == 's20')
    pg.locator('.slide.on [data-go="s30"]').first.click(); ok('the link on slide 20 opens slide 30', cur() == 's30')
    pg.locator('#secnav [data-sec="2"]').click(); ok('a part name opens the first slide of the part', cur() == 's06', cur())
    while pend():
        pg.keyboard.press('ArrowRight')
    pg.locator('.slide.on .term').first.click()
    ok('a term opens its definition', pg.locator('.pop').count() == 1)
    box = pg.locator('.pop').bounding_box()
    ok('the definition stays inside the stage', box['x'] >= 0 and box['x'] + box['width'] <= 1280 and box['y'] + box['height'] <= 720, box)
    pg.keyboard.press('Escape'); ok('Esc closes the definition', pg.locator('.pop').count() == 0)
    at('s05')
    geo = pg.evaluate("""() => {
      const c = (n) => { const r = n.getBoundingClientRect(); return r.left + r.width / 2; };
      const cards = [...document.querySelectorAll('.slide.on .cyflow .card')], arc = document.querySelector('.slide.on .cyloop .arc').getBoundingClientRect();
      return [cards.length, Math.abs(arc.left + 1 - c(cards[1])), Math.abs(arc.right - 1 - c(cards[3])), arc.bottom <= cards[1].getBoundingClientRect().top];
    }""")
    ok('slide 5: the loop runs from the test to the practice cases', geo[0] == 5 and geo[1] < 2 and geo[2] < 2 and geo[3], geo)
    pg.locator('.slide.on .cypart[data-go="s06"]').click(); ok('slide 5: a part of the session opens its first slide', cur() == 's06', cur())
    at('s15')
    geo = pg.evaluate("""() => {
      const r = (q) => document.querySelector('.slide.on ' + q).getBoundingClientRect();
      const g = [...document.querySelectorAll('.slide.on .group')].map((n) => n.getBoundingClientRect());
      const a = r('.gkind.k-syn'), b = r('.gkind.k-std');
      return [Math.abs(a.left - g[0].left), Math.abs(a.right - g[0].right), Math.abs(b.left - g[1].left), Math.abs(b.right - g[2].right), a.top >= g[0].bottom];
    }""")
    ok('slide 15: the marks under the steps span steps 1 to 4 and steps 5 to 9', all(v < 2 for v in geo[:4]) and geo[4], geo)
    pg.locator('.slide.on .stepbtn').nth(6).click(); ok('step 7 on the process slide opens slide 22', cur() == 's22')

    # Demonstrations.
    pg.locator('.slide.on .seg button').nth(1).click()
    ok('slide 22: the second task', 'Report topics' in pg.locator('.slide.on .seg button[aria-pressed="true"]').inner_text())
    at('s30')
    res = []
    for n in ('1', '10', '40'):
        pg.locator(f'.slide.on [data-n="{n}"]').click()
        pg.wait_for_timeout(120)
        res.append(pg.evaluate("() => (document.querySelector('.slide.on .body').innerText.match(/(\\d+) of 60/) || [])[1]"))
    ok('slide 30: 17, 31, and 34 of 60 reports with 1, 10, and 40 examples', res == ['17', '31', '34'], res)
    pg.locator('.slide.on input[type=text], .slide.on input:not([type])').first.fill('The clinic needs more nurses and vaccines')
    pg.wait_for_timeout(150)
    ok('slide 30: a typed sentence gets a topic', 'Health' in body())
    pg.keyboard.press('ArrowRight'); ok('slide 30: an arrow key in the text box keeps the slide', cur() == 's30')
    pg.keyboard.press('Escape'); pg.keyboard.press('ArrowRight'); ok('slide 30: after Esc the arrow key moves on', cur() == 's31', cur())
    at('s28'); pg.locator('.slide.on .btn').first.click(); pg.wait_for_timeout(2500)
    ok('slide 28: the sentences arrive', pg.locator('.slide.on .sents > *').count() >= 3)
    at('s02'); pg.locator('.slide.on .btn').first.click(); pg.wait_for_timeout(200)
    ok('slide 2: five codes are suggested', pg.locator('.slide.on .sgrow').count() == 5 and pg.locator('.slide.on .sgrow.hit').count() == 1)
    at('s12'); pg.locator('.slide.on .btn').first.click(); pg.wait_for_timeout(600)
    ok('slide 12: written cases are added', '120' in body())
    at('s23'); pg.locator('.slide.on .demo button').first.click(); pg.wait_for_timeout(200)
    ok('slide 23: a choice gets feedback', 'of 1' in pg.inner_text('.slide.on .demo'))
    at('s37'); pg.locator('.slide.on .body button').first.click(); pg.wait_for_timeout(200)
    ok('slide 37: an answer shows the suggestion', 'Suggested' in body())
    at('s08'); pg.locator('.slide.on .seg button').nth(2).click(); pg.wait_for_timeout(200)
    ok('slide 8: the very large model does not fit a laptop', 'Too large for this computer' in body())
    at('s03'); n = pg.locator('.slide.on .casebtn').count()
    for i in range(n):
        pg.locator('.slide.on .casebtn').nth(i).click()
        pg.wait_for_timeout(60)
    ok('slide 3: six tasks', n == 6, n)
    at('s21'); pg.locator('.slide.on input[type=checkbox]').check()
    ok('slide 21: the switch mixes practice cases into the test', 'Used for training and testing' in pg.inner_text('.slide.on .demo'))
    at('s18')
    for i in range(1, 6):
        pg.locator('.slide.on .chip').nth(i).click()
    ok('slide 18: 864 possible requests', '864' in pg.inner_text('.slide.on .demo'))
    pg.keyboard.press('f'); pg.wait_for_timeout(200)
    ok('controls: no console or page errors', not errs, errs[:5])
    pg.close()


with sync_playwright() as p:
    browser = p.chromium.launch()
    for w, h, theme in ((1280, 720, 'light'), (1280, 720, 'dark'), (1920, 1080, 'light')):
        layout(browser, w, h, theme)
    for theme in ('light', 'dark'):
        reading(browser, theme)
    controls(browser)
    browser.close()

print(f'{len(FAILED)} failed' if FAILED else 'all checks passed')
sys.exit(1 if FAILED else 0)
