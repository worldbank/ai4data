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
    ids, secs = pg.evaluate("() => { const s = [...document.querySelectorAll('.slide')]; return [s.map((x) => x.id), s.map((x) => x.dataset.sec)]; }")
    total, last = len(ids), ids[-1]
    whole = lambda sid: cur() == sid and pend() == 0
    presses, seen = 0, [cur()]
    while cur() != last and presses < 250:
        pg.keyboard.press('ArrowRight')
        presses += 1
        if cur() != seen[-1]:
            seen.append(cur())
    ok(f'forward walk visits {total} slides in order', seen == ids, f'{presses} presses')
    ok('each part opens with its divider', all(ids[secs.index(str(n))] == f'p{n}' for n in range(1, int(secs[-1]))), ids)
    ok('slide counter', pg.input_value('#goto') == str(total) and pg.inner_text('#total') == f'/ {total}', pg.inner_text('#total'))
    main = ids.index('pa')
    ok('the annex follows the closing slide', ids[main - 1] == 's43' and len(set(secs[main:])) == 1 and secs[main] not in secs[:main], ids[main - 1:])
    ok('the counter of an annex slide', pg.inner_text('#count') == f'Annex {total - main - 1} / {total - main - 1}', pg.inner_text('#count'))
    back = 0
    while cur() != 's01' and back < 250:
        pg.keyboard.press('ArrowLeft')
        back += 1
    ok('backward walk returns to the first slide in the same number of presses', back == presses, back)
    pg.keyboard.press('ArrowRight'); ok('the first step forward opens the divider of part 1', cur() == 'p1')
    pg.keyboard.press('ArrowRight')
    n0 = pend()
    pg.keyboard.press('ArrowRight')
    ok('a key press shows the next reveal', cur() == 's02' and pend() < n0)
    pg.keyboard.press('r'); ok('R hides the reveals of the slide again', cur() == 's02' and pend() == n0, pend())
    pg.keyboard.press('ArrowDown'); ok('the down arrow opens the next slide with every reveal shown', whole('s03'), cur())
    pg.keyboard.press('ArrowUp'); ok('the up arrow opens the previous slide with every reveal shown', whole('s02'), cur())
    pg.click('#bSkipNext'); ok('the button for the next slide', whole('s03'), cur())
    pg.click('#bSkipPrev'); ok('the button for the previous slide', whole('s02'), cur())
    pg.keyboard.press('1'); pg.keyboard.press('2'); pg.keyboard.press('Enter')
    ok('a number and Enter open that slide', whole(ids[11]) and pg.evaluate("() => document.activeElement.id") != 'goto', cur())
    pg.locator('#scrub button').nth(20).hover()
    ok('a mark of the strip shows its slide', pg.is_visible('#scrubtip') and 'Slide 21' in pg.inner_text('#scrubtip'))
    pg.locator('#scrub button').nth(20).click(); ok('a mark of the strip opens its slide', whole(ids[20]), cur())
    pg.keyboard.press('End'); ok('End', cur() == last)
    pg.keyboard.press('Home'); ok('Home', cur() == 's01')
    pg.keyboard.press('PageDown'); ok('PageDown', cur() == 'p1')
    pg.keyboard.press('PageUp'); ok('PageUp', cur() == 's01')
    pg.keyboard.press('Space'); ok('Space', cur() == 'p1')
    ok('a divider names its part and hides the strip of part names', 'Part 1 of 6' in pg.text_content('.slide.on .kicker') and not pg.is_visible('#secnav'))
    pg.locator('.slide.on .route [data-go="p4"]').click(); ok('a part on a divider opens the divider of that part', cur() == 'p4', cur())

    # Contents, notes, and links between slides.
    pg.keyboard.press('o'); ok('O opens the contents', pg.is_visible('#toc'))
    ok(f'the contents list {total} slides', pg.locator('#toclist button').count() == total)
    pg.locator('#toclist button').nth(ids.index('s23')).click()
    ok('a contents item opens its slide', cur() == 's23' and not pg.is_visible('#toc'))
    pg.click('#bToc'); pg.keyboard.press('Escape'); ok('Esc closes the contents', not pg.is_visible('#toc'))
    pg.keyboard.press('n')
    ok('N opens the notes of the current slide', pg.is_visible('#notespanel') and pg.inner_text('#notestext').startswith('Go through one round'))
    pg.keyboard.press('ArrowRight'); ok('the notes follow the slide', 'vote' in pg.inner_text('#notestext'), cur())
    pg.keyboard.press('n'); ok('N closes the notes', not pg.is_visible('#notespanel'))
    pg.locator('.slide.on .minirail button').nth(4).click(); ok('step number 5 opens s21', cur() == 's21')
    pg.locator('.slide.on [data-go="s29"]').first.click(); ok('the link on s21 opens s29', cur() == 's29')
    pg.locator('#secnav [data-sec="2"]').click(); ok('a part name opens the divider of the part', cur() == 'p2', cur())
    # The annex.
    at('s42')
    ok('the counter of the main part leaves out the annex', pg.inner_text('#count') == f'{main - 1} / {main}' and pg.inner_text('#total') == f'/ {main}', pg.inner_text('#count'))
    pg.locator(f'#secnav [data-sec="{secs[main]}"]').click(); ok('the name of the annex opens its divider', cur() == 'pa', cur())
    ok('the divider of the annex lists its slides', pg.locator('.slide.on .route button').count() == total - main - 1 and not pg.is_visible('#secnav'))
    pg.locator('.slide.on .route [data-go="s28"]').click(); ok('a slide on the divider of the annex opens', cur() == 's28', cur())
    for src, annex in (('s07', 's08'), ('s11', 's12'), ('s26', 's27'), ('s26', 's28'), ('s34', 's35'), ('s38', 's36'), ('s38', 's37')):
        at(src)
        pg.locator(f'.slide.on .src [data-go="{annex}"]').click(); there = cur()
        label = pg.inner_text('.slide.on .kicker .back')
        pg.locator('.slide.on .kicker .back').click()
        ok(f'{src}: the link opens {annex}, and its button returns', there == annex and cur() == src and label.startswith(f'Back to slide {ids.index(src) + 1}: '), (there, cur(), label))
    at('s06')
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
    ok('s05: the loop runs from the test to the practice cases', geo[0] == 5 and geo[1] < 2 and geo[2] < 2 and geo[3], geo)
    pg.locator('.slide.on .cypart[data-go="p2"]').click(); ok('s05: a part of the session opens its divider', cur() == 'p2', cur())
    at('s16')
    geo = pg.evaluate("""() => {
      const r = (q) => document.querySelector('.slide.on ' + q).getBoundingClientRect();
      const g = [...document.querySelectorAll('.slide.on .group')].map((n) => n.getBoundingClientRect());
      const a = r('.gkind.k-syn'), b = r('.gkind.k-std');
      return [Math.abs(a.left - g[0].left), Math.abs(a.right - g[0].right), Math.abs(b.left - g[1].left), Math.abs(b.right - g[2].right), a.top >= g[0].bottom];
    }""")
    ok('s16: the marks under the steps span steps 1 to 4 and steps 5 to 9', all(v < 2 for v in geo[:4]) and geo[4], geo)
    geo = pg.evaluate("""() => {
      const r = (q) => document.querySelector('.slide.on ' + q).getBoundingClientRect();
      const c = (b) => b.left + b.width / 2;
      const g = [...document.querySelectorAll('.slide.on .group')].map((n) => n.getBoundingClientRect());
      const a = r('.gloop .a'), b = r('.gloop .b'), e = r('.gloop .c');
      return [Math.abs(a.left + 1 - c(g[0])), Math.abs(e.right - 1 - c(g[2])), a.right >= b.left, b.right >= e.left, a.bottom <= g[0].top];
    }""")
    ok('s16: the loop runs from the third group of steps to the first', geo[0] < 2 and geo[1] < 2 and all(geo[2:]), geo)
    pg.locator('.slide.on .stepbtn').nth(6).click(); ok('step 7 on the process slide opens s23', cur() == 's23')

    # Demonstrations.
    stages = []
    for _ in range(4):
        stages.append(pg.locator('.slide.on .loop span.on b').inner_text())
        pg.locator('.slide.on [data-a="next"]').click(); pg.wait_for_timeout(80)
    ok('s23: the button goes through the four stages of a round and starts again',
       stages == ['1', '2', '3', '4'] and pg.locator('.slide.on .loop span.on b').inner_text() == '1', stages)
    at('s29')
    read = lambda: pg.locator('.slide.on [data-read]').inner_text()
    stops = int(pg.locator('.slide.on [data-at]').get_attribute('max'))
    ok('s29: the slide opens on the saved model', 'SAVED MODEL' in read().upper() and '68.4%' in read() and '93.8%' in read(), read())
    ok('s29: the saved model has the first codes of s31', '3 of 6' in pg.inner_text('.slide.on [data-match]'))
    pg.locator('.slide.on [data-at]').fill('0'); pg.wait_for_timeout(100)
    ok('s29: before training no first code is right', 'BEFORE TRAINING' in read().upper() and '0.0%' in read()
       and 'same code' in pg.inner_text('.slide.on [data-match]'), read())
    pg.locator('.slide.on [data-at]').fill('7'); pg.wait_for_timeout(100)
    ok('s29: one pass', 'one pass' in read().lower() and '35,515' in read() and '50.9%' in read(), read())
    pg.locator('.slide.on [data-at]').focus(); pg.keyboard.press('ArrowRight')
    ok('s29: an arrow key on the slider moves the point and keeps the slide', cur() == 's29' and '1.5 passes' in read().lower(), read())
    pg.locator('.slide.on [data-play]').click(); pg.wait_for_timeout(300)
    ok('s29: play starts before training', 'BEFORE TRAINING' in read().upper() and pg.inner_text('.slide.on [data-play]') == 'Pause')
    pg.locator('.slide.on [data-play]').click()
    ok('s29: pause', pg.inner_text('.slide.on [data-play]') != 'Pause')
    ok('s29: the points of the record', stops == 13, stops)
    pg.locator('.slide.on [data-at]').focus(); pg.keyboard.press('Escape'); pg.keyboard.press('ArrowRight')
    ok('s29: after Esc the arrow key moves on', cur() == 's31', cur())
    at('s27'); pg.locator('.slide.on .btn').first.click(); pg.wait_for_timeout(2700)
    ok('s27: the answers of the job titles arrive', pg.locator('.slide.on .sents > *').count() >= 3)
    at('s28'); kept = pg.locator('.slide.on .ck3 .drop').count(); pg.locator('.slide.on [data-a="go"][data-v="1"]').click(); pg.wait_for_timeout(150)
    ok('s28: the check shows kept and removed answers', kept == 1 and 'Job title 2 of 5' in body() and '8,031' in body(), kept)
    at('s31'); ok('s31: five codes are shown for each response', pg.locator('.slide.on .sgrow').count() == 5 and pg.locator('.slide.on .btn').count() == 0)
    at('s33'); first = body(); pg.locator('.slide.on .seg button').nth(2).click(); pg.wait_for_timeout(200)
    ok('s33: the comparison has three views', '68.4%' in first and '67.5%' in first and '59.5%' in body() and '49.7%' in body())
    at('s34'); ok('s34: times and memory of the small model', all(v in body() for v in ('0.048 s', '0.57 GB', '7×')))
    at('s02'); pg.locator('.slide.on .btn').first.click(); pg.wait_for_timeout(200)
    ok('s02: five codes are suggested', pg.locator('.slide.on .sgrow').count() == 5 and pg.locator('.slide.on .sgrow.hit').count() == 1)
    at('s12'); pg.locator('.slide.on .btn').first.click(); pg.wait_for_timeout(600)
    ok('s12: written cases are added', '120' in body())
    at('s24'); pg.locator('.slide.on .demo button').first.click(); pg.wait_for_timeout(200)
    ok('s24: a choice gets feedback', 'of 1' in pg.inner_text('.slide.on .demo'))
    at('s39'); pg.locator('.slide.on .body button').first.click(); pg.wait_for_timeout(200)
    ok('s39: an answer shows the suggestion', 'Suggested' in body())
    at('s08'); pg.locator('.slide.on .seg button').nth(2).click(); pg.wait_for_timeout(200)
    ok('s08: the very large model does not fit a laptop', 'Too large for this computer' in body())
    at('s03'); n = pg.locator('.slide.on .casebtn').count()
    for i in range(n):
        pg.locator('.slide.on .casebtn').nth(i).click()
        pg.wait_for_timeout(60)
    ok('s03: eleven tasks', n == 11, n)
    at('s32'); first = body(); pg.locator('.slide.on .seg button').nth(1).click(); pg.wait_for_timeout(200)
    ok('s32: the two scores of the rewritten answers', '66.0%' in first and '92.4%' in body())
    at('s35'); pg.locator('.slide.on .seg button').nth(2).click(); pg.wait_for_timeout(200)
    ok('s35: the example price changes the three amounts', all(v in body() for v in ('$21', '$39', '$172')))
    at('s22'); pg.locator('.slide.on input[type=checkbox]').check()
    ok('s22: the switch mixes practice cases into the test', 'Used for training and testing' in pg.inner_text('.slide.on .demo'))
    at('s19')
    for i in range(1, 6):
        pg.locator('.slide.on .chip').nth(i).click()
    ok('s19: 864 possible requests', '864' in pg.inner_text('.slide.on .demo'))
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
