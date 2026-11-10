"""Find, for each slide, the largest body scale that fits in every state its controls reach.

Reads deck.html and writes src/fit.css with one --z per slide. The search sets --z inline, so the
values already in fit.css do not affect it. `build_deck.py --fit` runs this script and rebuilds.

    python3 fit_slides.py [cap]        cap is the largest scale to try (default 1.42)
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
URL = (HERE / 'deck.html').as_uri()
CAP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.42
SAFETY = 0.96
SHOW = """(id) => {
  document.querySelectorAll('.slide').forEach((s) => s.classList.toggle('on', s.id === id));
  document.querySelectorAll('.pending').forEach((e) => e.classList.remove('pending'));
}"""
OVER = """(id) => {
  const s = document.getElementById(id), body = s.querySelector('.body');
  if (!body) return null;
  const st = document.querySelector('#stage').getBoundingClientRect();
  const src = s.querySelector('.src');
  const limit = src ? src.getBoundingClientRect().top - 2 : st.bottom - 0.022 * 45 * st.height / 45 - 8;
  const br = body.getBoundingClientRect();
  let worst = 0, why = '';
  const note = (px, w) => { if (px > worst) { worst = px; why = w; } };
  note(body.scrollHeight - body.clientHeight - 1, 'body scroll');
  for (const e of body.querySelectorAll('*')) {
    if (e.closest('svg') && e.tagName !== 'svg') continue;
    const r = e.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const scroller = e.parentElement.closest('.sents');
    if (!scroller) {
      note(r.bottom - br.bottom - 1, 'below body: ' + e.className);
      note(br.top - r.top - 1, 'above body: ' + e.className);
    }
    note(r.right - br.right - 1, 'right of body: ' + e.className);
    const cs = getComputedStyle(e);
    if (!e.classList.contains('sents') && !e.classList.contains('abs') && cs.overflowY !== 'visible') note(e.scrollHeight - e.clientHeight - 1, 'clips: ' + e.className);
    if (e.tagName !== 'INPUT' && !e.classList.contains('abs')) note(e.scrollWidth - e.clientWidth - 1, 'wide: ' + e.className + ' ' + e.textContent.slice(0, 20));
  }
  note(br.bottom - limit, 'body past the source line');
  return [worst, why];
}"""
CLICKS = """(id) => [...document.getElementById(id).querySelectorAll('.body button:not(.term):not([data-go]):not(:disabled), .body input[type=checkbox]')].length"""
CLICK = """([id, i]) => {
  const b = [...document.getElementById(id).querySelectorAll('.body button:not(.term):not([data-go]):not(:disabled), .body input[type=checkbox]')][i];
  if (b) b.click();
}"""


def worst(pg, sid):
    """Largest overflow over the default state and the states reached by one pass of clicks."""
    w, why = pg.evaluate(OVER, sid)
    n = pg.evaluate(CLICKS, sid)
    for i in range(min(n, 40)):
        pg.evaluate(CLICK, [sid, i])
        pg.wait_for_timeout(30)
        w2, why2 = pg.evaluate(OVER, sid)
        if w2 > w: w, why = w2, f'{why2} (after control {i})'
    return w, why


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1280, 'height': 720})
    pg.goto(URL)
    pg.wait_for_timeout(1500)
    if not pg.evaluate("() => document.fonts.check('16px Inter')"):
        raise SystemExit('The Inter font did not load (no network?). The scales depend on it, so fit.css was not changed.')
    ids = pg.evaluate("() => [...document.querySelectorAll('.slide')].filter((s) => s.querySelector('.body')).map((s) => s.id)")
    out = []
    for sid in ids:
        pg.evaluate(SHOW, sid)
        setz = lambda z: pg.evaluate("([id, z]) => document.getElementById(id).style.setProperty('--z', z)", [sid, z])
        setz(1); pg.wait_for_timeout(700 if sid == 's28' else 60)
        base, why = worst(pg, sid)
        if base > 0:
            lo, hi = 0.7, 1.0
        else:
            lo, hi = 1.0, CAP
            setz(hi)
            if worst(pg, sid)[0] <= 0: lo = hi
        while hi - lo > 0.02:
            mid = (lo + hi) / 2
            setz(mid)
            if worst(pg, sid)[0] <= 0: lo = mid
            else: hi = mid
        z = round(lo * SAFETY, 2) if lo < CAP else round(lo, 2)
        out.append((sid, z))
        print(sid, z, '' if base <= 0 else f'(overflowed at 1.0 by {base:.0f}px: {why})')
    (HERE / 'src/fit.css').write_text(''.join(f'#{sid} {{ --z: {z}; }}\n' for sid, z in out))
    b.close()
