# Slides

- `deck.html`: the deck, 40 slides in one HTML file. It opens in a browser from a local copy of the repository and needs no server. GitHub shows the file as source text.
- `outline.md`: the headline and the speaker notes of every slide. This is the file to review and comment on.
- `build_deck.py`: builds both files from the sources in `src/` and from the recorded data in `../assets/overview.json` and `src/occupation.json`.
- `src/`: the slide sources, the styles, and the script of the demonstrations.
- `fit_slides.py` and `check_deck.py`: the scripts that size the slides and test the deck in a browser.

The deck follows the [overview of the session](../README.md) in the same order. Each slide has a one-sentence headline, one figure or demonstration, and speaker notes.

## Presenting

| Key | Action |
|---|---|
| → , Space, Enter, Page Down | Next reveal or next slide |
| ← , Page Up | Previous slide |
| Home, End | First slide, last slide |
| O | Contents, with a link to every slide |
| N | Speaker notes of the current slide |
| F | Full screen |
| Esc | Close a definition, the contents, or the notes |

The part names at the top of a slide link to the first slide of each part. Words with a dotted underline show their definition when selected. The address of a slide is the file address followed by `#s` and the two-digit slide number, for example `deck.html#s22`.

In a window narrower than 820 pixels, the deck is laid out as one scrolling page with the notes under each slide.

The fonts (Inter and JetBrains Mono) load from Google Fonts. Without an internet connection the browser uses the system font, and long lines may wrap differently.

## Slide plan

| Slides | Part | Content |
|---|---|---|
| 1 | | Title |
| 2 to 5 | Tasks | A survey response to code, six tasks from development work, three ways to get a task done, the plan of the session |
| 6 to 9 | Small model | Large language models, small AI, the memory a model needs, the cost of building a small model |
| 10 to 14 | Synthetic data | Labelled examples, why they are missing, rare cases, synthetic data, the office comparison |
| 15 to 24 | Nine steps | The process, then one slide for each step with a demonstration on occupation coding |
| 25 to 35 | Report tagging | The task, the steps used, the request and the sentences, a model trained in the page, distillation, results, speed, limits |
| 36 to 40 | Small or large | Criteria, a six-task exercise for the room, checks before relying on a small model, summary, sources |

## Sources

| File | Content |
|---|---|
| `src/slides-01-14.html`, `src/slides-25-40.html` | Slides 1 to 14, 25, and 27 to 40, one `<section>` for each slide, with the speaker notes in `<aside class="notes">` |
| `STEPS` in `build_deck.py` | Slides 15 to 24 and 26: for each step, the office comparison, the model version, the notes, and its use for the report-tagging model |
| `src/app.js` | Navigation, the definitions of the terms, and the demonstrations |
| `src/style.css` | Styles for the stage, the reading layout, and the light and dark themes |
| `src/shell.html` | The frame of the page: the stage, the bar under it, the notes panel, and the contents |
| `src/fit.css` | The text scale of each slide, written by `fit_slides.py` |
| `src/occupation.json` | The recorded results of the occupation-coding model and its suggestions for the eight survey responses of slide 2 |

The scores, the request, the sentences, and the reports in the report-tagging part come from `../assets/overview.json`, the recorded data behind the figures of the overview page. The scores of the occupation-coding model in `src/occupation.json` were measured on 1,000 job titles from a job board, as means over three training runs. The survey responses on slide 2 and on the slides of the nine steps, and the codes shown for them, were written for the session as illustrations, and the slides say so. The five suggestions for each response on slide 2 are the recorded output of the occupation-coding model. Group names follow ISCO-08 (International Labour Organization), and the group descriptions on the slides are shortened paraphrases.

## Regenerating

```bash
python3 build_deck.py          # writes deck.html and outline.md
python3 build_deck.py --fit    # also recomputes src/fit.css, then builds again
python3 check_deck.py          # tests the deck in a browser
```

The build needs Python 3.9 or later and Pillow. The `--fit` option and the checks also need Playwright with Chromium:

```bash
pip install pillow playwright
playwright install chromium
```

Each slide has its own text scale so that its content fills the stage. `fit_slides.py` finds, for each slide, the largest scale at which the content fits in every state that the slide's controls can reach. A change to the text or the layout of a slide calls for a build with `--fit`.

`check_deck.py` renders every slide at 1280 × 720 in the light and dark themes and at 1920 × 1080, and reports content outside the stage. It then checks the reading layout at a width of 400 pixels and drives the keys, the contents, the notes, the links, and the demonstrations. `--shots DIR` saves a screenshot of every slide.

`build_deck.py --fragment PATH` also writes the page without its `<html>` and `<head>` elements, for a host that supplies them.
