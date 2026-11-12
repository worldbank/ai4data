# Slides

- `deck.html`: the deck, 49 slides in one HTML file. It opens in a browser from a local copy of the repository and needs no server. GitHub shows the file as source text.
- `outline.md`: the headline and the speaker notes of every slide. This is the file to review and comment on.
- `build_deck.py`: builds both files from the sources in `src/` and from the recorded data in `../assets/overview.json`, `src/occupation.json`, and `src/cost.json`.
- `src/`: the slide sources, the styles, and the script of the demonstrations.
- `fit_slides.py` and `check_deck.py`: the scripts that size the slides and test the deck in a browser.

The deck follows the [overview of the session](../README.md) in the same order. Each slide has a one-sentence headline, one figure or demonstration, and speaker notes.

The docs site shows `deck.html` on the Slides page of the session. The site build copies the file, so a rebuilt deck appears on the site after the next build.

## Presenting

| Key | Action |
|---|---|
| → , Space, Enter, Page Down | Next reveal, or the next slide after the last reveal |
| ← , Page Up | Previous reveal, or the previous slide before the first reveal |
| ↓ , ↑ | Next slide, previous slide |
| A number, then Enter | The slide with that number |
| R | Hide the reveals of the current slide again |
| Home, End | First slide, last slide |
| O | Contents, with a link to every slide |
| N | Speaker notes of the current slide |
| F | Full screen |
| Esc | Close a definition, the contents, or the notes |

A slide reached with → starts with its reveals hidden. Every other move opens a slide with all its reveals shown.

The bar under the stage has four buttons: `«` and `»` move by one slide, and `←` and `→` move by one reveal. Next to them are a box for a slide number and a strip with one mark for each slide. A mark shows the number and the headline of its slide under the pointer, and a click on the mark opens the slide. The taller marks are the first slides of the six parts.

Each part opens with a divider slide. The divider names the part, describes its content in one line, and lists the six parts with the current one marked. The part names at the top of the other slides link to these dividers. Words with a dotted underline show their definition when selected.

Each slide has an id, and the address of a slide is the file address followed by `#` and the id, for example `deck.html#s22`. The ids of the dividers are `p1` to `p6`, and the ids of the other slides are `s01` to `s43`. A slide keeps its id when slides are added before it, so the id differs from the slide number in the bar.

In a window narrower than 820 pixels, the deck is laid out as one scrolling page with the notes under each slide.

The fonts (Inter and JetBrains Mono) load from Google Fonts. Without an internet connection the browser uses the system font, and long lines may wrap differently.

## Slide plan

| Slides | Part | Content |
|---|---|---|
| 1 | | Title |
| 2 to 6 | Tasks | A survey response to code, eleven tasks from data work and services, three ways to get a task done, a diagram of the process with the parts of the session |
| 7 to 11 | Small model | Large language models, small AI, the memory a model needs, the cost of building a small model |
| 12 to 18 | Synthetic data | Labelled examples, why they are missing, rare cases, synthetic data, the office comparison, labelled job titles rewritten as survey answers |
| 19 to 29 | Nine steps | The process, then one slide for each step with a demonstration on occupation coding |
| 30 to 42 | Report tagging | The task, the steps used, the request and the sentences, a model trained in the page, distillation, results, speed, the measured cost of building each model, limits |
| 43 to 49 | Small or large | Criteria, a six-task exercise for the room, checks before relying on a small model, the requirements of a first attempt, summary, sources |

The first slide of each part is its divider.

## Sources

| File | Content |
|---|---|
| `src/slides-01-15.html`, `src/slides-26-43.html` | The slides with the ids `s01` to `s15`, `s26`, and `s28` to `s43`, one `<section>` for each slide, with the speaker notes in `<aside class="notes">` |
| `STEPS` in `build_deck.py` | The slides `s16` to `s25` and `s27`: for each step, the office comparison, the model version, the notes, and its use for the report-tagging model |
| `DIVIDERS` in `build_deck.py` | The dividers `p1` to `p6`: for each part, the line under its name and the notes. The names of the parts come from `SECS` in `src/app.js` |
| `src/app.js` | Navigation, the definitions of the terms, and the demonstrations |
| `src/style.css` | Styles for the stage, the reading layout, and the light and dark themes |
| `src/shell.html` | The frame of the page: the stage, the bar under it, the notes panel, and the contents |
| `src/fit.css` | The text scale of each slide, written by `fit_slides.py` |
| `src/occupation.json` | The recorded results of the occupation-coding model, its suggestions for the eight survey responses of slide `s02`, and the rewritten job title and scores of slide `s15` |
| `src/cost.json` | The requests, tokens, kept examples, training runs, and minutes of the three builds on slide `s36`, taken from the run logs |

The scores, the request, the sentences, and the reports in the report-tagging part come from `../assets/overview.json`, the recorded data behind the figures of the overview page. The scores of the occupation-coding model in `src/occupation.json` were measured on 1,000 job titles from the job board hahu.jobs, as means over three training runs. The titles come from the dataset [`tabiya/occupation_titles_esco`](https://huggingface.co/datasets/tabiya/occupation_titles_esco) (CC BY 4.0). The code of each title is the ESCO occupation assigned by the job board, mapped to its ISCO-08 unit group, and no trained coder reviewed these codes. Slide `s15` shows one title from the dataset with three survey-style answers that a large model wrote from it. Its scores are means over three training runs on 1,000 survey-style answers, which a third large model wrote from the test titles. Coded survey responses were not available for that test. The dollar amounts on slide `s36` multiply the logged tokens by an example price selected on the slide. The runs used open models on one server, and no price per token was paid. The survey responses on slide `s02` and on the slides of the nine steps, and the codes shown for them, were written for the session as illustrations, and the slides say so. The five suggestions for each response on slide `s02` are the recorded output of the occupation-coding model. Group names follow ISCO-08 (International Labour Organization), and the group descriptions on the slides are shortened paraphrases. The tasks and practice cases on slide `s03` are illustrations written for the session. The first row holds the applications named in the description of the session. The match on the semantic-search card is the first result recorded in the [repository README](../../../README.md) for the model `avsolatorio/GIST-all-MiniLM-L6-v2`.

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

`check_deck.py` renders every slide at 1280 × 720 in the light and dark themes and at 1920 × 1080, and reports content outside the stage. It then checks the reading layout at a width of 400 pixels and drives the keys, the buttons and the strip of the bar, the contents, the notes, the links, and the demonstrations. `--shots DIR` saves a screenshot of every slide.

`build_deck.py --fragment PATH` also writes the page without its `<html>` and `<head>` elements, for a host that supplies them.
