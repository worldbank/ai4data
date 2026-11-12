# Slides

- `deck.html`: the deck in one HTML file, with 42 slides in the main part and 8 in the annex. It opens in a browser from a local copy of the repository and needs no server. GitHub shows the file as source text.
- `outline.md`: the headline and the speaker notes of every slide. This is the file to review and comment on.
- `build_deck.py`: builds both files from the sources in `src/` and from the recorded data in `../assets/overview.json`, `../assets/occupation/`, `src/occupation.json`, `src/learn.json`, and `src/cost.json`.
- `src/`: the slide sources, the styles, and the script of the demonstrations.
- `fit_slides.py` and `check_deck.py`: the scripts that size the slides and test the deck in a browser.

The deck covers the concepts of the [overview of the session](../README.md). Its longest example is occupation coding, and the report-tagging model of the overview has two slides in the annex. Each slide has a one-sentence headline, one figure or demonstration, and speaker notes.

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

The bar under the stage has four buttons: `«` and `»` move by one slide, and `←` and `→` move by one reveal. Next to them are a box for a slide number and a strip with one mark for each slide. A mark shows the number and the headline of its slide under the pointer, and a click on the mark opens the slide. The taller marks are the first slides of the six parts, the closing slide, and the first slide of the annex. The counter at the top right gives the position in the main part, and on an annex slide the position in the annex.

Each part opens with a divider slide. The divider names the part, describes its content in one line, and lists the six parts with the current one marked. The part names at the top of the other slides link to these dividers. Words with a dotted underline show their definition when selected.

Each slide has an id, and the address of a slide is the file address followed by `#` and the id, for example `deck.html#s22`. The ids of the dividers are `p1` to `p7` and `pa` for the annex, and the ids of the other slides are `s01` to `s43`. A slide keeps its id when slides are added before it or when it moves to another part or to the annex, so the id differs from the slide number in the bar.

In a window narrower than 820 pixels, the deck is laid out as one scrolling page with the notes under each slide.

The fonts (Inter and JetBrains Mono) load from Google Fonts. Without an internet connection the browser uses the system font, and long lines may wrap differently.

## Slide plan

| Slides | Part | Content |
|---|---|---|
| 1 | | Title |
| 2 to 6 | Tasks | A survey response to code, eleven tasks from data work and services, three ways to get a task done, a diagram of the process with the parts of the session |
| 7 to 10 | Small model | Large language models, small AI, the cost of building a small model |
| 11 to 17 | Synthetic data | Labelled examples, why they are missing, synthetic data, the office comparison, distillation, labelled job titles rewritten as survey answers |
| 18 to 28 | Nine steps | The process, then one slide for each step with an illustrative example from occupation coding. These slides show no scores of the models |
| 29 to 35 | Occupation coding | The three sets of training texts, the recorded training run of the small model, five suggested codes for eight responses, the scores by training set, the comparison with three large models, speed. The line above each headline names the step of the process |
| 36 to 42 | Small or large | Criteria, a six-task exercise for the room, checks before relying on a small model, the requirements of a first attempt, summary, sources |
| 43 to 50 | Annex | The memory a model needs (`s08`), the real cases of a rare group (`s12`), the request that rewrites a job title as survey answers (`s27`), the check by a second model (`s28`), the measured cost of each build (`s35`), a second small model that tags reports with topics (`s36`) and its scores on 300 real reports (`s37`) |

The first slide of each part is its divider.

The annex holds slides with more detail. They are outside the main part so that the main part fits a session of 60 minutes. The main part ends at the sources slide, and the annex follows it. The source line of slides `s07`, `s11`, `s26`, `s34`, and `s38` links to the related annex slides, and each annex slide has a button that returns to its slide in the main part (the `data-back` attribute of its `<section>`). To move a slide to the annex, move its `<section>` to `src/slides-annex.html`, set `data-sec` to the number of the annex (the position of its name in `SECS`) and `data-back` to the id of its slide in the main part, and start its kicker with "Annex".

## Sources

| File | Content |
|---|---|
| `src/slides-01-15.html`, `src/slides-26-43.html` | The slides with the ids `s01` to `s15` and `s26` to `s43`, one `<section>` for each slide, with the speaker notes in `<aside class="notes">`. Slide `s30` (distillation) is in the first file, after `s14` |
| `src/slides-annex.html` | The slides of the annex: `s08`, `s12`, `s27`, `s28`, and `s35`. The divider `pa` is generated from `ANNEX` in `build_deck.py` and lists these slides |
| `STEPS` in `build_deck.py` | The slides `s16` to `s25`: the process slide and, for each step, the office comparison, the model version, and the notes |
| `DIVIDERS` in `build_deck.py` | The dividers `p1` to `p7`: for each part, the line under its name and the notes. The names of the parts come from `SECS` in `src/app.js`, where the last name is the annex |
| `src/app.js` | Navigation, the definitions of the terms, and the demonstrations |
| `src/style.css` | Styles for the stage, the reading layout, and the light and dark themes |
| `src/shell.html` | The frame of the page: the stage, the bar under it, the notes panel, and the contents |
| `src/fit.css` | The text scale of each slide, written by `fit_slides.py` |
| `src/occupation.json` | The recorded results of the occupation-coding models: the suggestions for the eight survey responses of slide `s02`, the job title and answers of slide `s15`, and the counts, examples, scores, and times of slides `s26` to `s34`, including the annex slides `s27` and `s28` |
| `src/learn.json` | The training run of slide `s29`. The file holds the mean scores of three runs of the occupation model on the 1,000 survey-style test answers: before training, at twelve points of training, and for the saved model. It also holds the first code of one run for six test answers |
| `src/cost.json` | The requests, tokens, kept examples, training runs, and minutes of the three builds on slide `s35`, taken from the run logs |

### Occupation coding

The job titles come from the dataset [`tabiya/occupation_titles_esco`](https://huggingface.co/datasets/tabiya/occupation_titles_esco) (CC BY 4.0), which holds titles from the job board hahu.jobs. The code of each title is the ESCO occupation assigned by the job board, mapped to its ISCO-08 unit group. No trained coder reviewed these codes. Group names follow ISCO-08 (International Labour Organization), and the group descriptions on the slides are shortened paraphrases.

The deck shows two occupation models. The first was trained on job titles that a large model wrote from the group definitions. Slide `s02` uses this model. The second was trained on three sets: the titles of the first model, the job-board titles, and survey-style answers that a large model wrote from the job-board titles. The slides `s26` to `s34` describe this model. Qwen3.8-27B wrote the answers, and Gemma 4 31B checked them.

Each score of slides `s32` and `s33` is the mean of three training runs. The two test sets hold 1,000 texts each, from job titles set aside before training. One holds the job titles. The other holds survey-style answers that Gemini 3.8 Flash wrote from those titles, because coded survey responses were not available. Three limits apply to the comparison on slide `s33`, and its speaker notes state them. Gemini 3.8 Flash is scored on answers that it wrote. The request for the test answers resembles the request for the training answers. The real job titles of the test come from the same job board as the training titles.

The times on slide `s34` are medians for one answer. The large models were timed over 50 single requests each, and the times of the hosted services include the network and the load of the provider.

The survey responses on slide `s02` and on the slides of the nine steps, and the codes shown for them, were written for the session as illustrations, and the slides say so. The slides of the nine steps show no output of a model, and the model codes in the test table of step 7 belong to the illustration. The five suggestions for each response on slide `s02` are the recorded output of the first model. Slide `s31` shows six responses from the survey-style test and two of the illustrations, with the recorded suggestions of the second model.

Slide `s29` shows the training of the second model. The three training runs behind slides `s32` and `s33` were repeated with the same texts, settings, and random starts. In each run the model was scored on the test answers before training, at twelve points up to six passes over the training texts, and after training. The saved models of the repeated runs give the scores of the first runs. The scores on the slide are means of the three runs. The six test answers show the run with the suggestions of slide `s31`. One pass reads 35,515 of the 44,058 training texts. The other texts are held out, and training keeps the pass with the lowest loss on held-out texts.

### Other slides

The report, the topics, and the scores on the report-tagging slides `s36` and `s37` come from `../assets/overview.json`, the recorded data behind the figures of the overview page. The dollar amounts on slide `s35` multiply the logged tokens by an example price selected on the slide. The runs used open models on one server, and no price per token was paid. The tasks and practice cases on slide `s03` are illustrations written for the session. The first row holds the applications named in the description of the session. The match on the semantic-search card is the first result recorded in the [repository README](../../../README.md) for the model `avsolatorio/GIST-all-MiniLM-L6-v2`.

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
