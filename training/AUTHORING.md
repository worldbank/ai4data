# Authoring a training session

This guide describes the folder layout, file names, and title formats of a training session, and the page that each file becomes in the Training section of the docs site. It also covers the writing rules for the session's text and the checks to run before a pull request. The site pages are generated from the session folder by `scripts/docs/build_training_site.py` at every site build, so a session that follows this guide appears on the site without changes to the site's code.

## Folder layout

```
training/
  README.md                       series overview; the Training index page
  AUTHORING.md                    this guide (GitHub only)
  session-NN-short-name/
    README.md                     session overview page
    preface.md                    Preface page
    slides/canva.json             Slides page (optional)
    notebooks/NN-short-name.ipynb one page per notebook
    resources/README.md           Participant guide, the first page of Participant resources
    resources/*.md                further pages of Participant resources
    assets/README.md              Worked-example data page
    assets/documents/*.md         the session's documents (optional)
    assets/documents/roles.json   Document collection page
    demo/                         GitHub only
```

`NN` is the two-digit session or notebook number. Everything else in the folder, such as decks in `.pptx` form, data files, and scripts, stays on GitHub, and the site pages link to it there.

## How the files become pages

| Source | Page in the sidebar | Requirements |
|---|---|---|
| `README.md` | the session's category, labelled `NN · <title>`, and its overview page | The first line is `# Session NN · <title>`. The sections `Roles` and `Definition of ready` stay on GitHub. |
| `preface.md` | Preface | A `# Title` line. |
| `slides/canva.json` | Slides, with the deck in Canva's embedded viewer | See [Slides](#slides). |
| `notebooks/NN-*.ipynb` | Notebooks, one page each, in file-name order | See [Notebooks](#notebooks). |
| `resources/README.md` | Participant guide | A `# Title` line. |
| `resources/<name>.md` | a page at `resources/<name>` | A `# Title` line. Pages follow the order in which `resources/README.md` first links to them; files without a link follow in alphabetical order. |
| `assets/documents/roles.json` | Document collection | See [Document collection](#document-collection). |
| `assets/README.md` | Worked-example data | A `# Title` line. |

The session also needs a row in the table of `training/README.md` that links to its README. A planned session has a row with its title and the format `Planned`, without a link.

Links between these files are written as relative links to the source files, for example `[glossary](resources/glossary.md)` or `[notebook 03](notebooks/03-retrieval-augmented-generation.ipynb)`. They work on GitHub, and the generator rewrites them to the corresponding site pages. Relative links to any other file point to that file on GitHub.

## Notebooks

- **File name and title.** A notebook is named `NN-short-name.ipynb`, and its first cell begins with `# NN · <title>`. The page URL is formed from the title, so a new title after the session is published needs a redirect from the old URL (see [Addresses of published pages](#addresses-of-published-pages)).
- **Sections.** Each numbered section starts a Markdown cell whose metadata `id` is `section-N` and whose first line is `## N. <section title>`. The site gives that heading the anchor `#section-N`.
- **Links to sections.** A contents list links to a section as `[N. <title>](#scrollTo=section-N)`, which Colab follows inside the notebook. A link to another notebook of the session uses its Colab address on the `main` branch, `https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/<session folder>/notebooks/<file>`, with an optional `#scrollTo=section-N`. On the site, both kinds of link lead to the corresponding page and heading.
- **Outputs.** Notebooks are committed with their outputs, and the site shows those outputs. Without an API key, every notebook runs in a fresh Colab environment and replays recorded model answers.
- **Output types.** Text and stream outputs appear as text blocks, Markdown outputs as Markdown, and images as image files. HTML outputs keep their markup. Scripts inside HTML outputs are removed, and an interactive output needs a client module in `website/src/clientModules/` for its behaviour on the site; `pcnDetail.js` is the example. `<style>` blocks are moved to a generated stylesheet. Widget outputs are left out.

## Slides

The Slides page shows a deck kept in Canva. `slides/canva.json` holds the page title and the public view link of the deck:

```json
{
  "title": "Slides",
  "view_url": "https://www.canva.com/design/<id>/<token>/view?utm_content=...&utlId=..."
}
```

The link is the one Canva gives under **Share → Public view link**, or the address that a `canva.link` short link redirects to. Its query string is kept, because Canva attributes views by it. Anyone with the link can view the deck, so the deck holds only material that may be public. The names of colleagues appear only with their agreement.

## Document collection

When a session searches a collection of documents, the documents are Markdown files in `assets/documents/`, each with front matter. The page shows the fields `id`, `type`, `country`, `date`, `source`, and `note`; `id`, `title`, and `date` are required. The application cites a chunk as `[id · section · year]`, and the page labels each chunk the same way.

`assets/documents/roles.json` groups the documents by their role in the session and holds the text of the page:

```json
{
  "intro": ["First paragraph of the page.", "Second paragraph."],
  "sources": "A sentence on where the figures in the documents come from.",
  "groups": [
    {
      "key": "evidence",
      "label": "Relevant evidence",
      "description": "What the documents of this group are and how the session uses them.",
      "documents": [{"id": "kenya-digital", "note": "The role of this document in one sentence."}]
    }
  ]
}
```

The group keys `evidence`, `definition`, `outdated`, and `distractor` have their own colours on the page; other keys are shown without a colour. Every document in the folder is listed in one group. The intro and sources text is Markdown, and relative links in it are resolved from `assets/documents/`.

## Writing

The writing rules of the [cookbook authoring guide](../cookbook/authoring/index.md) apply to every page of a session: descriptive titles, plain statements, no slogans, and no antithesis as a habit. Three further rules apply to training material.

- **Audience.** A session is written for a general audience of researchers, analysts, people who maintain data and documents, reviewers, and developers. The preface and the overview describe the session from its content.
- **Preface.** The preface explains the problem that the session addresses and the readers for whom it matters. It then sets out the ideas that the notebooks develop, how readers at different technical levels can use the session, and the contribution of each notebook in one or two sentences. Each paragraph explains; none of them announces the structure of the page.
- **Notebook text.** Markdown cells are explanations in connected paragraphs. Each cell says what the next code does, what its result shows, and what that result means for the application.

The materials contain no names of colleagues, no API keys or other credentials, no confidential documents, and no delivery dates.

## Addresses of published pages

Once a session is on the site, its page addresses are listed in `website/published-routes.txt`, and the site build fails when one of them stops resolving. A page moves when its source file is renamed or moved, when a notebook title changes, or when a resource file is renamed. In each case, an entry in `website/redirects.json` maps the old address to the new one. The process is described in `website/README.md`, under "Changing or removing a page".

## Checks before a pull request

The site build runs the generator, which prints a warning for each convention of this guide that a session breaks. The strict run turns the warnings into an error:

```bash
python3 scripts/docs/build_training_site.py --strict
cd website && npm run build
npx docusaurus serve --port 3400 --no-open
```

The served site is at `http://localhost:3400/ai4data/training/`. Before the pull request:

1. The strict run reports no warnings, and the site build succeeds.
2. `npm run routes:update` has added the new pages to `website/published-routes.txt`, and every moved page has an entry in `website/redirects.json`.
3. Every page of the session opens from the sidebar, and the section links of each notebook lead to their headings.
4. The notebooks run in a fresh Colab environment without an API key.
5. The session has a row in the table of `training/README.md`.
6. The text of the touched files has been checked against the writing rules above.
