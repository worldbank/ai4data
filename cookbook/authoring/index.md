---
id: index
title: Authoring guide for cookbooks
sidebar_label: Authoring guide
description: The standard structure, writing rules, and tooling for a cookbook on this site, with the scaffold and checker that enforce them.
hide_table_of_contents: false
---

# Authoring guide for cookbooks

A cookbook is a practical guide on one problem domain for one audience. It
is organized around the questions that audience asks, and each chapter
answers one question with recipes: tasks with a defined result, the skills
and time they need, and the code or template to carry them out. This guide
sets the standard that every cookbook on the site follows, so that readers
can move between cookbooks without relearning the structure and so that new
cookbooks can be produced quickly.

Two tools support the standard. `scripts/docs/new_cookbook.py` creates a
cookbook from the template in `cookbook/_template/` and registers it on the
site. `scripts/docs/check_cookbooks.py` verifies that every cookbook has the
required structure and follows the writing rules; it runs in continuous
integration and can be run locally.
`scripts/docs/renumber_cookbook.py` renumbers the chapters of a cookbook
after a chapter is added, removed, or moved (titles, sidebar positions,
recipe numbers, and in-page links) and lists the plain-text mentions of
chapter numbers that need a manual check.

## Definition and scope

| Element | Standard |
|---|---|
| Subject | One problem domain, for example AI-ready data dissemination, microdata documentation, or extraction of data from documents. |
| Audience | One primary audience, named on the overview page and on the landing card, for example national statistical organizations or data curators. |
| Organizing principle | The questions the audience asks about its own situation. Each chapter answers one question. |
| Length | One chapter per question the audience asks; the count follows from the questions. Most domains need six to ten. A domain that needs more is two cookbooks; a question that needs fewer than two recipes is part of another chapter. |
| Relation to documentation | A cookbook states what to do and why, and links to the documentation for how each method or tool works. It does not duplicate the documentation. |
| Register | A guidance document: formal, specific, and descriptive. See [Writing rules](#writing-rules). |

## Required parts

### Overview page (`index.mdx`)

1. Title and subtitle. The title names the guide ("Practical Guide to ..."); the subtitle names the audience ("A cookbook for ...").
2. Version and date, and a note stating that the guide is a working draft where that is the case.
3. An introduction of two or three paragraphs: the problem, the organizing questions, the standards the recipes build on.
4. "Prerequisites": the data, systems, people, and skills the recipes assume, at the foundational level and at the higher levels. "First steps": the three or four first steps for a team with limited time, with chapter links and time estimates, and one sentence on what comes after.
5. A "Chapters and questions" table: chapter, question, practical topics.
6. The maturity levels, using the shared vocabulary below.
7. The running example: one small, fictional dataset or case that every chapter uses, with its files.
8. "Scope of the examples": a statement that the examples are small on purpose and that each recipe states its limits.
9. A files table listing every downloadable file with the chapter that uses it.
10. Related resources, a suggested citation, and a version history.

### Chapters

Each chapter is one `.mdx` file with this structure, in this order. The
checker verifies the headings.

| Section | Content |
|---|---|
| Front matter | `id`, `title` ("N. Noun phrase"), `sidebar_label` ("N. Short"), `sidebar_position`, `description`. |
| H1 and question | The title, then a line `**Question:** ...` with the chapter's organizing question. |
| Rationale | Why the question matters, in two or three paragraphs. |
| Target state | The condition the chapter aims at, as a bulleted list. |
| Maturity levels | A table with one row per level and the steps that reach it. |
| Running example | An `:::info[Running example]` admonition that says what the chapter does to the example. |
| Recipes | Two to four recipes (see below). |
| Implementation options | Open standards and patterns, with alternatives. |
| World Bank examples | Program workstreams, tools, and reference implementations that apply, with links. Optional where none exist. |
| Resources | External tools, guides, standards documents, and example implementations needed to carry out the recipes, one line each with a link. Every item says what it is used for in the chapter. |
| Common mistakes | Three to five short items. |
| Verification | How to test that the chapter's steps worked. |
| Checklist | A `<Checklist id="..." items={[...]} />` component. |

### Recipes

A recipe uses the `Recipe` component:

```mdx
<Recipe title="N.M Imperative title naming the result" level="Foundational | AI-ready | AI-native" skills="Who can do this" time="How long">

**Result:** what the team has when the recipe is done, in one or two full sentences.

One or two paragraphs that explain the thing the recipe produces, the terms it uses, and why the organization wants it, before any instruction.

1. Step, with the reason it is done this way.
2. Step, with code or a template where useful.
3. How to check that it worked.

**Worked example.** The recipe carried out on the running example, with the actual input, the action, and the output.

**What this does not do.** Where the example stops and what the full version adds.

</Recipe>
```

- The title is an imperative that names the result ("Publish code lists and crosswalks"). Procedures are the one place where the imperative is the standard form.
- `level` uses the shared vocabulary. `skills` names a role. `time` is an indicative estimate for a team with the stated skills; the page labels it as indicative.
- Code is embedded from a file in `website/static/cookbook-files/<cookbook-id>/` with a code-import block, so the page and the download are one file:

  ````mdx
  ```python file=../../website/static/cookbook-files/<cookbook-id>/script.py
  ```
  ````

- Every recipe ends with a **What this does not do** paragraph. Simplified examples stay useful only when they say where they stop.
- A recipe is written for a reader who has not read the author's notes. It explains before it instructs: one or two paragraphs before the steps say what the recipe produces, what the terms mean, and why the organization wants the result. Every file or code block is introduced by a sentence that says what it is, and its columns or fields are explained in a short table or list beside it. A term of art (embedding, structured output, token, span) is explained at its first use in the chapter, in a full sentence where a parenthesis would be cramped. Each step says what to do and, where it is not obvious, why. The reader test: a statistician with no background in language models can say, after reading the recipe once, what to do first, why, and what the file in front of them means.
- Where a recipe has a concrete procedure, it has a **Worked example** paragraph: the recipe carried out on the running example, with the real input, the action, and the output. For a script, the example is the command and its output. For a template or a policy, it is the template filled in for the example organization. For a design decision, it is the decision the example organization took and the reason. A worked example uses the values of the running example's facts file and never invents a new figure.

### Register

The cookbooks are read by statisticians, curators, and developers who are
new to the subject. The prose is that of a colleague explaining the work in
full sentences, in a guidance-document register. The following rules hold
throughout a cookbook, and they override any habit of compressing text:

- Every sentence has a subject and a verb. A "Result" line, a running-example admonition, a description in the front matter, and a bullet in "Verification" or "Common mistakes" are full sentences, never a stack of nouns ("A small set of read-only tools with typed inputs, structured outputs, and a check").
- Explanation comes before instruction. A step that says "keep the set small" is preceded by the sentence that says why a small set matters to the model.
- No aphorisms. A short pronouncement ("A manifest is a plan.", "Twenty words is a floor.") followed by a list reads as a slogan. Say the thing in a sentence that carries its reason.
- No semicolon-balanced contrasts as a habit ("four tools produce correct calls; twenty produce wrong ones"). State the point, then the consequence, in two sentences.
- A list of more than four items becomes a bulleted list; a sentence cannot hold it. Two parentheticals in a row ("([chapter 2]) (one day)") become a sentence or a table.
- A bullet is a sentence or two, never a label with a colon and a fragment ("Manifest check: no errors; warnings decided.").
- Length follows the explanation needed. A recipe is as long as it takes to be understood on one reading; short is not a goal.


### Standards page (`standards.md`)

A table per area: the standard, its use, where the World Bank uses it, and
where the cookbook applies it. Prefer the standards the World Bank has
adopted; link each to its source. The checker warns when the page is
missing.

### Glossary and contributing pages

A glossary of the terms the cookbook introduces, and a contributing page
with the recipe template and the criteria for a recipe. Both can start from
the template files.

### Optional parts

- A self-assessment page using the `SelfAssessment` component with a
  question per chapter (`questions` and `base` props).
- An "Application by data type" page when the domain spans data types.
- Profiles, templates, and example records as downloadable files.

## Shared vocabulary

Maturity levels are the same in every cookbook, so that a reader who knows
one cookbook knows them all:

| Level | Meaning |
|---|---|
| **Foundational** | The practice exists in a form a person can use and a crawler can read. |
| **AI-ready** | The practice is structured, identified, and accessible through documented interfaces, so that software, including AI systems, can use it without manual steps. |
| **AI-native** | The practice offers AI-oriented interfaces or methods, with provenance and verification built in. |

Use "organization" for the audience institution, as the AI-readiness
assessment framework does, and name the specific kind on first use
("national statistical organization").

Glossary terms live in one file, `cookbook/_shared/glossary.md`, as
`**Term.** definition` entries. Each cookbook's `glossary.md` is generated
from it by `scripts/docs/build_glossaries.py`, which keeps the terms whose
name appears in that cookbook's chapters. Edit the shared file and run the
script; do not edit a cookbook's glossary page by hand.

The running example (the example organization, its series, datasets, and
documents, with their values and release dates) is defined once in
`website/static/cookbook-files/_example/facts.json`. Fixtures and chapters
use those identifiers and values. `scripts/docs/check_example_facts.py`
reports an identifier the facts file does not know, a fixture value that
differs from it, and a chapter sentence that quotes a different value.

## Code and files

- Files for download live in `website/static/cookbook-files/<cookbook-id>/`. The page embeds them by code import and links to them with `pathname:///cookbook-files/<cookbook-id>/<file>`.
- Scripts use established libraries for what libraries already do (pandas for tables, scikit-learn and scipy for metrics and statistics, rapidfuzz for fuzzy matching, jsonschema for schema validation, the reference implementation of a standard where one exists) and keep their own code for the logic that is specific to the recipe. The libraries are listed once in `website/static/cookbook-files/requirements.txt`, and a script names any dependency beyond that list in its docstring. Each script has a module docstring with purpose, usage, exit status, and a "what this does not check" statement.
- Scripts pass `ruff check` under the repository configuration and `ruff format`.
- Each cookbook with scripts has a test file `tests/test_cookbook_<id>.py` that runs them on the example files. Fixtures that would otherwise need the network are pinned under `tests/fixtures/`.
- The example data are fictional and say so. Real data are used only where the source is cited and the licence allows it.

## Writing rules

The repository's style rules apply in full. The points that matter most in
a cookbook:

- Titles and headings are noun phrases that name the section ("Target state", "Progression between maturity levels"). They are never a statement plus a comma and a tag, a teaser, or a slogan.
- Recipe titles are imperatives that name the result.
- The register is that of a guidance document: declarative sentences, no second person outside prompt templates and policy templates, no contractions, no intensifiers.
- Claims about improvement or scale are made only where the cookbook links to the measurement or the source.
- Standards are cited with a link to the publishing body. Where the World Bank has adopted a standard, the cookbook says so.
- Before finishing, run the greps in the repository's `CLAUDE.md` or run `scripts/docs/check_cookbooks.py`, which includes them.

## Creating a cookbook

```bash
python scripts/docs/new_cookbook.py \
  --id microdata-documentation \
  --title "Practical Guide to AI-Ready Microdata Documentation" \
  --audience "Data curators in national statistical organizations" \
  --chapters "variables:Variables and value labels:Can AI interpret the variables?" \
             "questionnaire:Questionnaire and concepts:Can AI relate questions to concepts?"
```

The script copies the template into `cookbook/<id>/`, writes one chapter
file per `slug:Title:Question` argument, creates
`website/static/cookbook-files/<id>/`, adds a sidebar in
`website/sidebars-cookbook.js`, and adds a card in
`website/src/content/cookbooks.js`. It then prints the files to edit.

## Visibility

A cookbook can stay in the repository without being public. The file
`website/cookbook-visibility.json` holds one list, `hidden`, of cookbook
ids. For every id on it:

- each page of the cookbook is built as unlisted: it carries a noindex tag
  and a banner, and it is left out of search and the sitemap;
- the landing page, the front-page overview, the reading paths, the
  operational companion, and the modern office page drop their cards and
  links into it;
- the folder `cookbook-files/<id>/` is removed from the build output;
- a direct link still opens the pages, so reviewers can read a draft.

The checker fails when a public cookbook links into a hidden one, so that
a hidden cookbook does not leak through a cross-reference. A new cookbook
is added to the list when it is scaffolded and removed from it when it has
been reviewed; nothing else changes between the two states. The development
server shows every cookbook whatever the list says.

## Checking a cookbook

```bash
python scripts/docs/check_cookbooks.py            # all cookbooks
python scripts/docs/check_cookbooks.py --id microdata-documentation
```

The checker reports errors (missing required parts, unregistered cookbook,
banned constructions in headings, a chapter or recipe number in a link or a
readiness-map label that differs from the file it points to, an index row
whose title differs from the chapter's) and warnings (missing optional
parts, recipes without a scope note, second person in prose, a bare
"chapter N" that the cookbook does not have). Errors fail the
continuous-integration job for changes under `cookbook/`.

Three more scripts keep the parts that repeat information in step:

```bash
python scripts/docs/sync_cookbook_indexes.py          # overview tables and landing cards
python scripts/docs/build_glossaries.py               # glossary pages from the shared glossary
python scripts/docs/check_example_facts.py            # fixtures and chapters against facts.json
python scripts/docs/refresh_cookbook_outputs.py <id>  # quoted outputs from the scripts
```

`sync_cookbook_indexes.py` rewrites the chapter and question columns of each
index page's "Chapters and questions" table from the chapter files, rewrites
the chapter count of the landing card in `website/src/content/cookbooks.js`,
and reports a file in `cookbook-files/<id>/` that the "Files" table does not
list, a table row whose file is missing, and a chapter without a
self-assessment question. The practical-topics column and the "Used in" text
of a new file are written by hand. The continuous-integration job runs all
four with `--check` and fails when anything differs.

## Review before publication

- Every standard named on the standards page has been checked against its source.
- Every external claim links to its evidence.
- Every script runs on the example files and has a test.
- Every recipe has a scope note.
- Translations and non-English examples have been reviewed by a speaker of the language.
- The overview names the version, the date, and the draft status.
- The landing card's description, audience, and chapter count match the overview.
