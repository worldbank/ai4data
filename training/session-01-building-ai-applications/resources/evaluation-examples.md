# Testing on fixed questions

The session's eight questions, each with the evidence a correct answer uses and the behaviour it shows. Section 6 of notebook 03 runs the retrieval check on them: for each question, whether the expected documents and the needed numbers are among the retrieved passages. A test of the whole application against the same questions, with the columns described below, is the subject of the program's [evaluation cookbook](https://worldbank.github.io/ai4data/cookbook/evaluation-suites/).

## The questions

| Id | Category | Question | Expected behaviour |
|---|---|---|---|
| q1 | main | How have internet access and economic development evolved in Rwanda and Kenya? Compare the recent available data, summarize the main patterns, and cite the sources. | Latest year for each indicator and country, names the indicators and the source, compares levels and trends |
| q2 | factual | What share of Rwanda's population used the internet in 2023? | 28.8 percent, with the indicator and the source |
| q3 | ambiguous | How connected is Kenya? | Asks which measure is meant, or states the indicator chosen |
| q4 | missing data | What share of Kenya's population used the internet in 2025? | States that 2025 is unavailable and gives 2024 at 35.0 percent |
| q5 | out of scope | How have Kenya's coffee export earnings changed in recent years? | States that the material has no figures; gives no number |
| q6 | factual | Which of the two countries had faster GDP growth in 2023? | Rwanda, 8.6 against 5.7 percent |
| q7 | definition | How is the internet-users indicator defined, and what should be kept in mind when comparing years? | The three-month, any-device definition and at least one caveat |
| q8 | outdated evidence | What are the latest available figures for internet use in Kenya and Rwanda? | 35.0 and 31.7 percent for 2024; quoting the 2015 values of the 2016 summary is a failure |

## What a test of the whole application reports

One row per question, with four columns.

| Column | What it measures | How |
|---|---|---|
| retrieval | Did the top-4 passages include a passage from an expected document | Document ids of the hits against `expected_documents` |
| numbers | How many expected values appear in the answer | Each expected value searched for in the answer within rounding |
| cited | Does the answer cite a retrieved document or name the source | Document id of a hit in the answer, or the source name |
| verify | Verified, flagged, and unmarked counts | The verifier over claim tags and the observations the tools returned |

## What a failing row looks like

A row with `numbers 0/1` and `retrieval hit` is a generation problem: the evidence was there and the answer did not use it. A row with `retrieval miss` is a retrieval problem: fix chunking, metadata, ranking, or the query before touching the instructions. A row with a `flagged` count is a model that changed a number; the instructions and the model are the suspects. A row with `unmarked` numbers is a model that produced figures without claims; the instructions are the first suspect, the tools the second.

## Growing the set of questions

Every failure found in use becomes a question with its expected behaviour. The test is rerun on every change to the instructions, the tools, the index, or the model, and a change that makes a row worse is examined before it ships. The evaluation cookbook covers tests at scale, including a second model as a judge for behaviours a rule cannot check.
