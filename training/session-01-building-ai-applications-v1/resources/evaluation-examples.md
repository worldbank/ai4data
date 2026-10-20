# Evaluation examples

What notebook 06 measures, with the eight questions of the session and the two configurations it compares.

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
| q8 | outdated evidence | What is the latest available figure for internet use in Kenya? | 35.0 percent for 2024; quoting 2015 is a failure |

## The columns

| Column | What it measures | How |
|---|---|---|
| retrieval | Did the top-4 passages include a passage from an expected document | Document ids of the hits against `expected_documents` |
| numbers | How many expected values appear in the answer | Each expected value searched for in the answer within rounding |
| cited | Does the answer cite a retrieved document or name the source | Document id of a hit in the answer, or the source name |
| verify | Verified, flagged, and unmarked counts | The verifier over claim tags and the observations the tools returned |

## The two configurations in the recorded run

Configuration A, retrieval only. Configuration B, tools and claims.

| Id | A retrieval | A numbers | B retrieval | B numbers | B verify |
|---|---|---|---|---|---|
| q1 | hit | 4/4 | hit | 4/4 | 4 verified |
| q2 | hit | 1/1 | hit | 1/1 | 1 verified |
| q3 | hit | 0/0 | hit | 0/0 | no data used |
| q4 | hit | 0/0 | hit | 0/0 | 1 verified |
| q5 | hit | 0/0 | hit | 0/0 | no data used |
| q6 | hit | 2/2 | hit | 2/2 | 2 verified |
| q7 | hit | 0/0 | hit | 0/0 | no data used |
| q8 | hit | 0/1 | hit | 1/1 | 1 verified |

Reading it:

- q8 separates the configurations. A answers from the 2016 summary when it ranks first; B fetches the series and reports 2024.
- q3, q5, and q7 have no expected numbers, so the numbers column is 0/0 for both. The behaviour, asking, declining, defining, has to be read in the answer. A suite can automate it with a second model as a judge, with its own evaluation.
- The verify column is empty for A because no number came from a tool. Verification needs claims, and claims need data tools.

## What a failing row looks like

A row with `numbers 0/1` and `retrieval hit` is a generation problem: the evidence was there and the answer did not use it. A row with `retrieval miss` is a retrieval problem: fix chunking, metadata, ranking, or the query before touching the instructions. A row with a `flagged` count is a model that changed a number; the instructions and the model are the suspects. A row with `unmarked` numbers is a model that produced figures without claims; the instructions are the first suspect, the tools the second.

## Growing the suite

Every failure found in use becomes a question with its expected behaviour. The suite is rerun on every change to the instructions, the tools, the index, or the model, and a change that makes a row worse is examined before it ships. The program's [evaluation cookbook](https://worldbank.github.io/ai4data/cookbook/evaluation-suites/) covers suites at scale.
