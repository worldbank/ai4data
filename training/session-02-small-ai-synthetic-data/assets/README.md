# Session 02 assets

This folder holds the recorded data behind the overview page, the slides, and the two notebooks. The notebooks read the files of `occupation/` and `topics/` from the repository, so they run without a key for a large model.

## Overview page

| File | Content | Source |
|---|---|---|
| `overview.json` | The numbers, requests, sentences, and reports behind the figures of the overview page and the report-tagging slides | Recorded runs of the report-tagging task |

## Occupation coding (notebook 01)

| File | Content | Source |
|---|---|---|
| `occupation/titles.jsonl.gz` | 10,134 job titles, each with an ISCO-08 unit group code | [tabiya/occupation_titles_esco](https://huggingface.co/datasets/tabiya/occupation_titles_esco), CC BY 4.0 |
| `occupation/test-titles.jsonl.gz` | 1,000 further job titles with codes, kept for the test | Same dataset |
| `occupation/answers.jsonl.gz` | 30,299 survey-style answers written from the 10,134 job titles, three per title. Each row holds the code of the title, the style of the answer, the code chosen by a second model, and the result of the check (`kept`) | Written by Qwen3.8-27B and checked by Gemma 4 31B |
| `occupation/test-answers.jsonl.gz` | The 1,000 test titles, each rewritten as one survey-style answer | Written by Gemini 3.8 Flash |
| `occupation/definition-titles.jsonl.gz` | 15,865 job titles for all 436 unit groups | Written by a large model from the ISCO-08 group definitions and checked by a second model |
| `occupation/illustrative-answers.json` | 29 short answers in the style of a household survey, each with an expected code | Composed for the slides of this session |
| `occupation/results.json` | The request sent to the large model, the token counts of the rewriting step, and the recorded scores of models trained on five combinations of the sets | Recorded runs |

The job titles come from the job board hahu.jobs in Ethiopia and were published by the Tabiya project. The job board assigned each title an occupation of the European classification ESCO. The first four digits of an ESCO occupation code are the ISCO-08 unit group, and the files keep those four digits. No trained coder reviewed the codes.

The survey-style answers, the titles written from the definitions, and the illustrative answers are synthetic. None of them is an answer from a survey, and no trained coder reviewed their codes.

The ISCO-08 structure and definitions are not stored in this folder. Notebook 01 downloads the file from the website of the International Labour Organization when it runs.

## Report tagging (notebook 02)

| File | Content | Source |
|---|---|---|
| `topics/topics.csv` | The 86 topics, each with its parent topic where it has one | The World Bank's topic list |
| `topics/sentences.jsonl.gz` | 20,296 sentences, each with one or more topics and the identifier of the request that produced it | Written by Qwen3.8-27B in 1,000 requests |
| `topics/reports-threshold.jsonl.gz` | 488 abstracts of Policy Research Working Papers from 2016 to 2020 with their topics, used to choose the threshold. The topics are the subjects of the catalogue record, mapped to the list of 86 | World Bank Documents and Reports database; each row holds the URL of its record |
| `topics/reports-test.jsonl.gz` | 300 further abstracts with their topics, used for the test | Same database |

The sentences are synthetic. The abstracts and their topics are real records.
