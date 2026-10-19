---
license: cc-by-4.0
language:
  - en
pretty_name: "LFS occupation coding (example)"
task_categories:
  - text-classification
size_categories:
  - n<1K
tags:
  - tabular
  - official-statistics
  - occupation-coding
---

# Dataset card: LFS occupation coding (example)

## Dataset summary

Six hundred job descriptions from the example organization's labour
force survey, each with the ISCO-08 occupation code assigned by a
trained coder and the respondent's region, urban or rural residence,
sex, age group, education level, and survey weight. The task the
dataset supports is automatic occupation coding: predicting the
four-digit code from the job and industry descriptions. All records are
invented for this guide.

## Motivation

Occupation coding is the most time-consuming step of processing the
labour force survey. A model that proposes a code with a confidence
lets coders concentrate on the hard cases. A shared, documented dataset
lets the organization and its partners train and compare such models on
the same examples.

## Composition

600 records, one per employed person, from 344 households. Columns:
record_id, hhid, region, urban, sex, age_group, educ, job_title,
industry_text, isco_code (the label), isco_major, weight. The
dictionary file describes each column. Records are split into training
(69%), validation (11%), and test (20%) by household, so that no
household appears in two splits.

## Collection process

Job and industry descriptions were recorded by interviewers in face-to-
face interviews during the second quarter of 2025, as the respondent
gave them, in English. Codes were assigned by the organization's coding
unit with the ISCO-08 index; a second coder re-coded a ten percent
sample with 94 percent agreement at four digits.

## Preprocessing

Descriptions were trimmed of leading and trailing spaces and nothing
else; spelling was not corrected, because a coder sees the raw text.
Records with no job description (3 of the original extract) were
removed. Direct identifiers were removed; the household identifier was
replaced by a sequence number.

## Representativeness and known gaps

The dataset is not a sample of the employed population. Compared with
the 2024 population census (persons aged 15 to 64): the Northern Region
holds 14.3 percent of records against 27 percent of the population,
urban residents 66.2 percent against 41 percent, and men 63.5 percent
against 49 percent. Persons aged 15 to 24 are 21.0 percent of records
against 30 percent of the population.
The survey weight (weight) corrects for the sampling design, not for
these differences in the extract. Occupations in agriculture and
fishing are under-represented relative to their share of employment.
The 600 records cover 44 of the 436 ISCO-08 unit groups, none with 30
records or more; a model trained on this file alone will code most
occupations poorly, and the file serves as a worked example of the
format.

## Intended uses

- Training and evaluating automatic occupation coders for the
  organization's own surveys.
- Benchmarking coding models across organizations that use ISCO-08.
- Teaching and demonstrations of text classification on official
  statistics.

## Out-of-scope uses

- Estimating the occupational structure of the population. The extract
  is not representative and carries no design information beyond the
  weight.
- Identifying respondents. No direct identifier remains, and the terms
  of use prohibit re-identification attempts.
- Training models for other countries' classifications without
  re-coding; the codes follow ISCO-08 and the national index of the
  example organization.

## Licence and terms

Creative Commons Attribution 4.0 (CC BY 4.0). Use for training and
evaluating models is permitted; attribution is required in any
publication, model card, or product that used the data, in the form
given under Citation. The organization asks that models trained on the
data cite it in their model card.

## Versions

Version 1.0.0 (October 2026): first release. The test split is frozen;
a new version with a changed test split would change the version's
first number.

## Citation

National Statistics Office (example). 2026. LFS occupation coding
(example), version 1.0.0. https://stats.example/datasets/lfs-occupation-ml

## Contact

data@stats.example
