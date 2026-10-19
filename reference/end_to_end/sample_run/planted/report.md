# End-to-end run

## 1. Documentation

6 records loaded: 5 series in the indicator schema and 1 survey in the microdata schema.

## 2. Validation at the foundational level

| Record | Type | Decision | Findings |
|---|---|---|---|
| FS_UNDERNOURISH_PCT | indicator | published | none |
| ED_PRIMARY_NER_PCT | indicator | published | none |
| HE_U5_MORT_RATE | indicator | published | none |
| LF_UNEMP_PCT | indicator | withheld | missing: series_description.definition_long |
| PV_HEADCOUNT_NPL_PCT | indicator | published | none |
| EX-LFS-2025-Q2 | microdata | published | none |

## 3. Publication

5 records published to the catalog directory: FS_UNDERNOURISH_PCT, ED_PRIMARY_NER_PCT, HE_U5_MORT_RATE, PV_HEADCOUNT_NPL_PCT, EX-LFS-2025-Q2.

15 files written: records, values, schema.org descriptions, and the index.

NADA: not attempted (no catalog URL given).

## 4. Agent access over MCP

| Tool | Arguments | Result |
|---|---|---|
| search_catalog | {"query": "undernourishment hunger food", "limit": 3} | ok |
| get_record | {"idno": "FS_UNDERNOURISH_PCT"} | ok |
| get_values | {"series": "FS_UNDERNOURISH_PCT", "ref_area": "NAT"} | ok |
| get_values | {"series": "HE_U5_MORT_RATE", "ref_area": "NAT"} | ok |
| get_values | {"series": "ED_PRIMARY_NER_PCT", "ref_area": "NAT"} | ok |
| get_values | {"series": "LF_UNEMP_PCT", "ref_area": "NAT"} | error |

## 5. Answer

Question: What are the latest published figures for the prevalence of undernourishment, the under-five mortality rate, the net enrolment rate in primary education, and the unemployment rate?

Answer (composed by the template with a planted wrong number and a planted untagged number):

```text
In 2024, the prevalence of undernourishment was <claim id="FS_UNDERNOURISH_PCT:NAT:2024" policy="rounded" decimals="1">7.2</claim> % of population, according to the release of 2025-03-14. In 2020, the under-five mortality rate was 41 deaths per 1000 live births, according to the release of 2022-11-30. In 2024, the net enrolment rate in primary education was <claim id="ED_PRIMARY_NER_PCT:NAT:2024" policy="rounded" decimals="1">92.0</claim> % of age group, according to the release of 2025-02-01. No published series for the unemployment rate is in the catalog, so no figure is given for it.
```

## 6. Verification with Proof-Carrying Numbers

| Number | Claim | Policy | Outcome | Detail |
|---|---|---|---|---|
| 7.2 | FS_UNDERNOURISH_PCT:NAT:2024 | rounded | flagged | record holds 7.6 for 2024 |
| 92.0 | ED_PRIMARY_NER_PCT:NAT:2024 | rounded | verified | matches 92.0 for 2024 under rounded |
| 41 | none | none | unmarked | no claim tag, so no mark is drawn |

1 verified, 1 flagged, 1 unmarked.

A verified number matches the record it cites under the stated policy. The check says nothing about whether the record is the right one for the question, whether the statistical method behind it is sound, or whether the sentence around the number interprets it correctly.

result: FAIL
