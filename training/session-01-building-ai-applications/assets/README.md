# Worked-example assets

Everything the six notebooks share.

| File | Content | Source |
|---|---|---|
| `indicators.csv` | 150 observations: five indicators, Kenya and Rwanda, 2010 to 2024 | World Bank Indicators API v2, retrieved 13 October 2026, CC BY 4.0 |
| `indicator_metadata.json` | Name, definition, unit, source organization, and topics for each indicator | Same API |
| `documents/*.md` | Eight short documents with a YAML header: four country briefs written for the workshop from the indicator data, one indicator definition note, one outdated regional summary from 2016, two distractors | Written for the workshop; each header names its source and date |
| `questions.json` | Eight evaluation questions with the expected evidence, expected numbers, and the expected behaviour | Written for the workshop |
| `recorded_responses.json` | Model outputs recorded for each scenario the notebooks run, used when no API key is configured | Recorded; regenerate with `notebooks/record_responses.py` when a key is available |

## Indicators

| Id | Name |
|---|---|
| IT.NET.USER.ZS | Individuals using the Internet (% of population) |
| IT.CEL.SETS.P2 | Mobile cellular subscriptions (per 100 people) |
| NY.GDP.PCAP.CD | GDP per capita (current US$) |
| NY.GDP.MKTP.KD.ZG | GDP growth (annual %) |
| SP.POP.TOTL | Population, total |

## Documents and what each is for

| Id | Title | Role in the session |
|---|---|---|
| kenya-digital | Kenya: internet use and mobile subscriptions, 2010 to 2024 | Relevant evidence |
| rwanda-digital | Rwanda: internet use and mobile subscriptions, 2010 to 2024 | Relevant evidence |
| kenya-economy | Kenya: GDP per capita and growth, 2010 to 2024 | Relevant evidence |
| rwanda-economy | Rwanda: GDP per capita and growth, 2010 to 2024 | Relevant evidence |
| internet-indicator | How the internet-users indicator is defined and measured | Definition; answers the metadata question |
| regional-2016 | Regional digital summary, 2016 | Outdated figures; shows why dates in metadata matter |
| coffee | Coffee exports in East Africa | Irrelevant to the question; shows a retrieval failure |
| mobile-money | Mobile money in Kenya and Rwanda | Related topic without the requested indicators; a plausible distractor |

The figures in the briefs are rounded from `indicators.csv`. The 2016 summary reproduces the values as they stood for 2015 and states expectations that later data did not confirm. No document is an official publication.

## Planted problems

| Problem | Where it appears | Capability that addresses it |
|---|---|---|
| Ambiguous question ("How connected is Kenya?") | `questions.json`, q3 | Instructions that ask for clarification or state the indicator chosen |
| Irrelevant retrieval result (coffee note) | Notebook 03 | Metadata filters and ranking |
| Outdated evidence (2016 summary) | Notebook 03 | Dates in chunk metadata; recency in ranking |
| Request for data that does not exist (Kenya, 2025) | Notebook 04 | A tool that returns "no observation" and an answer that says so |
| Unsupported numerical claim | Notebook 06 | Claim tags and the verifier |
