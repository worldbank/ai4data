# Guidance for assistants using the statistics-example server

## What this server provides

Read-only access to the five published series of the example organization
(unemployment rate, prevalence of undernourishment, under-five mortality,
poverty headcount, primary net enrolment) through four tools:
`search_series`, `get_series_metadata`, `get_observations`, and
`list_code_values`. Values are official statistics as published, with the
release date and status of each observation.

## How to use the tools

1. Call `search_series` first when the series identifier is unknown; use the
   `idno` it returns in every later call.
2. Call `get_series_metadata` before interpreting values: it returns the
   definition, unit, periodicity, breaks, and limitations.
3. Call `get_observations` with codes from `list_code_values`; never guess
   a region or period code.
4. Request the smallest range that answers the question; the tools return
   at most 500 observations per call.

## How to cite

Every data response carries `SOURCE_URL`, `RELEASE`, `license`, and
`citation`. Quote the value with its unit and period, name the organization,
and give the source URL and release date. Example: "Unemployment rate, 2025
Q2: 6.1% of the labour force (National Statistical Organization, released
2025-08-20, https://stats.example/series/LF_UNEMP_PCT)."

## Licence

Data are published under CC BY 4.0. Attribution is required; the citation
above satisfies it.

## Limitations

- Values marked `OBS_STATUS = P` are provisional and may be revised.
- A break (`OBS_STATUS = B`) means the series is not comparable with earlier periods; say so.
- Regional values exist only for the regions in `list_code_values`; the
  server has no district-level data.
- Series start in the year given by `time_period_start`; there are no
  earlier values anywhere in this server.

## When to refuse or decline

- Do not estimate, extrapolate, or interpolate values the tools did not return; say that the value is not available.
- Do not answer questions about individuals or confidential microdata; this server has none.
- If the tools return an error, report it; do not fill the gap from memory.

## Contact

Questions and corrections: data@stats.example. Release calendar:
`stats://release-calendar`.
