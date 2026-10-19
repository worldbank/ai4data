# Synthesis record: Labour Force Survey 2025 Q2, public synthetic file

## Source data

Labour Force Survey 2025 Q2 (`EX-LFS-2025-Q2`), person file, 18 variables,
41,206 records, licensed-access version before disclosure control. The
synthetic file has the same variables and 41,206 records.

## Method and parameters

Sequential CART synthesis (synthpop 1.8) with the visit sequence region,
sex, age, education, labour force status, occupation, industry, hours,
income; minimum leaf size 20; smoothing on income. Seed 2025. Run on the
organization's server; no data left the organization.

## Utility results

Marginals: total variation distance below 0.02 on every categorical
variable; median income within 1 percent. Associations: Cramér's V within
0.05 of the real file on every pair. Target analyses: the employment rate
by sex and region within 0.4 points; the Gini of income within 0.01. Full
report attached.

## Disclosure risk results

No exact copies. Closest-record distance: synthetic-to-real median 0.21
against 0.19 between real halves. Attribute inference of income band
from region, sex, age, and education: 0.46 against a majority baseline
of 0.41. Reviewed by the disclosure control unit on 2026-09-30.

## Intended uses

Teaching; development and testing of analysis code before an application
for the licensed file; method development; demonstrations.

## Prohibited uses and limits

Not for estimation or publication of statistics; not a substitute for the
real microdata; relationships beyond those listed in the utility report
are not guaranteed; small groups are not reliable.

## Labelling and licence

Every file name, the record, and the first row of the data carry the word
SYNTHETIC. Licence: CC BY 4.0, with the citation "Synthetic version of
EX-LFS-2025-Q2, National Statistical Organization, 2026".

## Contact

Microdata unit, microdata@stats.example.
