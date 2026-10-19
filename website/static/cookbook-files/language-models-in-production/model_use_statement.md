# Statement of model use: Labour Force Survey 2025 Q2

## Where a model was used

- Occupation coding (GSBPM 5.2): a locally run language model assigned
  ISCO-08 codes to 61 percent of open-text occupation responses at or
  above a confidence of 0.80; the remaining 39 percent were coded by
  staff. A 5 percent sample of the automatically coded responses was
  re-coded by staff.
- Edit failure explanation (GSBPM 5.3): a locally run model drafted
  plain-language explanations of edit failures for editors; editors
  decided every case.
- Release commentary (GSBPM 6.5): a hosted model drafted the commentary
  from the output tables; every number was verified against the tables
  and an analyst edited and approved the text.

## Models and versions

- Coding and explanation: an open-weight model, version recorded in the
  AI component register, run on the organization's servers; no record
  left the organization.
- Commentary: a hosted model, version recorded; only aggregate output
  tables were sent.

## Measured performance

- Coding: 4-digit accuracy 0.93 on the automatically coded share
  (re-coded sample of 1,250 responses), 0.98 at 1 digit; full results
  in the evaluation report for this release.
- Commentary: all 11 numbers verified; two comparative claims sourced
  from the series history.

## Human oversight

- Every automatically coded response above the threshold was accepted by
  the rule set by the methodology unit; the sample re-coding is reported
  above. Every edit decision and the commentary were made or approved by
  staff.

## Effect on the statistics

- Coding: the automated share changes the distribution of coder workload,
  not the classification; the re-coded sample shows no systematic
  difference by major group at the reported precision.
- Automatically coded responses entered the microdata under the rule the
  methodology unit approved, with the re-coded sample as the check on
  it; model proposals for edits and the commentary entered only after a
  person's decision.

## Contact

Methodology unit, methods@stats.example.
