# Incident report I-2025-01

**Component:** C05 Release commentary drafts
**Reported by:** Analyst, labour statistics unit, 2025-11-20
**Closed by:** AI governance board, 2025-12-05 (decision D009)

## What happened

The commentary draft for the labour market report 2025 Q2 (EX-REP-2025-02)
stated the youth unemployment rate as 17.5 percent. The published value for
the second quarter of 2025 is 16.8 percent; 17.5 percent is the first-quarter
value. The draft was produced by the hosted model from the output tables of
the release, which contained both quarters.

## How it was found

The number check of the commentary procedure compared every figure in the
draft with the release table and flagged 17.5 as a value that exists in the
table under a different reference period. The analyst corrected the draft
before approval. No wrong figure was published.

## Cause

The prompt gave the model the two-quarter table without naming the reference
period of the commentary.

## Immediate action

The draft prompt now names the reference period, and the model is asked to
quote the period beside every figure.

## Control that prevents a repeat

A reference-period case (a table with two periods, a commentary that must
quote the latest) runs in the commentary evaluation suite and passes.

## Effect on users

None. The incident is recorded because the draft reached the approval step
with an error that the control, and only the control, caught.
