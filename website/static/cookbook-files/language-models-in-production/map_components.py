"""Map model-assisted components to GSBPM phases and flag the risky ones.

Reads a component map (CSV: component, phase, sub_process, task,
model_role draft|flag|propose|code|explain|answer, review
always|sample|none, data_sensitivity public|internal|confidential,
model_location hosted|local, owner) and prints the components by GSBPM
phase, then the flags:

    unreviewed output    a role that produces text or codes users see
                         (answer, draft, code) with review "none"
    confidential hosted  confidential data sent to a hosted model
    deciding role        a role of "decide", which this guide does not
                         allow: models propose, flag, code, explain, and
                         draft; people decide

Exit status 1 when any flag is raised, so that the map can be checked in
a pipeline. Uses pandas.

Usage:
    python map_components.py gsbpm_map.csv

What this does not do: the map records the organization's decisions; it
does not judge whether a task should use a model at all. The chapter's
typology and the evaluation suite of each component answer that, and the
GSBPM sub-process numbers follow version 5.1 and may differ in the
organization's own version.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

VISIBLE = {"answer", "draft", "code"}


def flags_for(r: pd.Series) -> list[str]:
    """The rules a component can break: unreviewed visible output, confidential data hosted, a deciding role."""
    out = []
    if r["model_role"] in VISIBLE and r["review"] == "none":
        out.append(
            f"{r['component']}: {r['model_role']} output reaches people with no review"
        )
    if r["data_sensitivity"] == "confidential" and r["model_location"] == "hosted":
        out.append(f"{r['component']}: confidential data to a hosted model")
    if r["model_role"] == "decide":
        out.append(
            f"{r['component']}: a model does not decide; change the role to propose or flag"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("map")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.map, dtype=str).fillna("")

    print(f"{len(df)} components across {df['phase'].nunique()} phases\n")
    for phase, comps in df.groupby("phase", sort=False):
        print(phase)
        for c in comps.itertuples():
            print(
                f"  {c.sub_process:<34} {c.component:<28} {c.model_role:<8} review {c.review:<7} {c.data_sensitivity:<13} {c.model_location}"
            )
    flags = [f for _, r in df.iterrows() for f in flags_for(r)]
    print()
    if flags:
        print("flags:")
        for f in flags:
            print(f"  {f}")
        return 1
    print("no flags: every visible output is reviewed and confidential data stay local")
    return 0


if __name__ == "__main__":
    sys.exit(main())
