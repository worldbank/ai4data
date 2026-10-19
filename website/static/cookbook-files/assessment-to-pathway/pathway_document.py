"""Write the improvement pathway document from the profile, the targets, the sequence, and the effort.

Reads the question scores, the targets, the dependencies, and the
recipe effort table, and prints the pathway document in Markdown with
the sections an organization agrees and funds: summary, readiness
profile, targets and gaps, sequence in waves, effort and roles by wave,
evidence and re-assessment, and governance of the pathway. Exit status
1 when any of the underlying checks fails.

Usage:
    python pathway_document.py question_scores.csv targets.csv dependencies.csv recipe_effort.csv --organization "National Statistics Office (example)" --assessed 2025-06-20

Requires pandas and the other scripts of this cookbook in the same folder.

What this does not do: it writes the sections from the tables. The
narrative paragraphs that explain the organization's choices, and the
dates that a budget cycle imposes, are added by the planning team in
the document the script produces.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def load(name: str):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scores", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("effort", type=Path)
    parser.add_argument("--organization", default="The organization")
    parser.add_argument("--assessed", default="(date of the assessment)")
    args = parser.parse_args(argv)
    rp, gt, sp, ee = (
        load("readiness_profile"),
        load("gap_table"),
        load("sequence_plan"),
        load("effort_estimate"),
    )
    scores = pd.read_csv(args.scores, dtype=str).fillna("")
    targets = pd.read_csv(args.targets, dtype=str).fillna("")
    dep_df = pd.read_csv(args.dependencies, dtype=str).fillna("")
    effort = pd.read_csv(args.effort, dtype=str).fillna("")
    prof = rp.profile(scores)
    table = gt.gaps(scores, targets)
    deps = {
        r["dimension"]: [d for d in r["depends_on"].split(";") if d]
        for _, r in dep_df.iterrows()
    }
    try:
        ws = sp.waves(table, deps)
    except ValueError as e:
        print(f"problem: {e}")
        return 1
    sel = ee.estimate(table, effort)
    info = {r["dimension"]: r for _, r in table.iterrows()}
    pillars = prof.groupby("pillar")["average"].mean()
    overall = pillars.mean() / 3 * 100

    out = [
        f"# Improvement pathway: {args.organization}",
        "",
        f"Assessed on {args.assessed} with the AI-readiness assessment framework.",
        "",
    ]
    out += ["## Summary", ""]
    n_out = len(sel[sel["cookbook"] == "outside the series"])
    outside = f"{n_out} item" + (" lies" if n_out == 1 else "s lie")
    gapped = [d for w in ws for d in w]
    out.append(
        f"The organization scores {overall:.1f} of 100 overall (pillar I {pillars.get('1', 0):.2f}, pillar II {pillars.get('2', 0):.2f} on the 0 to 3 scale). {len(gapped)} of {len(table)} dimensions have a gap to their target, addressed in {len(ws)} waves. The pathway needs {int(sel['days_low'].sum())} to {int(sel['days_high'].sum())} person-days across {len(sel)} recipes, of which {outside} outside the cookbook series."
    )
    out += [
        "",
        "## Readiness profile",
        "",
        "| Dimension | Questions | Average | Level |",
        "|---|---|---|---|",
    ]
    for _, r in prof.iterrows():
        out.append(
            f"| {r['dimension']} {r['dimension_name']} | {r['applicable']} | {r['average']:.2f} | {r['level']} |"
        )
    out += [
        "",
        "## Targets and gaps",
        "",
        "| Dimension | Current | Target | Gap | Reason |",
        "|---|---|---|---|---|",
    ]
    for _, r in table.sort_values("dimension").iterrows():
        out.append(
            f"| {r['dimension']} | {r['current']} | {r['target']} | {r['gap']} | {r['reason']} |"
        )
    out += ["", "## Sequence", ""]
    for i, w in enumerate(ws, 1):
        out.append(f"### Wave {i}")
        out.append("")
        for d in w:
            r = info[d]
            after = [x for x in deps.get(d, []) if x in gapped]
            out.append(
                f"- {d} {r['name']}: {r['current']} to {r['target']}"
                + (f", after {', '.join(after)}" if after else "")
            )
        out.append("")
    out += [
        "## Effort and roles by wave",
        "",
        "| Wave | Dimension | Recipe | Source | Person-days | Roles | Kind of cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, w in enumerate(ws, 1):
        for d in w:
            for _, e in sel[sel["dimension"] == d].iterrows():
                out.append(
                    f"| {i} | {d} | {e['recipe']} | {e['cookbook']} | {int(e['days_low'])} to {int(e['days_high'])} | {e['roles']} | {e['cost_kind']} |"
                )
    out += [
        "",
        "## Evidence and re-assessment",
        "",
        "Each question below target has a row in the evidence register with the document that will show the change, an owner, and a due date. The re-assessment uses the same framework version and compares the profiles with the comparison script.",
        "",
    ]
    out += [
        "## Governance of the pathway",
        "",
        "The pathway is owned by the head of methodology and reported to the AI governance board each quarter with the evidence register's status and the overdue items. Changes to targets or sequence are decisions of the board and are recorded in the decision log.",
        "",
    ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
