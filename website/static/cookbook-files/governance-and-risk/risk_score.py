"""Score the risk of each AI component and list the controls its tier requires.

Reads the risk register (one row per component with three ratings from 1
to 3: cost of error, exposure, reversibility) and the controls table
(controls required per tier). The score is the product of the three
ratings, from 1 to 27; the tier is Low (1 to 6), Medium (7 to 12), or
High (13 to 27). With --register it compares the computed tier with the
tier recorded in the component register and reports a difference. Exit
status 1 when a rating is outside 1 to 3, a residual risk is not
accepted, or a tier differs from the register.

Usage:
    python risk_score.py risk_register.csv controls_by_tier.csv --register component_register.csv

Requires pandas.

What this does not check: whether the ratings are right. The board
decides them; the script makes the arithmetic and the consequences
visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RATINGS = ["cost_of_error", "exposure", "reversibility"]


def tier(score: int) -> str:
    if score <= 6:
        return "Low"
    if score <= 12:
        return "Medium"
    return "High"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("risk_register", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument(
        "--register", type=Path, help="component register to compare tiers with"
    )
    args = parser.parse_args(argv)

    risk = pd.read_csv(args.risk_register, dtype=str).fillna("")
    controls = pd.read_csv(args.controls, dtype=str).fillna("")
    problems: list[str] = []
    for col in RATINGS:
        risk[col] = pd.to_numeric(risk[col], errors="coerce")
        bad = risk[risk[col].isna() | (risk[col] < 1) | (risk[col] > 3)]
        for _, row in bad.iterrows():
            problems.append(f"{row['component_id']}: {col} is not a rating from 1 to 3")
    risk = risk.dropna(subset=RATINGS)
    risk["score"] = (risk[RATINGS].prod(axis=1)).astype(int)
    risk["tier"] = risk["score"].map(tier)

    recorded = {}
    if args.register:
        reg = pd.read_csv(args.register, dtype=str).fillna("")
        recorded = dict(zip(reg["component_id"], reg["risk_tier"]))

    print(
        f"{'id':<4} {'component':<30} {'cost':>4} {'exp':>4} {'rev':>4} {'score':>5}  tier"
    )
    for _, row in risk.sort_values("score", ascending=False).iterrows():
        flag = ""
        rec = recorded.get(row["component_id"])
        if rec and rec != row["tier"]:
            flag = f"  (register says {rec})"
            problems.append(
                f"{row['component_id']}: computed {row['tier']}, register says {rec}"
            )
        if not row["residual_risk_accepted_by"].strip():
            problems.append(f"{row['component_id']}: residual risk not accepted")
        print(
            f"{row['component_id']:<4} {row['component'][:30]:<30} {int(row['cost_of_error']):>4} "
            f"{int(row['exposure']):>4} {int(row['reversibility']):>4} {row['score']:>5}  {row['tier']}{flag}"
        )

    print("\ncontrols required:")
    for t in ("Low", "Medium", "High"):
        ids = risk.loc[risk["tier"] == t, "component_id"].tolist()
        if not ids:
            continue
        print(f"  {t} ({', '.join(ids)}):")
        for _, c in controls[controls["tier"] == t].iterrows():
            print(f"    - {c['control']} [{c['evidence']}]")

    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
