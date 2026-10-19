"""Step 7: write the report of the run, as Markdown and as JSON.

The report is deterministic: no timestamps, no absolute paths, so that a
committed sample can be compared with a fresh run in a test. The last line
is PASS when every record that was meant to publish published, every tool
call succeeded, and no number in the answer was flagged; otherwise FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .answer import ToolCall
from .publish import Published
from .validate import RecordResult, as_dict
from .verify import Outcome, summary

CAVEAT = (
    "A verified number matches the record it cites under the stated policy. "
    "The check says nothing about whether the record is the right one for the "
    "question, whether the statistical method behind it is sound, or whether "
    "the sentence around the number interprets it correctly."
)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build(
    *,
    level: str,
    results: list[RecordResult],
    published: Published,
    nada: list[dict[str, Any]],
    calls: list[ToolCall],
    question: str,
    answer: str,
    outcomes: list[Outcome],
    answer_source: str,
) -> dict[str, Any]:
    counts = summary(outcomes)
    # A withheld record is the gate working, not a failure. A tool call may
    # fail only for a product the catalog does not hold.
    calls_ok = all(
        c.ok
        or (
            c.tool == "get_values"
            and c.arguments.get("series") not in published.records
        )
        for c in calls
    )
    structure_ok = all(not r.structure for r in results)
    passed = structure_ok and calls_ok and counts["flagged"] == 0
    return {
        "level": level,
        "records": [as_dict(r) for r in results],
        "published": published.records,
        "files": published.files,
        "nada": nada,
        "tool_calls": [
            {"tool": c.tool, "arguments": c.arguments, "ok": c.ok} for c in calls
        ],
        "question": question,
        "answer": answer,
        "answer_source": answer_source,
        "verification": [o.as_dict() for o in outcomes],
        "counts": counts,
        "caveat": CAVEAT,
        "result": "PASS" if passed else "FAIL",
    }


def to_markdown(r: dict[str, Any]) -> str:
    lines = ["# End-to-end run", ""]
    lines += [
        "## 1. Documentation",
        "",
        f"{len(r['records'])} records loaded: {sum(1 for x in r['records'] if x['kind'] == 'indicator')} series in the indicator schema and {sum(1 for x in r['records'] if x['kind'] == 'microdata')} survey in the microdata schema.",
        "",
    ]
    lines += [f"## 2. Validation at the {r['level']} level", ""]
    rows = []
    for x in r["records"]:
        reasons = (
            [f"structure: {s}" for s in x["structure_errors"]]
            + [f"missing: {m}" for m in x["missing_fields"]]
            + [f"invalid: {i}" for i in x["validity_issues"]]
        )
        rows.append(
            [
                x["idno"],
                x["kind"],
                "published" if x["published"] else "withheld",
                "; ".join(reasons) or "none",
            ]
        )
    lines += [_md_table(["Record", "Type", "Decision", "Findings"], rows), ""]
    lines += [
        "## 3. Publication",
        "",
        f"{len(r['published'])} records published to the catalog directory: {', '.join(r['published'])}.",
        "",
        f"{len(r['files'])} files written: records, values, schema.org descriptions, and the index.",
        "",
    ]
    if r["nada"]:
        lines += [
            _md_table(
                ["Record", "NADA request", "Status"],
                [[n["idno"], n["url"], n["status"]] for n in r["nada"]],
            ),
            "",
        ]
    else:
        lines += ["NADA: not attempted (no catalog URL given).", ""]
    lines += [
        "## 4. Agent access over MCP",
        "",
        _md_table(
            ["Tool", "Arguments", "Result"],
            [
                [c["tool"], json.dumps(c["arguments"]), "ok" if c["ok"] else "error"]
                for c in r["tool_calls"]
            ],
        ),
        "",
    ]
    lines += [
        "## 5. Answer",
        "",
        f"Question: {r['question']}",
        "",
        f"Answer ({r['answer_source']}):",
        "",
        "```text",
        r["answer"],
        "```",
        "",
    ]
    lines += [
        "## 6. Verification with Proof-Carrying Numbers",
        "",
        _md_table(
            ["Number", "Claim", "Policy", "Outcome", "Detail"],
            [
                [
                    o["number"],
                    o["claim_id"] or "none",
                    o["policy"] or "none",
                    o["outcome"],
                    o["detail"],
                ]
                for o in r["verification"]
            ],
        ),
        "",
    ]
    c = r["counts"]
    lines += [
        f"{c['verified']} verified, {c['flagged']} flagged, {c['unmarked']} unmarked.",
        "",
        r["caveat"],
        "",
        f"result: {r['result']}",
        "",
    ]
    return "\n".join(lines)


def write(r: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    md = out_dir / "report.md"
    js = out_dir / "report.json"
    md.write_text(to_markdown(r), encoding="utf-8")
    js.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "answer.md").write_text(r["answer"] + "\n", encoding="utf-8")
    return md, js
