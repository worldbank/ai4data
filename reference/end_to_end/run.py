"""Run the end-to-end reference implementation.

Documents a set of series and a survey in the World Bank schemas, validates
them against the pinned schemas and a completeness profile, publishes the
records that pass to a catalog directory (and optionally to a NADA
instance), starts an MCP server over the catalog as a subprocess, calls its
tools over the protocol, composes an answer from the results with claim
tags, and verifies every number in the answer with the PCN engine. Writes
report.md, report.json, answer.md, and the catalog under --out.

Usage:
    python run.py --out build
    python run.py --out build --inject-error          # one wrong, one untagged number
    python run.py --out build --answer my_answer.md   # verify a model's answer
    python run.py --out build --nada-url https://catalog.example --nada-dry-run
    NADA_API_KEY=... python run.py --out build --nada-url https://catalog.example

Exit status: 0 when the report's result is PASS, 1 when it is FAIL.

What this does not do: it does not call a language model (the answer is
composed by a template unless --answer gives one), and it does not deploy
any of the applications it stands in for. It shows the pattern on the
cookbook's example organization so that each step and its output can be
read, tested, and replaced by the organization's own system.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e2e import answer as answer_mod
from e2e import publish, records, report, validate, verify
from e2e.answer import ASKED, QUESTION, ToolCall

REPO_ROOT = HERE.parent.parent
DEFAULT_SCHEMAS = REPO_ROOT / "tests" / "fixtures" / "wb-schemas"


def _result_payload(result: Any) -> Any:
    """The JSON the tool returned, from either SDK generation's result object."""
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured:
        # 1.x wraps a non-object return in {"result": ...}
        return (
            structured.get("result", structured)
            if isinstance(structured, dict) and set(structured) == {"result"}
            else structured
        )
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return None


async def _call_tools(
    server: Path, catalog: Path, plan: list[tuple[str, dict[str, Any]]]
) -> list[ToolCall]:
    """Run the server as a subprocess over stdio and make the planned calls."""
    params_kwargs = {
        "command": sys.executable,
        "args": [str(server), str(catalog)],
        "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
    }
    try:  # mcp 2.x
        from mcp import Client, StdioServerParameters

        calls: list[ToolCall] = []
        async with Client(StdioServerParameters(**params_kwargs)) as client:
            for tool, arguments in plan:
                result = await client.call_tool(tool, arguments)
                calls.append(ToolCall(tool, arguments, _result_payload(result)))
        return calls
    except ImportError:  # mcp 1.x
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        calls = []
        async with (
            stdio_client(StdioServerParameters(**params_kwargs)) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            for tool, arguments in plan:
                result = await session.call_tool(tool, arguments)
                calls.append(ToolCall(tool, arguments, _result_payload(result)))
        return calls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("build"),
        help="output directory (default: build)",
    )
    ap.add_argument(
        "--fixtures",
        type=Path,
        default=HERE / "fixtures",
        help="folder with series.json, survey.json, values.csv, and the profiles",
    )
    ap.add_argument(
        "--schema-dir",
        type=Path,
        default=DEFAULT_SCHEMAS,
        help="pinned World Bank schema files",
    )
    ap.add_argument(
        "--level",
        choices=["foundational", "ai-ready"],
        default="foundational",
        help="completeness level required for publication",
    )
    ap.add_argument(
        "--inject-error",
        action="store_true",
        help="write one wrong tagged number and one untagged number into the answer",
    )
    ap.add_argument(
        "--answer",
        type=Path,
        help="verify this answer (Markdown or text with claim tags) in place of the composed one",
    )
    ap.add_argument("--nada-url", help="base URL of a NADA instance to publish to")
    ap.add_argument(
        "--nada-api-key",
        default=os.environ.get("NADA_API_KEY"),
        help="catalog administration API key (or NADA_API_KEY)",
    )
    ap.add_argument(
        "--nada-dry-run",
        action="store_true",
        help="build the NADA requests and report them without sending",
    )
    args = ap.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1. Documentation
    doc = records.load(args.fixtures)
    print(
        f"[1/6] documentation: {len(doc.series)} series records and 1 survey record loaded from {args.fixtures.name}/"
    )

    # 2. Validation
    store = validate.SchemaStore(args.schema_dir)
    profiles = validate.load_profiles(args.fixtures)
    results = validate.validate(doc.records, profiles, store, args.level)
    passed = [(kind, rec) for (kind, rec), res in zip(doc.records, results) if res.ok]
    withheld = [res for res in results if not res.ok]
    print(
        f"[2/6] validation at the {args.level} level: {len(passed)} records pass, {len(withheld)} withheld"
        + (
            ": " + ", ".join(f"{r.idno} ({'; '.join(r.reasons)})" for r in withheld)
            if withheld
            else ""
        )
    )

    # 3. Publication
    published = publish.publish_catalog(passed, doc.values, out)
    nada: list[dict[str, Any]] = []
    if args.nada_url:
        nada = publish.publish_to_nada(
            passed, args.nada_url, args.nada_api_key, args.nada_dry_run
        )
    print(
        f"[3/6] publication: {len(published.records)} records and {len(published.files)} files in {published.catalog_dir.relative_to(out) if published.catalog_dir.is_relative_to(out) else published.catalog_dir}/"
        + (f"; NADA: {', '.join(n['status'] for n in nada)}" if nada else "")
    )

    # 4. Agent access over MCP
    plan: list[tuple[str, dict[str, Any]]] = [
        ("search_catalog", {"query": "undernourishment hunger food", "limit": 3})
    ]
    plan += [("get_record", {"idno": "FS_UNDERNOURISH_PCT"})]
    plan += [
        ("get_values", {"series": series, "ref_area": "NAT"}) for series, _, _ in ASKED
    ]
    calls = asyncio.run(_call_tools(HERE / "server.py", published.catalog_dir, plan))
    print(
        f"[4/6] agent access: {len(calls)} tool calls over MCP (stdio), {sum(1 for c in calls if c.ok)} succeeded"
    )

    # 5. Answer
    if args.answer:
        answer_text = args.answer.read_text(encoding="utf-8").strip()
        source = f"from {args.answer.name}"
    else:
        answer_text = answer_mod.compose(calls, inject_error=args.inject_error)
        source = "composed by the template" + (
            " with a planted wrong number and a planted untagged number"
            if args.inject_error
            else ""
        )
    print(f"[5/6] answer: {len(answer_text.split())} words, {source}")

    # 6. Verification
    claims = verify.claims_from_calls(calls)
    outcomes = verify.verify(answer_text, claims)
    counts = verify.summary(outcomes)
    print(
        f"[6/6] verification with PCN: {counts['verified']} verified, {counts['flagged']} flagged, {counts['unmarked']} unmarked, against {len(claims)} claims"
    )

    r = report.build(
        level=args.level,
        results=results,
        published=published,
        nada=nada,
        calls=calls,
        question=QUESTION,
        answer=answer_text,
        outcomes=outcomes,
        answer_source=source,
    )
    md, _ = report.write(r, out)
    print(f"report: {md}")
    print(f"result: {r['result']}")
    return 0 if r["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
