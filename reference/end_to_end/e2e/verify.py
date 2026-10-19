"""Step 6: verify every number in the answer with the PCN engine.

Claims come from the tool results: each observation returned by get_values
becomes a claim keyed by its claim_id. The answer is parsed for claim tags
of the form <claim id="..." policy="..." decimals="...">7.6</claim>, and
each tag is compared with its claim under its policy by pcn_core, the
Python core of Proof-Carrying Numbers. The outcome per tag is verified or
flagged; a tag whose identifier is unknown is flagged too. Numbers that
appear outside any tag are reported as unmarked, which is the fail-closed
rule: no tag, no mark. Periods, dates, and four-digit integers (years, and
constants such as "per 1000") are left out of the unmarked list.

A verified outcome means the number matches the record it cites under the
policy. It does not establish that the record is the right one for the
question or that the sentence around it is correct.

Requires pcn-core, installed from the PCN repository (see requirements.txt).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pcn_core import Claim, compare_by_policy

from .answer import ToolCall

TAG = re.compile(
    r'<claim\s+id="(?P<id>[^"]+)"\s+policy="(?P<policy>[a-z]+)"(?:\s+decimals="(?P<decimals>\d+)")?(?:\s+tolerance="(?P<tolerance>[0-9.]+)")?\s*>(?P<text>[^<]*)</claim>'
)
NUMBER = re.compile(r"(?<![\w.-])\d+(?:[.,]\d+)?(?![\w-])")
PERIOD = re.compile(r"^\d{4}(-Q[1-4]|-\d{2}(-\d{2})?)?$")


@dataclass
class Outcome:
    text: str
    claim_id: str | None
    policy: str | None
    outcome: str  # verified | flagged | unmarked
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.text,
            "claim_id": self.claim_id,
            "policy": self.policy,
            "outcome": self.outcome,
            "detail": self.detail,
        }


def claims_from_calls(calls: list[ToolCall]) -> dict[str, Claim]:
    claims: dict[str, Claim] = {}
    for call in calls:
        if call.tool != "get_values" or not call.ok:
            continue
        for obs in call.result["observations"]:
            claims[obs["claim_id"]] = Claim(
                value=obs["OBS_VALUE"], country=obs["REF_AREA"], date=obs["TIME_PERIOD"]
            )
    return claims


def _policy(m: re.Match) -> dict[str, Any]:
    kind = m.group("policy")
    if kind == "rounded":
        return {"type": "rounded", "decimals": int(m.group("decimals") or 0)}
    if kind == "tolerance":
        return {"type": "tolerance", "tolerance": float(m.group("tolerance") or 0.02)}
    return {"type": kind}


def verify(answer: str, claims: dict[str, Claim]) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for m in TAG.finditer(answer):
        claim_id, text = m.group("id"), m.group("text").strip()
        policy = _policy(m)
        claim = claims.get(claim_id)
        if claim is None:
            outcomes.append(
                Outcome(
                    text,
                    claim_id,
                    policy["type"],
                    "flagged",
                    "no claim with this identifier in the tool results",
                )
            )
        elif compare_by_policy(text, claim, policy):
            outcomes.append(
                Outcome(
                    text,
                    claim_id,
                    policy["type"],
                    "verified",
                    f"matches {claim.value} for {claim.date} under {policy['type']}",
                )
            )
        else:
            outcomes.append(
                Outcome(
                    text,
                    claim_id,
                    policy["type"],
                    "flagged",
                    f"record holds {claim.value} for {claim.date}",
                )
            )
    # Numbers outside any tag: strip the tags, then drop periods and dates.
    stripped = TAG.sub(" ", answer)
    stripped = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", stripped)
    for m in NUMBER.finditer(stripped):
        token = m.group(0)
        if PERIOD.match(token) or len(token) == 4 and token.isdigit():
            continue
        outcomes.append(
            Outcome(token, None, None, "unmarked", "no claim tag, so no mark is drawn")
        )
    return outcomes


def summary(outcomes: list[Outcome]) -> dict[str, int]:
    return {
        k: sum(1 for o in outcomes if o.outcome == k)
        for k in ("verified", "flagged", "unmarked")
    }
