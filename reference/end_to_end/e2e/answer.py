"""Step 5: compose an answer from the MCP tool results, with claim tags.

The answer is written by a template so that the run needs no model: for
each series the question names, the composer takes the latest observation
from the get_values result and writes one sentence with the number inside
a claim tag naming the observation. That is what a model is instructed to
do under Proof-Carrying Numbers; the template does it deterministically so
that the verification step has something to check on every run.

With `inject_error`, the composer writes one wrong number inside its tag
and leaves one number without a tag, which is how a model's mistakes look
to the verifier: the first is flagged, the second gets no mark.

An answer from a real model can be passed to run.py with --answer and is
verified by the same step against the same claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The series the question asks about, and how many decimals the answer shows.
ASKED = [
    ("FS_UNDERNOURISH_PCT", "the prevalence of undernourishment", 1),
    ("HE_U5_MORT_RATE", "the under-five mortality rate", 0),
    ("ED_PRIMARY_NER_PCT", "the net enrolment rate in primary education", 1),
    ("LF_UNEMP_PCT", "the unemployment rate", 1),
]

QUESTION = (
    "What are the latest published figures for the prevalence of undernourishment, "
    "the under-five mortality rate, the net enrolment rate in primary education, "
    "and the unemployment rate?"
)


@dataclass
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    result: Any

    @property
    def ok(self) -> bool:
        return not (isinstance(self.result, dict) and "error" in self.result)


def _fmt(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _latest(values: dict[str, Any]) -> dict[str, Any]:
    return max(values["observations"], key=lambda o: o["TIME_PERIOD"])


def compose(calls: list[ToolCall], inject_error: bool = False) -> str:
    """One sentence per asked series, from the get_values results in `calls`."""
    by_series = {c.arguments.get("series"): c for c in calls if c.tool == "get_values"}
    sentences = []
    for i, (series, label, decimals) in enumerate(ASKED):
        call = by_series.get(series)
        if call is None or not call.ok:
            sentences.append(
                f"No published series for {label} is in the catalog, so no figure is given for it."
            )
            continue
        obs = _latest(call.result)
        unit = call.result["UNIT_MEASURE"]
        release = call.result["RELEASE"]
        value = obs["OBS_VALUE"]
        status = " (provisional)" if obs["OBS_STATUS"] == "P" else ""
        number = _fmt(value, decimals)
        if inject_error and i == 0:
            number = _fmt(value - 0.4, decimals)  # a wrong figure inside its tag
        tagged = f'<claim id="{obs["claim_id"]}" policy="rounded" decimals="{decimals}">{number}</claim>'
        if inject_error and i == 1:
            tagged = number  # a figure the model forgot to tag
        sentences.append(
            f"In {obs['TIME_PERIOD']}, {label} was {tagged} {unit}{status}, "
            f"according to the release of {release}."
        )
    return " ".join(sentences)
