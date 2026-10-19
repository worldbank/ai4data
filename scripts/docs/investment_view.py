"""Aggregate what the recipes of each Pillar II dimension take: roles, effort, recurring work.

Reads the cookbook links of each dimension in website/src/content/readinessMap.js,
opens the linked chapters (a link to a cookbook's root includes all its chapters),
and collects every recipe's level, skills line, and time estimate. For each
dimension and level it writes the distinct roles, the one-off effort as a range
of person-days, the recurring work the recipes name, and the number of recipes,
to website/src/content/investment.json. The page renders that file.

Usage:
    python scripts/docs/investment_view.py          # writes the JSON
    python scripts/docs/investment_view.py --print  # prints the table

What this does not do: it adds up the recipes' own estimates, which are
written for the running examples and a first implementation. Hardware,
hosting, and prerequisite dimensions are written by hand on the page.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAP = REPO / "website" / "src" / "content" / "readinessMap.js"
OUT = REPO / "website" / "src" / "content" / "investment.json"
LEVELS = ["Foundational", "AI-ready", "AI-native"]

WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "half": 0.5,
}
UNIT_DAYS = {"minute": 1 / 96, "hour": 1 / 8, "day": 1, "week": 5, "month": 21}
RECURRING = re.compile(
    r"\b(per|each|every|a month|a quarter|ongoing|after that|part of|with each|on each|monthly|weekly|quarterly|yearly)\b",
    re.IGNORECASE,
)
PER_PERIOD = re.compile(
    r"\b(release|month|quarter|week|period|review|run|change|year|batch|round|ongoing|after that|part of)\b",
    re.IGNORECASE,
)

ROLE_RULES = [
    ("Data curator", r"curator"),
    ("Methodologist", r"methodolog|statistician|survey"),
    ("Developer or data engineer", r"developer|python|backend|web|engineer|sdk"),
    ("Data scientist", r"data scientist|gpu|machine"),
    ("Dissemination officer", r"disseminat|web editor|analyst|communication"),
    ("Catalog or data manager", r"catalog|data manager|librarian|metadata lead|archiv"),
    ("IT operations and security", r"operations|infrastructure|security|administrator"),
    ("Disclosure control specialist", r"disclosure|legal|data protection"),
    ("Management", r"management|head of|lead\b|finance|owner"),
    (
        "Subject-matter staff",
        r"subject-matter|coding supervisor|coders|editors|speakers|native|reviewer|classification",
    ),
]


def parse_time(text: str) -> tuple[float, float, list[str]]:
    """One-off effort in person-days (min, max) and the recurring parts, from a recipe's time line."""
    lo = hi = 0.0
    recurring: list[str] = []
    for part in re.split(r";|,\s*then\s*|\bthen\b", text):
        part = part.strip()
        if not part:
            continue
        if RECURRING.search(part) and not re.search(
            r"to set up|the first time|for the first|for a prototype|for the template",
            part,
            re.IGNORECASE,
        ):
            recurring.append(part)
            continue
        nums = []
        for tok in re.findall(
            r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|half)\b", part, re.IGNORECASE
        ):
            nums.append(float(tok) if tok[0].isdigit() else WORDS[tok.lower()])
        if not nums and re.search(r"\b(a|an)\b", part, re.IGNORECASE):
            nums = [1.0]
        unit = next(
            (u for u in UNIT_DAYS if re.search(rf"\b{u}", part, re.IGNORECASE)), None
        )
        if unit is None or not nums:
            continue
        vals = [0.5] if "half" in part.lower() and unit == "day" else nums
        scaled = [v * UNIT_DAYS[unit] for v in vals]
        if "about" in part.lower() and len(scaled) == 1:
            scaled = [scaled[0], scaled[0]]
        lo += min(scaled)
        hi += max(scaled)
    return lo, hi, recurring


def roles_from(skills: str) -> list[str]:
    found = [name for name, pat in ROLE_RULES if re.search(pat, skills, re.IGNORECASE)]
    return found or ["Staff of the unit"]


def hidden_cookbooks() -> set[str]:
    """Ids listed as hidden in website/cookbook-visibility.json; their recipes stay out of the view."""
    path = REPO / "website" / "cookbook-visibility.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("hidden", []))


def chapters_for(link: str) -> list[Path]:
    """The chapter files behind one readiness-map link."""
    m = re.match(r"/cookbook/([^/#]+)/?([^/#]*)", link)
    if not m:
        return []
    cb, slug = m.group(1), m.group(2)
    if cb in hidden_cookbooks():
        return []
    root = REPO / "cookbook" / cb
    if slug:
        p = root / f"{slug}.mdx"
        return [p] if p.exists() else []
    return sorted(
        p for p in root.glob("*.mdx") if p.name not in ("index.mdx", "start-here.mdx")
    )


def recipes_in(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    out = []
    for m in re.finditer(
        r'<Recipe\s+title="([^"]+)"\s+level="([^"]+)"\s+skills="([^"]+)"\s+time="([^"]+)"',
        text,
    ):
        out.append(
            {
                "title": m.group(1),
                "level": m.group(2),
                "skills": m.group(3),
                "time": m.group(4),
                "chapter": f"{path.parent.name}/{path.stem}",
            }
        )
    return out


def dimension_links() -> dict[str, list[str]]:
    src = MAP.read_text(encoding="utf-8")
    cb = re.search(r"const CB = '([^']+)'", src).group(1)
    out: dict[str, list[str]] = {}
    for block in re.finditer(
        r"\n  \{\n    id: '(\d\.\d)',(.*?)\n  \},", src, re.DOTALL
    ):
        dim, body = block.group(1), block.group(2)
        links = re.findall(r"\{to: (?:`\$\{CB\}([^`]*)`|'([^']+)'), label", body)
        out[dim] = [cb + a if a else b for a, b in links]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--print", action="store_true", dest="show")
    args = parser.parse_args(argv)

    result = {}
    for dim, links in dimension_links().items():
        if not dim.startswith("2."):
            continue
        seen: set[str] = set()
        per_level: dict[str, dict] = {
            lv: {
                "roles": defaultdict(int),
                "lo": 0.0,
                "hi": 0.0,
                "recurring": [],
                "recipes": [],
            }
            for lv in LEVELS
        }
        for link in links:
            for chapter in chapters_for(link):
                for r in recipes_in(chapter):
                    key = r["chapter"] + r["title"]
                    if key in seen:
                        continue
                    seen.add(key)
                    lv = per_level[r["level"]]
                    lo, hi, rec = parse_time(r["time"])
                    lv["lo"] += lo
                    lv["hi"] += hi
                    lv["recurring"] += rec
                    for role in roles_from(r["skills"]):
                        lv["roles"][role] += 1
                    lv["recipes"].append(
                        {
                            "title": r["title"],
                            "chapter": r["chapter"],
                            "time": r["time"],
                            "skills": r["skills"],
                        }
                    )
        result[dim] = {
            lv: {
                "roles": sorted(v["roles"], key=lambda k: -v["roles"][k]),
                "effort_days": [round(v["lo"]), round(v["hi"])],
                "per_period": sorted(
                    {r for r in v["recurring"] if PER_PERIOD.search(r)}
                ),
                "per_item": sorted(
                    {r for r in v["recurring"] if not PER_PERIOD.search(r)}
                ),
                "recipes": v["recipes"],
            }
            for lv, v in per_level.items()
        }
    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.show:
        for dim, levels in result.items():
            print(f"\n{dim}")
            for lv, v in levels.items():
                print(
                    f"  {lv:<12} recipes {len(v['recipes']):>2}  effort {v['effort_days'][0]:>3}-{v['effort_days'][1]:<3} person-days  roles: {', '.join(v['roles'])}"
                )
                if v["per_period"]:
                    print(
                        f"               per period: {'; '.join(v['per_period'][:6])}"
                    )
                if v["per_item"]:
                    print(f"               per item:   {'; '.join(v['per_item'][:6])}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
