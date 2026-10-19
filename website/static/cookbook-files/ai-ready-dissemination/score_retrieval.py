"""Score a search system on a known-item question set.

Reads a question set (CSV with question_id, language, question, expected_id)
and a catalog, runs each question through a search function, and prints
Recall@5, Recall@10, and MRR overall and per language.

The built-in search is a plain keyword overlap over the catalog's `name` and
`definition_long` fields (World Bank indicator schema names). It exists so
that the script runs without any dependencies and gives a baseline. Replace
`search()` with a call to your own search API to score it.

Uses pandas and scikit-learn (TF-IDF for the baseline search).

Usage:
    python score_retrieval.py eval_questions.csv example_catalog.csv
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN = re.compile(r"\w+")  # Unicode-aware: words in any script
ID_FIELD = "idno"
TEXT_FIELDS = ("name", "definition_long")
K_VALUES = (5, 10)


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


class KeywordSearch:
    """A TF-IDF keyword search over the catalog records: the baseline to replace with the real search."""

    def __init__(self, catalog: pd.DataFrame) -> None:
        self.ids = catalog[ID_FIELD].tolist()
        texts = catalog[list(TEXT_FIELDS)].fillna("").agg(" ".join, axis=1)
        self.vectorizer = TfidfVectorizer(token_pattern=TOKEN.pattern, lowercase=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def __call__(self, question: str, k: int = 10) -> list[str]:
        """Return up to k catalog ids ranked by similarity; records that share no word are left out."""
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[
            0
        ]
        ranked = sorted(
            (i for i in range(len(self.ids)) if scores[i] > 0),
            key=lambda i: (-scores[i], i),
        )
        return [self.ids[i] for i in ranked[:k]]


def search(question: str, catalog: list[dict[str, str]], k: int = 10) -> list[str]:
    """Rank the catalog for one question; kept as a function so that a real search can replace it."""
    return KeywordSearch(pd.DataFrame(catalog))(question, k)


def rank_in(ranked: list[str], expected: str) -> float:
    return ranked.index(expected) + 1 if expected in ranked else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("questions")
    parser.add_argument("catalog")
    args = parser.parse_args(argv)
    catalog = pd.read_csv(args.catalog, dtype=str)
    questions = pd.read_csv(args.questions, dtype=str)
    questions = questions[questions["expected_id"] != "NONE"].copy()
    find = KeywordSearch(catalog)

    print(f"{'id':<5} {'lang':<5} {'rank':>4}  question")
    ranks = []
    for q in questions.itertuples():
        rank = rank_in(find(q.question, k=max(K_VALUES)), q.expected_id)
        ranks.append(rank)
        print(
            f"{q.question_id:<5} {q.language:<5} {'-' if pd.isna(rank) else int(rank):>4}  {q.question}"
        )
    questions["rank"] = ranks
    for k in K_VALUES:
        questions[f"R@{k}"] = (questions["rank"] <= k).astype(float)
    questions["MRR"] = (1 / questions["rank"]).fillna(0.0)

    groups = pd.concat(
        [questions.assign(group="all"), questions.assign(group=questions["language"])]
    )
    table = groups.groupby("group", sort=False)[
        [f"R@{k}" for k in K_VALUES] + ["MRR"]
    ].agg(["mean", "size"])
    print()
    print(
        f"{'group':<6} {'n':>3} "
        + " ".join(f"{'R@' + str(k):>6}" for k in K_VALUES)
        + f" {'MRR':>6}"
    )
    for group in ["all"] + sorted(g for g in table.index if g != "all"):
        r = table.loc[group]
        recalls = " ".join(f"{r[(f'R@{k}', 'mean')]:>6.2f}" for k in K_VALUES)
        print(
            f"{group:<6} {int(r[('MRR', 'size')]):>3} {recalls} {r[('MRR', 'mean')]:>6.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
