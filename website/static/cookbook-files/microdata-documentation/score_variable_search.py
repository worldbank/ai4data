"""Score variable-level search on a known-variable question set.

Users of microdata look for variables ("age of household head"), and most
often across several surveys. This script runs each question in a question
set through a search function over the data dictionary and reports Recall@5
and MRR, overall and per language.

The built-in search is a keyword overlap over the label, the question text,
and the concept of each variable. It exists so that the script runs with no
dependencies and gives a baseline. Replace `search()` with a call to the
catalog's variable search to score it.

Uses pandas and scikit-learn (TF-IDF for the baseline search).

Usage:
    python score_variable_search.py variable_questions.csv lfs_2025q2_dictionary.csv

What this does not do: thirteen questions over one dictionary give a
demonstration. A usable set has 30 to 50 questions per language and covers
several surveys, so that the score reflects cross-survey search.
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN = re.compile(r"\w+")
TEXT_FIELDS = ("label", "question", "concept")
K = 5


class KeywordSearch:
    """A TF-IDF keyword search over the dictionary: the baseline to replace with the catalog's variable search."""

    def __init__(self, dictionary: pd.DataFrame) -> None:
        df = dictionary.fillna("").drop_duplicates("name")
        self.names = df["name"].tolist()
        texts = (
            df[list(TEXT_FIELDS)].agg(" ".join, axis=1)
            + " "
            + df["name"].str.replace("_", " ")
        )
        self.vectorizer = TfidfVectorizer(token_pattern=TOKEN.pattern, lowercase=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def __call__(self, question: str, k: int = K) -> list[str]:
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[
            0
        ]
        ranked = sorted(
            (i for i in range(len(self.names)) if scores[i] > 0),
            key=lambda i: (-scores[i], i),
        )
        return [self.names[i] for i in ranked[:k]]


def search(question: str, dictionary: list[dict[str, str]], k: int = K) -> list[str]:
    """Return up to k variable names ranked by keyword similarity. Replace with the real search."""
    return KeywordSearch(pd.DataFrame(dictionary))(question, k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("questions")
    parser.add_argument("dictionary")
    args = parser.parse_args(argv)
    find = KeywordSearch(pd.read_csv(args.dictionary, dtype=str))
    questions = pd.read_csv(args.questions, dtype=str)
    questions = questions[questions["expected_name"] != "NONE"].copy()

    print(f"{'id':<4} {'lang':<5} {'rank':>4}  question")
    ranks = []
    for q in questions.itertuples():
        ranked = find(q.question)
        rank = (
            ranked.index(q.expected_name) + 1
            if q.expected_name in ranked
            else float("nan")
        )
        ranks.append(rank)
        print(
            f"{q.question_id:<4} {q.language:<5} {'-' if pd.isna(rank) else int(rank):>4}  {q.question}"
        )
    questions["rank"] = ranks
    questions["R@5"] = (questions["rank"] <= K).astype(float)
    questions["MRR"] = (1 / questions["rank"]).fillna(0.0)
    groups = pd.concat(
        [questions.assign(group="all"), questions.assign(group=questions["language"])]
    )
    table = groups.groupby("group", sort=False).agg(
        n=("rank", "size"), r5=("R@5", "mean"), mrr=("MRR", "mean")
    )

    print()
    print(f"{'group':<6} {'n':>3} {'R@5':>6} {'MRR':>6}")
    for group in ["all"] + sorted(g for g in table.index if g != "all"):
        r = table.loc[group]
        print(f"{group:<6} {int(r.n):>3} {r.r5:>6.2f} {r.mrr:>6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
