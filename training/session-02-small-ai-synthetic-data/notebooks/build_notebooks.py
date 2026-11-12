"""Generates the session 02 notebooks and, with --execute, runs them to record their outputs.

    python build_notebooks.py             # write the .ipynb files
    python build_notebooks.py --execute   # write and execute (a GPU is expected)
    python build_notebooks.py 01          # limit either command to the notebooks whose name starts with 01

The notebooks are generated from this file so that their text can be reviewed and edited in one place.
Edit here, then regenerate. The kernel for --execute is taken from WORKSHOP_KERNEL (default python3).

Without --execute, a notebook whose code cells are unchanged keeps its recorded outputs, so a change to the
text needs no new run. A change to a code cell drops the outputs of that notebook, and the command says so.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import explainers

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
COLAB = "https://colab.research.google.com/github/worldbank/ai4data/blob/main/training/session-02-small-ai-synthetic-data/notebooks"
SESSION_LINE = "**Session 02 · Building specialized small AI using synthetic data**"


def read_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


# =============================================================================
# Pictures: animated explainers drawn at build time and embedded in Markdown cells
# =============================================================================
ALT = {
    "topic_stages": "Stages of the report-tagging pipeline: topics, sentences written by a large model, checks, training of the small model, a threshold from real abstracts, a test on real abstracts.",
    "request": "One request to the large model names eight topics. Four of the returned sentences are shown with their topics.",
    "topic_loop": "A cycle of four steps: test the model on real cases, find the topics with the lowest scores, have the large model write more sentences for them, train again.",
    "occupation_stages": "Stages of the occupation-coding pipeline: labelled job titles, answers written by a large model, a check by a second model, titles written from the definitions for the groups without a job title, training of the small model, tests.",
    "hierarchy": "The unit group 5212 Street food salespersons under minor group 521, sub-major group 52, and major group 5. The four levels of the classification have 10, 43, 130, and 436 groups.",
    "rewrite": "One job title with its code and three answers written from it by the large model. Each answer keeps the code of the title.",
    "check": "Four answers with the code of their job title and the code chosen by a second model. Answers with two different codes are removed.",
    "occupation_loop": "A cycle of four steps: test the model, find the groups with the most errors, have the large model write more answers for them, train again.",
}


def topic_pictures() -> dict[str, str]:
    overview = json.loads((ASSETS / "overview.json").read_text(encoding="utf-8"))
    sentences = read_jsonl(ASSETS / "topics/sentences.jsonl.gz")
    request = overview["synthetic"]["request"]
    mine = [s for s in sentences if s["request"] == request]
    shown = [mine[1], mine[0], mine[2], next(s for s in mine[3:] if len(s["topics"]) == 2)]
    return {
        "topic_stages": explainers.stages_svg(
            "From a list of topics to a model that tags reports",
            [
                ("data", ["86 topics"], "the list of the catalogue"),
                ("large", ["Sentences"], f"{len(sentences):,} written by a large model"),
                ("data", ["Checks"], "by script"),
                ("small", ["Training"], "the small model learns from the sentences"),
                ("data", ["Threshold"], "set on 488 real abstracts"),
                ("small", ["Test"], "on 300 real abstracts"),
            ],
        ),
        "request": explainers.request_svg(
            overview["synthetic"]["topics"], [(s["text"], s["topics"]) for s in shown], 20, len({s["request"] for s in sentences})
        ),
        "topic_loop": explainers.loop_svg(
            "One round of improvement",
            [
                ("small", ["Test the model", "on real abstracts"]),
                ("data", ["Find the topics", "with the lowest scores"]),
                ("large", ["Write more sentences", "for those topics"]),
                ("small", ["Train again", "on all the sentences"]),
            ],
        ),
    }


STYLES = ["Short answer", "Job and place of work", "Tasks in everyday words"]
# Names of the few groups that the pictures show. The notebooks download the full list from the ILO.
PICTURE_NAMES = {"2221": "Nursing professionals"}
# Answers shown in the picture of the check: (id of the job title, kind of answer).
CHECK_ROWS = [("hahu-04276", 2), ("hahu-05631", 2), ("hahu-04078", 1), ("hahu-04438", 2)]


def occupation_pictures() -> dict[str, str]:
    answers = read_jsonl(ASSETS / "occupation/answers.jsonl.gz")
    nurse = sorted((a for a in answers if a["id"] == "hahu-04276"), key=lambda a: a["style"])
    assert len(nurse) == 3 and all(a["kept"] for a in nurse)
    by_key = {(a["id"], a["style"]): a for a in answers}
    rows = [by_key[k] for k in CHECK_ROWS]
    assert [a["kept"] for a in rows] == [True, False, False, True] and rows[1]["second_code"] and not rows[2]["second_code"]
    return {
        "occupation_stages": explainers.stages_svg(
            "From labelled job titles to a model that codes survey answers",
            [
                ("data", ["Job titles"], "labelled, from job advertisements"),
                ("large", ["Answers"], "written by a large model from the titles"),
                ("large", ["Check"], "by a second large model"),
                ("large", ["Other groups"], "titles written from the definitions"),
                ("small", ["Training"], "the small model learns from the three sets"),
                ("small", ["Tests"], "on three sets kept apart"),
            ],
        ),
        "hierarchy": explainers.hierarchy_svg(
            [
                ("5", "Service and sales workers", 10, "major groups"),
                ("52", "Sales workers", 43, "sub-major groups"),
                ("521", "Street and market salespersons", 130, "minor groups"),
                ("5212", "Street food salespersons", 436, "unit groups"),
            ]
        ),
        "rewrite": explainers.rewrite_svg(nurse[0]["title"], nurse[0]["code"], PICTURE_NAMES[nurse[0]["code"]], [a["answer"] for a in nurse], STYLES),
        "check": explainers.check_svg([(a["answer"], a["title"], a["code"], a["second_code"] or None) for a in rows]),
        "occupation_loop": explainers.loop_svg(
            "One round of improvement",
            [
                ("small", ["Test the model", "on real answers"]),
                ("data", ["Find the groups", "with the most errors"]),
                ("large", ["Write more texts", "for those groups"]),
                ("small", ["Train again", "on all the texts"]),
            ],
        ),
    }


def occupation_facts() -> dict[str, str]:
    """Numbers in the text of notebook 01, computed from the stored files."""
    titles = read_jsonl(ASSETS / "occupation/titles.jsonl.gz")
    answers = read_jsonl(ASSETS / "occupation/answers.jsonl.gz")
    hand = json.loads((ASSETS / "occupation/illustrative-answers.json").read_text(encoding="utf-8"))
    major = [t["code"][0] for t in titles]
    upper = sum(m in "123" for m in major) / len(major)
    kept = {k: sum(a["kept"] for a in answers if a["style"] == k) / sum(a["style"] == k for a in answers) for k in (1, 2, 3)}
    groups = len({t["code"] for t in titles})
    # Test texts that also occur among the training texts, compared without regard to capital letters and spacing.
    plain = lambda text: " ".join(text.lower().split())
    written = read_jsonl(ASSETS / "occupation/definition-titles.jsonl.gz")
    test_titles = read_jsonl(ASSETS / "occupation/test-titles.jsonl.gz")
    test_answers = read_jsonl(ASSETS / "occupation/test-answers.jsonl.gz")
    assert not {t["id"] for t in test_titles} & {t["id"] for t in titles}
    board = {plain(t["title"]) for t in titles}
    trained = board | {plain(w["title"]) for w in written} | {plain(a["answer"]) for a in answers if a["kept"]}
    assert not any(plain(a["answer"]) in trained for a in test_answers)
    return {
        "@@F:same_titles@@": str(sum(plain(t["title"]) in trained for t in test_titles)),
        "@@F:same_board@@": str(sum(plain(t["title"]) in board for t in test_titles)),
        "@@F:same_hand@@": str(sum(plain(h["answer"]) in trained for h in hand)),
        "@@F:hand_missing@@": str(sum(h["code"] not in {t["code"] for t in titles} for h in hand)),
        "@@F:titles@@": f"{len(titles):,}",
        "@@F:classes@@": str(groups),
        "@@F:missing@@": str(436 - groups),
        "@@F:hand@@": str(len(hand)),
        "@@F:hand_step@@": f"{1 / len(hand):.2f}",
        "@@F:major_text@@": (
            f"Major groups 1, 2, and 3 (managers, professionals, and technicians) hold {upper:.0%} of the titles. "
            f"Major group 6 (skilled agricultural, forestry, and fishery workers) has {major.count('6')} titles, "
            f"and major group 9 (elementary occupations) has {major.count('9')}."
        ),
        "@@F:kept_text@@": (
            f"The check keeps {kept[1]:.0%} of the short answers, {kept[2]:.0%} of the answers with a place of work, "
            f"and {kept[3]:.0%} of the answers that describe tasks. A short answer carries the least information, "
            "so the second model more often replies that none of the listed groups fits."
        ),
    }


def pictures() -> dict[str, str]:
    return {**topic_pictures(), **occupation_pictures()}


_PICTURES: dict[str, str] = {}


def explainer_md(name: str) -> str:
    if not _PICTURES:
        _PICTURES.update(pictures())
    svg = _PICTURES[name]
    return f"![{ALT[name]}](data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()})"


def md(text: str, id: str | None = None):
    """A Markdown cell. `id` gives the cell a stable id, which a `#scrollTo=<id>` link targets in Colab."""
    text = re.sub(r"@@EXPLAINER:(\w+)@@", lambda m: explainer_md(m.group(1)), text)
    cell = new_markdown_cell(text.strip())
    if id:
        cell["id"] = id
        cell["metadata"]["id"] = id
    return cell


def code(text: str):
    return new_code_cell(text.strip())


SETUP = '''# Setup. In Colab this clones the repository and installs the packages; locally it uses the folder as is.
import os, sys, subprocess, pathlib, warnings

REPO_URL = "https://github.com/worldbank/ai4data.git"
BRANCH = "main"   # the branch that the Colab setup clones
SESSION = "training/session-02-small-ai-synthetic-data/notebooks"

if not pathlib.Path("visuals.py").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO_URL, "/content/ai4data"], check=True)
    os.chdir(f"/content/ai4data/{SESSION}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, os.getcwd())

# Optional: a key for an OpenAI-compatible API, used by one cell that writes new examples. The notebook runs without it.
try:
    from google.colab import userdata  # type: ignore
    for name in ("WORKSHOP_API_KEY", "WORKSHOP_BASE_URL", "WORKSHOP_MODEL"):
        if userdata.get(name):
            os.environ[name] = userdata.get(name)
except Exception:
    pass

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # keeps the download messages out of the outputs
warnings.filterwarnings("ignore")

import datasets, torch, transformers
import visuals as v

datasets.disable_progress_bars()
transformers.logging.set_verbosity_error()

ASSETS = pathlib.Path("../assets")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU. No GPU was found, so the notebook trains on a sample.")
print("transformers", transformers.__version__, "| torch", torch.__version__.split("+")[0])'''

RUN_NOTE = """## How to run it

Run the cells in order. The training cell needs a GPU. In Colab, the menu **Runtime → Change runtime type** offers a T4 GPU at no cost. Without a GPU the notebook trains on a sample of the examples, and the scores are then lower than the recorded ones.

No API key is needed. The examples that the large model wrote are stored in the repository, and the notebook reads them from there. One optional cell sends a new request to a large model when a key is set."""


# =============================================================================
# 02 · Tagging reports with topics
# =============================================================================
NB02 = [
    md(f"""
# 02 · Tagging reports with topics

{SESSION_LINE} **· notebook 2 of 2**

## Objective

A catalogue of reports assigns topics to each report, so that a reader can find every report on water or on labor markets. A specialist who assigns the topics by hand reads each report, and a large collection holds hundreds of thousands of reports. This notebook builds a small model that reads the abstract of a report and assigns up to five of 86 topics.

The model is trained without a single labelled report. A large language model wrote 20,296 sentences of the kind found in development reports and listed the topics of each sentence. The small model learns from those sentences. Real abstracts with the topics of the catalogue enter at two later points: 488 of them set one number, the threshold, and 300 others test the result.

@@EXPLAINER:topic_stages@@

## Contents

1. [The task and the topics](#scrollTo=section-1) loads the 86 topics.
2. [Sentences written by a large model](#scrollTo=section-2) shows one request and the sentences that came back.
3. [Checks on the sentences](#scrollTo=section-3) counts repeated sentences, topics outside the list, and the sentences of each topic.
4. [Training the small model](#scrollTo=section-4) fine-tunes a pre-trained model on the sentences.
5. [From sentences to a report](#scrollTo=section-5) scores each sentence of an abstract and takes the mean.
6. [The threshold](#scrollTo=section-6) is chosen on 488 real abstracts.
7. [The test on real abstracts](#scrollTo=section-7) measures the model on 300 abstracts and compares it with large models.
8. [Size and speed](#scrollTo=section-8) measures the model and the time for one abstract.
9. [Rounds of improvement and limits](#scrollTo=section-9) describes the steps that follow a first model.

{RUN_NOTE}
"""),
    md("""
The next cell prepares the environment and reports the device. The first run in Colab installs the packages and takes longer than a later run. The cell is setup only.
"""),
    code(SETUP),
    md("""
The notebook uses four widely used libraries and one small helper file.

| Need | Library |
|---|---|
| Tables | [pandas](https://pandas.pydata.org/) |
| The pre-trained model, its tokenizer, and the training loop | [Transformers](https://huggingface.co/docs/transformers) |
| Examples in the form that the training loop reads | [Datasets](https://huggingface.co/docs/datasets) |
| Scores | [scikit-learn](https://scikit-learn.org/) |

`visuals.py` draws the pictures from the results of the cells. It holds no part of the pipeline.
"""),
    md("""
## 1. The task and the topics

The task has a fixed list of answers: the 86 topics of the catalogue. A report can have several topics, and a topic can have a parent topic. The next cell loads the list and shows eight rows of it.
""", id="section-1"),
    code('''
import pandas as pd

topics = pd.read_csv(ASSETS / "topics/topics.csv")
TOPICS = topics["topic"].tolist()
print(len(TOPICS), "topics.", int(topics["parent"].notna().sum()), "of them have a parent topic.")
topics.sample(8, random_state=1).fillna("")
'''),
    md("""
A task of this kind is called multi-label classification: for each topic, the model answers whether the topic applies to the text. The model therefore gives 86 probabilities for a text, one for each topic.
"""),
    md("""
## 2. Sentences written by a large model

Training needs texts whose topics are known. The organization in this example has no such texts, so a large language model writes them. One request names eight topics drawn at random and asks for 20 sentences, each with the list of its topics. The picture shows one request and four of the sentences that came back.

@@EXPLAINER:request@@

The next cell prints the text of that request. Its rules ask for varied lengths and styles, because a model trained on uniform sentences handles real text poorly. They also ask that a third of the sentences cover two or three topics, because real reports do.
""", id="section-2"),
    code('''
import json

overview = json.loads((ASSETS / "overview.json").read_text(encoding="utf-8"))
PROMPT = overview["synthetic"]["prompt"]["allow"]
print(PROMPT)
'''),
    md("""
A thousand requests of this form were sent to an open large model (Qwen3.8-27B) on a server, and the replies were stored in the repository. The next cell loads them and prints six sentences of the request shown above.
"""),
    code('''
sentences = pd.read_json(ASSETS / "topics/sentences.jsonl.gz", lines=True)
print(f"{len(sentences):,} sentences from {sentences['request'].nunique():,} requests\\n")
for row in sentences[sentences["request"] == overview["synthetic"]["request"]].head(6).itertuples():
    print("-", row.text)
    print("     topics:", ", ".join(row.topics))
'''),
    md("""
The next cell is optional. With a key for an OpenAI-compatible API in `WORKSHOP_API_KEY` and a model name in `WORKSHOP_MODEL`, it sends the same request to that model and prints five of the new sentences. Without a key the cell prints a message and the notebook continues with the stored sentences.
"""),
    code('''
if os.environ.get("WORKSHOP_API_KEY"):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["WORKSHOP_API_KEY"], base_url=os.environ.get("WORKSHOP_BASE_URL"))
    form = '\\n\\nAnswer with JSON of this form: {"examples": [{"text": "...", "labels": ["..."]}]}'
    reply = client.chat.completions.create(
        model=os.environ["WORKSHOP_MODEL"],
        messages=[{"role": "user", "content": PROMPT + form}],
        response_format={"type": "json_object"},
        temperature=1.0,
    )
    for item in json.loads(reply.choices[0].message.content)["examples"][:5]:
        print("-", item["text"], "\\n     topics:", ", ".join(item["labels"]))
else:
    print("No key is set. The notebook continues with the stored sentences.")
'''),
    md("""
## 3. Checks on the sentences

A large model follows a request imperfectly. It can repeat a sentence, invent a topic outside the list, or return a sentence without a topic. A short script finds these cases before training. The next cell runs the three checks and then counts the sentences of each topic.
""", id="section-3"),
    code('''
by_topic = sentences.explode("topics")

print("Repeated sentences:           ", int(sentences["text"].duplicated().sum()))
print("Sentences without a topic:    ", int((sentences["topics"].map(len) == 0).sum()))
print("Topics outside the list of 86:", int((~by_topic["topics"].isin(TOPICS)).sum()))

per_topic = by_topic["topics"].value_counts()
print(f"\\nSentences per topic: {per_topic.min()} to {per_topic.max()}, median {int(per_topic.median())}")
print("Topics per sentence:", sentences["topics"].map(len).value_counts().sort_index().to_dict())
'''),
    md("""
The three checks find nothing to remove in this set. Every topic has at least 197 sentences, because the requests drew the topics at random with equal chances. About a third of the sentences have one topic, and about six in ten have two.

The checks concern form only. A script cannot tell whether a sentence is really about its topics. For a task with one answer per text, a second large model can label each text again, and texts with two different answers are removed. Notebook 01 uses that check.
"""),
    md("""
## 4. Training the small model

The small model starts from DeBERTa-v3-small, a model with 142 million parameters that was pre-trained on a large amount of English text. Pre-training taught it how words relate in a sentence. It knows nothing about the 86 topics. Fine-tuning continues the training on the sentences of section 3, with a new output layer of 86 values, one for each topic.

A model reads numbers, so a tokenizer first cuts each text into pieces called tokens and replaces each token with its number in the model's vocabulary. The next cell loads the tokenizer and shows the tokens of one short sentence.
""", id="section-4"),
    code('''
from transformers import AutoTokenizer

BASE = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(BASE)

example = "Mobile money accounts raised household savings in rural districts."
encoded = tokenizer(example)
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
print(encoded["input_ids"])
'''),
    md("""
The answer for a sentence also becomes numbers: a row of 86 values with 1 at the topics of the sentence and 0 elsewhere. The next cell defines the two conversions and sets one request in ten aside. The model never trains on the sentences set aside, so the loss on them shows whether it learns patterns that carry over to new sentences. The split is by request: all sentences of a request go to the same side, because sentences from one reply resemble each other.
"""),
    code('''
import numpy as np
from datasets import Dataset

position = {topic: i for i, topic in enumerate(TOPICS)}


def targets(topic_lists):
    """One row per text and one column per topic: 1 where the topic applies, 0 elsewhere."""
    table = np.zeros((len(topic_lists), len(TOPICS)), dtype=np.float32)
    for row, names in enumerate(topic_lists):
        table[row, [position[name] for name in names]] = 1
    return table


def encode(frame):
    data = Dataset.from_dict({"text": frame["text"].tolist(), "labels": targets(frame["topics"]).tolist()})
    return data.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=128), batched=True)


requests = sentences["request"].drop_duplicates().sample(frac=1, random_state=0)
set_aside = set(requests[: len(requests) // 10])
train_part = sentences[~sentences["request"].isin(set_aside)]
check_part = sentences[sentences["request"].isin(set_aside)]
if DEVICE == "cpu":   # a sample, to shorten training without a GPU
    train_part, check_part = train_part.sample(2000, random_state=0), check_part.sample(300, random_state=0)
print(f"{len(train_part):,} sentences for training, {len(check_part):,} set aside")
'''),
    md("""
The next cell trains the model with the `Trainer` of the Transformers library. The settings are common defaults for fine-tuning a model of this size: four passes over the sentences (epochs), 32 sentences per step, and a learning rate of 0.00005 that rises during the first steps and then falls to zero. The recorded run below was made on one NVIDIA H100 GPU, and the last line of the output gives its duration. The cell takes longer on a Colab T4 GPU.

The table that the cell displays has one row per epoch. The training loss measures the distance between the model's answers and the right answers on the training sentences. The validation loss measures the same distance on the sentences set aside.
"""),
    code('''
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

set_seed(42)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(TOPICS), problem_type="multi_label_classification"   # 86 independent yes-or-no answers
)
settings = TrainingArguments(
    output_dir="topic-model",
    num_train_epochs=4 if DEVICE == "cuda" else 1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    report_to="none",
    disable_tqdm=False,   # shows the progress bar and the table of losses
    seed=42,
)
trainer = Trainer(
    model=model,
    args=settings,
    train_dataset=encode(train_part),
    eval_dataset=encode(check_part),
    data_collator=DataCollatorWithPadding(tokenizer),   # pads the sentences of a step to the same length
)
run = trainer.train()
print(f"Training took {run.metrics['train_runtime']:.0f} seconds.")
assert np.isfinite(run.training_loss), "The loss is not a number. See the note on library versions below."
'''),
    md("""
Both losses fall in every epoch, and the validation loss stays close to the training loss. The model therefore learned patterns that hold for sentences it had not seen. A validation loss that rises while the training loss falls would mean that the model memorizes its training sentences, and training should then stop earlier.

A note on library versions: `requirements.txt` asks for a version of Transformers below 5. In a test for this session, DeBERTa-v3-small trained normally with version 4.57 and returned a loss that was not a number with versions 5.20 and 5.22. The last line of the cell stops the notebook if that happens.
"""),
    md("""
## 5. From sentences to a report

The model learned from single sentences, and an abstract has several. The next cell therefore splits an abstract into sentences, asks the model for the 86 probabilities of each sentence, and takes the mean over the sentences. A topic that several sentences support gets a high mean. A topic that one sentence mentions in passing gets a low one.
""", id="section-5"),
    code('''
import re

import torch

BOUNDARY = re.compile(r"(?<=[.!?])\\s+(?=[A-Z0-9\\"'(])")   # a sentence ends at . ! or ? followed by a capital letter or a digit


def split_sentences(text):
    parts = [part.strip() for part in BOUNDARY.split(" ".join(text.split()))]
    return [part for part in parts if len(part.split()) >= 4] or [text.strip()]


def sentence_scores(texts, batch=64):
    """The 86 probabilities of each text."""
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            inputs = tokenizer(texts[start:start + batch], truncation=True, max_length=128, padding=True, return_tensors="pt").to(model.device)
            out.append(torch.sigmoid(model(**inputs).logits).float().cpu().numpy())
    return np.concatenate(out)


def report_scores(texts):
    """The 86 probabilities of each abstract: the mean over its sentences."""
    return np.stack([sentence_scores(split_sentences(text)).mean(axis=0) for text in texts])
'''),
    md("""
The reports for the threshold and for the test are abstracts of Policy Research Working Papers published from 2016 to 2020, taken from the World Bank's Documents and Reports database. The topics of each report come from the subjects that the database lists for it. The next cell loads both files and shows the path of one test abstract through the model. The abstract reviews studies of the way a change in a central bank's interest rate reaches the lending rates of banks. The picture is animated: the sentences appear with their most probable topic, and the bars then show the mean over the sentences for the six best topics.
"""),
    code('''
threshold_reports = pd.read_json(ASSETS / "topics/reports-threshold.jsonl.gz", lines=True, dtype={"id": str})
test_reports = pd.read_json(ASSETS / "topics/reports-test.jsonl.gz", lines=True, dtype={"id": str})
print(len(threshold_reports), "abstracts for the threshold,", len(test_reports), "abstracts for the test")

report = test_reports.iloc[@@REPORT@@]
parts = split_sentences(report["text"])
scores = sentence_scores(parts)
mean = scores.mean(axis=0)
best = np.argsort(-mean)[:6]

v.report_view(
    report["title"],
    parts,
    tops=[(TOPICS[row.argmax()], float(row.max())) for row in scores],
    means=[(TOPICS[i], float(mean[i])) for i in best],
    threshold=0.05,
    catalogue=report["topics"],
)
'''),
    md("""
The means are much lower than the probabilities of single sentences, because each sentence is about one or two topics and adds almost nothing to the other 84. A report therefore gets a topic when its mean passes a low threshold. The picture uses 0.05, and the next section explains where that number comes from.

In the recorded run the model assigns three topics to this abstract. Financial Sector and Macroeconomics are also in the catalogue record. Financial Stability is absent from the record, so the test counts it as an error, although a reader could defend it for a paper on bank lending rates. The record also lists Trade, which no sentence of the abstract mentions. A model that reads the abstract alone cannot find that topic.
"""),
    md("""
## 6. The threshold

The threshold is the one setting that needs real reports with known topics. It is chosen on 488 abstracts that play no part in the test. For each threshold in a short list, the next cell assigns the topics whose mean is at or above the threshold, five at most, and compares them with the topics of the catalogue.

The comparison uses three scores. Precision is the share of assigned topics that the catalogue also lists. Recall is the share of the catalogue's topics that the model assigned. F1 combines the two, and it is high only when both are high. The scores are computed over all reports and topics together, which is called the micro average.
""", id="section-6"),
    code('''
from sklearn.metrics import precision_recall_fscore_support


def assign(scores, threshold, at_most=5):
    """1 for each topic that a report gets: mean at or above the threshold, and among the five best."""
    chosen = np.zeros_like(scores, dtype=int)
    for row, order in enumerate(np.argsort(-scores, axis=1)[:, :at_most]):
        chosen[row, order[scores[row, order] >= threshold]] = 1
    return chosen


def evaluate(scores, reports, threshold):
    truth = targets(reports["topics"]).astype(int)
    measured = truth.sum(axis=0) > 0          # topics that occur in these reports
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth[:, measured], assign(scores, threshold)[:, measured], average="micro", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "topics measured": int(measured.sum())}


threshold_scores = report_scores(threshold_reports["text"])
GRID = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
f1_at = [evaluate(threshold_scores, threshold_reports, g)["f1"] for g in GRID]
THRESHOLD = GRID[int(np.argmax(f1_at))]
v.threshold_chart(GRID, f1_at, THRESHOLD)
'''),
    md("""
The score is highest at a low threshold and falls quickly above 0.1. At the common default of 0.5 the model assigns almost no topic, because a mean over many sentences rarely reaches one half. A model trained on sentences and used on whole abstracts needs this calibration step, and a few hundred real cases are enough for it.
"""),
    md("""
## 7. The test on real abstracts

The 300 test abstracts took no part in training or in the choice of the threshold. The next cell scores them once, with the threshold of section 6.
""", id="section-7"),
    code('''
test_scores = report_scores(test_reports["text"])
result = evaluate(test_scores, test_reports, THRESHOLD)
print(f"Threshold {THRESHOLD:g}, at most 5 topics per report, {result['topics measured']} topics measured")
print(f"Precision {result['precision']:.3f}   Recall {result['recall']:.3f}   F1 {result['f1']:.3f}")
'''),
    md("""
In the recorded run, about half of the assigned topics are in the catalogue (precision), and the model finds about four in ten of the catalogue's topics (recall). The catalogue can omit a topic that an abstract covers, and such a case counts as an error of the model. The 300 abstracts cover 44 of the 86 topics, so the test says nothing about the other 42.

The next cell places the result beside recorded results on the same 300 abstracts. Three large models received each abstract with the list of topics and were asked for the topics directly. The fourth recorded result is the small model after five rounds of improvement, which section 9 describes.
"""),
    code('''
recorded = overview["workflow"]
rows = [("Small model of this notebook\\ntrained on the first set of sentences", result["f1"], "small")]
after_rounds = sum(r["arms"][2]["f1"] for r in recorded["rounds"]) / len(recorded["rounds"])
rows += [("Small model after five further rounds\\nrecorded, mean of two runs", after_rounds, "small")]
rows += [(f"{m['model']}\\nlarge model asked directly, recorded", m["f1"], "large") for m in recorded["large"]]
v.result_bars("Micro F1 on the 300 test abstracts", rows, top_value=0.6)
'''),
    md("""
In the recorded run, the small model of this notebook is about 0.07 below the best large model. After the recorded rounds of improvement the small model is 0.01 below it. A new run of this notebook can give a slightly different score, because the result of training depends on the GPU and on the versions of the libraries.

The overall score hides differences between topics. The next cell computes F1 for each topic with at least eight test abstracts and prints the five best and the five weakest.
"""),
    code('''
truth = targets(test_reports["topics"]).astype(int)
assigned = assign(test_scores, THRESHOLD)
_, _, f1_topic, support = precision_recall_fscore_support(truth, assigned, average=None, zero_division=0)
table = pd.DataFrame({"topic": TOPICS, "test abstracts": support, "assigned by the model": assigned.sum(axis=0), "F1": f1_topic.round(2)})
table = table[table["test abstracts"] >= 8].sort_values("F1", ascending=False)
pd.concat([table.head(5), table.tail(5)]).set_index("topic")
'''),
    md("""
The weakest topics of the recorded run show two kinds of problem. The model assigns Fragility, Conflict, and Violence to 57 abstracts, and the catalogue lists it for 17. Synthetic sentences that describe a topic too broadly produce this pattern. The model assigns Water Resources Management to one abstract, and the catalogue lists it for 19. This second pattern appears when real abstracts use wording that is absent from the synthetic sentences. Both findings tell the team which topics need more sentences or better ones.
"""),
    md("""
## 8. Size and speed

The reasons for building a small model are practical: it fits on an ordinary computer, and it answers quickly. The next cell counts the parameters of the trained model, computes the memory that they need, and times the model on the 300 test abstracts.
""", id="section-8"),
    code('''
import time

parameters = sum(p.numel() for p in model.parameters())
print(f"{parameters / 1e6:.0f} million parameters, {parameters * 4 / 1e9:.2f} GB at 4 bytes per parameter")

start = time.perf_counter()
report_scores(test_reports["text"])
seconds = (time.perf_counter() - start) / len(test_reports)
print(f"{seconds * 1000:.1f} milliseconds per abstract on this device ({DEVICE})")
'''),
    md("""
A large model of 27 billion parameters needs about 31 GB of memory in an 8-bit version and a GPU of that size. The model of this notebook needs about half a gigabyte and also runs on a laptop without a GPU. In a recorded measurement on four CPU threads, it took about 0.2 seconds per abstract.
"""),
    md("""
## 9. Rounds of improvement and limits

The model of this notebook is a first version. The process of the session continues with rounds of improvement.

@@EXPLAINER:topic_loop@@

In two recorded runs, five rounds raised the F1 score on the test abstracts from 0.47 to 0.52 on average. Each round tested the model on real abstracts kept for that purpose, selected the topics with the lowest scores, and had the large model write 3,000 further sentences for them. A control added the same number of sentences for topics drawn at random and reached 0.48. The gain therefore came from the choice of topics. The larger number of sentences alone added about 0.01.

These results have four limits.

- The right answers are the topics of a catalogue record. A record can omit a topic that the abstract covers.
- The test covers 44 of the 86 topics.
- All abstracts are working papers in English from one series. Other kinds of report and other languages need their own test.
- The model assigns topics from the abstract only. It has no information on the rest of the report.

Before such a model enters use, people who know the catalogue should compare its topics with their own on a sample of reports, and the test should be repeated when new kinds of reports arrive.

[Notebook 01]({COLAB}/01-occupation-coding.ipynb) builds a model for a second task, coding the answers to a survey question on occupation, and adds a check by a second large model.
""".replace("{COLAB}", COLAB), id="section-9"),
]


# =============================================================================
# 01 · Coding survey answers on occupation
# =============================================================================
NB01 = [
    md(f"""
# 01 · Coding survey answers on occupation

{SESSION_LINE} **· notebook 1 of 2**

## Objective

A labour force survey asks each respondent about the main job, and the interviewer writes the answer down in the respondent's words. A coder then assigns each answer a code of the International Standard Classification of Occupations (ISCO-08), which has 436 detailed groups. This notebook builds a small model that reads an answer and suggests five groups, so that the coder chooses from a short list.

The organization in this example has no survey answers with codes. It has @@F:titles@@ job titles from job advertisements, each with a code. An advertisement words a job differently from a respondent. A large language model therefore rewrites each title as three answers of the kind a respondent gives, and each answer keeps the code of its title. A second large model checks the answers.

The advertisements cover @@F:classes@@ of the 436 groups. For the full list of groups, a large model also wrote job titles from the definitions of the classification. The small model learns from the three sets of texts.

@@EXPLAINER:occupation_stages@@

## Contents

1. [The task and the classification](#scrollTo=section-1) loads the groups of ISCO-08.
2. [Labelled job titles from another source](#scrollTo=section-2) loads the job titles and counts them by major group.
3. [Survey-style answers written by a large model](#scrollTo=section-3) shows the request and the answers that came back.
4. [The check by a second model](#scrollTo=section-4) removes the answers that a second model codes differently.
5. [Groups without a job title](#scrollTo=section-5) adds titles that a large model wrote from the definitions.
6. [Training the small model](#scrollTo=section-6) fine-tunes a pre-trained model on the three sets.
7. [Three tests](#scrollTo=section-7) measure the model and compare it with recorded results for other training sets.
8. [Suggestions for one answer](#scrollTo=section-8) shows the five groups that the model suggests for a typed answer.
9. [Limits and rounds of improvement](#scrollTo=section-9) lists the questions that the tests leave open.

{RUN_NOTE}
"""),
    md("""
The next cell prepares the environment and reports the device. The first run in Colab installs the packages and takes longer than a later run. The cell is setup only.
"""),
    code(SETUP),
    md("""
The notebook uses four widely used libraries and one small helper file.

| Need | Library |
|---|---|
| Tables | [pandas](https://pandas.pydata.org/) |
| The pre-trained model, its tokenizer, and the training loop | [Transformers](https://huggingface.co/docs/transformers) |
| Examples in the form that the training loop reads | [Datasets](https://huggingface.co/docs/datasets) |
| The download of the classification | [Requests](https://requests.readthedocs.io/) |

`visuals.py` draws the pictures from the results of the cells. It holds no part of the pipeline.
"""),
    md("""
## 1. The task and the classification

ISCO-08 arranges occupations on four levels. Ten major groups divide into 43 sub-major groups, 130 minor groups, and 436 unit groups. A code has one digit for each level, so the unit group 5212 lies in major group 5, sub-major group 52, and minor group 521. A statistical office codes each survey answer to a unit group.

@@EXPLAINER:hierarchy@@

The next cell downloads the classification from the website of the International Labour Organization and keeps the name of each group. The notebook uses the names to display results. If the download fails, the notebook continues and shows codes without names.
""", id="section-1"),
    code('''
import io

import pandas as pd
import requests

ISCO_URL = "https://webapps.ilo.org/ilostat-files/ISCO/newdocs-08-2021/ISCO-08/ISCO-08%20EN%20Structure%20and%20definitions.xlsx"
LEVELS = {"1": "major groups", "2": "sub-major groups", "3": "minor groups", "4": "unit groups"}

try:
    reply = requests.get(ISCO_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    reply.raise_for_status()
    isco = pd.read_excel(io.BytesIO(reply.content), dtype=str)
    isco = isco.rename(columns={"Level": "level", "ISCO 08 Code": "code", "Title EN": "name"})[["level", "code", "name"]]
    print(", ".join(f"{count} {LEVELS[level]}" for level, count in isco["level"].value_counts().sort_index().items()))
except Exception as error:
    print("The classification could not be downloaded. Results show codes without names.", error)
    isco = pd.DataFrame(columns=["level", "code", "name"])

NAMES = dict(zip(isco["code"], isco["name"]))


def group_name(code):
    return NAMES.get(code, "")


isco[isco["code"].str.startswith("52")].set_index("code")
'''),
    md("""
The table shows the part of the classification under sub-major group 52, Sales Workers. Neighbouring unit groups differ in details that a short answer can leave out. The answer "sells food" fits a street food salesperson (5212), a stall salesperson at a market (5211), and a shop assistant (5223). A list of five suggestions suits this situation: the coder sees the candidates side by side and chooses with the rest of the questionnaire at hand.
"""),
    md("""
## 2. Labelled job titles from another source

Training needs texts whose code is known, and the organization in this example has no coded survey answers. A public dataset supplies a related kind of text: job titles from the job board hahu.jobs in Ethiopia, published by the Tabiya project under the CC BY 4.0 licence. The job board assigned each title an occupation of the European classification ESCO, and the first four digits of an ESCO code are the ISCO-08 unit group. No trained coder reviewed these codes.

The next cell loads the titles kept for training and shows ten of them with their groups.
""", id="section-2"),
    code('''
titles = pd.read_json(ASSETS / "occupation/titles.jsonl.gz", lines=True, dtype=str)
print(f"{len(titles):,} job titles in {titles['code'].nunique()} of the 436 unit groups")

sample = titles.sample(10, random_state=3)
sample.assign(group=sample["code"].map(group_name))[["title", "code", "group"]].set_index("title")
'''),
    md("""
A title in an advertisement names a position, often with a grade, a department, or a note from the employer. A respondent describes the work: "teacher at the primary school", "sell vegetables at the market". A model trained on titles alone has seen few texts of the second kind.

The titles also cover the labour market unevenly. The next cell counts them by major group.
"""),
    code('''
by_major = titles["code"].str[0].value_counts().sort_index()
v.result_bars(
    "Job titles by major group",
    [(f"{digit} · {group_name(digit) or 'major group ' + digit}", count, "muted") for digit, count in by_major.items()],
    fmt="{:,.0f}", label_chars=54,
)
'''),
    md("""
@@F:major_text@@ A job board advertises salaried positions. A household survey also records farmers, street vendors, and domestic workers, and the job board carries few advertisements for such work. Sections 3 and 4 change the wording of the titles and keep their groups. Section 5 adds the groups that are absent.
"""),
    md("""
## 3. Survey-style answers written by a large model

The large model receives one job title with the name and the definition of its group. It writes three answers that a respondent with this job could give: a short answer of one to four words, an answer that adds the place of work, and an answer that describes the tasks without the job title. Each answer takes the code of its title, so no person codes the new texts.

@@EXPLAINER:rewrite@@

The next cell prints the request. Its rules ask for the plain words of a respondent and exclude a copy of the title.
""", id="section-3"),
    code('''
import json

results = json.loads((ASSETS / "occupation/results.json").read_text(encoding="utf-8"))
PROMPT = results["prompt"]
print(PROMPT)
'''),
    md("""
The request was sent once for each title to an open large model (Qwen3.8-27B) on a server, and the replies are stored in the repository. The next cell loads them and prints the answers for three titles. Each line ends with the result of the check that section 4 describes.
"""),
    code('''
answers = pd.read_json(ASSETS / "occupation/answers.jsonl.gz", lines=True, dtype={"id": str, "code": str, "second_code": str})
print(f"{len(answers):,} answers for {answers['id'].nunique():,} job titles\\n")


def show_answers(title):
    group = answers[answers["title"] == title]
    code = group["code"].iloc[0]
    print(f"{title}   [{code} {group_name(code)}]")
    for row in group.itertuples():
        print(f"   {row.style}. {row.answer}   ({'kept' if row.kept else 'removed'} in section 4)")
    print()


for title in ["Pediatric Nurse & NICU", "Security Officers", "Construction Surveyor"]:
    show_answers(title)
'''),
    md("""
The next cell is optional. With a key for an OpenAI-compatible API in `WORKSHOP_API_KEY` and a model name in `WORKSHOP_MODEL`, it sends the request for one job title to that model and prints the three answers. Without a key the cell prints a message and the notebook continues with the stored answers.
"""),
    code('''
if os.environ.get("WORKSHOP_API_KEY"):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["WORKSHOP_API_KEY"], base_url=os.environ.get("WORKSHOP_BASE_URL"))
    one = titles.iloc[0]
    # The recorded run also sent the definition of the group. The name alone is enough for a trial.
    request = PROMPT.format(title=one["title"], label=f"{one['code']} {group_name(one['code'])}")
    reply = client.chat.completions.create(
        model=os.environ["WORKSHOP_MODEL"],
        messages=[{"role": "user", "content": request + '\\n\\nAnswer with JSON of this form: {"answers": ["...", "...", "..."]}'}],
        response_format={"type": "json_object"},
        temperature=1.0,
    )
    print(one["title"])
    for answer in json.loads(reply.choices[0].message.content)["answers"]:
        print("   -", answer)
else:
    print("No key is set. The notebook continues with the stored answers.")
'''),
    md("""
## 4. The check by a second model

An answer can move away from the code of its title. A short answer such as "driver" fits several groups, and the writer can describe a neighbouring occupation. A second large model (Gemma 4 31B) therefore reads each answer without the title and without the code. It chooses among the group of the title, the groups next to it in the classification, groups drawn at random, and the reply that none of the listed groups fits. An answer is kept when the second model chooses the group of the title.

@@EXPLAINER:check@@

The next cell counts the answers kept and removed for each of the three kinds of answer.
""", id="section-4"),
    code('''
KINDS = {1: "Short answer", 2: "Job and place of work", 3: "Tasks in everyday words"}

counts = answers.groupby("style")["kept"].agg(["sum", "size"])
v.kept_removed([(KINDS[kind], int(row["sum"]), int(row["size"] - row["sum"])) for kind, row in counts.iterrows()])
print(f"{int(answers['kept'].sum()):,} of {len(answers):,} answers are kept.")
'''),
    md("""
@@F:kept_text@@ The next cell prints six removed answers with the group that the second model chose.
"""),
    code('''
for row in answers[~answers["kept"]].sample(6, random_state=2).itertuples():
    second = f"{row.second_code} {group_name(row.second_code)}" if row.second_code else "none of the listed groups"
    print(f"“{row.answer}”   from the job title “{row.title}”")
    print(f"     group of the title:  {row.code} {group_name(row.code)}")
    print(f"     second model:        {second}\\n")
'''),
    md("""
The removed answers are of three kinds. Some describe a neighbouring occupation. Some are correct and too short to identify one group among 436. For others the code of the job board is doubtful, and the second model disagrees with the code more than with the answer. The check cannot separate these cases. It removes all three, and the set that remains is smaller and more consistent.

The check compares an answer with the code and never reads the job title. The next cell prints the answers for a title whose code on the job board is doubtful.
"""),
    code('''
show_answers("Quality Auditor")
'''),
    md("""
The job board placed the title "Quality Auditor" in group 3154, Air Traffic Controllers. The large model wrote its answers from the name and the definition of the group, so they describe an air traffic controller, and the second model confirmed them. The three answers are correct texts for group 3154. The title keeps its doubtful code in the training set, because no step of the pipeline checks the codes of the job board.
"""),
    md("""
## 5. Groups without a job title

The job titles, and therefore the answers written from them, cover @@F:classes@@ of the 436 unit groups. A model can suggest only the groups of its training texts. The groups without a job title include 6310 Subsistence Crop Farmers and 5212 Street Food Salespersons, which a household survey records.

A large model can write texts for a group without any labelled example, because ISCO-08 publishes a definition of each group with its tasks and with examples of occupations. In a separate recorded run, a large model read the definition of each group and wrote job titles for it, and a second model checked them as in section 4. The next cell loads these titles and shows ten of them for groups that the job board lacks.
""", id="section-5"),
    code('''
written = pd.read_json(ASSETS / "occupation/definition-titles.jsonl.gz", lines=True, dtype=str)
print(f"{len(written):,} titles written from the definitions, in {written['code'].nunique()} unit groups")

new_groups = written[~written["code"].isin(set(titles["code"]))]
sample = new_groups.sample(10, random_state=1)
sample.assign(group=sample["code"].map(group_name))[["title", "code", "group"]].set_index("title")
'''),
    code('''
v.result_bars(
    "Unit groups with at least one training text",
    [
        ("Job titles and survey-style answers", titles["code"].nunique(), "muted"),
        ("With titles written from the definitions", pd.concat([titles["code"], written["code"]]).nunique(), "small"),
    ],
    top_value=436, fmt="{:.0f}",
)
'''),
    md("""
The job titles carry codes that an employer or a job board assigned to real advertisements. The survey-style answers add the wording of a respondent for the same groups. The titles written from the definitions add the remaining groups, in the wording of a job title. No set holds survey-style answers for the groups that the job board lacks, and section 9 returns to this gap.
"""),
    md("""
## 6. Training the small model

The training set joins the three sets. Each text has one right group. A task of this kind is called single-label classification: the model gives one probability for each group, and the probabilities sum to 1. The next cell builds the set and counts its texts by source.
""", id="section-6"),
    code('''
kept = answers[answers["kept"]]
examples = pd.concat([
    titles[["id", "title", "code"]].rename(columns={"title": "text"}).assign(source="job title"),
    kept[["id", "answer", "code"]].rename(columns={"answer": "text"}).assign(source="survey-style answer"),
    written[["id", "title", "code"]].rename(columns={"title": "text"}).assign(source="title written from a definition"),
]).drop_duplicates("text").reset_index(drop=True)

CODES = sorted(examples["code"].unique())
position = {code: i for i, code in enumerate(CODES)}
print(f"{len(examples):,} texts in {len(CODES)} unit groups")
print(examples["source"].value_counts().to_string())
'''),
    md("""
The small model starts from DeBERTa-v3-small, a model with 142 million parameters that was pre-trained on a large amount of English text. A tokenizer cuts each text into pieces called tokens and replaces each token with its number in the model's vocabulary. The next cell loads the tokenizer and shows the tokens of one answer. The mark ▁ at the start of a token stands for a space, and the tokens [CLS] and [SEP] mark the start and the end of the text.
"""),
    code('''
from transformers import AutoTokenizer

BASE = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(BASE)

example = "sells roasted maize by the roadside"
print(tokenizer.convert_ids_to_tokens(tokenizer(example)["input_ids"]))
'''),
    md("""
The next cell sets one job title in ten aside, together with the answers written from it. The model never trains on these texts, so its results on them show whether it learns patterns that carry over to new texts. The split is by title, because an answer resembles the title that it was written from.
"""),
    code('''
import numpy as np
from datasets import Dataset


def encode(frame):
    data = Dataset.from_dict({"text": frame["text"].tolist(), "label": [position[code] for code in frame["code"]]})
    return data.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=64), batched=True)


ids = examples["id"].drop_duplicates().sample(frac=1, random_state=0)
set_aside = set(ids[: len(ids) // 10])
train_part = examples[~examples["id"].isin(set_aside)]
check_part = examples[examples["id"].isin(set_aside)]
if DEVICE == "cpu":   # a sample, to shorten training without a GPU
    train_part, check_part = train_part.sample(3000, random_state=0), check_part.sample(300, random_state=0)
print(f"{len(train_part):,} texts for training, {len(check_part):,} set aside")
'''),
    md("""
The next cell trains the model with the `Trainer` of the Transformers library: @@EPOCHS_WORD@@ passes over the texts (epochs), 32 texts per step, and a learning rate of 0.00005 that rises during the first steps and then falls to zero. The function `accuracy` computes two measures on the texts set aside after each epoch. Top 1 is the share of texts whose right group is the model's first suggestion. Top 5 is the share whose right group is among the first five.

The recorded run below was made on one NVIDIA H100 GPU, and the last line of the output gives its duration. The cell takes longer on a Colab T4 GPU.
"""),
    code('''
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed


def accuracy(prediction):
    best_five = np.argsort(-prediction.predictions, axis=1)[:, :5]
    right = best_five == prediction.label_ids[:, None]
    return {"top 1": right[:, 0].mean(), "top 5": right.any(axis=1).mean()}


set_seed(42)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(CODES))
settings = TrainingArguments(
    output_dir="occupation-model",
    num_train_epochs=@@EPOCHS@@ if DEVICE == "cuda" else 1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    report_to="none",
    disable_tqdm=False,   # shows the progress bar and the table of results
    seed=42,
)
trainer = Trainer(
    model=model,
    args=settings,
    train_dataset=encode(train_part),
    eval_dataset=encode(check_part),
    data_collator=DataCollatorWithPadding(tokenizer),   # pads the texts of a step to the same length
    compute_metrics=accuracy,
)
run = trainer.train()
print(f"Training took {run.metrics['train_runtime']:.0f} seconds.")
assert np.isfinite(run.training_loss), "The loss is not a number. Check that the installed version of transformers is below 5."
'''),
    md("""
@@TRAIN_TEXT@@
"""),
    md("""
## 7. Three tests

Three sets test the model. The first two come from 1,000 advertisements of the job board, and none of these advertisements took part in training or in the rewriting step. A job board repeats titles across advertisements: @@F:same_titles@@ of the 1,000 test titles also occur word for word among the training texts, @@F:same_board@@ of them as the title of another advertisement and the others among the texts written by the large models. None of the 1,000 test answers occurs among the training texts. Of the @@F:hand@@ illustrative answers, @@F:same_hand@@ occur word for word among the training texts.

- **Job titles.** 1,000 titles of the same job board, kept apart from the start.
- **Survey-style answers.** The same 1,000 titles, each rewritten as one survey answer by a third large model (Gemini 3.8 Flash) with a different request. No check removed any of these answers.
- **Illustrative answers.** @@F:hand@@ short answers composed for the slides of this session in the style of a household survey, each with an expected code. No trained coder reviewed them.

The next cell defines the measure and applies it to the three sets.
""", id="section-7"),
    code('''
import torch


def probabilities(texts, batch=128):
    """One row per text and one column per group of CODES."""
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            inputs = tokenizer(texts[start:start + batch], truncation=True, max_length=64, padding=True, return_tensors="pt").to(model.device)
            out.append(torch.softmax(model(**inputs).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def top(texts, codes, sizes=(1, 3, 5)):
    """The share of texts whose right code is among the first 1, 3, and 5 suggestions."""
    order = np.argsort(-probabilities(list(texts)), axis=1)[:, : max(sizes)]
    right = np.array(CODES)[order] == np.array(list(codes))[:, None]
    return {f"top {size}": round(float(right[:, :size].any(axis=1).mean()), 3) for size in sizes}


test_titles = pd.read_json(ASSETS / "occupation/test-titles.jsonl.gz", lines=True, dtype=str)
test_answers = pd.read_json(ASSETS / "occupation/test-answers.jsonl.gz", lines=True, dtype=str)
illustrative = pd.read_json(ASSETS / "occupation/illustrative-answers.json", dtype=str)

scores = {
    "job titles": top(test_titles["title"], test_titles["code"]),
    "survey-style answers": top(test_answers["answer"], test_answers["code"]),
    "illustrative answers": top(illustrative["answer"], illustrative["code"]),
}
pd.DataFrame(scores).T.assign(texts=[len(test_titles), len(test_answers), len(illustrative)])
'''),
    md("""
@@TEST_TEXT@@

The next cell compares training sets. Each recorded model was trained on one set or on a combination, three times with different random seeds, and the table gives the mean share of texts whose right group is among five suggestions.
"""),
    code('''
TESTS = {"report_titles": "job titles", "survey_style": "survey-style answers", "handwritten": "illustrative answers"}

recorded = pd.DataFrame(
    {name: [arm["tests"][test]["top5"] for arm in results["arms"]] for test, name in TESTS.items()},
    index=[arm["label"] for arm in results["arms"]],
).assign(texts=[arm["examples"] for arm in results["arms"]])
recorded.loc["The model of this notebook"] = [scores[name]["top 5"] for name in TESTS.values()] + [len(examples)]
recorded.astype({"texts": int})
'''),
    code('''
SETS = {arm["arm"]: arm for arm in results["arms"]}
rows = [(f"{arm['label']}\\nrecorded, mean of {arm['runs']} runs", arm["tests"]["survey_style"]["top5"], "muted") for arm in results["arms"]]
rows.append(("The model of this notebook\\nthe three sets", scores["survey-style answers"]["top 5"], "small"))
v.result_bars("Share of survey-style answers with the right group among five suggestions", rows, top_value=1.0)
'''),
    md("""
@@COMPARE_TEXT@@
"""),
    code('''
rows = [("The model of this notebook\\nfirst suggestion", scores["job titles"]["top 1"], "small")]
rows += [(f"{m['model']}\\nlarge model asked directly, recorded", m["top1"], "large") for m in results["large_on_titles"]]
v.result_bars("Share of job titles with the right group as the first suggestion", rows, top_value=1.0)
'''),
    md("""
@@LARGE_TEXT@@
"""),
    md("""
## 8. Suggestions for one answer

In use, the model receives one answer and returns its five most probable groups. The next cell follows one answer through the steps. The picture is animated: the answer becomes tokens, the model gives one probability for each group, and the five highest become the suggestions.
""", id="section-8"),
    code('''
def suggest(answer, size=5):
    """The most probable groups for one answer: (code, name, probability)."""
    row = probabilities([answer])[0]
    return [(CODES[i], group_name(CODES[i]), float(row[i])) for i in np.argsort(-row)[:size]]


ANSWER = "sells roasted maize by the roadside"
v.answer_path(ANSWER, tokenizer.tokenize(ANSWER), suggest(ANSWER), probabilities([ANSWER])[0])
'''),
    md("""
The next cell shows the suggestions for three of the illustrative answers, with the expected code of each. A green bar marks the expected code when it is among the suggestions. The variable `MINE` holds a further answer, and any text can replace it.
"""),
    code('''
for row in illustrative.iloc[@@SHOWN@@].itertuples():
    v.suggestions(row.answer, suggest(row.answer), truth=row.code)

MINE = "i repair phones in a small shop"
v.suggestions(MINE, suggest(MINE), note="No expected code is recorded for this answer.")
'''),
    md("""
@@SUGGEST_TEXT@@
"""),
    md("""
## 9. Limits and rounds of improvement

The results of section 7 have five limits.

- No survey answer coded by a trained coder took part in this notebook. The survey-style test and the illustrative answers stand in for such answers. A test on answers from a real survey is required before the model enters use.
- The large model that wrote the survey-style test saw the name of the right group, and the model that wrote the training answers saw its name and definition. The answers can contain words of the name. The score on the survey-style test can therefore be higher than the score on real answers.
- The codes of the job titles come from the job board. Errors in them enter the training set and both tests on those titles.
- For @@F:missing@@ of the 436 unit groups, the training set holds titles written from the definitions and no survey-style answer. Farming, street trade, and domestic work are among them.
- The illustrative set has @@F:hand@@ answers, so one answer changes its score by about @@F:hand_step@@.

The process of the session continues with rounds of improvement.

@@EXPLAINER:occupation_loop@@

A round tests the model on real cases kept for that purpose, finds the groups with the most errors, and has the large model write further texts for those groups. @@ROUNDS_TEXT@@

[Notebook 02]({COLAB}/02-report-tagging.ipynb) builds a model for a second task, assigning topics to reports, where a text can have several right answers and no labelled text is used for training.
""".replace("{COLAB}", COLAB), id="section-9"),
]


# Text of notebook 01 that states results of the recorded run. It is written after an execution, from the outputs.
FACTS = occupation_facts()
NB01_TEXT = {
    "@@EPOCHS@@": "8",
    "@@EPOCHS_WORD@@": "eight",
    "@@SHOWN@@": "[2, 3, 9]",
    "@@TRAIN_TEXT@@": (
        "The recorded run took about four minutes. On the texts set aside, the share with the right group as the first suggestion "
        "rose from 0.51 after the first epoch to 0.75 after the eighth. The share with the right group among five suggestions "
        "rose from 0.73 to 0.93. The validation loss stopped falling after the sixth epoch, so the run has enough epochs.\n\n"
        "The texts set aside come from the same three sources as the training texts and resemble them. Section 7 tests the "
        "model on texts from other advertisements."
    ),
    "@@TEST_TEXT@@": (
        "The right group is among the five suggestions for 0.846 of the job titles, 0.951 of the survey-style answers, and "
        "0.793 of the illustrative answers (23 of 29). The right group is the first suggestion for 0.617, 0.709, and 0.414 of "
        "the texts (12 of the 29 illustrative answers). A list of five suggestions holds the right group for 0.23 to 0.38 more "
        "of the texts than the first suggestion alone. This difference is the reason for giving a coder a list to choose from."
    ),
    "@@COMPARE_TEXT@@": (
        "Rewriting the job titles as survey-style answers raises the score on the survey-style test from 0.774 to 0.924. Both "
        "models take their codes from the same 10,134 labelled titles. The score on the job titles is 0.812 with the titles "
        "alone and 0.830 with both sets, so the added answers do not lower it.\n\n"
        "The three models trained without the titles written from the definitions score between 0.207 and 0.345 on the "
        f"illustrative answers. {FACTS['@@F:hand_missing@@']} of the {FACTS['@@F:hand@@']} illustrative answers have a code "
        "that no job title carries, and a model cannot suggest a group absent from its training texts. With all three sets, the "
        "score is 0.747.\n\n"
        "The illustrative answers also give a result against combining the sets. A model trained only on the titles written "
        "from the definitions scores 0.759 on them, about the same as the model trained on all three sets. Its first suggestion "
        "is right for 0.529 of the illustrative answers, and the first suggestion of the model trained on all three sets is "
        "right for 0.368 (the file `results.json` holds these scores under `top1`). The difference is about five of the 29 "
        "answers. A test on coded answers from a real survey is needed to settle whether the job-board texts lower the quality "
        "of the first suggestion on survey answers.\n\n"
        "The model of this notebook is a single run. Its three scores are within 0.05 of the recorded means for all three sets."
    ),
    "@@LARGE_TEXT@@": (
        "The first suggestion of the small model is right for 0.617 of the 1,000 job titles. The three large models score "
        "0.497, 0.433, and 0.406. Each large model received the titles with the codes and names of the 436 unit groups, "
        "without definitions and without examples.\n\n"
        "The comparison favours the small model in one respect. Its training set holds 10,134 titles of the same job board, "
        "so the model has learned the coding habits of that board, and the test codes follow the same habits. A recorded model "
        "trained only on the titles written from the definitions scores 0.351 on this test, below the three large models. "
        "No large model was tested on the survey-style answers."
    ),
    "@@SUGGEST_TEXT@@": (
        "In the three illustrative cases, the expected group is in the list, and the first suggestion is a neighbouring group. "
        "For “drives a motorcycle taxi”, the model gives 98% to Car, Taxi and Van Drivers and 1% to the expected group, "
        "Motorcycle Drivers. A high probability therefore does not confirm the first suggestion, and a coder reviews the list "
        "in every case. The answer in `MINE` divides between two groups of repair workers, 7422 at 43% and 7421 at 38%. "
        "A coder decides between them with the definitions of the two groups."
    ),
    "@@ROUNDS_TEXT@@": (
        "No round was run for this notebook. The results of section 7 indicate the content of a first round: survey-style "
        f"answers for the {FACTS['@@F:missing@@']} unit groups without a job title, written from the definitions and checked "
        "as in section 4. That run has not been made, and its effect on the scores is unknown."
    ),
}

NOTEBOOKS = {"01-occupation-coding.ipynb": NB01, "02-report-tagging.ipynb": NB02}
SETTINGS = {"@@REPORT@@": "176", **FACTS, **NB01_TEXT}


def write_all(only: list[str], keep_outputs: bool) -> list[Path]:
    written = []
    for name, cells in NOTEBOOKS.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        nb = new_notebook(cells=[nbformat.from_dict(dict(c)) for c in cells])
        for cell in nb.cells:
            for key, value in SETTINGS.items():
                cell["source"] = cell["source"].replace(key, value)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
        path = HERE / name
        if keep_outputs and path.exists():
            keep_recorded(nb, nbformat.read(path, as_version=4), name)
        nbformat.write(nb, path)
        written.append(path)
    return written


def keep_recorded(nb, old, name: str) -> None:
    """Copies the recorded outputs of `old` into `nb` when the two notebooks have the same code cells."""
    new_code = [c for c in nb.cells if c.cell_type == "code"]
    old_code = [c for c in old.cells if c.cell_type == "code"]
    if [c.source for c in new_code] != [c.source for c in old_code]:
        if any(c.get("outputs") for c in old_code):
            print(f"{name}: a code cell changed, so the recorded outputs were dropped. Run with --execute.")
        return
    for new, recorded in zip(new_code, old_code):
        new["outputs"], new["execution_count"] = recorded.get("outputs", []), recorded.get("execution_count")


def execute(paths: list[Path]) -> None:
    from nbclient import NotebookClient

    kernel = os.environ.get("WORKSHOP_KERNEL", "python3")
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=3600, kernel_name=kernel, resources={"metadata": {"path": str(HERE)}}).execute()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.write(nb, path)
        print("executed", path.name)


if __name__ == "__main__":
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    written = write_all(only, keep_outputs="--execute" not in sys.argv)
    print("wrote", len(written), "notebooks")
    if "--execute" in sys.argv:
        execute(written)
