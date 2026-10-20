"""Notebook 07, the lab: a small application to chat with your own documents.

Everything here runs inside a notebook, with no API key. The pieces are the
ones of notebooks 02, 03 and 06: split and index the documents, retrieve the
passages that match a question, ask a language model to answer from them, and
check every number of the answer against the passages. The model is a small
open one (Qwen2.5-1.5B-Instruct by default) that runs on a Colab CPU, and
faster on the free T4 GPU.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import frontmatter
from IPython.display import HTML, clear_output, display
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import workshop as w

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MY_DOCUMENTS = Path("my_documents")
SUPPORTED = (".md", ".txt", ".pdf")

INSTRUCTIONS = """You are a research assistant. Answer the question using only the passages given, in at most four short sentences.
Quote numbers exactly as they appear in the passages. Do not add anything the passages do not say.
If the passages do not contain the answer, reply exactly: I could not find that in the documents."""

# A number, with thousands separators and decimals; a full stop that ends a sentence is not part of it.
NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)(?!\.\d)")

NOT_FOUND = "I could not find that in the documents."


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------


class LocalChatModel(BaseChatModel):
    """A small open model, run in the notebook itself, behind the same chat interface as the other notebooks."""

    model_id: str = MODEL_ID
    max_new_tokens: int = 200
    tokenizer: Any = None
    model: Any = None

    @property
    def _llm_type(self) -> str:
        return "local-open-model"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        import torch

        roles = {"system": "system", "human": "user", "ai": "assistant"}
        chat = [{"role": roles.get(m.type, "user"), "content": m.content} for m in messages]
        inputs = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def load_model(model_id: str = MODEL_ID, max_new_tokens: int = 200) -> LocalChatModel:
    """Download the model on first use and put it on the GPU when there is one."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu = torch.cuda.is_available()   # any GPU that Colab gives (T4, L4, A100...) is used first
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16 if gpu else torch.float32)
    model.to("cuda" if gpu else "cpu").eval()
    if gpu:
        print(f"{model_id} loaded on the GPU ({torch.cuda.get_device_name(0)}).")
    else:
        print(f"{model_id} loaded on the CPU, so answers will take up to a minute. For quick answers: Runtime > Change runtime type > T4 GPU, then run the notebook again from the top.")
    return LocalChatModel(model_id=model_id, max_new_tokens=max_new_tokens, tokenizer=tokenizer, model=model)


# ----------------------------------------------------------------------------
# The documents
# ----------------------------------------------------------------------------


def session_documents() -> list[Path]:
    return sorted(w.DOCS_DIR.glob("*.md"))


def download_session_documents(name: str = "session_documents.zip") -> Path:
    """Zip the eight documents of the session, so that they can be saved and uploaded again."""
    with zipfile.ZipFile(name, "w") as z:
        for p in session_documents():
            z.write(p, p.name)
    try:
        from google.colab import files  # type: ignore

        files.download(name)
    except ImportError:
        print(f"Saved {name} in this folder.")
    return Path(name)


def upload_documents() -> list[Path]:
    """Ask for files (.md, .txt or .pdf; a .zip is unpacked) and keep them in the folder my_documents."""
    MY_DOCUMENTS.mkdir(exist_ok=True)
    for old in MY_DOCUMENTS.iterdir():
        old.unlink()
    try:
        from google.colab import files  # type: ignore

        uploaded = files.upload()
    except ImportError as e:
        raise RuntimeError("Upload works in Colab. Elsewhere, put your files in the folder my_documents.") from e
    for name, data in uploaded.items():
        target = MY_DOCUMENTS / Path(name).name
        target.write_bytes(data)
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as z:
                z.extractall(MY_DOCUMENTS)
            target.unlink()
    return sorted(p for p in MY_DOCUMENTS.rglob("*") if p.suffix.lower() in SUPPORTED)


def read_file(path: Path) -> Document:
    """One Document per file. Markdown keeps its YAML header as metadata; a PDF is read page by page."""
    suffix = path.suffix.lower()
    meta: dict[str, Any] = {"id": re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-"), "title": path.stem, "file": path.name}
    if suffix == ".md":
        post = frontmatter.load(path)
        meta.update({k: (v if isinstance(v, (int, float)) else str(v)) for k, v in post.metadata.items()})
        return Document(page_content=post.content.strip(), metadata=meta)
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        return Document(page_content=text.strip(), metadata=meta)
    return Document(page_content=path.read_text(errors="ignore").strip(), metadata=meta)


def prepare(paths: list[Path], chunk_size: int = 700) -> list[Document]:
    """Split each document: by section when it has Markdown headings, in windows otherwise (the choice of notebook 02)."""
    chunks: list[Document] = []
    for path in paths:
        doc = read_file(path)
        if not doc.page_content:
            continue
        if path.suffix.lower() == ".md" and re.search(r"^#{1,2} ", doc.page_content, re.M):
            chunks += w.split_sections([doc])
        else:
            pieces = w.split_fixed([doc], chunk_size=chunk_size, chunk_overlap=80)
            for i, c in enumerate(pieces, 1):
                c.metadata["section"] = f"part {i}"
            chunks += pieces
    return chunks


def chunk_table(chunks: list[Document]) -> None:
    """How each document was split, as a table."""
    counts: dict[str, list[int]] = {}
    for c in chunks:
        counts.setdefault(str(c.metadata.get("file", c.metadata.get("id"))), []).append(len(c.page_content))
    rows = [(name, len(sizes), f"{sum(sizes) // len(sizes):,}") for name, sizes in counts.items()]
    w.show_table(["document", "chunks", "characters per chunk"], rows, source=f"{len(chunks)} chunks in all")


# ----------------------------------------------------------------------------
# One question, one answer
# ----------------------------------------------------------------------------


@dataclass
class NumberCheck:
    text: str
    found_in: list[str]
    snippet: str = ""


@dataclass
class Reply:
    question: str
    answer: str
    passages: list[tuple[Document, float]]
    checks: list[NumberCheck] = field(default_factory=list)
    answered: bool = True


def _values(text: str) -> list[tuple[str, float]]:
    out = []
    for m in NUMBER.finditer(text):
        try:
            out.append((m.group(0), float(m.group(0).replace(",", ""))))
        except ValueError:
            pass
    return out


def check_numbers(answer: str, passages: list[tuple[Document, float]]) -> list[NumberCheck]:
    """Look for every number of the answer in the passages that were used. Citations and years are not figures."""
    body = re.sub(r"\[[^\]]*\]", " ", answer)
    checks = []
    for m in NUMBER.finditer(body):
        token = m.group(0)
        if re.fullmatch(r"(19|20)\d\d", token) or body[: m.start()].rstrip().endswith("per"):
            continue   # a year, or the "100" of "per 100 people"
        value = float(token.replace(",", ""))
        found, snippet = [], ""
        for doc, _ in passages:
            for t, v in _values(doc.page_content):
                if abs(v - value) < 1e-9:
                    found.append(str(doc.metadata.get("id")))
                    if not snippet:
                        i = doc.page_content.find(t)
                        snippet = doc.page_content[max(0, i - 60) : i + len(t) + 40].replace("\n", " ")
                    break
        checks.append(NumberCheck(token, found, snippet))
    return checks


class DocumentChat:
    """The application: retrieve, answer from the passages, check the numbers."""

    def __init__(self, model: BaseChatModel, store, k: int = 4, min_score: float = 0.25):
        self.model, self.store, self.k, self.min_score = model, store, k, min_score

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        hits = self.store.similarity_search_with_score(question, k=self.k)
        return [(d, s) for d, s in hits if s >= self.min_score]

    def ask(self, question: str) -> Reply:
        passages = self.retrieve(question)
        if not passages:
            return Reply(question, NOT_FOUND, [], answered=False)
        context = "\n\n".join(f"[{d.metadata.get('id')}] ({d.metadata.get('section', '')})\n{d.page_content}" for d, _ in passages)
        reply = self.model.invoke([("system", INSTRUCTIONS), ("human", f"PASSAGES:\n{context}\n\nQUESTION: {question}")])
        answer = reply.content.strip()
        return Reply(question, answer, passages, check_numbers(answer, passages), answered=NOT_FOUND not in answer)


# ----------------------------------------------------------------------------
# What the user sees
# ----------------------------------------------------------------------------

BUBBLE = "padding:10px 14px;border-radius:14px;margin:6px 0;max-width:90%;font:14px/1.6 Helvetica,Arial,sans-serif;color:#16315a"


def reply_html(reply: Reply) -> str:
    """The question and the answer as a chat, a mark after each number, and the passages under the answer."""
    marks = {c.text: c for c in reply.checks}
    out, pos = [], 0
    answer = reply.answer
    for m in NUMBER.finditer(answer):
        token = m.group(0)
        inside_citation = answer.rfind("[", 0, m.start()) > answer.rfind("]", 0, m.start())
        if token not in marks or inside_citation or re.fullmatch(r"(19|20)\d\d", token):
            continue
        c = marks[token]
        if c.found_in:
            glyph, colour = "✓", "#16a34a"
            tip = f"Found in a passage: [{c.found_in[0]}]\n…{c.snippet}…"
        else:
            glyph, colour = "⚠", "#d97706"
            tip = "Not found in any of the passages that were used.\nThe model may have worked it out, or made it up. Check it."
        out.append(escape(answer[pos : m.start()]))
        out.append(f'<span style="border-bottom:2px dotted {colour};cursor:help" title="{escape(tip)}">{escape(token)}<b style="color:{colour};margin-left:2px">{glyph}</b></span>')
        pos = m.end()
    out.append(escape(answer[pos:]))
    passages = "".join(
        f'<div style="margin:6px 0;padding:6px 10px;border-left:3px solid #c9d3e6"><b>[{escape(str(d.metadata.get("id")))}] {escape(str(d.metadata.get("section", "")))}</b> '
        f'<span style="color:#7a879c">score {s:.2f}</span><br>{escape(d.page_content[:420])}{"…" if len(d.page_content) > 420 else ""}</div>'
        for d, s in reply.passages
    )
    cited = sorted({i for c in reply.checks for i in c.found_in})
    sources = (
        '<div style="margin-top:8px;font:12.5px Helvetica,Arial,sans-serif;color:#5b6b85">Numbers found in: '
        + " · ".join(f'<b>{escape(i)}</b>' for i in cited)
        + "</div>"
        if cited
        else ""
    )
    fold = (
        f'<details style="margin-top:6px;font:12.5px Helvetica,Arial,sans-serif;color:#5b6b85"><summary style="cursor:pointer">The {len(reply.passages)} passages the answer was written from</summary>{passages}</details>'
        if reply.passages
        else '<div style="margin-top:6px;font:12.5px Helvetica,Arial,sans-serif;color:#5b6b85">No passage was close enough to the question, so the model was not asked.</div>'
    )
    return (
        f'<div style="{BUBBLE};background:#e8eefb;margin-left:auto">{escape(reply.question)}</div>'
        f'<div style="{BUBBLE};background:#f3f4f6">{"".join(out)}{sources}{fold}</div>'
    )


def launch(chat: DocumentChat) -> None:
    """A chat box under the cell: type a question, press Ask."""
    import ipywidgets as widgets

    log = widgets.Output()
    box = widgets.Text(placeholder="Ask a question about your documents", layout=widgets.Layout(width="75%"))
    button = widgets.Button(description="Ask", button_style="primary", layout=widgets.Layout(width="90px"))
    status = widgets.HTML("")

    def ask(_=None):
        question = box.value.strip()
        if not question:
            return
        box.value = ""
        button.disabled = True
        status.value = "<i>Reading the passages and writing the answer…</i>"
        try:
            reply = chat.ask(question)
            with log:
                display(HTML(reply_html(reply)))
        finally:
            status.value = ""
            button.disabled = False

    button.on_click(ask)
    box.on_submit(ask)
    legend = widgets.HTML(
        '<div style="font:12.5px Helvetica,Arial,sans-serif;color:#5b6b85"><b style="color:#16a34a">✓</b> the number appears in a passage · '
        '<b style="color:#d97706">⚠</b> it does not · no mark on years and citations. Hover a mark to see why.</div>'
    )
    display(widgets.VBox([legend, log, widgets.HBox([box, button]), status]))
