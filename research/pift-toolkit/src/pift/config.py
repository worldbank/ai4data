"""
Configuration: the single file that adapts the toolkit to a catalogue.

A :class:`Config` is loaded from a YAML file (see ``configs/example.yaml``). It
describes the catalogue's records, how each field is serialized, the query
facets used for synthetic supervision, the base encoder, and the training and
generation defaults. Everything downstream (serialization, generation, mining,
fine-tuning, evaluation, search) reads from this object, so adapting the
pipeline to a new catalogue means editing one YAML file rather than touching
code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FieldSpec:
    """One serialized field.

    role:
      - ``protected``: never dropped during augmentation (e.g. the title); a
        record is meaningless without it.
      - ``fixed``: droppable, length-capped, never truncated by the budget.
      - ``elastic``: shares whatever budget the fixed fields leave (e.g. long
        description / methodology), truncated by max-min fair allocation.
    """

    key: str
    label: str
    role: str = "fixed"
    extract: Optional[dict] = None
    max_chars: Optional[int] = None

    def effective_cap(self, elastic_hard_cap: int) -> int:
        if self.max_chars is not None:
            return self.max_chars
        if self.role == "elastic":
            return elastic_hard_cap
        return 300  # default fixed-field cap

    def __post_init__(self):
        if self.role not in ("protected", "fixed", "elastic"):
            raise ValueError(
                f"field {self.key!r}: role must be protected|fixed|elastic, got {self.role!r}"
            )


@dataclass
class Config:
    raw: dict
    fields: list[FieldSpec] = field(default_factory=list)
    path: Optional[Path] = None

    # ---- catalogue ----
    @property
    def id_field(self) -> str:
        return self.raw["catalogue"]["id_field"]

    @property
    def records_glob(self) -> Optional[str]:
        return self.raw["catalogue"].get("records_glob")

    @property
    def records_file(self) -> Optional[str]:
        return self.raw["catalogue"].get("records_file")

    # ---- serialization ----
    @property
    def serialization(self) -> dict:
        s = dict(self.raw.get("serialization", {}))
        s.setdefault("separator", " | ")
        s.setdefault("total_chars", 1700)
        s.setdefault("field_dropout", 0.15)
        s.setdefault("label_scheme", "label")  # "label" or "key"
        if s["label_scheme"] not in ("label", "key"):
            raise ValueError("serialization.label_scheme must be 'label' or 'key'")
        return s

    @property
    def protected_labels(self) -> set:
        return {f.label for f in self.fields if f.role == "protected"}

    @property
    def elastic_labels(self) -> set:
        return {f.label for f in self.fields if f.role == "elastic"}

    def label_for_key(self, key: str) -> Optional[str]:
        for f in self.fields:
            if f.key == key:
                return f.label
        return None

    # ---- facets (query generation + dropout protection) ----
    @property
    def facets(self) -> dict:
        """facet name -> list of field LABELS that must survive dropout in the
        positive document for that facet (the evidence the pair teaches)."""
        return self.raw.get("facets", {"keyword": [], "natural": []})

    # ---- base model ----
    @property
    def base_model(self) -> dict:
        bm = dict(self.raw.get("base_model", {}))
        bm.setdefault("hf_id", "intfloat/multilingual-e5-small")
        bm.setdefault("query_prefix", "")
        bm.setdefault("doc_prefix", "")
        return bm

    # ---- generation / training (defaults; overridable on the CLI) ----
    @property
    def generation(self) -> dict:
        g = dict(self.raw.get("generation", {}))
        g.setdefault("provider", "anthropic")
        g.setdefault("model", "claude-haiku-4-5")
        g.setdefault("languages", ["en"])
        g.setdefault("queries_per_record", 4)
        g.setdefault("eval_provider", g["provider"])
        g.setdefault("eval_model", "claude-sonnet-4-6")
        g.setdefault("eval_fraction", 0.1)
        return g

    @property
    def training(self) -> dict:
        t = dict(self.raw.get("training", {}))
        t.setdefault("loss", "cmnrl")          # cmnrl (unguided) | cgist (guided)
        t.setdefault("guide_model", None)
        t.setdefault("epochs", 5)
        t.setdefault("batch_size", 128)
        t.setdefault("mini_batch_size", 32)
        t.setdefault("lr", 3e-5)
        t.setdefault("n_negatives", 3)
        t.setdefault("max_seq_length", 512)
        t.setdefault("field_dropout", self.serialization["field_dropout"])
        return t

    def protect_for_facet(self, facet: Optional[str]) -> set:
        if not facet:
            return set()
        return set(self.facets.get(facet, []))


def load_config(path: str | os.PathLike) -> Config:
    import yaml

    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    fields = [FieldSpec(**f) for f in raw.get("fields", [])]
    if not fields:
        raise ValueError(f"{p}: config must define at least one field under 'fields:'")
    protected = [f for f in fields if f.role == "protected"]
    if not protected:
        raise ValueError(
            f"{p}: at least one field must have role 'protected' (e.g. the title), "
            "otherwise a record can be fully dropped during augmentation"
        )
    cfg = Config(raw=raw, fields=fields, path=p)
    # touch validators so misconfiguration fails fast at load time
    _ = cfg.serialization, cfg.base_model, cfg.training, cfg.generation
    return cfg
