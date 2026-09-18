# Configuration reference

Everything the toolkit does is driven by one YAML file. This page documents every
key. A complete working example is `configs/example.yaml`.

## `catalogue`

```yaml
catalogue:
  id_field: id                                   # unique record id; dotted paths allowed
  records_file: examples/sample_catalogue/records.jsonl   # JSON-Lines, one record per line
  # records_glob: "examples/sample_catalogue/*.json"      # ...or per-record JSON files
```

- `id_field` (required): path to a unique identifier within each record. The
  train/eval split is a hash of this value, so it must be stable and unique.
- Provide exactly one of `records_file` (JSON-Lines) or `records_glob` (a glob of
  per-record JSON files). Relative paths resolve against the config file's
  directory.

## `fields`

An ordered list. Each entry describes one serialized field.

```yaml
fields:
  - {key: title,       label: "Title",       role: protected}
  - {key: description, label: "Description", role: elastic}
  - key: topics
    label: "Topics"
    role: fixed
    max_chars: 250
    extract: {type: list_join, subkey: name, sep: ", "}
```

| Key | Required | Meaning |
|---|---|---|
| `key` | yes | Dotted path into the record (e.g. `source.organization`). |
| `label` | yes | Display label, and the stable identity used for dropout protection and facets. |
| `role` | yes | `protected`, `fixed`, or `elastic` (see below). |
| `max_chars` | no | Hard cap for this field's text. Defaults: 300 for `fixed`, large for `elastic`. |
| `extract` | no | How to turn the raw value into text (see below). Default: scalar. |

**Roles**

- `protected`: never dropped during augmentation. At least one field must be
  protected (typically the title); otherwise a record could be dropped entirely.
- `fixed`: may be dropped during augmentation; length-capped by `max_chars`;
  never truncated by the global budget.
- `elastic`: long free-text fields (description, methodology). They share whatever
  character budget the fixed fields leave, by max-min fair allocation, so a short
  description frees room for a long methodology and vice versa.

**Extraction shapes** (`extract.type`)

- `scalar` (default): the value at `key`, rendered as text.
- `list_scalar`: `key` is a list of scalars; joined by `sep` (default `", "`),
  de-duplicated.
- `list_join`: `key` is a list of objects; `subkey` is pulled from each, then
  joined by `sep`, de-duplicated. Example: topics as `[{"name": "Health"}, ...]`.

## `serialization`

```yaml
serialization:
  separator: " | "
  total_chars: 1700      # approx token budget * ~4 chars/token
  field_dropout: 0.15
  label_scheme: label    # "label" (human labels) or "key" (raw field names)
```

- `total_chars`: the per-record character budget. Calibrate to your encoder's max
  sequence length; ~4 characters per token is a reasonable rule of thumb for
  English, lower for dense technical text, so leave headroom.
- `field_dropout`: probability a non-protected field is dropped during training
  augmentation.
- `label_scheme`: `label` emits your human labels; `key` emits the raw field
  names. `key` needs no label curation and tests whether the recipe works on the
  catalogue's native names; both are valid, just be consistent between training
  and serving (the toolkit handles this for you).

## `facets`

Maps each query facet to the field **labels** whose content it depends on.

```yaml
facets:
  keyword: []
  definition: ["Description"]
  geo_year: ["Geographic coverage", "Time coverage"]
```

This serves two purposes:

1. **Generation**: a facet is skipped for a record that lacks its evidence
   fields, so you never ask for a methodology query about a record with no
   methodology.
2. **Training**: the listed fields are protected from dropout in that pair's
   positive document, so the evidence the pair teaches is never removed.

Facets with an empty list (`keyword`, `natural`) apply to every record.

## `base_model`

```yaml
base_model:
  hf_id: intfloat/multilingual-e5-small
  query_prefix: "query: "
  doc_prefix: "passage: "
```

- `hf_id`: any SentenceTransformer-compatible encoder on the Hugging Face Hub or a
  local path.
- `query_prefix` / `doc_prefix`: prefixes the model family expects. E5 needs
  `query: ` / `passage: `; BGE and GTE have their own; many models need none.
  Leave empty if unsure, but check the model card: a prefix-trained model used
  bare will underperform.

## `generation`

```yaml
generation:
  provider: anthropic            # anthropic | openai | heuristic
  model: claude-haiku-4-5
  eval_provider: anthropic
  eval_model: claude-sonnet-4-6  # different/stronger model for the eval split
  languages: [en]                # e.g. [en, es, fr, sw, ar, zh]
  queries_per_record: 4
  eval_fraction: 0.1             # held-out share of records by id hash
```

- Use a small, cheap model for the training split and a different, stronger model
  for the eval split, so the benchmark is not just measuring generator style.
- `languages`: documents stay in their source language; queries are written in
  each listed language (cross-lingual retrieval). Add the languages your users
  actually search in.
- `heuristic` provider needs no API key and builds queries from the fields. Use
  it for the offline demo and tests; for production training use a real model.

## `training`

```yaml
training:
  loss: cmnrl              # cmnrl (unguided) | cgist (guided)
  guide_model: null       # required for cgist
  epochs: 5
  batch_size: 128
  mini_batch_size: 32     # memory knob for the cached losses; lower this on OOM
  lr: 3.0e-5
  n_negatives: 3
  max_seq_length: 512
```

- On out-of-memory, lower `mini_batch_size`, not `batch_size`: the batch is the
  negative pool, and the cached loss keeps it fixed while `mini_batch_size`
  controls memory.
- `n_negatives` must be no greater than the negatives produced by `pift mine`
  (`--n-negatives`). If mining produced fewer, fine-tuning warns and uses what is
  available.
- Any of these can be overridden on the CLI (`--epochs`, `--batch-size`,
  `--loss`, `--base`).
