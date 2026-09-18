# Deployment

The fine-tuned model is a standard SentenceTransformer directory, so you can serve
it with whatever vector stack you already run. This page covers the two decisions
that matter: loading the model correctly, and scaling the index.

## Load the model with its prefixes

A fine-tuned model directory contains `pift_config.json`, which records the query
and document prefixes. Apply them, or retrieval quality drops for prefix-trained
families (E5, BGE, GTE).

The toolkit's `Encoder` reads that file automatically:

```python
from pift.config import load_config
from pift.encoder import Encoder

config = load_config("configs/example.yaml")
enc = Encoder("models/my-encoder")          # prefixes loaded from pift_config.json
doc_vecs = enc.encode_documents(list_of_serialized_records)
qry_vec  = enc.encode_queries(["maternal mortality in Kenya"])
```

If you load the raw SentenceTransformer yourself, read the prefixes from
`pift_config.json` and prepend them: documents get `doc_prefix`, queries get
`query_prefix`. Always L2-normalize embeddings so a dot product is cosine
similarity.

## Serialize documents the same way you trained

Index the **canonical** serialization (no permutation, no dropout), which is
exactly what `pift evaluate` and `pift search` use:

```python
from pift.serialize import serialize
text = serialize(record, config)            # canonical
vec = enc.encode_documents([text])[0]
```

Re-serialize and re-embed when a record changes. Because the model is
order-invariant, you do not have to freeze the field order across re-indexing
events, which is the operational benefit of PI-FT.

## Scale the index

`pift search` keeps a normalized embedding matrix in memory and does brute-force
dot products. That is fine for catalogues up to roughly 100k records on a laptop.
Beyond that, move the vectors into a vector database and keep everything else the
same:

- **FAISS** (in-process, no server): build an `IndexFlatIP` for exact search, or
  an `IVF`/`HNSW` index for approximate search at scale.
- **Qdrant / Weaviate / Milvus** (standalone services): push the same normalized
  vectors with the record id as the payload.
- **pgvector** (if you already run Postgres): a `vector` column with a cosine
  operator class.

In every case the encode-and-serialize steps above are unchanged; only the
nearest-neighbor lookup moves.

## A minimal search service

```python
# app.py  (FastAPI sketch)
from fastapi import FastAPI
from pift.config import load_config
from pift.search import SearchIndex

config = load_config("configs/example.yaml")
index = SearchIndex(config, model="models/my-encoder")   # builds the index once
app = FastAPI()

@app.get("/search")
def search(q: str, k: int = 10):
    return [{"id": h.record_id, "score": h.score} for h in index.query(q, k)]
```

For production, build the index at startup (or load precomputed vectors from your
vector store), put the encoder on a GPU if query volume warrants it, and batch
incoming queries.

## Re-training cadence

Re-run `generate → mine → finetune` when the catalogue grows materially, when you
add a language, or when evaluation shows drift. Keep the generated query sets and
triplets under version control or object storage so a re-train is reproducible.
The held-out eval set should stay fixed across re-trains so the metric is
comparable over time.
