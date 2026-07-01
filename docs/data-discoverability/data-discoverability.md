# Data Discoverability

## Overview

Data discoverability underpins one of the foundational pillars of AI-ready data systems. It refers to the ability to efficiently locate and access datasets that are relevant to a user's needs, even when the user cannot specify exact dataset names or metadata. This is especially important in the context of development data, where information is often fragmented, diverse, and described in different ways across sources. Effective data discoverability empowers users—including researchers, policymakers, and practitioners—to find and leverage data that might otherwise remain hidden or underutilized.

---

## Why Traditional Search Falls Short

Keyword-based search—matching query terms to document text—works well when users know the exact terminology used in a catalog. In practice, development data users often do not:

- A researcher searching for "household consumption poverty" may not know that the World Bank uses "poverty headcount ratio at $2.15 a day (2017 PPP)" as the canonical indicator name.
- A policy analyst looking for "school attendance" may not anticipate that the same concept appears under "net enrollment rate," "attendance ratio," or "out-of-school children" depending on the source.
- A data engineer querying for "malaria burden" will miss indicators titled "incidence of malaria (per 1,000 population at risk)" unless they know the precise phrasing.

This vocabulary gap is endemic across large, multi-source catalogs. The World Bank's WDI database alone contains over 1,400 indicators, the Microdata Library catalogs thousands of datasets across topics, and cross-catalog search compounds the problem further.

Modern AI search systems address this by moving from string matching to **semantic matching**: understanding what a query *means* and returning results that express the same meaning, even in different words.

---

## How AI-Powered Semantic Search Works

Modern approaches to data discoverability leverage a combination of techniques:

### Dense Vector (Semantic) Search

Each dataset, indicator, or variable is represented as a dense numerical vector (embedding) that encodes its semantic meaning. User queries are embedded in the same space. Similarity search returns items whose embeddings are closest to the query embedding—regardless of exact word overlap.

This enables:
- *Synonymous query matching*: "child death rate" → finds "under-5 mortality rate"
- *Conceptual matching*: "income inequality" → finds "Gini index", "income share of poorest 40%"
- *Cross-language queries*: (with multilingual models) "taux de pauvreté" → finds "poverty headcount ratio"

Dense search is powered by vector similarity indices (e.g., FAISS {cite}`johnson2019faiss`, HNSW) that enable sub-millisecond nearest-neighbor lookup over millions of items.

### Sparse (Lexical) Search

Keyword-based search using inverted indices (BM25 {cite}`robertson2009bm25`) remains valuable for:
- Exact code lookups (e.g., indicator code `NY.GDP.MKTP.KD.ZG`)
- Short, specific queries where semantic models add noise
- Efficient filtering before semantic re-ranking

### Hybrid Search

Combining dense and sparse retrieval—known as hybrid search—typically outperforms either alone. A query returns candidates from both methods, which are then re-ranked by a learned or heuristic fusion function. This provides both semantic understanding and lexical precision.

---

## The @ai4data/search Package

The [`@ai4data/search`](https://www.npmjs.com/package/@ai4data/search) JavaScript/TypeScript package provides a client-side semantic search library designed for development data portals and web applications.

Key features:

- **HNSW index** for approximate nearest-neighbor semantic search (powered by [`hnswlib-wasm`](https://github.com/nicholaswmin/hnswlib-wasm))
- **BM25 lexical search** for keyword fallback and hybrid ranking
- **Web Worker support** — Search runs in a background thread to avoid blocking the main UI thread
- **Pre-built indices** — Indices can be precomputed server-side and served as static assets, enabling instant in-browser search without a backend

```typescript
import { SearchIndex } from "@ai4data/search";

const index = new SearchIndex();
await index.load("https://example.org/indicators-index.bin");

const results = await index.search("child mortality under 5", { topK: 10 });
// Returns: [{ id, title, score, ... }, ...]
```

Install:

```bash
npm install @ai4data/search
```

---

## Semantic Search for Development Data

A practical semantic search deployment for development data involves:

1. **Indexing** — For each dataset or indicator, build a text representation from its metadata (name + description + keywords + topic) and encode it with a sentence-transformer model.
2. **Index construction** — Build a vector index (HNSW or FAISS) over the encoded representations.
3. **Query serving** — At query time, encode the user's query and run nearest-neighbor search against the index.
4. **Re-ranking (optional)** — Apply BM25 or a cross-encoder to re-rank the top-K candidates.

The [`packages/ai4data/search/`](../../packages/ai4data/search/) directory contains the JavaScript implementation. For Python-based indexing and search, the `[search]` optional dependency group provides the necessary libraries:

```bash
uv pip install ai4data[search]
```

---

## Fine-Tuning Embedding Models for Structured Metadata

Catalogue search quality depends not only on the retrieval architecture above, but on how well the embedding model represents structured records. When metadata is a set of labeled fields rather than a single document, standard fine-tuning can make models sensitive to field order — a problem that surfaces when indexes are rebuilt or federated across systems that serialize records differently.

The [Fine-Tuning Embedding Models for Structured Metadata](embedding-fine-tuning.md) subsection documents a permutation-invariant fine-tuning (PI-FT) pipeline {cite}`solatorio2026fieldorder`: a YAML-driven toolkit for generating training data, fine-tuning a small encoder, evaluating order robustness, and deploying search over structured catalogues.

---

## Connecting Discovery to AI Assistants

An emerging dimension of data discoverability is making datasets accessible to AI assistants directly, without going through a search UI. The [Model Context Protocol (MCP)](../mcp/mcp.md) provides a standardized interface through which AI assistants can query data catalogs, retrieve indicator values, and generate analysis—bringing search capabilities directly into AI workflows.

---

## References

- {cite}`johnson2019faiss` — Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
- {cite}`robertson2009bm25` — Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
