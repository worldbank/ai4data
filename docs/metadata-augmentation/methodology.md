# Metadata Augmentation: Technical Details

This chapter provides technical depth on the five-step augmentation pipeline: embedding strategy, clustering approach, token budget enforcement, LLM configuration, and schema design. It is intended for data engineers and researchers who want to understand, extend, or adapt the implementation.

The implementation is in [`src/ai4data/metadata/augmentation/`](../../src/ai4data/metadata/augmentation/).

---

## Step 1: Loading Data Dictionaries

Adapters (in [`adapters.py`](../../src/ai4data/metadata/augmentation/adapters.py)) load variable data from diverse source formats into a canonical `DictionaryVariable` list.

### Canonical Schema

Each variable is represented as a `DictionaryVariable` Pydantic model:

```python
class DictionaryVariable(StrictBaseModel):
    variable_name: str           # machine-readable identifier (e.g., "HV001")
    label: str                   # human-readable label (e.g., "Cluster number")
    description: Optional[str]   # extended description / question text
    value_labels: Optional[Dict[str, str]]  # code -> label (e.g., {"1": "Male"})
```

### Adapters

**`ConfigurableDictionaryAdapter`** — For CSV and JSON sources with configurable column mapping:

```python
from ai4data.metadata.augmentation import ConfigurableDictionaryAdapter

# Map source column names to canonical names
adapter = ConfigurableDictionaryAdapter({
    "variable_name": "name",
    "label": "labl",
    "description": "qstn",
})
variables = adapter.load_csv("dictionary.csv")
```

**`NADACatalogAdapter`** — For [NADA microdata catalog](https://nada.ihsn.org/) JSON format, which uses nested objects (`qstn.qstnlit`, `catgry[].catValu`):

```python
from ai4data.metadata.augmentation import NADACatalogAdapter

adapter = NADACatalogAdapter()
variables = adapter.load_json("nada_catalog.json")
```

---

## Step 2: Semantic Embedding

Embedding is implemented in [`embeddings.py`](../../src/ai4data/metadata/augmentation/embeddings.py) using the `sentence-transformers` library.

### Model Choice

**Default model: `BAAI/bge-small-en-v1.5`**

This model was chosen for the following reasons:
- **Performance.** BGE-small achieves competitive performance on semantic similarity benchmarks (MTEB) for short English texts.
- **Size.** ~33M parameters. Encodes hundreds of variable labels per second on CPU.
- **Normalization.** With `normalize_embeddings=True`, cosine similarity equals dot product, enabling fast inner-product search.
- **No instruction prefix needed.** Unlike older INSTRUCTOR models, BGE-small works well without task-specific instructions for short label texts.

For multilingual data dictionaries, substitute a multilingual model:

```python
from ai4data.metadata.augmentation import DataDictionaryAugmentor

augmentor = DataDictionaryAugmentor(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
)
```

### Text Construction

The `EmbeddingEncoder.build_text()` method concatenates label and description (if present), separated by a period:

```python
# For variable: label="Cluster number", description=None
# → "Cluster number"

# For variable: label="Child is stunted", description="Stunting based on HAZ < -2SD"
# → "Child is stunted. Stunting based on HAZ < -2SD"
```

This gives the embedding model more context when descriptions are available, improving cluster quality for variables with short or ambiguous labels.

### Dimensionality Reduction

For large dictionaries (N > 150 variables by default), TruncatedSVD reduces the embedding dimension before clustering. This improves clustering quality by removing noise dimensions and reduces memory requirements for the distance computation.

```python
from ai4data.metadata.augmentation.clustering import reduce_dimensions

reduced = reduce_dimensions(embeddings, n_components=64, threshold=150)
```

SVD is skipped for small dictionaries where the full embedding space is appropriate.

---

## Step 3: Clustering

Implemented in [`clustering.py`](../../src/ai4data/metadata/augmentation/clustering.py).

### Algorithm: AgglomerativeClustering (Ward)

**Why Agglomerative Clustering?**
- **Deterministic.** No random initialization issues (unlike k-means).
- **No centroid instability.** Ward linkage uses pairwise distances, not centroids.
- **Hierarchical.** Can produce a dendrogram if needed for visualization or manual review.
- **Ward linkage** minimizes within-cluster variance—the most appropriate criterion for compact, well-separated clusters in embedding space.

### Automatic Cluster Count Estimation

When `n_clusters` is not specified, the pipeline estimates the optimal number using **silhouette score**:

```python
from ai4data.metadata.augmentation.clustering import estimate_n_clusters

k = estimate_n_clusters(embeddings, n_range=(3, 30))
```

The silhouette score measures how similar each point is to its own cluster relative to other clusters (range −1 to +1; higher is better). The algorithm:
1. Evaluates candidates k = 3, 4, ..., min(30, N−1)
2. For large N (> 500), uses a random subsample of 500 for speed
3. Returns the k with the highest silhouette score
4. Falls back to heuristic `max(3, sqrt(N/2))` when the score search fails

### Token Budget Enforcement

After clustering, a post-hoc split step ensures that no cluster's variable list exceeds the LLM context budget:

```python
from ai4data.metadata.augmentation.clustering import split_clusters_for_token_budget

labels = split_clusters_for_token_budget(
    labels, variables, max_tokens_per_cluster=450
)
```

The function splits oversized clusters by halving them iteratively. Token count per cluster is estimated conservatively (10 tokens per variable by default). This ensures every LLM call receives a variable list that fits within the context window without truncation.

---

## Step 4: LLM Theme Generation

Implemented in [`prompts.py`](../../src/ai4data/metadata/augmentation/prompts.py) and [`augmentor.py`](../../src/ai4data/metadata/augmentation/augmentor.py).

### Prompt Design

**System prompt** establishes the task and constraints:
- Role: "data catalog specialist for social science and development datasets"
- Goal: generate theme name, description, example variables
- Constraints: 2–6 word title-case theme name, 1–2 sentence description, up to 5 variable names from the INPUT list only
- Output: valid JSON only, no markdown or extra text

**User prompt** renders the numbered variable list:
```
# TASK
Generate a theme name and description for the following cluster of survey variables.

# VARIABLES
1. HV001: Cluster number
2. HV002: Household number
3. HV003: Respondent's line number
...

Output ONLY valid JSON matching the schema.
```

### Structured Output Schema

The LLM response is validated against `ThemeGenerationResult`:

```python
class ThemeGenerationResult(StrictBaseModel):
    theme_name: str        # e.g., "Household Identification"
    description: str       # e.g., "Variables identifying survey clusters and households."
    example_variables: List[str]  # e.g., ["HV001", "HV002", "HV003"]
```

Structured output enforcement uses the provider's JSON Schema API when available (`response_format={"type": "json_schema", ...}`), falling back to `{"type": "json_object"}` for providers that don't support strict schema.

### LLM Provider Configuration (via litellm)

The pipeline uses [litellm](https://docs.litellm.ai/) for provider-agnostic LLM calls. Any litellm-supported model string works:

```python
# Anthropic Claude (default)
augmentor = DataDictionaryAugmentor(model="claude-haiku-4-5-20251001")

# OpenAI
augmentor = DataDictionaryAugmentor(model="gpt-4o-mini")

# Google Gemini
augmentor = DataDictionaryAugmentor(model="gemini/gemini-2.0-flash")

# Local model via Ollama
augmentor = DataDictionaryAugmentor(model="ollama/llama3.1")
```

Set your API key as an environment variable:
```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

### Error Handling

Each cluster call is wrapped in a try/except. If the LLM call fails or the response cannot be parsed, the cluster receives a fallback theme ("Uncategorized") and execution continues. This ensures partial results are always returned even when individual calls fail—important for large dictionaries where one API error should not abort the entire run.

---

## Step 5: Output Schema

The complete augmented dictionary is represented as an `AugmentedDictionary` Pydantic model:

```python
class AugmentedDictionary(StrictBaseModel):
    dataset_id: Optional[str]                # source dataset identifier
    themes: List[Theme]                      # one per cluster
    variable_assignments: List[ThemeAssignment]  # one per input variable
    metadata: Optional[Dict[str, Any]]       # run config, model, timestamp
```

Where:

```python
class Theme(StrictBaseModel):
    theme_name: str
    description: str
    example_variables: List[str]   # 1–5 representative variable names

class ThemeAssignment(StrictBaseModel):
    variable_name: str
    theme_name: str
    cluster_id: int
```

The `metadata` field records the model name, embedding model, number of variables, number of clusters, and timestamp—ensuring run reproducibility and provenance tracking.

### Export Formats

```python
# JSON file
augmentor.export("augmented_dictionary.json")

# pandas DataFrame (variable_name, theme_name, cluster_id)
df = augmentor.to_dataframe()

# Direct Pydantic model access
result = augmentor.result
themes = {t.theme_name: t.description for t in result.themes}
```

---

## References

- [`src/ai4data/metadata/augmentation/`](../../src/ai4data/metadata/augmentation/) — Implementation
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — Default embedding model
- [LiteLLM documentation](https://docs.litellm.ai/) — LLM provider configuration
- [NADA microdata catalog](https://nada.ihsn.org/) — Source format for NSO microdata
- Reimert, R., & de Leeuw, J. (2022). "AgglomerativeClustering and Ward Linkage in sklearn." *scikit-learn documentation*.
