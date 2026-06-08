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

After clustering, a post-hoc merge step ensures that no cluster's variable list exceeds the LLM context budget:

```python
from ai4data.metadata.augmentation.clustering import merge_clusters_for_token_budget

labels = merge_clusters_for_token_budget(
    labels, variables, max_tokens_per_cluster=450
)
```

The merge splits oversized clusters by halving them iteratively. Token count per cluster is estimated conservatively (10 tokens per variable by default). This ensures every LLM call receives a variable list that fits within the context window without truncation.

---

## Step 4: LLM Variable Group Curation

Implemented in [`prompts.py`](../../src/ai4data/metadata/augmentation/prompts.py) and [`augmentor.py`](../../src/ai4data/metadata/augmentation/augmentor.py).

### Prompt Design

**System prompt** establishes the task and constraints:
- Role: expert metadata curator for social science and development datasets
- Goal: curate one DDI-style variable group from candidate variables (may be a subset)
- LLM-generated fields: `label`, `universe`, `notes`, `txt`, `definition`, `variables` (JSON array)
- Constraints: 2–6 word title-case label, catalog-style description, include all variables that fit (no artificial cap; cluster token budget is the practical limit)
- Output: valid JSON only, no markdown or extra text

**User prompt** renders the numbered variable list:
```
# TASK
Create one DDI-style variable group from the following candidate variables.

# VARIABLES
1. HV001: Cluster number
2. HV002: Household number
3. HV003: Respondent's line number
...

Return ONLY valid JSON matching the required schema.
```

### LLM vs Deterministic Fields

| Field | Source |
|-------|--------|
| `label`, `universe`, `notes`, `txt`, `definition`, `variables` | LLM |
| `vgid` | Deterministic (`make_vgid(label, cluster_id)`) |
| `variable_groups` | Always `""` |
| `group_type` | Always `"subject"` |
| `variables` (space-separated string) | Framework joins validated LLM array |

### Structured Output Schema

The LLM response is validated against `VariableGroupCurationResult`:

```python
class VariableGroupCurationResult(StrictBaseModel):
    label: str
    universe: str = ""
    notes: str = ""
    txt: str
    definition: str
    variables: List[str]  # min 1; all names must exist in candidate set
```

Validation via `VariableGroupCurationResult.from_llm_response(content, candidate_names=...)` rejects duplicate names and any variable not in the cluster candidate set. The full DDI record is assembled by `VariableGroup.from_curation()`.

Structured output enforcement uses the provider's JSON Schema API when available (`response_format={"type": "json_schema", ...}`), falling back to `{"type": "json_object"}` for providers that don't support strict schema.

---

## Step 4b: Self-Consistency QA Agent

Implemented in [`qa.py`](../../src/ai4data/metadata/augmentation/qa.py).

After curation, the pipeline optionally runs a **QA agent** — a second LLM call that assesses whether the proposed group's parts are mutually coherent (self-consistency prompting). This catches cases where the label, description, or definition do not match the selected variables.

### Manuscript mapping

| Manuscript concept | DDI field | QA checks |
|---|---|---|
| Theme name | `label` | Aligns with txt and variable labels |
| Description | `txt` | Describes selected variables, not unrelated concepts |
| Examples | `variables` (+ input labels) | Support the stated label |
| Grouping rationale | `definition` | Matches label and selected variables |

### QA output schema

```python
class VariableGroupQAResult(StrictBaseModel):
    is_self_consistent: bool
    rationale: str = ""
```

Results are stored on each `VariableGroup` as `qa_passed` and `qa_rationale`. When QA is disabled, `qa_passed` is `None`.

### Configuration

```python
# Default: curation and QA both use claude-sonnet-4-6
augmentor = DataDictionaryAugmentor()

# Cost tradeoff: cheaper curation, Sonnet QA
augmentor = DataDictionaryAugmentor(
    model="gpt-4o-mini",
    qa_model="claude-sonnet-4-6",
)

# Disable QA
augmentor = DataDictionaryAugmentor(enable_qa=False)
```

Run metadata records `qa_model`, `enable_qa`, `n_qa_passed`, `n_qa_failed`, and `n_qa_skipped`.

### QA failure behavior

- **Inconsistent curation** (`qa_passed=False`): curation output is kept; assignments are still emitted; rationale explains the inconsistency.
- **QA call failure** (`qa_passed=None`): curation output is kept; `qa_rationale` is `"QA call failed"`.

---

### LLM Provider Configuration (via litellm)

The pipeline uses [litellm](https://docs.litellm.ai/) for provider-agnostic LLM calls. Any litellm-supported model string works:

```python
# Anthropic Claude (default)
augmentor = DataDictionaryAugmentor(model="claude-sonnet-4-6")

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

Each cluster call is wrapped in a try/except. If the LLM call fails, the response cannot be parsed, or variable validation fails, the cluster receives a fallback variable group (`VG_UNCATEGORIZED_{cluster_id}`) and execution continues. This ensures partial results are always returned even when individual calls fail—important for large dictionaries where one API error should not abort the entire run.

---

## Step 5: Output Schema

The complete augmented dictionary is represented as an `AugmentedDictionary` Pydantic model:

```python
class AugmentedDictionary(StrictBaseModel):
    dataset_id: Optional[str]
    variable_groups: List[VariableGroup]              # one per cluster
    variable_assignments: List[VariableGroupAssignment]  # curated variables only
    metadata: Optional[Dict[str, Any]]
```

Where:

```python
class VariableGroup(StrictBaseModel):
    vgid: str
    variables: str           # space-separated curated variable names
    variable_groups: str = ""
    group_type: str = "subject"
    label: str
    universe: str = ""
    notes: str = ""
    txt: str
    definition: str
    cluster_id: int
    qa_passed: Optional[bool] = None
    qa_rationale: str = ""

class VariableGroupAssignment(StrictBaseModel):
    variable_name: str
    vgid: str
    label: str
    cluster_id: int
```

The `metadata` field records the model name, embedding model, number of variables, number of clusters, and timestamp—ensuring run reproducibility and provenance tracking.

### Export Formats

```python
# JSON file
augmentor.export("augmented_dictionary.json")

# pandas DataFrame (variable_name, vgid, label, cluster_id)
df = augmentor.to_dataframe()

# Direct Pydantic model access
result = augmentor.result
groups = {g.vgid: g.txt for g in result.variable_groups}
```

---

## References

- [`src/ai4data/metadata/augmentation/`](../../src/ai4data/metadata/augmentation/) — Implementation
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — Default embedding model
- [LiteLLM documentation](https://docs.litellm.ai/) — LLM provider configuration
- [NADA microdata catalog](https://nada.ihsn.org/) — Source format for NSO microdata
- Reimert, R., & de Leeuw, J. (2022). "AgglomerativeClustering and Ward Linkage in sklearn." *scikit-learn documentation*.
