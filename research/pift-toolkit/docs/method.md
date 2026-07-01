# The method: permutation-invariant fine-tuning

This page explains what the toolkit does and why, at a level useful for someone
deciding whether to adopt it. For how to run it, see
[pipeline.md](pipeline.md); for configuration, see
[configuration.md](configuration.md).

## The problem

A structured record (a statistical indicator, a dataset entry, a product, a
document with metadata) is a set of labeled fields: a title, a description, a
unit, a source, and so on. To retrieve it with a text-embedding model you first
serialize the fields into a single string. That serialization forces a choice of
field order.

That choice is usually treated as a throwaway detail, yet it changes retrieval
quality once a model is fine-tuned. When you
fine-tune an encoder on one fixed field order, the model can learn to use a
field's absolute position as a shortcut instead of reading the field label.
Nothing in ordinary fine-tuning discourages this. The failure only shows up later,
when the index is rebuilt under a different order: a portal is redesigned, a
federated system ingests records serialized by a different producer, or a
downstream system reformats before embedding. In the paper's experiments a
standard fine-tune lost 7.4 nDCG@10 points to such an order change. A model that
should not care about field order quietly did.

## The fix

Make field order uninformative during training. Instead of one fixed
serialization, present each record under a field order sampled fresh on every
access, and drop non-essential fields at random. When the same record keeps
reappearing under different orders, position stops predicting anything and the
only stable cue left is the field label. The encoder is pushed to read the schema
rather than memorize a template.

Two standard results explain why this induces invariance:

- Averaging a function over all orderings of its input yields a
  permutation-invariant function; training on one random ordering per step is the
  usual stochastic estimate of that average (Janossy pooling).
- Augmenting with a group's orbit is equivalent to constraining the model to the
  invariant subspace for that group.

Either way, the model gains nothing from order-dependent behavior. In the paper
this cut the order-change penalty from 7.4 points to 0.2, at no cost to
in-distribution accuracy.

In code this is the permutation and dropout in
[`serialize.render_segments`](../src/pift/serialize.py), applied on the fly by the
training data loader in [`finetune._make_transform`](../src/pift/finetune.py). It
is the "two lines in the data loader" the paper refers to. Everything else is
standard contrastive fine-tuning.

## Supporting components

Field-order permutation is the central idea. Three further components make it
work well on real catalogues, and each is configurable:

1. **Full-schema serialization with budgeting.** Permutation needs fields to
   permute, so the serializer emits every populated field rather than a
   hand-picked subset. Short fields are kept whole; long free-text fields
   (`elastic` role) share whatever token budget is left, by max-min fair
   allocation, so the document stays within the encoder's length limit without
   truncating a long field from the tail.

2. **Facet-protected dropout.** A query that targets the methodology must not
   have the methodology field dropped from its positive document, or the pair
   teaches nothing. The config maps each query facet to the fields it depends on,
   and those fields are protected from dropout in that pair.

3. **Duplication-aware negatives.** Metadata catalogues are full of
   near-duplicate records across collections. A near-duplicate of the positive is
   not a real negative. Mining drops any candidate whose embedding is more
   similar to the positive than a threshold (default cosine 0.95), and the guided
   loss (`cgist`) additionally masks in-batch false negatives using a guide
   model.

## Supervision without usage logs

Search tuning normally relies on click logs, which only cover queries users have
already issued, so they give no signal for records nobody has searched yet. The
toolkit generates the signal instead: an LLM writes grounded, facet-targeted
queries for every record, in every language you request, each anchored to the
record it should retrieve. The held-out evaluation set is generated with a
different (stronger) model and over a disjoint set of records, so the metric is
not just measuring the generator's style.

## Losses

- `cmnrl` (default): Cached MultipleNegativesRankingLoss. Unguided. The cached
  (GradCache) form decouples the negative pool (the batch) from memory (the
  mini-batch), so encoders of different sizes can train against the same
  negatives.
- `cgist`: Cached GISTEmbedLoss. Guided. A guide model masks in-batch false
  negatives, which matters in a near-duplicate-rich corpus. Set `guide_model` in
  the config. A strong open multilingual encoder is a good default guide; the
  paper also shows that using a first-generation fine-tune of the same recipe as
  the guide (self-distillation) compounds the gains.

## Graded evaluation with an LLM judge

The held-out metrics score retrieval against the single labeled positive per
query. In a catalogue full of near-duplicates that label is sparse, so it
understates how good a result list is: a system can return a perfectly useful
record that happens not to be the labeled one. The optional LLM judge scores each
retrieved (query, record) pair on a 0-3 rubric, so two systems the binary metrics
call tied can still be separated by how relevant their top results are.

Relevance is a property of the (query, record) pair rather than of the retriever,
so judgements are cached on disk and reused across every model in a comparison.
That is what makes
`pift benchmark` over several models affordable: each pair is judged at most once.
See [pipeline.md](pipeline.md) for the commands.

## What you get out

A fine-tuned SentenceTransformer directory you can load anywhere, plus a
`pift_config.json` recording the query/document prefixes so evaluation and
serving apply them automatically. The same encoder powers `pift evaluate` and
`pift search`.
