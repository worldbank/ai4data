# Fine-Tuning Embedding Models for Structured Metadata

Semantic search over a data catalogue depends on embedding models that map queries and records into the same vector space. When records are **structured** — a title, description, unit, source, and other labeled fields — they must be flattened into one string before encoding. That serialization forces a field order, and standard fine-tuning can make the model rely on field *position* instead of field *label*. Rebuilding the index under a different order then silently degrades retrieval quality.

This subsection documents a **configuration-driven pipeline** for **permutation-invariant fine-tuning (PI-FT)** of embedding models on structured metadata catalogues. It packages the method from the paper [*Field Order Should Not Matter: Permutation-Invariant Embedding Model Fine-Tuning for Structured Metadata Retrieval*](https://arxiv.org/abs/2606.30473) {cite}`solatorio2026fieldorder`.

You bring a set of structured records (each a small schema of labeled fields) and a YAML config; the pipeline generates training supervision, mines hard negatives, fine-tunes a small open encoder, evaluates it (including an order-robustness test), and serves search.

The package lives at [`research/pift-toolkit/`](https://github.com/worldbank/ai4data/tree/main/research/pift-toolkit) in this repository.

**Please cite the underlying paper** when referring to the PI-FT method, DevDataBench, or the experimental results. **Cite this documentation** when pointing readers to the toolkit usage guide in this book.

---

## What it solves

PI-FT removes field-order fragility by serializing each record under a freshly shuffled field order during training. The pipeline also generates grounded, facet-targeted queries with an LLM when click logs are unavailable, giving coverage of every record and facet.

---

## Documentation in this subsection

| Page | Contents |
|------|----------|
| **The method** | Problem, fix, and design rationale for permutation-invariant fine-tuning |
| **Pipeline guide** | The five stages, commands, and outputs |
| **Configuration reference** | Every key in the YAML config |
| **Deployment** | Loading the model, prefixes, and scaling the index |

---

## Quick start

```bash
cd research/pift-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"
```

Copy and edit `configs/example.yaml` for your catalogue, then run the pipeline stages described in the Pipeline guide.

---

## References

**Paper**

Solatorio, A. V., Dupriez, O., & Macalaba, R. (2026). Field Order Should Not Matter: Permutation-Invariant Embedding Model Fine-Tuning for Structured Metadata Retrieval. *arXiv preprint arXiv:2606.30473*. [https://arxiv.org/abs/2606.30473](https://arxiv.org/abs/2606.30473)

```bibtex
@article{solatorio2026fieldorder,
  title={Field Order Should Not Matter: Permutation-Invariant Embedding Model Fine-Tuning for Structured Metadata Retrieval},
  author={Solatorio, Aivin V. and Dupriez, Olivier and Macalaba, Rafael},
  journal={arXiv preprint arXiv:2606.30473},
  year={2026},
  url={https://arxiv.org/abs/2606.30473},
  eprint={2606.30473},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

**Cite this documentation as:**
World Bank. 2026. "Fine-Tuning Embedding Models for Structured Metadata." AI for Data – Data for AI. Available at [https://worldbank.github.io/ai4data](https://worldbank.github.io/ai4data).
