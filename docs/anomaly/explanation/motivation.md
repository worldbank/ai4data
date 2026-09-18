# Motivation: Why LLM Elicitation for Anomaly Explanation

This chapter describes the problem of explaining anomalies in development data, the rationale for using large language models (LLMs), and the design choice of *elicitation*—structured, schema-constrained outputs—rather than free-form generation.

---

## The Problem

Development indicators—from World Development Indicators (WDI) and Corporate Scorecard to country-specific dashboards—often exhibit anomalies: abrupt changes, gaps, or values that appear implausible. Statistical and machine-learning detectors can **flag** these anomalies with high recall; the harder task is **explaining** them.

Why did GDP drop 15% in a given year? Why does a poverty indicator show a sudden discontinuity? Is it a real economic shock (conflict, pandemic, policy reform), a data error (placeholder, ingestion bug), or a methodological change (rebasing, census revision)?

Historically, this work has been manual and expert-intensive. Data stewards draw on institutional knowledge, external reports, and historical events to attribute causes. That approach does not scale across thousands of indicators, geographies, and time periods. As the volume of development data grows, automated support for explanation becomes essential for quality assurance and user trust.

---

## Why LLMs?

Large language models bring two critical capabilities to anomaly explanation:

**Contextual world knowledge.** LLMs encode information about macroeconomic events, conflicts, natural disasters, pandemics, major policy reforms, and statistical revisions. When presented with a time window and indicator—for example, "GDP growth, Country X, 2019–2020"—they can propose plausible, historically grounded causes.

**Structured reasoning over time.** By framing the task as anomaly *windows* (e.g., 2019–2020) rather than single points, we align with how real-world events unfold. The model reasons over the relevant period and produces explanations that fit that context.

The key design choice is **elicitation**, not generation: we do not ask the LLM for creative narrative. We elicit structured outputs—classifications, evidence citations, confidence scores—that are constrained by schema and prompt rules, making them amenable to validation and audit.

---

## LLM Elicitation as a Design Choice

**Elicitation** means prompting the model to produce outputs that conform to a predefined schema—much like form-filling rather than essay-writing. This aligns with established frameworks for data quality and official statistics:

**OpenAI Structured Outputs** (OpenAI, 2024) and **Gemini Structured Output** (Google AI, 2024) support JSON Schema enforcement so that responses are type-safe, parseable, and reliable. Schema adherence reduces hallucination and format errors.

**World Bank Data Quality Standards** (World Bank, 2024) stress traceability, verifiability, and transparent documentation. The guidelines require that explanations reference well-documented events when available.

By constraining outputs to a rigid taxonomy (`data_error`, `external_driver`, `measurement_system_update`, `insufficient_data`) and requiring evidence strength and verifiability levels, we ensure that explanations are actionable for reviewers and downstream systems.

---

## Scale and Efficiency

Manual expert review does not scale to thousands of anomalies across many indicators and countries. LLM elicitation provides:

- **Triage** — Prioritize anomalies by confidence, evidence strength, or classification for human review. Focus expert attention on the most uncertain or high-impact cases.
- **Audit trail** — Structured outputs (classification, evidence_source, verifiability) create a record that can be exported, analyzed, and used to improve future models.
- **Human-in-the-loop** — The [Reviewer Feedback System](../../anomaly-detection/feedback-system.md) collects verdicts, suggested corrections, and free-text comments. This feedback can support fine-tuning, prompt refinement, or audit coverage analysis.

---

## Alternative Approaches Considered

**Rule-based systems.** One could build a lookup table of known events (COVID-19 pandemic, 2008 financial crisis, specific country rebasing events) and match them to anomaly windows by year and geography. This approach is precise when coverage is good but fails for novel events, rare geographies, or complex multi-cause anomalies. It also requires constant maintenance as new events occur.

**Embedding-only retrieval.** A retrieval-augmented approach would embed anomaly contexts and retrieve similar past cases from a curated knowledge base. While promising, this requires a large curated corpus of explained anomalies, which does not yet exist at the required scale and coverage. LLM world-knowledge is a practical alternative that scales immediately.

**Unconstrained LLM generation.** Asking the LLM to "explain this anomaly in 2–3 sentences" produces readable text but inconsistent format—different phrasings, missing evidence citations, no confidence scores, and no structured verifiability levels. Downstream systems cannot reliably parse or compare free-form outputs, making aggregate analysis and retraining impractical.

Elicitation combines the breadth of LLM world-knowledge with the structure needed for auditable, machine-readable quality assurance.

---

## Related Work

The design draws on several lines of research and practice:

- **Faithful explanation and attribution.** Work on faithfulness in language model explanations (e.g., OpenAI, 2024) motivates strict schema enforcement to reduce hallucination and post-hoc rationalization.
- **UN Data Quality Assessment Framework (DQAF).** The UN Statistics Division's quality framework emphasizes traceability, coherence, and timeliness as dimensions of statistical data quality. The pipeline's classification taxonomy and verifiability levels are designed to produce outputs that speak directly to these dimensions.
- **LLMs for data quality.** Narayan et al. (2022) demonstrated that foundation models can perform data wrangling tasks—entity matching, error detection, data transformation—with surprising effectiveness even without task-specific fine-tuning, suggesting that LLM world-knowledge is broadly applicable to data quality tasks.
- **World Bank statistical review processes.** Internal review protocols for the WDI and Scorecard databases involve indicator-level sign-off by country economists and thematic experts, providing a well-understood human review target for the automated pipeline to support.

---

## References

- [OpenAI: Structured model outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Google AI: Structured outputs (Gemini)](https://ai.google.dev/gemini-api/docs/structured-output)
- [World Bank: Data Quality and Effectiveness](https://datahelpdesk.worldbank.org/knowledgebase/articles/906534-data-quality-and-effectiveness)
- Narayan, A. et al. (2022). "Can Foundation Models Wrangle Your Data?" *Proceedings of VLDB*, 16(4).
- UN Statistics Division. *Data Quality Assessment Framework (DQAF)*. https://unstats.un.org/unsd/methodology/dataquality/
