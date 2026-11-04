---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in this guide.
---

# Glossary

**Adapter.** A small set of trained parameters added to a model for a task, so that fine-tuning fits on modest hardware (parameter-efficient fine-tuning).

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Data snapshot.** A bounded visual region of a document page containing structured or semi-structured information for analytical interpretation or operational reuse, such as a statistical table, chart, analytical map, or monitoring dashboard. A crop should retain the context needed to interpret it.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Decision point.** The step at which a named person decides on a model's output (a code, a proposal, an explanation, a draft).

**Evaluation set.** Decided records kept to test changes to prompts, manifests, rubrics, and models.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better or follows the organization's conventions.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Grader.** A language model, with a rubric, that scores answers the automatic checks cannot score; validated against human scores before use.

**Grader (judge).** A language model that scores answers with a rubric where no script can; validated against human scores before use.

**Grouped split.** A draw of whole groups (households, documents) into the parts of a dataset, so that no group is divided between training and test.

**Harness.** A program that sends questions to an agent connected to the server and logs the traces.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Leakage.** Any way the answer reaches a model other than through its features: shared groups across splits, a feature that encodes the label, duplicated inputs.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**ML-ready dataset.** A fixed, versioned set of examples with a documented structure, a stated purpose, a split for training and testing, a statement of whom the examples represent, and a licence that covers model training.

**Paired comparison.** The same questions through two versions, so that differences are between versions.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Representativeness.** The degree to which a dataset's composition matches the population it will be used on, measured group by group and stated in the card.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Rubric.** A written scale that says what each score means, so that two people, or a person and a model, score the same thing the same way.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Stable identifier.** An identifier for a series or dataset that does not change across releases or site redesigns.

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Council deliberation.** Independent proposals, cross-critiques, and revisions by distinct models that examine extraction evidence before an orchestrator synthesizes a candidate policy.

**Cumulative experiment state.** The compact record of accepted decisions, known failure modes, and unresolved questions carried forward from one experiment round to the next.

**Diagnostic judge.** A model that assesses the generated metadata against the supplied image, schema, and extraction policy and compares it with a full human reference when available.

**Discovery sample.** A snapshot whose human-reviewed semantic annotation is available to the diagnostic judge and human reviewer during policy development.

**Extraction policy.** The versioned prompts and rules that guide the reference extractor in describing a snapshot.

**Round manifest.** The machine-readable inventory of a round's sample IDs, settings, versions, status, and artifact paths.
