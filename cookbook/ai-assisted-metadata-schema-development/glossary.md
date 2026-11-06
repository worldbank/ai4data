---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in this guide.
---

# Glossary

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Artifact class.** The kind of object a schema describes, defined by its analytical purpose, boundaries, and expected diversity. A member of the class is an artifact.

**Calibration.** The comparison of model scores with curator scores on a sample, per dimension, to find systematic bias.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Canonical metadata concept.** A shared descriptive meaning proposed across field profiles, with a name, definition, examples, and rationale for consolidation.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Concept coverage.** The descriptive evidence obtained by assigning development field profiles to reviewed concepts or a residual category; it supports human review.

**Concept inventory.** An intermediate collection of descriptive concepts, definitions, examples, and refinement decisions that precedes operational field design.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Crosswalk.** A mapping from one classification to another, for example from a national occupation classification to ISCO.

**Data snapshot.** The image of one table or figure cropped from a page, with its coordinates, class, and source document; the unit of extraction and of citation.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Document record.** The catalog record of a document in the World Bank document schema, with the series and surveys it draws on.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Evidence pack.** The list, by assessment question, of the artefact that answers it, its location, and its date, compiled for an AI-readiness assessment.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Grounding.** Restricting a language model to what was retrieved or given for the task: an answer that uses only the retrieved records, a draft that uses only what the record and its named sources support.

**Held-out validation.** Assessment of an operational schema against artifacts excluded from its initial development, with candidate findings and explicit claim limits.

**Human adjudication.** A documented human decision about the evidence, scope, materiality, and appropriate treatment of a proposed concept or schema change.

**Incident.** An AI component's output reaching a reviewer, a product, or a user with an error that the controls should have caught, or that a control caught at the last step; handled by a fixed procedure and recorded.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Layout detection.** A model that finds regions of a page image and returns their class and bounding box.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Metadata field profile.** An exact-name aggregation of metadata observations that preserves representative evidence and occurrence statistics for semantic comparison.

**Metadata observation.** An artifact-level proposal for a reusable descriptive property, retaining its observed value, evidence source, description, and rationale.

**ML-ready dataset.** A fixed, versioned set of examples with a documented structure, a stated purpose, a split for training and testing, a statement of whom the examples represent, and a licence that covers model training.

**Operational metadata schema.** A specification of usable metadata fields, definitions, evidence rules, examples, and annotation guidance for an intended application.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Recall.** The share of labelled mentions that the extractor found.

**Representativeness.** The degree to which a dataset's composition matches the population it will be used on, measured group by group and stated in the card.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Standards crosswalk.** A comparison of local schema fields with external terms or patterns, recording semantic relationships and sources for human refinement decisions.

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Utility.** How closely the synthetic file reproduces the real one at the marginal, association, and analysis levels.

**Variant.** A way of naming a dataset other than its canonical name: an acronym, an informal name, a translation, a misspelling.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.
