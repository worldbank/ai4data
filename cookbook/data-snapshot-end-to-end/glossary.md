---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the data-snapshot-end-to-end cookbook, in plain language.
---

# Glossary

**Adapter.** A small set of trained parameters added to a model for a task, so that fine-tuning fits on modest hardware (parameter-efficient fine-tuning).

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Benchmark (population).** The share of each group in the population, from the census or population estimates, against which a dataset's composition is compared.

**Bounding box.** The rectangle that encloses a detected table or figure on a page, written as the left, top, right, and bottom edges as fractions of the page width and height, with the origin at the top left.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Confidence threshold.** The model confidence above which a code is accepted automatically, set from the measured accuracy curve.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Data snapshot.** A bounded visual region of a document page containing structured or semi-structured information for analytical interpretation or operational reuse, such as a statistical table, chart, analytical map, or monitoring dashboard. A crop should retain the context needed to interpret it.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Leakage.** Any way the answer reaches a model other than through its features: shared groups across splits, a feature that encodes the label, duplicated inputs.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**OCR.** Optical character recognition: reading text from the image of a page, which is what a scanned document needs before anything can be extracted from it.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Recall.** The share of labelled mentions that the extractor found.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Vision-language model.** A model that reads an image and returns text or structured output; used to extract chart data from snapshots.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
