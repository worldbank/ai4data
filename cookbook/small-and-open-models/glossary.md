---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Small and Open Models for Statistical Offices.
---

# Glossary

**Adapter.** A small set of trained parameters added to a model for a task, so that fine-tuning fits on modest hardware (parameter-efficient fine-tuning).

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Benchmark (population).** The share of each group in the population, from the census or population estimates, against which a dataset's composition is compared.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Cost-quality frontier.** The candidates that no other candidate beats on both suite score and cost per query.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Digest.** The SHA-256 hash of a model file, recorded in the lock file and verified before loading.

**Edit rule.** A machine-readable condition a record must satisfy (range, consistency), owned by the organization and applied before and after any model proposal.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Feature.** A column a model may use as input.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better or follows the organization's conventions.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**Grounding.** Restricting a language model to what was retrieved or given for the task: an answer that uses only the retrieved records, a draft that uses only what the record and its named sources support.

**GSBPM.** The Generic Statistical Business Process Model, the UNECE reference model of the phases and sub-processes of statistical production, from specifying needs to disseminating and evaluating.

**Harness.** A program that sends questions to an agent connected to the server and logs the traces.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Key-value cache.** The memory a model uses per token of context during generation; grows with context length and concurrency.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Lock file.** The record of a model's source, version, and file digests that the server verifies before loading.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Prompt injection.** Text placed in content a system reads (a document, a web page, a query) that is written as an instruction to the model in the hope that the model follows it; tested for before release.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**Quantization.** Storing weights at fewer bits (8 or 4 in place of 16) to reduce memory, with a suite check for any loss.

**Question bank.** A repository of question wordings keyed by the concept each measures and the surveys that use each.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Read-only.** A tool that reads and returns data and has no path to change anything; the property the server's safety rests on.

**Report card.** A run's scores per slice with intervals, the set version, and the component versions, published with a release.

**Required score.** The suite score a task's model has to reach, set from the cost of error before any candidate is tested.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Serving stack.** The software that loads a model and answers requests (a single-machine runtime or a batching server).

**Slice.** A subset of the questions (a language, a question type) scored on its own, so that a change's effect on it is visible.

**Small model.** A model of roughly one to fifteen billion parameters that runs on one GPU or on CPU, enough for narrow tasks with the right context.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Switch-off test.** The test that production proceeds with the model layer stopped, with the fallback per task exercised.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Tool use.** The ability of a language model to call functions or APIs during an answer, so that values come from the source.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Variant.** A way of naming a dataset other than its canonical name: an acronym, an informal name, a translation, a misspelling.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
