---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Metadata Curation with Language Models.
---

# Glossary

**Acceptance rate.** The share of model suggestions that curators accept or edit, per field, task, and model.

**Adapter.** A small set of trained parameters added to a model for a task, so that fine-tuning fits on modest hardware (parameter-efficient fine-tuning).

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**Agents manifest.** The file that defines the agents of the review pipeline (detectors, critic, categorizer, severity scorer) and their instructions.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Calibration.** The comparison of model scores with curator scores on a sample, per dimension, to find systematic bias.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Cohen's kappa.** A measure of agreement between two labellers that discounts the agreement expected by chance; 1 is perfect agreement, 0 is chance.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Data dictionary.** The documentation of a data file's variables: names, labels, types, value codes, missing codes, universes, questions.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Decision log.** The table with one row per decision of the governance board: approval, change, incident, review, or policy, with the role that decided, the follow-up, and its status.

**Draft status.** The mark on a model-written field that keeps it out of the published catalog until a decision.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Error taxonomy.** The coded kinds of error a component makes, each pointing to a different fix: for an extractor, missed, spurious, boundary, wrong type; for curation suggestions, wrong flag, invented fact, wrong vocabulary term, style.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Evaluation set.** Decided records kept to test changes to prompts, manifests, rubrics, and models.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Grounding.** Restricting a language model to what was retrieved or given for the task: an answer that uses only the retrieved records, a draft that uses only what the record and its named sources support.

**Held-out part.** Questions kept out of development so that a change tuned on the rest is tested on questions it has not seen.

**Incident.** An AI component's output reaching a reviewer, a product, or a user with an error that the controls should have caught, or that a control caught at the last step; handled by a fixed procedure and recorded.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Match type.** How a mention was matched: exact, phrase (contains), fuzzy, semantic, or none.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX and, for some data types, schema.org, Croissant, and DCAT.

**NADA.** The open-source data catalog from the International Household Survey Network, used for the World Bank Microdata Library. It supports DDI Codebook, Dublin Core, ISO 19115/19139, and IPTC. NADA AI, a service being integrated into NADA, adds semantic search and an MCP server over the catalog.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Review board.** The interface in which a curator sees the current and proposed values as a diff and decides.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Rubric.** A written scale that says what each score means, so that two people, or a person and a model, score the same thing the same way.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**SKOS and XKOS.** SKOS is the web standard for publishing controlled vocabularies (concepts with labels and identifiers); XKOS extends it for statistical classifications and their correspondences.

**Small model.** A model of roughly one to fifteen billion parameters that runs on one GPU or on CPU, enough for narrow tasks with the right context.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Variable group.** A DDI structure that organizes variables by theme for navigation.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.

**World Bank metadata schemas.** JSON Schema definitions published by the Development Data Group for indicators, microdata, documents, geospatial data, tables, images, scripts, and videos, used by its catalogs and the Metadata Editor.

**XKOS.** The DDI Alliance extension of SKOS for statistical classifications and the correspondences between them.
