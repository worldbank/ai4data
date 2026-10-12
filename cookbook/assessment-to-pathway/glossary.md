---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in this guide.
---

# Glossary

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Code list.** A published table of the codes used in a dataset (for geography, periods, categories) with their labels. In SDMX, a code list is part of the data structure definition.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Decision log.** The table with one row per decision of the governance board: approval, change, incident, review, or policy, with the role that decided, the follow-up, and its status.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Evidence pack.** The list, by assessment question, of the artefact that answers it, its location, and its date, compiled for an AI-readiness assessment.

**Evidence register.** The table with one row per assessment question below target, naming the document that will show the new answer, its owner, its due date, and its status.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Governance board.** The body, with written terms of reference, that approves AI components, closes incidents, decides reviews, and accepts residual risk.

**Grounding.** Restricting a language model to what was retrieved or given for the task: an answer that uses only the retrieved records, a draft that uses only what the record and its named sources support.

**Improvement pathway.** The document, generated from the profile, the targets, the sequence, and the estimate, that a board approves and a funder reads, with the narrative and the dates added by the planning team.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Licence matrix.** The table with one row per published product giving its licence, the licence URL, whether AI use is allowed, under conditions, or prohibited, and whether the licence is machine-readable.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**MCP.** The Model Context Protocol, the open standard through which AI applications discover and call tools, read resources, and use prompts offered by a server, such as a statistics API.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX, schema.org, Croissant, and DCAT.

**ML-ready dataset.** A fixed, versioned set of examples with a documented structure, a stated purpose, a split for training and testing, a statement of whom the examples represent, and a licence that covers model training.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Re-assessment.** A repeat of the AI-readiness assessment on the same framework version at the end of a cycle, answered from the evidence register's documents and compared with the first profile.

**Readiness profile.** The summary of an AI-readiness assessment: an average and a level per dimension, an average per pillar, and an overall score, computed from the question-level answers.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Semantic search.** Search by meaning; see dense retrieval.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Statement of model use.** The section of a release's quality report that says where models were used, which, how they performed, who decided, and what the effect was.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Synthetic data.** Records generated by a model fitted to real data so that they have the same shape and similar relationships while containing no real person's record.

**Target level.** The maturity level a dimension is to reach by the end of a planning cycle, set with a written reason and approved by the board.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Wave.** A set of dimensions that can be worked in parallel because everything they depend on is already at target or in an earlier wave.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
