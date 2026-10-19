---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Evaluation Suites for Statistical AI.
---

# Glossary

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI visibility check.** A fixed set of questions asked of AI assistants on a schedule, recording whether the organization is cited and whether the figures are right.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Benchmark (population).** The share of each group in the population, from the census or population estimates, against which a dataset's composition is compared.

**Bootstrap interval.** A confidence interval obtained by resampling the questions with replacement many times; it shows how much a pass rate or a difference could move with a different draw of questions.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Citation validity.** The share of citations in generated answers that resolve to a real record and support the statement they are attached to.

**Cohen's kappa.** A measure of agreement between two labellers that discounts the agreement expected by chance; 1 is perfect agreement, 0 is chance.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Decision log.** The table with one row per decision of the governance board: approval, change, incident, review, or policy, with the role that decided, the follow-up, and its status.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Error taxonomy.** The coded kinds of error a component makes, each pointing to a different fix: for an extractor, missed, spurious, boundary, wrong type; for curation suggestions, wrong flag, invented fact, wrong vocabulary term, style.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Evaluation set.** Decided records kept to test changes to prompts, manifests, rubrics, and models.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Grader.** A language model, with a rubric, that scores answers the automatic checks cannot score; validated against human scores before use.

**Grader (judge).** A language model that scores answers with a rubric where no script can; validated against human scores before use.

**Harness.** A program that sends questions to an agent connected to the server and logs the traces.

**Held-out part.** Questions kept out of development so that a change tuned on the rest is tested on questions it has not seen.

**Incident.** An AI component's output reaching a reviewer, a product, or a user with an error that the controls should have caught, or that a control caught at the last step; handled by a fixed procedure and recorded.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Known-item question.** A test question with one expected identifier, known in advance, used to measure whether a search returns the right record.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Labelled sample.** Sentences labelled by a person with the dataset mention they contain, or none, used to measure the extractor.

**Labelling guide.** The written rules labellers follow, maintained from their disagreements.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Mention.** A reference to a dataset in a document, named or unnamed.

**MRR.** Mean reciprocal rank: 1 when the right item comes first, one half when it comes second, and so on, averaged over the questions.

**nDCG@k.** Normalized discounted cumulative gain at k. A retrieval measure that credits partially relevant results and rewards placing the best ones first.

**Number check.** The comparison of every figure in an assistant's answer with the values of the source table, which marks each as verified or unverified before the answer is used.

**Paired comparison.** The same questions through two versions, so that differences are between versions.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Recall.** The share of labelled mentions that the extractor found.

**Recall@k.** The share of questions for which the right item appears in the first k results.

**Report card.** A run's scores per slice with intervals, the set version, and the component versions, published with a release.

**Required score.** The suite score a task's model has to reach, set from the cost of error before any candidate is tested.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Review board.** The interface in which a curator sees the current and proposed values as a diff and decides.

**Rubric.** A written scale that says what each score means, so that two people, or a person and a model, score the same thing the same way.

**Slice.** A subset of the questions (a language, a question type) scored on its own, so that a change's effect on it is visible.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Tool use.** The ability of a language model to call functions or APIs during an answer, so that values come from the source.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
