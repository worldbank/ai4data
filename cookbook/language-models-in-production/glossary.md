---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Language Models in Statistical Production.
---

# Glossary

**Acceptance rate.** The share of model suggestions that curators accept or edit, per field, task, and model.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**Automated share.** The share of responses coded at or above the confidence threshold without a coder; measured by a re-coded sample each round.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Confidence threshold.** The model confidence above which a code is accepted automatically, set from the measured accuracy curve.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Decision point.** The step at which a named person decides on a model's output (a code, a proposal, an explanation, a draft).

**Derived variable.** A variable computed from others by a stated rule, for example labour force status from the employment questions.

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or business or reveals something about one: removal of identifiers, top-coding, coarsening of geography, suppression of rare combinations; applied before any release.

**Edit rule.** A machine-readable condition a record must satisfy (range, consistency), owned by the organization and applied before and after any model proposal.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better or follows the organization's conventions.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Grounding.** Restricting a language model to what was retrieved or given for the task: an answer that uses only the retrieved records, a draft that uses only what the record and its named sources support.

**GSBPM.** The Generic Statistical Business Process Model, the UNECE reference model of the phases and sub-processes of statistical production, from specifying needs to disseminating and evaluating.

**Harmonization.** The matching of mention variants to canonical identifiers.

**Incident.** An AI component's output reaching a reviewer, a product, or a user with an error that the controls should have caught, or that a control caught at the last step; handled by a fixed procedure and recorded.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Labelled sample.** Sentences labelled by a person with the dataset mention they contain, or none, used to measure the extractor.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX, schema.org, Croissant, and DCAT.

**Method of record.** The documented, reproducible imputation or estimation method the organization applies; a model proposes and explains beside it.

**Number check.** The comparison of every figure in an assistant's answer with the values of the source table, which marks each as verified or unverified before the answer is used.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Question bank.** A repository of question wordings keyed by the concept each measures and the surveys that use each.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Re-coded sample.** A random sample of the automated share coded blind by staff each round to measure the automated accuracy.

**Report card.** A run's scores per slice with intervals, the set version, and the component versions, published with a release.

**Required score.** The suite score a task's model has to reach, set from the cost of error before any candidate is tested.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Statement of model use.** The section of a release's quality report that says where models were used, which, how they performed, who decided, and what the effect was.

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Switch-off test.** The test that production proceeds with the model layer stopped, with the fallback per task exercised.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Universe.** The population a variable applies to, as the question was asked.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
