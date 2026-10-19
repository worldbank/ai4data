---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in this guide.
---

# Glossary

**Acceptance rate.** The share of model suggestions that curators accept or edit, per field, task, and model.

**Access tier.** A class of release conditions: public use file, licensed file, or secure access, each with its own disclosure control and agreement.

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Confidence threshold.** The model confidence above which a code is accepted automatically, set from the measured accuracy curve.

**Cost record.** One row per AI component and month with the provider, the number of requests, and the cost, summarized for the governance board.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Croissant.** The MLCommons format, built on schema.org, for describing a machine-learning dataset so that tools can load it: its files with their checksums, the fields of each record, and the splits.

**DCAT.** The Data Catalog Vocabulary, a W3C standard for describing data catalogs and their datasets in machine-readable form so that catalogs can be harvested.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Decision log.** The table with one row per decision of the governance board: approval, change, incident, review, or policy, with the role that decided, the follow-up, and its status.

**Decision point.** The step at which a named person decides on a model's output (a code, a proposal, an explanation, a draft).

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or business or reveals something about one: removal of identifiers, top-coding, coarsening of geography, suppression of rare combinations; applied before any release.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Evidence pack.** The list, by assessment question, of the artefact that answers it, its location, and its date, compiled for an AI-readiness assessment.

**Evidence register.** The table with one row per assessment question below target, naming the document that will show the new answer, its owner, its due date, and its status.

**Exit plan.** The steps by which an AI component would be switched off or moved to a second model or provider, tested once on the component's evaluation suite.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Governance board.** The body, with written terms of reference, that approves AI components, closes incidents, decides reviews, and accepts residual risk.

**GSBPM.** The Generic Statistical Business Process Model, the UNECE reference model of the phases and sub-processes of statistical production, from specifying needs to disseminating and evaluating.

**Incident.** An AI component's output reaching a reviewer, a product, or a user with an error that the controls should have caught, or that a control caught at the last step; handled by a fixed procedure and recorded.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Licence matrix.** The table with one row per published product giving its licence, the licence URL, whether AI use is allowed, under conditions, or prohibited, and whether the licence is machine-readable.

**Machine client.** Software, including an AI assistant or agent, that retrieves data from the organization's API or MCP server and accepts the terms of use for doing so.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**MCP.** The Model Context Protocol, the open standard through which AI applications discover and call tools, read resources, and use prompts offered by a server, such as a statistics API.

**Mention.** A reference to a dataset in a document, named or unnamed.

**ML-ready dataset.** A fixed, versioned set of examples with a documented structure, a stated purpose, a split for training and testing, a statement of whom the examples represent, and a licence that covers model training.

**Number check.** The comparison of every figure in an assistant's answer with the values of the source table, which marks each as verified or unverified before the answer is used.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**Provider register.** The table with one row per model provider, including the organization's own server, giving the services, the components, the data terms, the contract end, the second option, and the exit plan.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Re-assessment.** A repeat of the AI-readiness assessment on the same framework version at the end of a cycle, answered from the evidence register's documents and compared with the first profile.

**Re-coded sample.** A random sample of the automated share coded blind by staff each round to measure the automated accuracy.

**Recall.** The share of labelled mentions that the extractor found.

**Report card.** A run's scores per slice with intervals, the set version, and the component versions, published with a release.

**Required score.** The suite score a task's model has to reach, set from the cost of error before any candidate is tested.

**Residual risk.** The risk that remains after a component's controls are in place, accepted by the role its tier names and recorded in the risk register and the decision log.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Risk tier.** Low, Medium, or High, from the product of a component's three ratings (cost of error, exposure, reversibility); each tier carries a fixed list of controls.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**schema.org Dataset.** A vocabulary for describing datasets on web pages, read by general crawlers and dataset search engines.

**SDMX.** Statistical Data and Metadata eXchange, the standard for exchanging aggregate statistical data, their structure, and code lists; its concept names (SERIES, REF_AREA, TIME_PERIOD, OBS_VALUE) are the column names of this series' data files.

**Semantic search.** Search by meaning; see dense retrieval.

**Small model.** A model of roughly one to fifteen billion parameters that runs on one GPU or on CPU, enough for narrow tasks with the right context.

**Statement of model use.** The section of a release's quality report that says where models were used, which, how they performed, who decided, and what the effect was.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Switch-off test.** The test that production proceeds with the model layer stopped, with the fallback per task exercised.

**Task boundary.** The table that names, for each task where a model is used, the role of the model (propose, explain, draft, retrieve and rank) and the person who decides, and the tasks that stay with people.

**Task card.** The one-page list per role of the everyday tasks an AI assistant may help with, with the model's role, the data tier, and the review for each, and the tasks that stay with the person.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Use log.** One row per use of an AI assistant for an everyday task, with the unit, the role, the task, the tool, the data tier, the reviewer, and the outcome, summarized for the unit and the governance board.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
