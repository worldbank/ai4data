---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in this guide.
---

# Glossary

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Data tier.** The class of a text for the purpose of entering it into an assistant: public (published), internal (drafts, embargoed, for the organization only), or confidential (anything at the level of a respondent, a record, or a member of staff, and any identifier).

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Decision log.** The table with one row per decision of the governance board: approval, change, incident, review, or policy, with the role that decided, the follow-up, and its status.

**Decision point.** The step at which a named person decides on a model's output (a code, a proposal, an explanation, a draft).

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Edit rule.** A machine-readable condition a record must satisfy (range, consistency), owned by the organization and applied before and after any model proposal.

**Exit plan.** The steps by which an AI component would be switched off or moved to a second model or provider, tested once on the component's evaluation suite.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Governance board.** The body, with written terms of reference, that approves AI components, closes incidents, decides reviews, and accepts residual risk.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**Incident.** An AI component's output reaching a reviewer, a product, or a user with an error that the controls should have caught, or that a control caught at the last step; handled by a fixed procedure and recorded.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Local assistant.** A chat interface on an open-weight model running on the organization's own server, with no connection to an external service, used for the confidential tasks the task card allows.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Number check.** The comparison of every figure in an assistant's answer with the values of the source table, which marks each as verified or unverified before the answer is used.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Prompt library.** The organization's versioned file of questions for the tasks on the task card, each with its source, format, and constraints, its data tier, and its review.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Risk tier.** Low, Medium, or High, from the product of a component's three ratings (cost of error, exposure, reversibility); each tier carries a fixed list of controls.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Task boundary.** The table that names, for each task where a model is used, the role of the model (propose, explain, draft, retrieve and rank) and the person who decides, and the tasks that stay with people.

**Task card.** The one-page list per role of the everyday tasks an AI assistant may help with, with the model's role, the data tier, and the review for each, and the tasks that stay with the person.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Unit session.** A two-hour session in which a unit works through the task card, the tiers, a library prompt, the number check, and the use log on its own texts.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Use log.** One row per use of an AI assistant for an everyday task, with the unit, the role, the task, the tool, the data tier, the reviewer, and the outcome, summarized for the unit and the governance board.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
