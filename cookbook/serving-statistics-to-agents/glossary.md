---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Serving Official Statistics to AI Agents.
---

# Glossary

**Access tier.** A class of release conditions: public use file, licensed file, or secure access, each with its own disclosure control and agreement.

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI visibility check.** A fixed set of questions asked of AI assistants on a schedule, recording whether the organization is cited and whether the figures are right.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Client configuration.** The settings a user pastes into an assistant to connect it to the server: endpoint, transport, credentials.

**Code list.** A published table of the codes used in a dataset (for geography, periods, categories) with their labels. In SDMX, a code list is part of the data structure definition.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**COUNTER Code of Practice for Research Data.** Rules for logging and reporting dataset views and downloads, including the separation of machine access from regular access.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Exit plan.** The steps by which an AI component would be switched off or moved to a second model or provider, tested once on the component's evaluation suite.

**Feature.** A column a model may use as input.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Grader.** A language model, with a rubric, that scores answers the automatic checks cannot score; validated against human scores before use.

**Grader (judge).** A language model that scores answers with a rubric where no script can; validated against human scores before use.

**Guidance resource.** A resource that tells the model how to use the tools: search first, read metadata before interpreting, cite the source and release.

**Harness.** A program that sends questions to an agent connected to the server and logs the traces.

**Incident.** An AI component's output reaching a reviewer, a product, or a user with an error that the controls should have caught, or that a control caught at the last step; handled by a fixed procedure and recorded.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Known-item question.** A test question with one expected identifier, known in advance, used to measure whether a search returns the right record.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**MCP.** The Model Context Protocol, the open standard through which AI applications discover and call tools, read resources, and use prompts offered by a server, such as a statistics API.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Model Context Protocol (MCP).** An open standard through which AI assistants discover and call an organization's tools and read its resources.

**NADA.** The open-source data catalog from the International Household Survey Network, used for the World Bank Microdata Library. It supports DDI Codebook, Dublin Core, ISO 19115/19139, and IPTC.

**Number check.** The comparison of every figure in an assistant's answer with the values of the source table, which marks each as verified or unverified before the answer is used.

**OBS_STATUS.** The SDMX observation status attribute, with codes such as `A` normal, `P` provisional, `B` break in series, `E` estimated.

**OpenAPI.** The standard way to describe an API in a file: which addresses exist, what parameters they take, and what they return.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Prompt injection.** Text placed in content a system reads (a document, a web page, a query) that is written as an instruction to the model in the hope that the model follows it; tested for before release.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**Provenance fields.** The fields every data response carries: series identifier, reference area, unit, release date, source URL, licence, citation, and observation status per value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Read-only.** A tool that reads and returns data and has no path to change anything; the property the server's safety rests on.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Rubric.** A written scale that says what each score means, so that two people, or a person and a model, score the same thing the same way.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**SDMX.** Statistical Data and Metadata eXchange, the standard for exchanging aggregate statistical data, their structure, and code lists; its concept names (SERIES, REF_AREA, TIME_PERIOD, OBS_VALUE) are the column names of this series' data files.

**Streamable HTTP.** The transport for servers used over the network; stdio is the transport for servers run locally by a client.

**Structured content.** A tool result returned as JSON that conforms to the tool's output schema, alongside a text rendering.

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Switch-off test.** The test that production proceeds with the model layer stopped, with the fallback per task exercised.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Tool use.** The ability of a language model to call functions or APIs during an answer, so that values come from the source.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Universe.** The population a variable applies to, as the question was asked.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.
