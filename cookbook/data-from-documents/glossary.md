---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Extracting Data from Documents.
---

# Glossary

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI visibility check.** A fixed set of questions asked of AI assistants on a schedule, recording whether the organization is cited and whether the figures are right.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Benchmark (population).** The share of each group in the population, from the census or population estimates, against which a dataset's composition is compared.

**Bounding box.** The rectangle that encloses a detected table or figure on a page, written as the left, top, right, and bottom edges as fractions of the page width and height, with the origin at the top left.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Code list.** A published table of the codes used in a dataset (for geography, periods, categories) with their labels. In SDMX, a code list is part of the data structure definition.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Data snapshot.** The image of one table or figure cropped from a page, with its coordinates, class, and source document; the unit of extraction and of citation.

**DataCite.** The registration agency and metadata schema for dataset DOIs. The World Bank Microdata Library assigns DataCite DOIs.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Derived column.** A table column computed from others, such as a rate from a numerator and a denominator; a check for extracted values.

**Document record.** The catalog record of a document in the World Bank document schema, with the series and surveys it draws on.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better or follows the organization's conventions.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Known-item question.** A test question with one expected identifier, known in advance, used to measure whether a search returns the right record.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Layout detection.** A model that finds regions of a page image and returns their class and bounding box.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX, schema.org, Croissant, and DCAT.

**NADA.** The open-source data catalog from the International Household Survey Network, used for the World Bank Microdata Library. It supports DDI Codebook, Dublin Core, ISO 19115/19139, and IPTC.

**Number check.** The comparison of every figure in an assistant's answer with the values of the source table, which marks each as verified or unverified before the answer is used.

**OBS_STATUS.** The SDMX observation status attribute, with codes such as `A` normal, `P` provisional, `B` break in series, `E` estimated.

**OCR.** Optical character recognition: reading text from the image of a page, which is what a scanned document needs before anything can be extracted from it.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**Provenance fields.** The fields every data response carries: series identifier, reference area, unit, release date, source URL, licence, citation, and observation status per value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Recall.** The share of labelled mentions that the extractor found.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Review board.** The interface in which a curator sees the current and proposed values as a diff and decides.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Rubric.** A written scale that says what each score means, so that two people, or a person and a model, score the same thing the same way.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**schema.org Dataset.** A vocabulary for describing datasets on web pages, read by general crawlers and dataset search engines.

**SDMX.** Statistical Data and Metadata eXchange, the standard for exchanging aggregate statistical data, their structure, and code lists; its concept names (SERIES, REF_AREA, TIME_PERIOD, OBS_VALUE) are the column names of this series' data files.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Table record.** The catalog record of a statistical table in the World Bank table schema, with columns, rows, sources, definitions, citation, and relations.

**Target level.** The maturity level a dimension is to reach by the end of a planning cycle, set with a written reason and approved by the board.

**Text layer.** The machine-readable text inside a PDF; absent in scanned documents until OCR produces it.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Tidy data.** A table layout with one observation per row and one variable per column.

**Tidy file.** A data file with one record per row, one variable per column, and the same columns in every row; for extracted data, one row per value with provenance columns (document, page, bounding box, class, title, row, column, value, unit, status).

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Total row.** The row of a table that sums the others; a check for extracted values.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Variant.** A way of naming a dataset other than its canonical name: an acronym, an informal name, a translation, a misspelling.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Vision-language model.** A model that reads an image and returns text or structured output; used to extract chart data from snapshots.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
