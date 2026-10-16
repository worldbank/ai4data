---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to AI-Ready Microdata Documentation.
---

# Glossary

**Access tier.** A class of release conditions: public use file, licensed file, or secure access, each with its own disclosure control and agreement.

**Adapter.** A small set of trained parameters added to a model for a task, so that fine-tuning fits on modest hardware (parameter-efficient fine-tuning).

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**BM25.** A keyword ranking function used by most search engines. It matches exact words and weights rare ones more.

**Calibration.** The comparison of model scores with curator scores on a sample, per dimension, to find systematic bias.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Code list.** A published table of the codes used in a dataset (for geography, periods, categories) with their labels. In SDMX, a code list is part of the data structure definition.

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Concept map.** A table that links each variable to the concept it measures and the classification it follows, with version, level, and URI.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Crosswalk.** A mapping from one classification to another, for example from a national occupation classification to ISCO.

**Data appraisal.** The DDI field for known quality issues of a study: coverage gaps, non-response, comparability limits.

**Data dictionary.** The documentation of a data file's variables: names, labels, types, value codes, missing codes, universes, questions.

**Data snapshot.** The image of one table or figure cropped from a page, with its coordinates, class, and source document; the unit of extraction and of citation.

**DataCite.** The registration agency and metadata schema for dataset DOIs. The World Bank Microdata Library assigns DataCite DOIs.

**DCAT.** The Data Catalog Vocabulary, a W3C standard for describing data catalogs and their datasets in machine-readable form so that catalogs can be harvested.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**DDI Codebook.** The Data Documentation Initiative standard for documenting a study and its variables; the World Bank microdata schema is its JSON form.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Dense retrieval.** Search that represents queries and records as numeric vectors (embeddings) and ranks by similarity of meaning.

**Derived variable.** A variable computed from others by a stated rule, for example labour force status from the employment questions.

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or business or reveals something about one: removal of identifiers, top-coding, coarsening of geography, suppression of rare combinations; applied before any release.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better or follows the organization's conventions.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Harmonization.** The matching of mention variants to canonical identifiers.

**Harmonized name.** A variable name used for the same concept across surveys, tied to one definition.

**ISCED.** The International Standard Classification of Education, whose levels classify educational attainment.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Known-item question.** A test question with one expected identifier, known in advance, used to measure whether a search returns the right record.

**Known-variable question set.** Questions the way users ask for variables, each with the variable that should come first, used to score variable search.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**MCP.** The Model Context Protocol, the open standard through which AI applications discover and call tools, read resources, and use prompts offered by a server, such as a statistics API.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX and, for some data types, schema.org, Croissant, and DCAT.

**Missing-value code.** A code in a variable that marks a missing, inapplicable, refused, or unknown response, which has to be labelled so that it is never read as a value.

**MRR.** Mean reciprocal rank: 1 when the right item comes first, one half when it comes second, and so on, averaged over the questions.

**NADA.** The open-source data catalog from the International Household Survey Network, used for the World Bank Microdata Library. It supports DDI Codebook, Dublin Core, ISO 19115/19139, and IPTC. NADA AI, a service being integrated into NADA, adds semantic search and an MCP server over the catalog.

**OCR.** Optical character recognition: reading text from the image of a page, which is what a scanned document needs before anything can be extracted from it.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.

**OpenAPI.** The standard way to describe an API in a file: which addresses exist, what parameters they take, and what they return.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**Provenance fields.** The fields every data response carries: series identifier, reference area, unit, release date, source URL, licence, citation, and observation status per value.

**Public use file.** A microdata file released to anyone under terms of use, after disclosure control.

**Quality dimensions.** Completeness, semantic alignment, specificity, and consistency, scored from 1 to 5 with a rubric.

**Question bank.** A repository of question wordings keyed by the concept each measures and the surveys that use each.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Read-only.** A tool that reads and returns data and has no path to change anything; the property the server's safety rests on.

**Recall.** The share of labelled mentions that the extractor found.

**Recall@k.** The share of questions for which the right item appears in the first k results.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**schema.org Dataset.** A vocabulary for describing datasets on web pages, read by general crawlers and dataset search engines.

**Semantic search.** Search by meaning; see dense retrieval.

**SKOS and XKOS.** SKOS is the web standard for publishing controlled vocabularies (concepts with labels and identifiers); XKOS extends it for statistical classifications and their correspondences.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Statistical disclosure control.** Methods that prevent the identification of individuals from published data.

**Study record.** The study-level documentation of a survey: title, abstract, dates, coverage, universe, design, access conditions.

**Target level.** The maturity level a dimension is to reach by the end of a planning cycle, set with a written reason and approved by the board.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Universe.** The population a variable applies to, as the question was asked.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Variable group.** A DDI structure that organizes variables by theme for navigation.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.

**World Bank metadata schemas.** JSON Schema definitions published by the Development Data Group for indicators, microdata, documents, geospatial data, tables, images, scripts, and videos, used by its catalogs and the Metadata Editor.

**XKOS.** The DDI Alliance extension of SKOS for statistical classifications and the correspondences between them.
