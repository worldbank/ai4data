---
id: glossary
title: Shared glossary
---

# Shared glossary

Every term the cookbooks define, in one place. `scripts/docs/build_glossaries.py`
writes each cookbook's glossary page from this file, keeping the terms whose
name (or a match word in the comment under it) appears in that cookbook's
chapters. Edit terms here; never edit the generated pages.

**Acceptance rate.** The share of model suggestions that curators accept or edit, per field, task, and model.

**Access tier.** A class of release conditions: public use file, licensed file, or secure access, each with its own disclosure control and agreement.

**Adapter.** A small set of trained parameters added to a model for a task, so that fine-tuning fits on modest hardware (parameter-efficient fine-tuning).

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**Agents manifest.** The file that defines the agents of the review pipeline (detectors, critic, categorizer, severity scorer) and their instructions.

**AI visibility check.** A fixed set of questions asked of AI assistants on a schedule, recording whether the organization is cited and whether the figures are right.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Attribute inference.** The risk that a sensitive attribute of real people can be predicted from their other characteristics through the synthetic file; measured against a baseline.

**Automated share.** The share of responses coded at or above the confidence threshold without a coder; measured by a re-coded sample each round.

**Benchmark (population).** The share of each group in the population, from the census or population estimates, against which a dataset's composition is compared.

**BM25.** A keyword ranking function used by most search engines. It matches exact words and weights rare ones more.

**Bootstrap interval.** A confidence interval obtained by resampling the questions with replacement many times; it shows how much a pass rate or a difference could move with a different draw of questions.

**Bounding box.** The rectangle that encloses a detected table or figure on a page, written as the left, top, right, and bottom edges as fractions of the page width and height, with the origin at the top left.
<!-- match: bounding box; bbox -->

**Calibration.** The comparison of model scores with curator scores on a sample, per dimension, to find systematic bias.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Canonical name.** The one name, with its acronym and identifier, under which a dataset's mentions are counted.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.
<!-- match: checksum; sha-256 -->

**Citation validity.** The share of citations in generated answers that resolve to a real record and support the statement they are attached to.

**Client configuration.** The settings a user pastes into an assistant to connect it to the server: endpoint, transport, credentials.

**Closest-record distance.** The distance from a synthetic record to its nearest real record on the quasi-identifiers; compared with the distance between two halves of the real file.

**Co-use.** The use of another organization's dataset in the same document as the organization's own.

**Code list.** A published table of the codes used in a dataset (for geography, periods, categories) with their labels. In SDMX, a code list is part of the data structure definition.

**Cohen's kappa.** A measure of agreement between two labellers that discounts the agreement expected by chance; 1 is perfect agreement, 0 is chance.
<!-- match: kappa -->

**Component register.** The table with one row per AI component the organization runs: owner, model, provider, data, human review, risk tier, approval and review dates, and exit plan.

**Concept map.** A table that links each variable to the concept it measures and the classification it follows, with version, level, and URI.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Confidence threshold.** The model confidence above which a code is accepted automatically, set from the measured accuracy curve.

**Content-Oriented Guidelines.** The SDMX guidelines that define cross-domain concepts (`REF_AREA`, `TIME_PERIOD`, `OBS_VALUE`, `OBS_STATUS`, `UNIT_MEASURE`, and others) and cross-domain code lists for reuse across statistical domains.

**Cost record.** One row per AI component and month with the provider, the number of requests, and the cost, summarized for the governance board.

**Cost-quality frontier.** The candidates that no other candidate beats on both suite score and cost per query.

**COUNTER Code of Practice for Research Data.** Rules for logging and reporting dataset views and downloads, including the separation of machine access from regular access.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Cramér's V.** A measure of association between two categorical variables, from 0 (unrelated) to 1 (fully determined).
<!-- match: cramér; cramer -->

**Croissant.** The MLCommons format, built on schema.org, for describing a machine-learning dataset so that tools can load it: its files with their checksums, the fields of each record, and the splits.

**Crosswalk.** A mapping from one classification to another, for example from a national occupation classification to ISCO.

**Data appraisal.** The DDI field for known quality issues of a study: coverage gaps, non-response, comparability limits.

**Data dictionary.** The documentation of a data file's variables: names, labels, types, value codes, missing codes, universes, questions.

**Data snapshot.** The image of one table or figure cropped from a page, with its coordinates, class, and source document; the unit of extraction and of citation.

**Data tier.** The class of a text for the purpose of entering it into an assistant: public (published), internal (drafts, embargoed, for the organization only), or confidential (anything at the level of a respondent, a record, or a member of staff, and any identifier).

**DataCite.** The registration agency and metadata schema for dataset DOIs. The World Bank Microdata Library assigns DataCite DOIs.

**Dataset card.** The one document that tells a user what a dataset is, why it exists, how it was collected and labelled, whom it represents, what it is for and not for, its licence, its version, and how to cite it.

**DCAT.** The Data Catalog Vocabulary, a W3C standard for describing data catalogs and their datasets in machine-readable form so that catalogs can be harvested.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**DDI Codebook.** The Data Documentation Initiative standard for documenting a study and its variables; the World Bank microdata schema is its JSON form.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Decision log.** The table with one row per decision of the governance board: approval, change, incident, review, or policy, with the role that decided, the follow-up, and its status.

**Decision point.** The step at which a named person decides on a model's output (a code, a proposal, an explanation, a draft).

**Dense retrieval.** Search that represents queries and records as numeric vectors (embeddings) and ranks by similarity of meaning.

**Derived column.** A table column computed from others, such as a rate from a numerator and a denominator; a check for extracted values.

**Derived variable.** A variable computed from others by a stated rule, for example labour force status from the employment questions.

**Digest.** The SHA-256 hash of a model file, recorded in the lock file and verified before loading.

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or business or reveals something about one: removal of identifiers, top-coding, coarsening of geography, suppression of rare combinations; applied before any release.
<!-- match: disclosure -->

**Document record.** The catalog record of a document in the World Bank document schema, with the series and surveys it draws on.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Draft status.** The mark on a model-written field that keeps it out of the published catalog until a decision.

**Edit rule.** A machine-readable condition a record must satisfy (range, consistency), owned by the organization and applied before and after any model proposal.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Error taxonomy.** The coded kinds of error a component makes, each pointing to a different fix: for an extractor, missed, spurious, boundary, wrong type; for curation suggestions, wrong flag, invented fact, wrong vocabulary term, style.
<!-- match: error taxonomy; taxonomy -->

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Evaluation set.** Decided records kept to test changes to prompts, manifests, rubrics, and models.

**Evidence pack.** The list, by assessment question, of the artefact that answers it, its location, and its date, compiled for an AI-readiness assessment.

**Evidence register.** The table with one row per assessment question below target, naming the document that will show the new answer, its owner, its due date, and its status.

**Exact copy.** A synthetic record identical to a real one; a failed check that sends the file back to the method.

**Exit plan.** The steps by which an AI component would be switched off or moved to a second model or provider, tested once on the component's evaluation suite.

**Feature.** A column a model may use as input.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better or follows the organization's conventions.
<!-- match: fine-tun -->

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Fully synthetic.** A file in which every value is generated; the subject of this guide.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Governance board.** The body, with written terms of reference, that approves AI components, closes incidents, decides reviews, and accepts residual risk.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**Grader.** A language model, with a rubric, that scores answers the automatic checks cannot score; validated against human scores before use.

**Grader (judge).** A language model that scores answers with a rubric where no script can; validated against human scores before use.
<!-- match: grader -->

**Grounding.** Restricting a language model to what was retrieved or given for the task: an answer that uses only the retrieved records, a draft that uses only what the record and its named sources support.
<!-- match: grounded; grounding -->

**Group key.** The column that names the unit whose records must stay together in a split: a household, a firm, a document.

**Grouped split.** A draw of whole groups (households, documents) into the parts of a dataset, so that no group is divided between training and test.

**GSBPM.** The Generic Statistical Business Process Model, the UNECE reference model of the phases and sub-processes of statistical production, from specifying needs to disseminating and evaluating.

**Guidance resource.** A resource that tells the model how to use the tools: search first, read metadata before interpreting, cite the source and release.

**Hallucination.** A statement in a generated answer that has no support in the retrieved material.

**Harmonization.** The matching of mention variants to canonical identifiers.

**Harmonized name.** A variable name used for the same concept across surveys, tied to one definition.

**Harness.** A program that sends questions to an agent connected to the server and logs the traces.

**Held-out part.** Questions kept out of development so that a change tuned on the rest is tested on questions it has not seen.

**Improvement pathway.** The document, generated from the profile, the targets, the sequence, and the estimate, that a board approves and a funder reads, with the narrative and the dates added by the planning team.

**Incident.** An AI component's output reaching a reviewer, a product, or a user with an error that the controls should have caught, or that a control caught at the last step; handled by a fixed procedure and recorded.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**ISCED.** The International Standard Classification of Education, whose levels classify educational attainment.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**JSON-LD.** JSON with a vocabulary attached, so that each key has a defined meaning that different tools read the same way; the format in which schema.org and Croissant records are embedded in web pages.

**Key-value cache.** The memory a model uses per token of context during generation; grows with context length and concurrency.

**Known-item question.** A test question with one expected identifier, known in advance, used to measure whether a search returns the right record.
<!-- match: known-item -->

**Known-variable question set.** Questions the way users ask for variables, each with the variable that should come first, used to score variable search.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Labelled sample.** Sentences labelled by a person with the dataset mention they contain, or none, used to measure the extractor.

**Labelling guide.** The written rules labellers follow, maintained from their disagreements.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Layout detection.** A model that finds regions of a page image and returns their class and bounding box.

**Leakage.** Any way the answer reaches a model other than through its features: shared groups across splits, a feature that encodes the label, duplicated inputs.

**Licence matrix.** The table with one row per published product giving its licence, the licence URL, whether AI use is allowed, under conditions, or prohibited, and whether the licence is machine-readable.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Local assistant.** A chat interface on an open-weight model running on the organization's own server, with no connection to an external service, used for the confidential tasks the task card allows.

**Lock file.** The record of a model's source, version, and file digests that the server verifies before loading.

**Machine client.** Software, including an AI assistant or agent, that retrieves data from the organization's API or MCP server and accepts the terms of use for doing so.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Match type.** How a mention was matched: exact, phrase (contains), fuzzy, semantic, or none.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**MCP.** The Model Context Protocol, the open standard through which AI applications discover and call tools, read resources, and use prompts offered by a server, such as a statistics API.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX, schema.org, Croissant, and DCAT.

**Method of record.** The documented, reproducible imputation or estimation method the organization applies; a model proposes and explains beside it.

**Missing-value code.** A code in a variable that marks a missing, inapplicable, refused, or unknown response, which has to be labelled so that it is never read as a value.

**ML-ready dataset.** A fixed, versioned set of examples with a documented structure, a stated purpose, a split for training and testing, a statement of whom the examples represent, and a licence that covers model training.

**Model Context Protocol (MCP).** An open standard through which AI assistants discover and call an organization's tools and read its resources.

**MRR.** Mean reciprocal rank: 1 when the right item comes first, one half when it comes second, and so on, averaged over the questions.

**NADA.** The open-source data catalog from the International Household Survey Network, used for the World Bank Microdata Library. It supports DDI Codebook, Dublin Core, ISO 19115/19139, and IPTC.

**Named-entity extraction.** A model that finds spans of text referring to entities of a given type, here dataset mentions, with a confidence.

**nDCG@k.** Normalized discounted cumulative gain at k. A retrieval measure that credits partially relevant results and rewards placing the best ones first.

**Number check.** The comparison of every figure in an assistant's answer with the values of the source table, which marks each as verified or unverified before the answer is used.

**OBS_STATUS.** The SDMX observation status attribute, with codes such as `A` normal, `P` provisional, `B` break in series, `E` estimated.

**OCR.** Optical character recognition: reading text from the image of a page, which is what a scanned document needs before anything can be extracted from it.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.
<!-- match: open-weight -->

**OpenAPI.** The standard way to describe an API in a file: which addresses exist, what parameters they take, and what they return.

**Paired comparison.** The same questions through two versions, so that differences are between versions.

**Partially synthetic.** A file in which only sensitive variables are replaced; a disclosure control method documented with the public use file.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.
<!-- match: precision; recall -->

**Printed value label.** A number printed on a chart next to its bar or point, which makes the extracted value verifiable.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Prompt injection.** Text placed in content a system reads (a document, a web page, a query) that is written as an instruction to the model in the hope that the model follows it; tested for before release.

**Prompt library.** The organization's versioned file of questions for the tasks on the task card, each with its source, format, and constraints, its data tier, and its review.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**Provenance fields.** The fields every data response carries: series identifier, reference area, unit, release date, source URL, licence, citation, and observation status per value.

**Provider register.** The table with one row per model provider, including the organization's own server, giving the services, the components, the data terms, the contract end, the second option, and the exit plan.

**Public use file.** A microdata file released to anyone under terms of use, after disclosure control.

**Quality dimensions.** Completeness, semantic alignment, specificity, and consistency, scored from 1 to 5 with a rubric.

**Quantization.** Storing weights at fewer bits (8 or 4 in place of 16) to reduce memory, with a suite check for any loss.

**Quasi-identifier.** A variable an outsider could know about a person from elsewhere (region, sex, age, education), whose combination can identify someone in a file.

**Question bank.** A repository of question wordings keyed by the concept each measures and the surveys that use each.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Re-assessment.** A repeat of the AI-readiness assessment on the same framework version at the end of a cycle, answered from the evidence register's documents and compared with the first profile.

**Re-coded sample.** A random sample of the automated share coded blind by staff each round to measure the automated accuracy.

**Read-only.** A tool that reads and returns data and has no path to change anything; the property the server's safety rests on.

**Readiness profile.** The summary of an AI-readiness assessment: an average and a level per dimension, an average per pillar, and an overall score, computed from the question-level answers.

**Recall.** The share of labelled mentions that the extractor found.

**Recall@k.** The share of questions for which the right item appears in the first k results.
<!-- match: recall@ -->

**Reference list.** Known uses of the data, assembled from staff knowledge and citation tracking, against which coverage is measured.

**Reference model.** A simple model trained on the dataset and scored on its test part, published as a baseline so that users can compare their scores and see the error by group.

**Relational synthesis.** Generation of a child table (persons) conditional on a generated parent table (households), preserving the structure.

**Report card.** A run's scores per slice with intervals, the set version, and the component versions, published with a release.

**Representativeness.** The degree to which a dataset's composition matches the population it will be used on, measured group by group and stated in the card.

**Required score.** The suite score a task's model has to reach, set from the cost of error before any candidate is tested.

**Residual risk.** The risk that remains after a component's controls are in place, accepted by the role its tier names and recorded in the risk register and the decision log.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Review board.** The interface in which a curator sees the current and proposed values as a diff and decides.

**Risk tier.** Low, Medium, or High, from the product of a component's three ratings (cost of error, exposure, reversibility); each tier carries a fixed list of controls.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Rubric.** A written scale that says what each score means, so that two people, or a person and a model, score the same thing the same way.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**schema.org Dataset.** A vocabulary for describing datasets on web pages, read by general crawlers and dataset search engines.

**SDMX.** Statistical Data and Metadata eXchange, the standard for exchanging aggregate statistical data, their structure, and code lists; its concept names (SERIES, REF_AREA, TIME_PERIOD, OBS_VALUE) are the column names of this series' data files.

**Semantic search.** Search by meaning; see dense retrieval.

**Sequential synthesis.** Fitting and generating one variable at a time conditional on the previous ones, with CART or parametric models.

**Serving stack.** The software that loads a model and answers requests (a single-machine runtime or a batching server).

**SKOS and XKOS.** SKOS is the web standard for publishing controlled vocabularies (concepts with labels and identifiers); XKOS extends it for statistical classifications and their correspondences.
<!-- match: skos; xkos -->

**Slice.** A subset of the questions (a language, a question type) scored on its own, so that a change's effect on it is visible.

**Small model.** A model of roughly one to fifteen billion parameters that runs on one GPU or on CPU, enough for narrow tasks with the right context.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Stable identifier.** An identifier for a series or dataset that does not change across releases or site redesigns.

**Statement of model use.** The section of a release's quality report that says where models were used, which, how they performed, who decided, and what the effect was.

**Statistical disclosure control.** Methods that prevent the identification of individuals from published data.

**Streamable HTTP.** The transport for servers used over the network; stdio is the transport for servers run locally by a client.

**Structured content.** A tool result returned as JSON that conforms to the tool's output schema, alongside a text rendering.

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Study record.** The study-level documentation of a survey: title, abstract, dates, coverage, universe, design, access conditions.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Switch-off test.** The test that production proceeds with the model layer stopped, with the fallback per task exercised.

**Synthesis record.** The document that accompanies a synthetic file: source, method and seed, utility and risk results, intended and prohibited uses, label, licence, contact.

**Synthetic data.** Records generated by a model fitted to real data so that they have the same shape and similar relationships while containing no real person's record.
<!-- match: synthetic -->

**Table record.** The catalog record of a statistical table in the World Bank table schema, with columns, rows, sources, definitions, citation, and relations.

**Target analysis.** A statistic or model that users of the file will compute, compared between the real and the synthetic file in the utility report.

**Target level.** The maturity level a dimension is to reach by the end of a planning cycle, set with a written reason and approved by the board.

**Task boundary.** The table that names, for each task where a model is used, the role of the model (propose, explain, draft, retrieve and rank) and the person who decides, and the tasks that stay with people.

**Task card.** The one-page list per role of the everyday tasks an AI assistant may help with, with the model's role, the data tier, and the review for each, and the tasks that stay with the person.

**Text layer.** The machine-readable text inside a PDF; absent in scanned documents until OCR produces it.

**Thin group.** A group with too few examples for a model to learn its patterns; listed in the card with the minimum count used.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Tidy data.** A table layout with one observation per row and one variable per column.

**Tidy file.** A data file with one record per row, one variable per column, and the same columns in every row; for extracted data, one row per value with provenance columns (document, page, bounding box, class, title, row, column, value, unit, status).
<!-- match: tidy -->

**Time-based test part.** A test part made of the latest round of the source, so that a score describes what a model will do on the next round.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Tool execution error.** A tool result marked as an error with a message the model can act on, as distinct from a protocol error.

**Tool use.** The ability of a language model to call functions or APIs during an answer, so that values come from the source.

**Total row.** The row of a table that sums the others; a check for extracted values.

**Total variation distance.** A measure of the difference between two categorical distributions, from 0 (identical) to 1 (no overlap).
<!-- match: total variation -->

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Typology of use.** The distinction between mention and use, and among primary, secondary, and background use.

**Unit session.** A two-hour session in which a unit works through the task card, the tiers, a library prompt, the number check, and the use log on its own texts.

**Universe.** The population a variable applies to, as the question was asked.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Use log.** One row per use of an AI assistant for an everyday task, with the unit, the role, the task, the tool, the data tier, the reviewer, and the outcome, summarized for the unit and the governance board.

**Utility.** How closely the synthetic file reproduces the real one at the marginal, association, and analysis levels.

**Variable group.** A DDI structure that organizes variables by theme for navigation.

**Variant.** A way of naming a dataset other than its canonical name: an acronym, an informal name, a translation, a misspelling.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Vision-language model.** A model that reads an image and returns text or structured output; used to extract chart data from snapshots.
<!-- match: vision-language -->

**Visit sequence.** The order in which variables are synthesized in a sequential method.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.

**Wave.** A set of dimensions that can be worked in parallel because everything they depend on is already at target or in an earlier wave.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.

**World Bank metadata schemas.** JSON Schema definitions published by the Development Data Group for indicators, microdata, documents, geospatial data, tables, images, scripts, and videos, used by its catalogs and the Metadata Editor.

**XKOS.** The DDI Alliance extension of SKOS for statistical classifications and the correspondences between them.

**Zero-shot model.** A model that performs a task from a description of the labels alone, without examples of that task in its training.
<!-- match: zero-shot -->
