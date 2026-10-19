---
id: glossary
title: Glossary
sidebar_label: Glossary
sidebar_position: 11
description: Terms used in this guide, in plain language.
---

# Glossary

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**AI-use policy.** The one-page statement, approved at the top of the organization, of which tasks a model may assist, which uses are prohibited, which controls apply, and who owns each instrument.

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Benchmark (population).** The share of each group in the population, from the census or population estimates, against which a dataset's composition is compared.

**Bounding box.** The rectangle that encloses a detected table or figure on a page, written as the left, top, right, and bottom edges as fractions of the page width and height, with the origin at the top left.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Canonical name.** The one name, with its acronym and identifier, under which a dataset's mentions are counted.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Code list.** A published table of the codes used in a dataset (for geography, periods, categories) with their labels. In SDMX, a code list is part of the data structure definition.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**Croissant.** The MLCommons format, built on schema.org, for describing a machine-learning dataset so that tools can load it: its files with their checksums, the fields of each record, and the splits.

**Data snapshot.** The image of one table or figure cropped from a page, with its coordinates, class, and source document; the unit of extraction and of citation.

**DataCite.** The registration agency and metadata schema for dataset DOIs. The World Bank Microdata Library assigns DataCite DOIs.

**Dataset card.** The one document that tells a user what a dataset is, why it exists, how it was collected and labelled, whom it represents, what it is for and not for, its licence, its version, and how to cite it.

**DCAT.** The Data Catalog Vocabulary, a W3C standard for describing data catalogs and their datasets in machine-readable form so that catalogs can be harvested.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Digest.** The SHA-256 hash of a model file, recorded in the lock file and verified before loading.

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or business or reveals something about one: removal of identifiers, top-coding, coarsening of geography, suppression of rare combinations; applied before any release.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Feature.** A column a model may use as input.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better or follows the organization's conventions.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Group key.** The column that names the unit whose records must stay together in a split: a household, a firm, a document.

**Grouped split.** A draw of whole groups (households, documents) into the parts of a dataset, so that no group is divided between training and test.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**JSON-LD.** JSON with a vocabulary attached, so that each key has a defined meaning that different tools read the same way; the format in which schema.org and Croissant records are embedded in web pages.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Labelled sample.** Sentences labelled by a person with the dataset mention they contain, or none, used to measure the extractor.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Layout detection.** A model that finds regions of a page image and returns their class and bounding box.

**Leakage.** Any way the answer reaches a model other than through its features: shared groups across splits, a feature that encodes the label, duplicated inputs.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Lock file.** The record of a model's source, version, and file digests that the server verifies before loading.

**Manifest.** A file that lists the parts of something so that it can be checked or resumed: for a job, the records and their status; for a release, every file with its checksum; for an agent interface, the design document of its tools, inputs, outputs, examples, and resources.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX and, for some data types, schema.org, Croissant, and DCAT.

**ML-ready dataset.** A fixed, versioned set of examples with a documented structure, a stated purpose, a split for training and testing, a statement of whom the examples represent, and a licence that covers model training.

**Public use file.** A microdata file released to anyone under terms of use, after disclosure control.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Reference model.** A simple model trained on the dataset and scored on its test part, published as a baseline so that users can compare their scores and see the error by group.

**Representativeness.** The degree to which a dataset's composition matches the population it will be used on, measured group by group and stated in the card.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**schema.org Dataset.** A vocabulary for describing datasets on web pages, read by general crawlers and dataset search engines.

**SDMX.** Statistical Data and Metadata eXchange, the standard for exchanging aggregate statistical data, their structure, and code lists; its concept names (SERIES, REF_AREA, TIME_PERIOD, OBS_VALUE) are the column names of this series' data files.

**Small model.** A model of roughly one to fifteen billion parameters that runs on one GPU or on CPU, enough for narrow tasks with the right context.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Synthesis record.** The document that accompanies a synthetic file: source, method and seed, utility and risk results, intended and prohibited uses, label, licence, contact.

**Synthetic data.** Records generated by a model fitted to real data so that they have the same shape and similar relationships while containing no real person's record.

**Thin group.** A group with too few examples for a model to learn its patterns; listed in the card with the minimum count used.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Tidy data.** A table layout with one observation per row and one variable per column.

**Tidy file.** A data file with one record per row, one variable per column, and the same columns in every row; for extracted data, one row per value with provenance columns (document, page, bounding box, class, title, row, column, value, unit, status).

**Time-based test part.** A test part made of the latest round of the source, so that a score describes what a model will do on the next round.

**Total variation distance.** A measure of the difference between two categorical distributions, from 0 (identical) to 1 (no overlap).

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Utility.** How closely the synthetic file reproduces the real one at the marginal, association, and analysis levels.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.

**Wave.** A set of dimensions that can be worked in parallel because everything they depend on is already at target or in an earlier wave.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
