---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Synthetic Data for Sharing.
---

# Glossary

**Access tier.** A class of release conditions: public use file, licensed file, or secure access, each with its own disclosure control and agreement.

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Attribute inference.** The risk that a sensitive attribute of real people can be predicted from their other characteristics through the synthetic file; measured against a baseline.

**Closest-record distance.** The distance from a synthetic record to its nearest real record on the quasi-identifiers; compared with the distance between two halves of the real file.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Cramér's V.** A measure of association between two categorical variables, from 0 (unrelated) to 1 (fully determined).

**Data dictionary.** The documentation of a data file's variables: names, labels, types, value codes, missing codes, universes, questions.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Derived variable.** A variable computed from others by a stated rule, for example labour force status from the employment questions.

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or business or reveals something about one: removal of identifiers, top-coding, coarsening of geography, suppression of rare combinations; applied before any release.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or to one version of it, carries citation metadata, and makes citations countable.

**Edit rule.** A machine-readable condition a record must satisfy (range, consistency), owned by the organization and applied before and after any model proposal.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Exact copy.** A synthetic record identical to a real one; a failed check that sends the file back to the method.

**Feature.** A column a model may use as input.

**Fully synthetic.** A file in which every value is generated; the subject of this guide.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**GSBPM.** The Generic Statistical Business Process Model, the UNECE reference model of the phases and sub-processes of statistical production, from specifying needs to disseminating and evaluating.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX and, for some data types, schema.org, Croissant, and DCAT.

**NADA.** The open-source data catalog from the International Household Survey Network, used for the World Bank Microdata Library. It supports DDI Codebook, Dublin Core, ISO 19115/19139, and IPTC. NADA AI, a service being integrated into NADA, adds semantic search and an MCP server over the catalog.

**Partially synthetic.** A file in which only sensitive variables are replaced; a disclosure control method documented with the public use file.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**Public use file.** A microdata file released to anyone under terms of use, after disclosure control.

**Quasi-identifier.** A variable an outsider could know about a person from elsewhere (region, sex, age, education), whose combination can identify someone in a file.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Relational synthesis.** Generation of a child table (persons) conditional on a generated parent table (households), preserving the structure.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Sequential synthesis.** Fitting and generating one variable at a time conditional on the previous ones, with CART or parametric models.

**Synthesis record.** The document that accompanies a synthetic file: source, method and seed, utility and risk results, intended and prohibited uses, label, licence, contact.

**Synthetic data.** Records generated by a model fitted to real data so that they have the same shape and similar relationships while containing no real person's record.

**Target analysis.** A statistic or model that users of the file will compute, compared between the real and the synthetic file in the utility report.

**Target level.** The maturity level a dimension is to reach by the end of a planning cycle, set with a written reason and approved by the board.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Total variation distance.** A measure of the difference between two categorical distributions, from 0 (identical) to 1 (no overlap).

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Utility.** How closely the synthetic file reproduces the real one at the marginal, association, and analysis levels.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Visit sequence.** The order in which variables are synthesized in a sequential method.

**Wave.** A set of dimensions that can be worked in parallel because everything they depend on is already at target or in an earlier wave.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
