# Preface to session 01

This page explains why session 01 is useful to participants with different backgrounds and levels of technical experience. It then explains why each notebook of the session is part of it.

## What the session is about

Many applications built on language models do the same job: they answer questions from a collection of documents and data. An assistant over an organization's reports, a chat interface to a database, a tool for searching research literature, and a helpdesk over policy documents all work this way. They are built from a small set of components: instructions and context, a prepared collection of documents, retrieval, tools and structured outputs, and workflows or agents to combine them. A verification step can be added to check the numbers in an answer.

The session builds one such application from these components, one notebook at a time, around a single research question:

> How have internet access and economic development evolved in Rwanda and Kenya? Compare the recent available data, summarize the main patterns, and cite the sources.

Each notebook starts from a limit of the application built so far and adds one component to address it. Each component also has its own ways of failing, and the notebooks show those failures on the same question. The session therefore covers both building an application of this kind and judging one built by others.

## Who gains from the session

Each component is explained in prose and then shown in code. The prose can be read without the code, and the code runs without an API key.

**Researchers and analysts.** Assistants are now used to search literature, look up figures, and draft summaries. The session shows how far an answer from the model's training data can be trusted: in notebook 01, four of six figures checked are close to the published values, two are off, and none comes with a source. The later notebooks ground the answer in documents and then in data. They also show which parts of a grounded answer remain unchecked, and notebook 06 checks each number of an answer against its source record.

**People who manage documents and data.** The answers of an application depend on how its collection of documents is prepared. Notebook 02 shows how metadata and the division of documents into chunks decide which passages a search returns. In notebook 03, a summary from 2016 outranks the current briefs because its wording matches the question, and a filter on the document date removes it. Cases like this connect the way a collection is documented and maintained to the quality of an application's answers.

**Managers and people who commission or oversee applications.** People in these roles decide whether an application is fit for use, often without building one. The session introduces the terms used in building applications: retrieval, tools, structured outputs, workflows, agents, and verification. For each component it states the purpose, the cost, and a test of whether the component works. Two such tests appear in the notebooks: a retrieval check on a fixed set of questions, and a verifier's mark on each number of an answer. The same terms and tests apply when reviewing an application proposed by a team or a vendor.

**Developers and data scientists.** The code uses common open-source libraries: LangChain for documents, retrieval, and model calls, LangGraph for the agent loop, FastMCP for the tool server, and Pydantic for output schemas. Verification uses `pcn-core`. The patterns carry over to other projects: a search with a date filter, a tool with a typed form, a tool server for any compatible client, and an agent with a step limit. The lab notebook assembles them into a working application over documents chosen by the participant.

## Taking part at different technical levels

The materials support three levels of engagement.

1. **Reading.** Each notebook is published with its outputs, including the model's answers. Reading needs no setup and no knowledge of Python.
2. **Running without a key.** In Google Colab, every notebook runs from a fresh environment without an API key. The model's answers are then recorded ones, labelled as such. The search, the data tool, the tool server, the agent loop, and the verifier run for real. A participant can change a question, a document, or a setting and run the cell again.
3. **Running live or on new documents.** With a key for any OpenAI-compatible model service, the same cells call a model live. The lab notebook runs a small open model inside Colab. The participant's documents stay in the notebook and are not sent to an outside service.

No prior experience with language-model APIs is assumed. Reading Python helps with the code cells. The explanation cells describe each cell and its result.

## Why each notebook is part of the session

- **01 · From a language model to an application.** A model answering from its training data gives figures that cannot be traced to a source. The notebook places evidence in the prompt and compares the two answers. It establishes the first two components, instructions and context, and the principle that the application decides what the model reads.
- **02 · Preparing knowledge for an application.** A prompt holds a few pages, so an application has to select its evidence from a larger collection. Selection needs a prepared collection. Its documents carry metadata, are divided into chunks of whole sections, and are indexed for search by meaning. The notebook builds that collection and shows how its choices affect what a later search can find.
- **03 · Retrieval-augmented generation.** The application now searches the collection for each question and answers from the retrieved passages, with citations. The notebook shows three ways search fails: a question with two parts, an outdated passage that outranks current ones, and a question the collection cannot answer. Each failure comes with a fix, and the notebook shows how to check whether a wrong answer came from the search or from the model.
- **04 · Tools and structured outputs.** Retrieved passages can lack figures needed for an answer. For the main question, the retrieved passages held two of the four figures, and a lookup tool over the data returned all four. The notebook gives the model that tool, makes the answer a structured output that other software can read, and serves the same tool over the Model Context Protocol (MCP) for use by other applications.
- **05 · Workflows and agents.** The main question needs several searches and lookups, and something has to decide their order. The notebook compares a workflow, in which the order is written in code, with an agent, in which the model chooses its own steps. It compares the two on cost, predictability, and testing. It also shows the step limit at work: in a recorded run, the agent keeps calling tools, and the limit ends the run after six steps.
- **06 · Verification.** An answer built from the right evidence can still misstate a number. The model tags each figure with the id of its source record, and a verifier compares every tagged figure with that record. Each number is marked verified, flagged, or unmarked, and the marks show the reader which figures were checked.
- **07 · Lab: chat with your documents.** Participants assemble the search of notebooks 02 and 03, the answering of notebook 01, and the number check of notebook 06 into a small application, and use it on documents of their own.
