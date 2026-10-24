# Preface

This page describes the reasons for the training series, its audiences, and the ways to take part at different levels of technical experience. Its last section explains why each session is part of the series and what each notebook of the session adds.

## Why the series exists

Language models now answer many questions about development data. A person who asks an assistant how internet use has changed in Kenya gets a fluent answer with figures in it. When the model answers from its training data alone, some of those figures are close to the official values, some are wrong, and none can be traced to a source. Notebook 01 of session 01 shows this on a real question: of six figures checked, four are close to the official values and two are off.

Two needs follow. Users of AI applications need to tell whether an answer rests on evidence and whether its numbers match the data. Publishers of data need to know how applications read their data, and the program's work on AI-ready data prepares metadata, APIs, and records for that use. The series addresses both needs through working examples.

## Who the series is for

The sessions are written for several groups at once. Each capability is explained in prose and then shown in code. The prose carries the argument and can be read without the code. The code runs without an API key.

**Statisticians and data producers.** AI applications cite the figures published by national statistical offices and other data teams. The sessions show three places where an application reads published data. A search reads documents and their metadata. A tool calls a data API. A server based on the Model Context Protocol offers the same tool to other applications. The sessions also show the effect of weak material: in session 01, an outdated summary in the collection outranks the current briefs in a search. Cases of this kind connect documentation and dissemination practice to the behaviour of an application.

**Researchers and analysts.** Researchers use assistants to find literature, look up figures, and draft summaries. The sessions show how an answer changes when it is grounded in supplied documents or in data. They also show which parts of a grounded answer remain unchecked. The last notebook of session 01 checks each number in an answer against its source record.

**Managers and staff who commission or oversee applications.** People in these roles decide whether an application is fit for use, often without building one. The sessions introduce the terms of application development: retrieval, tools, workflows, agents, and verification. For each component, the sessions state its purpose and its cost, and they show how to test whether it works. A retrieval check on a fixed set of questions is one such test. The verifier's mark on each number is another.

**Developers and data scientists.** The code uses common open-source libraries: LangChain, LangGraph, FastMCP, and Pydantic. Verification uses the program's `pcn-core`. Every notebook runs in Google Colab, and the code can be reused in other projects. The lab notebook builds a complete small application over documents chosen by the participant.

## Taking part at different technical levels

The materials support three levels of engagement.

1. **Reading.** Each notebook appears on this site with its committed outputs, including the model's answers. Reading needs no setup and no knowledge of Python.
2. **Running without a key.** In Colab, every notebook runs from a fresh environment without an API key. The model's answers are then recorded ones, labelled as such. The search, the data tool, the server, the agent, and the verifier run for real. A participant can change a question, a document, or a setting and run the cell again.
3. **Running live or on new documents.** With a key for any compatible model service, the same cells call a model live. The lab notebook runs a small open model inside Colab. Participants' documents stay in the notebook and are not sent to an outside service.

No prior experience with language-model APIs is assumed. Reading Python helps with the code cells. The explanation cells describe each cell and its result.

## Why each session matters

### Session 01 · Building AI applications for research and data

[Session 01](session-01-building-ai-applications/README.md) builds one application around one research question about internet access and economic development in Kenya and Rwanda. Each notebook starts from a problem in the previous version of the application and adds one component to address it. The session sets the shared terms and the reference design of the series, and later sessions build on both.

- **01 · From a language model to an application.** An answer written from training data alone cannot be traced to a source. The notebook places evidence in the prompt and shows how the answer changes. It establishes that the application decides what the model reads.
- **02 · Preparing knowledge for an application.** An application selects evidence from a prepared collection of documents. The notebook shows how metadata, the division of documents into chunks, and the search index determine which passages a later search can find.
- **03 · Retrieval-augmented generation.** The application searches the collection for each question and answers from the retrieved passages. The notebook shows three failures of search, each with a fix: a question with two parts, an outdated passage that outranks current ones, and a question the collection cannot answer. It also shows how to check whether a wrong answer came from the search or from the model.
- **04 · Tools and structured outputs.** Exact figures come from the data through a lookup tool. The notebook also gives the answer a fixed structure readable by other software, and offers the same tool to other applications through the Model Context Protocol. The public Data360 server serves as a second example.
- **05 · Workflows and agents.** A question about two countries needs several searches and lookups. The notebook compares a fixed sequence of steps written in code with an agent choosing its own steps. It also shows the limit on the number of steps an agent may take.
- **06 · Verification.** An answer built from the right evidence can still misstate a number. In this notebook the model tags each figure with the id of its source record, and a verifier checks every tagged figure against that record. Each number is marked verified, flagged, or unmarked.
- **07 · Lab: chat with your documents.** Participants assemble the components of notebooks 02, 03, and 06 into a small application and use it on documents of their choice.
