# From fluent answers to verifiable answers

This preface sets out the ideas behind session 01, its intended readers, and the place of each notebook in the argument of the session.

## The problem

A language model writes fluent text on almost any subject, figures included. Fluency is a property of the wording, and it tells a reader nothing about the evidence behind the figures. A figure recalled from training and a figure copied from a published table read the same way on the page.

For anyone who uses such answers in research, analysis, or decisions, two questions follow. How does an answer come to rest on evidence? How can a reader tell whether it does?

## The application around the model

On its own, a model answers from what it absorbed in training. An application is the software around the model. It decides what the model reads, where figures come from, in what order the steps run, and what happens to the answer afterwards. Most of what makes an answer reliable is decided there, in the application, and the session treats each of those decisions as a component.

The session rests on three ideas: grounding, the structure of failure, and the separation of writing from checking.

**Grounding.** An answer is grounded when it rests on evidence placed in front of the model. Evidence comes in different kinds, and each kind has its own channel. Background and instructions go into the prompt. Passages are retrieved from a collection of documents. Exact figures are looked up in the data through a tool.

**The structure of failure.** Each component fails in its own way. A search returns the wrong passages, a tool is asked for data that does not exist, and an agent keeps calling tools without reaching an answer. A failure can be fixed once its origin is known, so the session traces every failure it shows to the component responsible for it.

**Separating writing from checking.** Verification gives the checking of numbers to a separate step. The model labels each figure with its source, and a verifier compares the figure with the source. Confidence in a number then comes from the comparison.

## Readers from different backgrounds

The design described in the session is shared by many applications: assistants over reports, interfaces to databases, tools for literature search, and helpdesks over policy documents. People meet this design from different sides, and the session is written for several of them.

**Researchers and analysts** use the answers. For them the session is a way to judge an answer by asking what evidence it rests on and whether its figures have been checked.

**People who maintain collections of documents and data** supply the evidence. For them the session shows that the description and upkeep of a collection, such as its metadata, its dates, and the way documents are divided, shape what an application can find.

**Managers and reviewers** decide whether an application is fit for use. For them the components form a vocabulary for asking how an application works, and the tests in the session show what evidence of reliability looks like.

**Developers and data scientists** build the applications. For them the session presents each pattern with its trade-offs, in widely used open-source libraries: LangChain, LangGraph, FastMCP, and Pydantic.

## Readers at different technical levels

The argument of the session is carried by the explanations, and the code demonstrates it. A reader without programming experience can follow the argument from the explanations and the published outputs. A reader who programs can run the notebooks, change a question or a document, and reuse the code elsewhere. No API key is needed at any level: without one, the notebooks replay recorded answers from the model and run everything else for real.

## The place of each notebook

- **01 · From a language model to an application.** This notebook takes the step from a model to an application. Placing evidence in the prompt changes the answer, because the model writes from the material the application supplies.
- **02 · Preparing knowledge for an application.** What a search can find is settled before any question is asked, when documents are described, dated, and divided.
- **03 · Retrieval-augmented generation.** Selecting evidence for each question becomes part of the application. Selection is a source of error of its own, distinct from the writing of the answer, and it has its own fixes.
- **04 · Tools and structured outputs.** Documents suit definitions and context, and exact figures are better taken from the data itself. A tool gives the model access to the data. The notebook also gives the answer a structure readable by other software, and it offers the tool to other applications through a shared protocol.
- **05 · Workflows and agents.** Once an application has several sources, something must decide the order of the steps. Code can fix the order in advance, or the model can choose it. The choice trades predictability against flexibility, and an agent needs limits on how long it may run.
- **06 · Verification.** Writing and checking become separate steps. Each figure carries a reference to its source, and every reference is checked. The reader sees which numbers were verified, which were flagged, and which could not be checked.
- **07 · Lab: chat with your documents.** The lab applies the same components to a collection chosen by the participant. It tests whether the ideas of the session carry over to new material.
