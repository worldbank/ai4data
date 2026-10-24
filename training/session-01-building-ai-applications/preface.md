# From fluent answers to verifiable answers

This preface explains the problem addressed in session 01 and the three ideas developed in its notebooks. It then describes how the session can be used by readers with different backgrounds and different levels of technical experience. The last section describes each notebook in turn.

## The problem

A language model can write fluent text on almost any subject, and that text often contains figures such as percentages, totals, and years. The quality of the writing gives no information about the origin of those figures. When a model answers from its training alone, a figure may be correct, out of date, or invented, and a reader of the answer has no means of establishing which of these is the case. Anyone who uses such answers in research, analysis, or decision-making needs answers that are based on identified evidence, together with a way to check each figure against that evidence. Session 01 builds an application that meets these two needs.

## The application around the model

An application is the software that surrounds a language model. The application determines which material the model receives with each question, and it processes the answer after the model has written it. Session 01 builds such an application in stages. Each notebook adds a component, such as a search over documents, a tool that queries a database, a workflow that fixes the order of the steps, or a verifier that checks figures.

### Grounding

An answer is grounded when its statements and figures come from evidence that the application has supplied to the model. The session supplies evidence through three channels. Notebook 01 writes instructions and background information directly into the prompt. In notebooks 02 and 03, the application searches a collection of documents for each question and adds the retrieved passages to the prompt. In notebook 04, the model obtains exact figures, such as the share of a country's population using the internet in a given year, by calling a tool that queries a database of indicators. The session uses documents for definitions, methods, and context, and it uses the database for exact values.

### Tracing failures to components

When an application returns a wrong answer, the cause can usually be located in one component, and the correction depends on which component failed. Several notebooks contain a deliberately introduced failure. In notebook 03, the search ranks a 2016 summary highly for a question about the latest figures. The similarity score of a passage does not take its date into account, so the model receives outdated evidence. The correction belongs in the search, and the notebook adds a filter that excludes documents dated before 2020 from the ranking. In notebook 04, the model asks the tool for a year that has no observation, and the instructions in the prompt determine how the model reports the missing value. In notebook 05, an agent keeps calling tools without producing an answer until a limit on the number of steps ends the run.

### Separating writing from checking

A grounded answer can still contain a wrong figure, for instance when the model alters a value while copying it into the text. Notebook 06 therefore assigns the checking of figures to a separate step that does not involve the model. The model attaches a tag to each figure that names the source record of the figure. A verifier written in ordinary code compares each tagged figure with the named record. It marks the figure as verified when the two match and as flagged when they differ. A figure without a tag receives no mark, and the reader knows that it has not been checked. The verifier tests only whether a figure matches its record. The reader still has to judge whether the record is the right one for the question.

## Readers with different backgrounds

The design used in the session appears in assistants that answer questions about reports, in interfaces to statistical databases, and in tools for searching the research literature. The session is relevant to the people who build such applications and also to the people who use them, supply their content, or review them.

Researchers and analysts who read answers from these applications can learn from the session which questions to ask of an answer: the evidence on which it is based, and whether its figures have been checked. Notebooks 02 and 03 are of particular relevance to people who maintain collections of documents or data, because they show how metadata, publication dates, and the division of documents into chunks affect the results of a search. For managers and reviewers who decide whether an application is ready for use, the components of the session provide a checklist for examining how a particular application works. The code in the notebooks uses widely adopted open-source libraries, namely LangChain, LangGraph, FastMCP, and Pydantic, and developers and data scientists can adapt it for their own applications.

## Readers with different levels of technical experience

Each notebook combines explanations in prose with code that demonstrates them. The explanations and the outputs published on this site can be read without running any code. Running the notebooks requires a Python environment such as Google Colab. An API key for a language model is optional. Without a key, the notebooks replay answers recorded from the model and carry out all other steps in full, including the search, the tool calls, and the verification.

## The notebooks

- **01 · From a language model to an application.** The same question is sent to a model twice, first without context and then with background information in the prompt, and the two answers are compared. The notebook also shows the complete prompt sent to the model and the cost of the added context in tokens.
- **02 · Preparing knowledge for an application.** A collection of eight documents is prepared for search. Metadata is recorded for each document, the documents are divided into chunks, and the chunks are converted into embeddings. A comparison with a badly prepared version of the same collection shows the effect of preparation on the results of a search.
- **03 · Retrieval-augmented generation.** The application searches the prepared collection for each question and gives the retrieved passages to the model. For one question an outdated passage ranks highly, and a filter on the publication date removes it from the ranking. A final test checks the search on eight questions without involving the model.
- **04 · Tools and structured outputs.** The model obtains exact figures by requesting a tool call, and the application runs the tool. Later sections cover a request for a year with no data and an answer returned in a fixed structure for use by other software. The last section offers the same tool through the Model Context Protocol (MCP).
- **05 · Workflows and agents.** The notebook builds the application first as a workflow, in which code fixes the order of the steps, and then as an agent, which lets the model choose the steps. Later sections show a question that the tools cannot answer, a failed tool call, and an agent stopped by a step limit. A table at the end compares the two designs.
- **06 · Verification.** The model tags each figure with its source record, and a verifier compares the tagged figures with the records. The notebook shows a verified figure, a flagged figure, and an untagged figure, and it describes the limits of the check.
- **07 · Lab: chat with your documents.** Participants build a chat application over a collection of their own choice with the components of the earlier notebooks. The notebook ends with a complete version of such an application built on Data360.
