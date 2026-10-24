# How AI chatbots work

This preface explains the problem addressed in session 01, the readers for whom that problem matters, and the three ideas developed in the notebooks. It ends with notes on running the notebooks and a short description of each one.

## The problem

A language model can write fluent text on almost any subject, and the text often contains figures such as percentages, totals, and years. The quality of the writing gives no information about the origin of those figures. A figure produced from training alone may be correct, out of date, or invented, and the answer gives the reader no means of finding out. Session 01 builds an application that bases its answers on identified evidence and checks each figure against its source.

## Who the session is for

Researchers and analysts receive figures from applications of this kind and need to know whether each figure can be traced to a source. The session shows them which questions to ask of an answer. People who maintain collections of documents or data affect what an application can find through the metadata, dates, and structure of their documents, and notebooks 02 and 03 show that effect. Managers and reviewers who decide whether an application is ready for use need to examine how it works. The session divides an application into components that can be examined one at a time. Developers and data scientists build the software that decides which material reaches the model, and the notebooks implement each pattern with LangChain, LangGraph, FastMCP, and Pydantic.

## The application around the model

An AI chatbot consists of a language model and the software around it, which the session calls the application. The application decides which material reaches the model with each question, and it processes the answer afterwards. The session builds such an application in stages and adds one component in each notebook.

### Grounding

An answer is grounded when its statements and figures come from evidence supplied by the application. In notebook 01, background information is written into the prompt. Notebooks 02 and 03 add passages retrieved from a collection of documents, and notebook 04 obtains exact figures from a database of indicators through a tool.

### Tracing failures to components

A wrong answer can usually be traced to one component, and the correction depends on which component failed. In notebook 03, a 2016 summary ranks highly for a question about the latest figures, because the similarity score ignores dates. A filter on the publication date corrects the ranking. Notebook 04 shows a request for data that do not exist. In notebook 05, a limit on the number of steps stops an agent that keeps calling tools.

### Separating writing from checking

A grounded answer can still contain a wrong figure, for instance a value altered in copying. In notebook 06, the model tags each figure with its source record, and a verifier written in ordinary code compares the figure with that record. The verifier marks a figure as verified when it matches and flags it when it differs. A figure without a tag receives no mark. The reader still judges whether a record is the right one for the question.

## Readers at different technical levels

Readers without programming experience can follow the session from the explanations and the outputs published on this site. The notebooks run in Google Colab or any Python environment. An API key for a language model is optional. Without a key, the notebooks replay recorded answers from the model and run all other steps in full.

## The notebooks

- **01 · From a language model to an application.** A model answers the same question without context and then with background information in the prompt, and the notebook measures the cost of that context in tokens.
- **02 · Preparing knowledge for an application.** Eight documents receive metadata and are divided into chunks and embedded. A badly prepared copy of the collection shows the effect of poor preparation on a search.
- **03 · Retrieval-augmented generation.** The application retrieves passages for each question and passes them to the model. A filter on dates removes an outdated passage, and a test checks the search on eight questions without the model.
- **04 · Tools and structured outputs.** The model requests exact figures through a tool, and the answer is returned in a fixed structure. The last section offers the tool through the Model Context Protocol (MCP).
- **05 · Workflows and agents.** The application is built once as a workflow, with the order of the steps fixed in code, and once as an agent that chooses its own steps. A table compares the two designs.
- **06 · Verification.** Each figure is tagged with its source record and checked by a verifier, and the notebook describes the limits of that check.
- **07 · Lab: chat with your documents.** Participants build a chat application over documents of their own choice.
