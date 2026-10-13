# Session 01 deck outline

Generated from `build_deck.js`. This file is the reviewable text of every slide; edit the script and regenerate both.

## 1. Building AI applications for research and data

- Training series · Session 01
- 90 minutes: one application built step by step, in six notebooks

Notes: Welcome. The session builds one application from a bare model to a verified, evaluated system, in six notebooks. Two audiences: those who will build, and those who commission or evaluate. Questions at the notebook transitions.

## 2. Objectives and format

- Two audiences: builders; commissioners and evaluators
- Afterwards: name the components; choose an architecture and the evidence; rerun the notebooks
- The one question: given a research or data problem, which architecture, and what evidence
- Format: 37 min slides, 53 min demonstration, questions; six notebooks; runs without a key

Notes: State the two audiences so that each knows what to listen for. The question at the right is what the session is for; return to it on the last slide.

## 3. The worked example

- The question (as above)
- Data: five WDI series, Kenya and Rwanda, 2010 to 2024, retrieved 13 October 2026
- Documents: four briefs, one definition note, one outdated summary, two distractors
- Planted problems: ambiguity, irrelevant and outdated retrieval, missing data, an unsupported claim

Notes: One question through the whole session. The planted problems are why each capability exists; name them now so that participants watch for them.

## 4. What makes a language model an application

- The model is one component; the others select, equip, check, and measure
- Six components: instructions and context; prepared knowledge; retrieval; tools and schemas; workflow or agent; verification and evaluation
- One notebook per component

Notes: This diagram returns on every section slide with the current stage highlighted. The table is the map of the session.

## 5. Instructions, context, and the question

- Three parts: instructions, context, question; kept separate
- Instructions shape behaviour and cannot supply data
- Demonstration: notebook 01, zero context against supplied context; missing information; the ambiguous question

Demonstration: Notebook 01, sections 2 to 5

Notes: Demonstrate notebook 01: the zero-context answer against the data (wrong and unsourced), then the same question with four briefs, then the missing-year question, then the ambiguous question with and without the instruction to ask.

## 6. Why context has to be selected

- Four briefs fit in one call; a collection does not
- Every token costs and dilutes
- Hence preparation and retrieval

Notes: The bridge from notebook 01 to 02 and 03. Keep it short.

## 7. Preparing knowledge: metadata, chunks, embeddings

- Pipeline: documents, metadata, chunks, embeddings, index
- Metadata is knowledge; chunk boundaries decide what can be found; embeddings place text by meaning
- Demonstration: notebook 02

Demonstration: Notebook 02, sections 2, 3, and 5

Notes: Show the 2016 document's header, then the fixed window that cuts mid-sentence against the section chunk, then the poorly prepared index. Notebooks 02 and 03 are demonstrated from one file.

## 8. Retrieval-augmented generation

- Pipeline: question, retriever, top-k passages, prompt, model, cited answer
- Gives: cited answers from a collection of any size
- Fails: outdated passage ranks first; retrieval always returns something
- Fixes on the retrieval side: metadata filter, recency weight, query reformulation, threshold plus instructions

Demonstration: Notebook 03, sections 1 to 4

Notes: Demonstrate the worked example through the pipeline, then the retrieval failure on the latest-figure question, then the three fixes. A faithful answer to the wrong evidence is wrong; the model cannot repair a bad selection.

## 9. Retrieval and generation are two problems

- Retrieval problem: evidence not in the hits; test without a model; fix chunking, metadata, ranking, query
- Generation problem: evidence present and unused; fix instructions or model
- The retrieval table from notebook 03

Demonstration: Notebook 03, section 5

Notes: Show the retrieval table. This separation is the first evaluation idea of the session and returns in notebook 06.

## 10. Tools: numbers come from data

- A tool: function plus description; the model fills parameters, the application runs it
- Memory against data: hedged approximation against a sourced value with a claim id
- Absence of data is a testable result
- Demonstration: notebook 04

Demonstration: Notebook 04, sections 1 to 4

Notes: Demonstrate the memory answer against the tool, the tool schema, the model calling the tool, and the missing year. Point at claim_id: notebook 06 uses it.

## 11. Structured outputs and MCP

- A schema makes the answer a contract: sent with the request, validated on return
- MCP: one server, many clients; the Data360 MCP server as the example
- Notebooks keep tools local; the shape is the same

Demonstration: Notebook 04, sections 5 and 6

Notes: Demonstrate the structured output and the validation error. Name MCP once with the Data360 server; the demonstration at the end shows it live.

## 12. Which technique solves which problem

- Table: problem, technique, notebook (seven rows)
- The techniques combine; few applications need a general agent

Notes: The slide to photograph. Say that the rows are complementary and that the next slides cover the last three.

## 13. Workflows and agents

- Fixed workflow: order in code, one model call, predictable
- Model-directed loop: model chooses tools; adapts, can err
- Controls: step limit, trace, errors as data
- Demonstration: notebook 05

Demonstration: Notebook 05, sections 1 to 5

Notes: Demonstrate the fixed workflow, then the loop on the same question, then the out-of-scope question, the failed call with retry, and the step limit.

## 14. Choosing between a workflow and an agent

- Table: calls per run, new questions, cost, testing, use when
- The worked example wants the fixed workflow
- Prefer the simplest design that completes the task reliably

Notes: For the commissioning audience: ask which of the two a proposal is, and why.

## 15. Verification: every number checked against its record

- Claim tags name the observation; a verifier compares under a policy
- Verified, flagged, unmarked; fail-closed
- What the check leaves to the reader
- Demonstration: notebook 06, sections 1 and 2

Demonstration: Notebook 06, sections 1 and 2

Notes: Demonstrate the clean answer with eight verified claims, then the planted answer with a flagged and an unmarked number.

## 16. Evaluation: the application measured across questions

- The suite: eight questions, expected evidence and behaviour, two configurations, rerun on every change
- Columns: retrieval, numbers, cited, verify
- q8 separates the configurations; B is verifiable, A is not

Demonstration: Notebook 06, sections 3 to 5

Notes: Show the two tables. A table like this is what distinguishes a demonstration from a measured application.

## 17. Four checks for four questions

- Retrieval: right evidence found; Grounding: answer written from it; Verification: numbers match records; Evaluation: behaviour across questions
- Plausible is not a check

Notes: The slide for the commissioning audience: four questions to ask of any application.

## 18. The same ideas in the application the program runs

- Data360 Chat: MCP tools, bounded loop with trace, claim tags and verifier, evaluation suite
- Lessons: catalog search misses, empty dimensions, the verifier's limits, the suite grows from use
- Live demonstration: five pretested queries

Demonstration: Data360 Chat, demo/data360-chat-demo.md

Notes: Five minutes. Point at the tool calls, the citation, and the mark beside the number. Name one limitation and how the suite catches it.

## 19. Recap

- Six notebooks, six lessons (table)

Notes: Read the lessons column. Then return to the question of slide 2.

## 20. Resources and questions

- Take home: notebooks, module, guide, glossary, evaluation examples, next steps, assets
- In the program: agents cookbook, evaluation cookbook, PCN, Data360 MCP, chatbot, reference implementation
- The question to leave with
- Questions: 15 to 30 minutes

Notes: Links are in resources/README.md. Close on the question.
